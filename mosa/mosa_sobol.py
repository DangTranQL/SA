from scipy.stats import qmc
from sklearn.neighbors import KDTree
import numpy as np
from tqdm import tqdm
import random
from paretoset import paretoset
from contextlib import contextmanager
import matplotlib.pyplot as plt
from pymoo.indicators.gd import GD
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import HV
import warnings

warnings.filterwarnings("ignore")

@contextmanager
def dummy_tqdm(*args, **kwargs):
    class Dummy:
        def update(self, *a, **kw): pass
        def close(self): pass
    yield Dummy()

class custom_mosa():
    def __init__(self, initial_temp=1000, final_temp=0.01, cooling_rate=0.95, num_iterations=10, step_size=0.1):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.num_iterations = num_iterations
        self.step_size = step_size

    def setup(self, circuit, choice1, choice2, param_names, bounds, func_names, objectives, alpha=0.5):
        self.circuit = circuit
        self.choice1 = choice1
        self.choice2 = choice2
        self.param_names = param_names
        self.bounds = bounds
        self.func_names = func_names
        self.objectives = objectives
        self.alpha = alpha

        # archives for unpruned
        self.func_1 = []
        self.func_2 = []

        # pruned (set by prune())
        self.pareto_front = None
        self.gd = []
        self.igd = []
        self.hv = []
        self.hv_true = None
        self.ref_neg = np.load('data/ARneg_SensitivityPareto.npy')
        self.ref_posneg = np.load('data/PosNegFbGridSearchData_renamed.npz')[f'paretoset_SensPair_{self.choice1}_{self.choice2}']

        self.plot_pareto = None
        self.plot_f1 = None
        self.plot_f2 = None

    def prune(self, r):
        if r != 200:
            mask = paretoset(np.column_stack((self.plot_f1, self.plot_f2)), sense=['min', 'min'])

        # self.param_space = np.column_stack([np.array(v)[mask] for v in self.params.values()])
            self.plot_pareto = np.column_stack((self.plot_f1[mask], self.plot_f2[mask]))
        
        else:
            mask = paretoset(np.column_stack((self.func_1, self.func_2)), sense=['min', 'min'])

            self.pareto_front = np.column_stack((self.func_1[mask], self.func_2[mask]))

    def gd_igd_hv(self, r):
        if self.circuit == 'neg':
            ref = self.ref_neg
        else:
            ref = self.ref_posneg
        gd = GD(ref)
        igd = IGD(ref)

        buffer = 0.1
        t_min = ref.min(axis=0)
        t_max = ref.max(axis=0)
        hv_point = t_max + buffer*(t_max - t_min)
        hv = HV(ref_point=hv_point)
        self.hv_true = hv(ref)

        if r != 200:
            self.gd.append(gd(self.plot_pareto))
            self.igd.append(igd(self.plot_pareto))
            self.hv.append(hv(self.plot_pareto))
        else: 
            self.gd.append(gd(self.pareto_front))
            self.igd.append(igd(self.pareto_front))
            self.hv.append(hv(self.pareto_front))
    
    def _bounds_arrays(self):
        names = list(self.param_names)
        lows  = np.array([self.bounds[k][0] for k in names], float)
        highs = np.array([self.bounds[k][1] for k in names], float)
        return names, lows, highs

    def _sobol_batch(self, n):
        names, lows, highs = self._bounds_arrays()
        d = len(names)
        eng = qmc.Sobol(d, scramble=True)
        m = int(np.ceil(np.log2(max(2, n))))
        X01 = eng.random_base2(m)[:n]
        return qmc.scale(X01, lows, highs)

    def _farthest_points(self, n, X_existing, cand_mult=20000):
        names, lows, highs = self._bounds_arrays()
        d = len(names)
        eng = qmc.Sobol(d, scramble=True)
        m = int(np.ceil(np.log2(max(2, cand_mult))))
        C = qmc.scale(eng.random_base2(m)[:cand_mult], lows, highs)
        if X_existing is None or len(X_existing) == 0:
            return C[:n]
        tree = KDTree(X_existing)
        dmin, _ = tree.query(C, k=1)
        idx = np.argsort(dmin.ravel())[::-1][:n]
        return C[idx]

    def _nondominated_mask(self, F):
        # 2D, minimization
        n = F.shape[0]
        order = np.argsort(F[:, 0], kind='mergesort')
        F_sorted = F[order]
        mask_sorted = np.ones(n, dtype=bool)
        best_f2 = np.inf
        for i in range(n):
            f2 = F_sorted[i, 1]
            if f2 < best_f2 - 1e-15:
                best_f2 = f2
            else:
                mask_sorted[i] = False
        mask = np.zeros(n, dtype=bool)
        mask[order] = mask_sorted
        return mask

    def _crowding_distance(self, F_front):
        n = len(F_front)
        if n == 0: return np.array([])
        if n == 1: return np.array([1.0])
        D = np.zeros(n, float)
        for k in range(F_front.shape[1]):
            idx = np.argsort(F_front[:, k])
            D[idx[0]] = D[idx[-1]] = np.inf
            f = F_front[idx, k]
            den = (f[-1] - f[0]) if f[-1] > f[0] else 1.0
            for i in range(1, n-1):
                D[idx[i]] += (f[i+1] - f[i-1]) / den
        inf_mask = ~np.isfinite(D)
        if inf_mask.any():
            finite_max = np.max(D[~inf_mask]) if (~inf_mask).any() else 1.0
            D[inf_mask] = 10.0 * max(1.0, finite_max)
        return np.maximum(D, 1e-12)
    
    def run_mosa_adaptive(self, runs=3, batch_size=32, explore_frac=0.5, refine_step=0.05, use_tqdm=True):
        """
        MOSA with adaptive seeding:
        - Run 0: Sobol seeds (space-filling).
        - Runs >=1: exploration (farthest from archive) + exploitation (around Pareto, weighted by crowding distance).
        Each seed is evolved by the MOSA kernel (your acceptance rule).
        Archives are accumulated in self.func_1, self.func_2, and self.params[*].
        """
        bar = tqdm if use_tqdm else dummy_tqdm

        # reset archives
        self.func_1, self.func_2 = [], []

        X_all = np.empty((0, len(self.param_names)), float)
        F_all = np.empty((0, 2), float)

        with bar(total=runs, desc="Runs", position=0) as outer:
            for r in range(1, runs+1):
                # ---- choose starting points for this run ----
                if r == 1:
                    X0 = self._sobol_batch(batch_size)
                else:
                    # current Pareto + crowding distance
                    mask_nd = self._nondominated_mask(F_all)
                    X_pareto = X_all[mask_nd]
                    F_pareto = F_all[mask_nd]
                    cd = self._crowding_distance(F_pareto)

                    n_exp = int(round(batch_size * explore_frac))
                    n_expl = batch_size - n_exp
                    X_exp  = self._farthest_points(n_exp, X_all, cand_mult=20000)

                    # local refinement: Gaussian jitter around Pareto points (weighted by crowding distance)
                    if len(X_pareto) == 0:
                        X_expl = self._sobol_batch(n_expl)
                    else:
                        names, lows, highs = self._bounds_arrays()
                        rng = np.random.default_rng()
                        probs = cd / (cd.sum() + 1e-12)
                        idx = rng.choice(len(X_pareto), size=n_expl, p=probs)
                        centers = X_pareto[idx]
                        sigma = refine_step * (highs - lows)
                        X_expl = np.clip(rng.normal(loc=centers, scale=sigma), lows, highs)

                    X0 = np.vstack([X_exp, X_expl])

                # ---- build initial population (dicts + objective dicts) ----
                starts = [dict(zip(self.param_names, row)) for row in X0]
                if self.circuit == 'neg':
                    fstarts = [self.objectives([d[p] for p in self.param_names]) for d in starts]
                else:
                    fstarts = [self.objectives([d[p] for p in self.param_names], self.choice1, self.choice2) for d in starts]

                pop = [{'vars': {name: starts[j][name] for name in self.param_names},
                        'f': fstarts[j].copy()} for j in range(len(starts))]

                # ---- MOSA kernel (your acceptance rule) ----
                temp = self.initial_temp
                last_percent = 0.0
                with bar(total=100, desc="Temperatures", unit='%', leave=False, position=1) as temp_pbar:
                    while temp >= self.final_temp:
                        # if temp > 1:
                        #     self.step_size = 0.5
                        # else:
                        #     self.step_size = 0.05
                        with bar(total=self.num_iterations, desc="Iterations", leave=False, position=2) as iter_bar:
                            for _ in range(self.num_iterations):
                                with bar(total=len(pop), desc="Population", leave=False, position=3) as pop_bar:
                                    for idx in range(len(pop)):
                                        # idx = it % len(pop)  # safe even if num_iterations > population size
                                        vars_curr = pop[idx]['vars']
                                        f_curr    = pop[idx]['f']

                                        # propose neighbor (same step rule as your original code)
                                        vars_new = {}
                                        for name in self.param_names:
                                            lo, hi = self.bounds[name]
                                            vars_new[name] = np.clip(
                                                vars_curr[name] + np.random.uniform(-self.step_size, self.step_size),
                                                lo, hi
                                            )
                                        if self.circuit == 'neg':
                                            f_new = self.objectives([vars_new[p] for p in self.param_names])
                                        else:
                                            f_new = self.objectives([vars_new[p] for p in self.param_names], self.choice1, self.choice2)

                                        # multi-objective annealing acceptance
                                        gamma = 1.0
                                        pmax = 0.0
                                        for key in f_new:
                                            if f_new[key] < f_curr[key]:
                                                p = 1.0
                                            else:
                                                p = np.exp(-(f_new[key] - f_curr[key]) / temp)
                                            if p > pmax: 
                                                pmax = p
                                            gamma *= p
                                        gamma = self.alpha * pmax + (1.0 - self.alpha) * gamma

                                        if (gamma == 1.0) or (gamma > random.random()):
                                            pop[idx] = {'vars': {n: vars_new[n] for n in self.param_names},
                                                        'f': f_new.copy()}

                                        pop_bar.update(1)

                                iter_bar.update(1)

                        temp *= self.cooling_rate
                        percent = (self.initial_temp - temp) / (self.initial_temp - self.final_temp) * 100.0
                        temp_pbar.update(percent - last_percent)
                        last_percent = percent

                # ---- archive this run ----
                F = np.array([[ind['f'][self.func_names[0]], ind['f'][self.func_names[1]]] for ind in pop], float)
                X = np.array([[ind['vars'][n] for n in self.param_names] for ind in pop], float)

                # filter sentinels / non-finite
                good = (F[:,0] != 1e6) & (F[:,1] != 1e6) & np.isfinite(F).all(axis=1)
                F = F[good]; X = X[good]

                X_all = np.vstack([X_all, X]) if X_all.size else X
                F_all = np.vstack([F_all, F]) if F_all.size else F

                self.func_1.extend(F[:,0].tolist())
                self.func_2.extend(F[:,1].tolist())

                if r != 200:
                    self.plot_f1 = np.asarray(self.func_1, dtype=float)
                    self.plot_f2 = np.asarray(self.func_2, dtype=float)

                    mask = (self.plot_f1 != 1e6) & (self.plot_f2 != 1e6)
                    self.plot_f1 = self.plot_f1[mask]
                    self.plot_f2 = self.plot_f2[mask]
                
                else:
                    self.func_1 = np.asarray(self.func_1, dtype=float)
                    self.func_2 = np.asarray(self.func_2, dtype=float)

                    mask = (self.func_1 != 1e6) & (self.func_2 != 1e6)
                    self.func_1 = self.func_1[mask]
                    self.func_2 = self.func_2[mask]

                self.prune(r)
                self.gd_igd_hv(r)
                if r%5 == 0:
                    if self.circuit == 'neg':   
                        np.savez(f'batch32_iter10/mosa_sobol_neg_{r}.npz', f1=self.plot_pareto[:, 0], f2=self.plot_pareto[:, 1])
                    else:
                        np.savez(f'batch32_iter10/mosa_sobol_posneg_{r}.npz', f1=self.plot_pareto[:, 0], f2=self.plot_pareto[:, 1])
                elif r == 200:
                    if self.circuit == 'neg':
                        np.savez(f'batch32_iter10/mosa_sobol_neg_{r}.npz', f1=self.pareto_front[:, 0], f2=self.pareto_front[:, 1])
                    else:
                        np.savez(f'batch32_iter10/mosa_sobol_posneg_{r}.npz', f1=self.pareto_front[:, 0], f2=self.pareto_front[:, 1])

                outer.update(1)

                # print(f"Finish Run {r}\n")

        if self.circuit == 'neg':
            np.savez('batch32_iter10/mosa_sobol_neg_metrics.npz', gd=self.gd, igd=self.igd, hv=self.hv)
        else:
            np.savez('batch32_iter10/mosa_sobol_posneg_metrics.npz', gd=self.gd, igd=self.igd, hv=self.hv)
