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
import warnings

warnings.filterwarnings("ignore")

@contextmanager
def dummy_tqdm(*args, **kwargs):
    class Dummy:
        def update(self, *a, **kw): pass
        def close(self): pass
    yield Dummy()

class custom_mosa():
    def __init__(self, initial_temp=1000, final_temp=0.001, cooling_rate=0.95, num_iterations=1000, step_size=0.1):
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
        self.params = {key: [] for key in self.param_names}

        # pruned (set by prune())
        self.pareto_front = None
        self.param_space = None
        self.gd = None
        self.igd = None
    
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
    
    def run_mosa_adaptive(self, runs=3, batch_size=1000, explore_frac=0.5, refine_step=0.05, use_tqdm=False):
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
        self.params = {k: [] for k in self.param_names}

        X_all = np.empty((0, len(self.param_names)), float)
        F_all = np.empty((0, 2), float)

        with bar(total=runs, desc="Runs", position=0) as outer:
            for r in range(runs):
                # ---- choose starting points for this run ----
                if r == 0:
                    X0 = self._sobol_batch(batch_size)
                else:
                    # current Pareto + crowding distance
                    mask_nd = self._nondominated_mask(F_all)
                    X_pareto = X_all[mask_nd]
                    F_pareto = F_all[mask_nd]
                    cd = self._crowding_distance(F_pareto)

                    n_exp = int(round(batch_size * explore_frac))
                    n_expl = batch_size - n_exp
                    X_exp  = self._farthest_points(n_exp, X_all, cand_mult=max(20000, 20*batch_size))

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
                fstarts = [self.objectives([d[p] for p in self.param_names], self.choice1, self.choice2) for d in starts]

                pop = [{'vars': {name: starts[j][name] for name in self.param_names},
                        'f': fstarts[j].copy()} for j in range(len(starts))]

                # ---- MOSA kernel (your acceptance rule) ----
                temp = self.initial_temp
                last_percent = 0.0
                with bar(total=100, desc="Temperatures", unit='%', leave=False, position=1) as temp_pbar:
                    while temp >= self.final_temp:
                        with bar(total=self.num_iterations, desc="Iterations", leave=False, position=2) as iter_pbar:
                            for it in range(self.num_iterations):
                                idx = it % len(pop)  # safe even if num_iterations > population size
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
                                f_new = self.objectives([vars_new[p] for p in self.param_names], self.choice1, self.choice2)

                                # multi-objective annealing acceptance
                                gamma = 1.0
                                pmax = 0.0
                                for key in f_new:
                                    if f_new[key] < f_curr[key]:
                                        p = 1.0
                                    else:
                                        p = np.exp(-(f_new[key] - f_curr[key]) / temp)
                                    if p > pmax: pmax = p
                                    gamma *= p
                                gamma = self.alpha * pmax + (1.0 - self.alpha) * gamma

                                if (gamma == 1.0) or (gamma > random.random()):
                                    pop[idx] = {'vars': {n: vars_new[n] for n in self.param_names},
                                                'f': f_new.copy()}

                                iter_pbar.update(1)

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
                for j, name in enumerate(self.param_names):
                    self.params[name].extend(X[:, j].tolist())

                outer.update(1)

        # finalize archive arrays
        self.func_1 = np.asarray(self.func_1, float)
        self.func_2 = np.asarray(self.func_2, float)

    def prune(self):
        # compute non-dominated set from unpruned archives
        F = np.column_stack((np.asarray(self.func_1, float), np.asarray(self.func_2, float)))
        mask = paretoset(F, sense=['min', 'min'])

        # pruned arrays, shape (k, d) and (k, 2)
        self.param_space = np.column_stack([np.asarray(self.params[k], float)[mask] for k in self.param_names])
        self.pareto_front = F[mask]

    def pruned_plot(self):
        if self.pareto_front is None or self.param_space is None:
            raise RuntimeError("Call prune() before pruned_plot().")

        fig = plt.figure(figsize=(20, 10))
        ax2d = fig.add_subplot(1, 2, 1)
        ax2d.scatter(self.pareto_front[:, 0], self.pareto_front[:, 1])
        ax2d.set_title("Pruned Objectives")
        ax2d.set_xlabel(self.func_names[0])
        ax2d.set_ylabel(self.func_names[1])

        if len(self.param_names) == 2:
            ax2d2 = fig.add_subplot(1, 2, 2)
            ax2d2.scatter(self.param_space[:, 0], self.param_space[:, 1])
            ax2d2.set_title("Pruned Parameter Space")
            ax2d2.set_xlabel(self.param_names[0])
            ax2d2.set_ylabel(self.param_names[1])
        else:
            ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
            ax_3d.scatter(self.param_space[:, 0], self.param_space[:, 1], self.param_space[:, 2])
            ax_3d.set_xlabel(self.param_names[0])
            ax_3d.set_ylabel(self.param_names[1])
            ax_3d.set_zlabel(self.param_names[2])
            ax_3d.set_title("Pruned Parameter Space")
            ax_3d.grid(True)
            ax_3d.set_box_aspect([1, 1, 1])

        plt.tight_layout()
        plt.show()

    def unpruned_plot(self):
        if len(self.func_1) == 0:
            raise RuntimeError("No data to plot. Run run() or run_adaptive() first.")

        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.scatter(self.func_1, self.func_2)
        ax1.set_xlabel(self.func_names[0])
        ax1.set_ylabel(self.func_names[1])
        ax1.set_title("Unpruned Objectives")

        if len(self.param_names) == 2:
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.scatter(self.params[self.param_names[0]], self.params[self.param_names[1]])
            ax2.set_title("Unpruned Parameter Space")
            ax2.set_xlabel(self.param_names[0])
            ax2.set_ylabel(self.param_names[1])
        else:
            ax2_3d = fig.add_subplot(1, 2, 2, projection='3d')
            ax2_3d.scatter(self.params[self.param_names[0]], self.params[self.param_names[1]], self.params[self.param_names[2]])
            ax2_3d.set_xlabel(self.param_names[0])
            ax2_3d.set_ylabel(self.param_names[1])
            ax2_3d.set_zlabel(self.param_names[2])
            ax2_3d.set_title("Unpruned Parameter Space")
            ax2_3d.grid(True)
            ax2_3d.set_box_aspect([1, 1, 1])

        plt.tight_layout()
        plt.show()

    def plot(self, time):
        if self.pareto_front is None or self.param_space is None:
            raise RuntimeError("Call prune() before plot().")

        fig = plt.figure(figsize=(20, 20))

        # === Top row: Unpruned ===
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.scatter(self.func_1, self.func_2)
        ax1.set_xlabel(self.func_names[0])
        ax1.set_ylabel(self.func_names[1])
        ax1.set_title("Unpruned Objectives")

        if len(self.param_names) == 2:
            ax2 = fig.add_subplot(2, 2, 2)
            ax2.scatter(self.params[self.param_names[0]], self.params[self.param_names[1]])
            ax2.set_xlabel(self.param_names[0])
            ax2.set_ylabel(self.param_names[1])
            ax2.set_title("Unpruned Parameter Space")
        else:
            ax2 = fig.add_subplot(2, 2, 2, projection='3d')
            ax2.scatter(self.params[self.param_names[0]],
                        self.params[self.param_names[1]],
                        self.params[self.param_names[2]])
            ax2.set_xlabel(self.param_names[0])
            ax2.set_ylabel(self.param_names[1])
            ax2.set_zlabel(self.param_names[2])
            ax2.set_title("Unpruned Parameter Space")
            ax2.grid(True)
            ax2.set_box_aspect([1, 1, 1])

        # === Bottom row: Pruned (Pareto) ===
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.scatter(self.pareto_front[:, 0], self.pareto_front[:, 1])
        ax3.set_xlabel(self.func_names[0])
        ax3.set_ylabel(self.func_names[1])
        ax3.set_title("Pruned Objectives")

        if len(self.param_names) == 2:
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.scatter(self.param_space[:, 0], self.param_space[:, 1])
            ax4.set_xlabel(self.param_names[0])
            ax4.set_ylabel(self.param_names[1])
            ax4.set_title("Pruned Parameter Space")
        else:
            ax4 = fig.add_subplot(2, 2, 4, projection='3d')
            ax4.scatter(self.param_space[:, 0],
                        self.param_space[:, 1],
                        self.param_space[:, 2])
            ax4.set_xlabel(self.param_names[0])
            ax4.set_ylabel(self.param_names[1])
            ax4.set_zlabel(self.param_names[2])
            ax4.set_title("Pruned Parameter Space")
            ax4.grid(True)
            ax4.set_box_aspect([1, 1, 1])

        comment_text = f'GD = {self.gd}, IGD = {self.igd}\n Running Time = {time}'
        fig.text(0.5, 0.02, comment_text, ha='center', va='bottom', fontsize=14, wrap=True)

        # Adjust layout to prevent text from overlapping with the plot
        plt.tight_layout(rect=[0, 0.05, 1, 1])

        if self.circuit == 'posneg':
            plt.savefig(f'out/mosa_{self.circuit}_{self.choice1}{self.choice2}.jpg')
        else:
            plt.savefig(f'out/mosa_{self.circuit}.jpg')

    def gd_igd(self, ref):
        gd = GD(ref)
        igd = IGD(ref)
        self.gd = gd(self.pareto_front)
        self.igd = igd(self.pareto_front)