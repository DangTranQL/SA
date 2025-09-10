import numpy as np
from tqdm import tqdm
import random
from paretoset import paretoset
from contextlib import contextmanager
import matplotlib.pyplot as plt
from pymoo.indicators.gd import GD
from pymoo.indicators.igd import IGD
import warnings

# NEW: space-filling & distance tools
from scipy.stats import qmc
from sklearn.neighbors import KDTree

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

    def setup(self, param_names, bounds, func_names, objectives, alpha):
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

    # =========================
    # Helpers (private methods)
    # =========================
    def _bounds_arrays(self):
        names = list(self.param_names)
        lows  = np.array([self.bounds[k][0] for k in names], float)
        highs = np.array([self.bounds[k][1] for k in names], float)
        return names, lows, highs

    def _dicts_from_X(self, X):
        # rows of X -> list of dicts using self.param_names ordering
        return [dict(zip(self.param_names, row)) for row in X]

    def _evaluate_batch(self, X):
        # Evaluate objectives for matrix X (n, d) -> (n,2) float array
        batch = self._dicts_from_X(X)
        F = []
        for b in batch:
            f = self.objectives([b[p] for p in self.param_names])  # expects dict
            F.append([f[self.func_names[0]], f[self.func_names[1]]])
        return np.asarray(F, float)

    def _sobol_batch(self, n):
        # Owen-scrambled Sobol, scaled to bounds
        names, lows, highs = self._bounds_arrays()
        d = len(names)
        eng = qmc.Sobol(d, scramble=True)
        m = int(np.ceil(np.log2(max(2, n))))
        X01 = eng.random_base2(m)[:n]
        return qmc.scale(X01, lows, highs)

    def _farthest_points(self, n, X_existing, cand_mult=20000):
        # Sample many Sobol candidates, pick those farthest from existing archive
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

    def _local_refinement(self, n, X_pareto, weights, step=0.05):
        # Gaussian jitter around Pareto points; weights ~ crowding distance
        names, lows, highs = self._bounds_arrays()
        if X_pareto is None or len(X_pareto) == 0:
            return self._sobol_batch(n)

        rng = np.random.default_rng()
        w = np.asarray(weights, float)
        w = w / (w.sum() + 1e-12)
        idx = rng.choice(len(X_pareto), size=n, p=w)
        centers = X_pareto[idx]
        sigma = step * (highs - lows)
        X = rng.normal(loc=centers, scale=sigma)
        return np.clip(X, lows, highs)

    def _nondominated_mask(self, F):
        # Fast non-dominated mask in 2D (minimization)
        n = F.shape[0]
        order = np.argsort(F[:, 0], kind='mergesort')  # stable tie-break
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
        # Standard crowding distance for 2D (finite, positive)
        n = len(F_front)
        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([1.0])
        D = np.zeros(n, float)
        for k in range(F_front.shape[1]):
            idx = np.argsort(F_front[:, k])
            D[idx[0]] = D[idx[-1]] = np.inf
            f = F_front[idx, k]
            fmin, fmax = f[0], f[-1]
            denom = (fmax - fmin) if fmax > fmin else 1.0
            for i in range(1, n - 1):
                D[idx[i]] += (f[i + 1] - f[i - 1]) / denom
        # replace inf with large finite for weighting
        inf_mask = ~np.isfinite(D)
        if inf_mask.any():
            finite_max = np.max(D[~inf_mask]) if (~inf_mask).any() else 1.0
            D[inf_mask] = 10.0 * max(1.0, finite_max)
        return np.maximum(D, 1e-12)

    def _apply_badmask(self, F, X):
        # Drop rows with sentinel or non-finite values
        bad = (F[:, 0] == 1e6) | (F[:, 1] == 1e6) | ~np.isfinite(F).all(axis=1)
        return F[~bad], X[~bad]

    # =========================
    # NEW: adaptive, space-filling runner
    # =========================
    def run_adaptive(self, runs=3, batch_size=1000, explore_frac=0.5, use_tqdm=True, refine_step=0.05):
        """
        Batch-iterative sampling:
          - Run 0: Sobol (space-filling) batch
          - Later runs: explore (farthest) + exploit (local refinement around Pareto)
        Keeps self.func_1, self.func_2, self.params aligned for your plotting & pruning.
        """
        bar = tqdm if use_tqdm else dummy_tqdm

        # reset archives
        self.func_1 = []
        self.func_2 = []
        self.params = {key: [] for key in self.param_names}

        X_all = np.empty((0, len(self.param_names)), float)
        F_all = np.empty((0, 2), float)

        with bar(total=runs, desc="Runs", position=0) as outer:
            for r in range(runs):
                if r == 0:
                    X = self._sobol_batch(batch_size)
                else:
                    # current Pareto from archive
                    mask_nd = self._nondominated_mask(F_all)
                    X_pareto = X_all[mask_nd]
                    F_pareto = F_all[mask_nd]
                    cd = self._crowding_distance(F_pareto)

                    n_explore = int(round(batch_size * explore_frac))
                    n_exploit = batch_size - n_explore

                    X_exp  = self._farthest_points(n_explore, X_all, cand_mult=max(20000, 20 * batch_size))
                    X_expl = self._local_refinement(n_exploit, X_pareto, cd, step=refine_step)
                    X = np.vstack([X_exp, X_expl])

                # evaluate
                F = self._evaluate_batch(X)
                F, X = self._apply_badmask(F, X)  # drop 1e6/non-finite rows

                # merge into archives
                X_all = np.vstack([X_all, X]) if len(X_all) else X
                F_all = np.vstack([F_all, F]) if len(F_all) else F

                outer.update(1)

        # finalize class fields
        self.func_1 = F_all[:, 0].astype(float)
        self.func_2 = F_all[:, 1].astype(float)
        for j, name in enumerate(self.param_names):
            self.params[name] = X_all[:, j].tolist()

    # =========================
    # PRUNING & PLOTTING
    # =========================
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

    def plot(self):
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

        plt.tight_layout()
        plt.show()

    def gd_igd(self, ref):
        ind = GD(ref)
        print("\nGeneration Distance = ", ind(self.pareto_front))
