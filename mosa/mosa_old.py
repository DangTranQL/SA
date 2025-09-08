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

    def setup(self, param_names, bounds, func_names, objectives, alpha):
        self.param_names = param_names
        self.bounds = bounds
        self.func_names = func_names
        self.objectives = objectives
        self.alpha = alpha

        self.pareto_front = []
        self.func_1 = []
        self.func_2 = []
        self.params = {key: [] for key in self.param_names}

    def run(self, runs=1, use_tqdm=True):
        f1_list, f2_list = [], []
        bar = tqdm if use_tqdm else dummy_tqdm
        
        with bar(total=runs, desc="Runs", position=0) as outer_pbar:
            for _ in range(runs):
                temp = self.initial_temp
                pmax = 0
                last_percent = 0

                with bar(total=100, desc="Temperatures", unit='%', leave=False, position=1) as iter_pbar:
                    while temp >= self.final_temp:

                        with bar(total=self.num_iterations, desc="Iterations", leave=False, position=2) as iter_pbar_2:
                            for _ in range(self.num_iterations):
                                gamma = 1

                                vars_curr = {key: np.random.uniform(*self.bounds[key]) for key in self.param_names}
                                f_curr = self.objectives([vars_curr[param] for param in self.param_names])

                                self.pareto_front.append({
                                    'vars': {self.param_names[i]: vars_curr[self.param_names[i]] for i in range(len(self.param_names))},
                                    'f': f_curr.copy()
                                })

                                vars_new = {key: np.clip(vars_curr[key] + np.random.uniform(-self.step_size, self.step_size), *self.bounds[key]) for i, key in enumerate(self.param_names)}
                                f_new = self.objectives([vars_new[param] for param in self.param_names])

                                for key in f_new:
                                    if f_new[key] < f_curr[key]:
                                        pmax = p = 1
                                    else:
                                        p = np.exp(-(f_new[key] - f_curr[key]) / temp)
                                        if pmax < p:
                                            pmax = p    
                                    gamma *= p

                                gamma = self.alpha * pmax + (1 - self.alpha) * gamma

                                if gamma == 1 or gamma > random.random():
                                    vars_curr = vars_new
                                    f_curr = f_new.copy()

                                    self.pareto_front.append({
                                        'vars': {self.param_names[i]: vars_new[self.param_names[i]] for i in range(len(self.param_names))},
                                        'f': f_new.copy()
                                    })

                                iter_pbar_2.update(1)
                        
                        temp *= self.cooling_rate

                        percent_complete = (self.initial_temp - temp) / (self.initial_temp - self.final_temp) * 100
                        iter_pbar.update(percent_complete - last_percent)
                        last_percent = percent_complete

                f1_list.extend(d["f"][self.func_names[0]] for d in self.pareto_front)
                f2_list.extend(d["f"][self.func_names[1]] for d in self.pareto_front)

                for key in self.param_names:
                    self.params[key].extend(d["vars"][key] for d in self.pareto_front)

                self.pareto_front = []
                outer_pbar.update(1)

        self.func_1 = np.asarray(f1_list, dtype=float)
        self.func_2 = np.asarray(f2_list, dtype=float)

    def prune(self):
        mask = paretoset(np.column_stack((self.func_1, self.func_2)), sense=['min', 'min'])

        self.param_space = np.column_stack([np.array(v)[mask] for v in self.params.values()])
        self.pareto_front = np.column_stack((self.func_1[mask], self.func_2[mask]))

    def plot(self):
        fig = plt.figure(figsize=(20, 10))
        ax2d = fig.add_subplot(1, 2, 1)

        ax2d.scatter(self.pareto_front[:, 0], self.pareto_front[:, 1])
        ax2d.set_title("Pareto Plot")
        ax2d.set_xlabel(self.func_names[0])
        ax2d.set_ylabel(self.func_names[1])

        if len(self.param_names) == 2:
            ax2d2 = fig.add_subplot(1, 2, 2)
            ax2d2.scatter(self.param_space[:, 0], self.param_space[:, 1])
            ax2d2.set_title("Parameter Space")
            ax2d2.set_xlabel(self.param_names[0])
            ax2d2.set_ylabel(self.param_names[1])
        else:
            ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
            ax_3d.scatter(self.param_space[:, 0], self.param_space[:, 1], self.param_space[:, 2])
            ax_3d.set_xlabel(self.param_names[0])
            ax_3d.set_ylabel(self.param_names[1])
            ax_3d.set_zlabel(self.param_names[2])
            ax_3d.grid(True)
            ax_3d.set_box_aspect([1, 1, 1])

        plt.tight_layout()
        plt.show()

    def gd_igd(self, ref):
        ind = GD(ref)
        print("\nGeneration Distance = ", ind(self.pareto_front))