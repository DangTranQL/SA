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

class custom_mosa_old():
    def __init__(self, initial_temp=1000, final_temp=0.001, cooling_rate=0.95, num_iterations=1000, step_size=0.1):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.num_iterations = num_iterations
        self.step_size = step_size

    def setup(self, circuit, choice1, choice2, param_names, bounds, func_names, objectives, alpha):
        self.circuit = circuit
        self.choice1 = choice1
        self.choice2 = choice2
        self.param_names = param_names
        self.bounds = bounds
        self.func_names = func_names
        self.objectives = objectives
        self.alpha = alpha

        self.pareto_front = []
        self.func_1 = []
        self.func_2 = []
        self.params = {key: [] for key in self.param_names}
        self.gd = []
        self.igd = []
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

            self.param_space = np.column_stack([np.array(v)[mask] for v in self.params.values()])
            self.pareto_front = np.column_stack((self.func_1[mask], self.func_2[mask]))

    def gd_igd(self, r):
        if self.circuit == 'neg':
            ref = self.ref_neg
        else:
            ref = self.ref_posneg
        gd = GD(ref)
        igd = IGD(ref)
        if r != 200:
            self.gd.append(gd(self.plot_pareto))
            self.igd.append(igd(self.plot_pareto))
        else: 
            self.gd.append(gd(self.pareto_front))
            self.igd.append(igd(self.pareto_front))

    def run(self, runs=1, use_tqdm=True):
        bar = tqdm if use_tqdm else dummy_tqdm
        
        with bar(total=runs, desc="Runs", position=0) as outer_pbar:
            for r in range(1, runs+1):
                temp = self.initial_temp
                pmax = 0
                last_percent = 0

                vars_curr_list = [{key: np.random.uniform(*self.bounds[key]) for key in self.param_names} for _ in range(1000)]
                if self.circuit == 'neg':
                    f_curr_list = [self.objectives([vars_curr_list[i][param] for param in self.param_names]) for i in range(1000)]
                else:
                    f_curr_list = [self.objectives([vars_curr_list[i][param] for param in self.param_names], self.choice1, self.choice2) for i in range(1000)]
                self.pareto_front = [{'vars': {self.param_names[i]: vars_curr_list[j][self.param_names[i]] for i in range(len(self.param_names))}, 'f': f_curr_list[j].copy()} for j in range(1000)]

                with bar(total=100, desc="Temperatures", unit='%', leave=False, position=1) as iter_pbar:
                    while temp >= self.final_temp:

                        with bar(total=self.num_iterations, desc="Iterations", leave=False, position=2) as iter_pbar_2:
                            for i in range(self.num_iterations):
                                gamma = 1

                                vars_curr = self.pareto_front[i]['vars']
                                f_curr = self.pareto_front[i]['f']

                                vars_new = {key: np.clip(vars_curr[key] + np.random.uniform(-self.step_size, self.step_size), *self.bounds[key]) for i, key in enumerate(self.param_names)}
                                if self.circuit == 'neg':
                                    f_new = self.objectives([vars_new[param] for param in self.param_names])
                                else:
                                    f_new = self.objectives([vars_new[param] for param in self.param_names], self.choice1, self.choice2)

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
                                    self.pareto_front[i] = {
                                        'vars': {self.param_names[i]: vars_new[self.param_names[i]] for i in range(len(self.param_names))},
                                        'f': f_new.copy()
                                    }

                                iter_pbar_2.update(1)
                        
                        temp *= self.cooling_rate

                        percent_complete = (self.initial_temp - temp) / (self.initial_temp - self.final_temp) * 100
                        iter_pbar.update(percent_complete - last_percent)
                        last_percent = percent_complete

                self.func_1.extend(d["f"][self.func_names[0]] for d in self.pareto_front)
                self.func_2.extend(d["f"][self.func_names[1]] for d in self.pareto_front)

                for key in self.param_names:
                    self.params[key].extend(d["vars"][key] for d in self.pareto_front)

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
                    for key in self.param_names:
                        self.params[key] = np.array(self.params[key])[mask].tolist()

                self.prune(r)
                self.gd_igd(r)
                if r != 200:
                    if self.circuit == 'neg':   
                        np.savez(f'tests/mosa_random_neg_{r}.npz', f1=self.plot_pareto[:, 0], f2=self.plot_pareto[:, 1])
                    else:
                        np.savez(f'tests/mosa_random_posneg_{r}.npz', f1=self.plot_pareto[:, 0], f2=self.plot_pareto[:, 1])
                elif r == 200:
                    if self.circuit == 'neg':
                        np.savez(f'tests/mosa_random_neg_{r}.npz', f1=self.pareto_front[:, 0], f2=self.pareto_front[:, 1])
                    else:
                        np.savez(f'tests/mosa_random_posneg_{r}.npz', f1=self.pareto_front[:, 0], f2=self.pareto_front[:, 1])
                self.pareto_front = []
                outer_pbar.update(1)
        if self.circuit == 'neg':
            np.savez('tests/mosa_random_neg_gd_igd.npz', gd=self.gd, igd=self.igd)
        else:
            np.savez('tests/mosa_random_posneg_gd_igd.npz', gd=self.gd, igd=self.igd)

    def plot(self, time):
        run1 = np.load('point/mosa_random_neg_1.npz')
        run50 = np.load('point/mosa_random_neg_50.npz')
        run100 = np.load('point/mosa_random_neg_100.npz')
        run200 = np.load('point/mosa_random_neg_200.npz')

        fig = plt.figure(figsize=(20, 20))
        ax1 = fig.add_subplot(1, 2, 1)

        ax1.scatter(self.ref[:, 0], self.ref[:, 1], label="Grid Search", color='black')
        ax1.scatter(run1['f1'], run1['f2'], label='Run 1', color='blue')
        ax1.set_xlabel(self.func_names[0])
        ax1.set_ylabel(self.func_names[1])
        ax1.set_title("Pruned Objectives")

        comment_text = f'Running Time = {time}'
        fig.text(0.5, 0.02, comment_text, ha='center', va='bottom', fontsize=14, wrap=True)

        # Adjust layout to prevent text from overlapping with the plot
        plt.tight_layout(rect=[0, 0.05, 1, 1])

        if self.circuit == 'posneg':
            plt.savefig(f'out/random_{self.circuit}_{self.choice1}{self.choice2}.jpg')
        else:
            plt.savefig(f'out/random_{self.circuit}.jpg')

    # def plot(self, time):
    #     fig = plt.figure(figsize=(20, 20))

    #     # === Top row: Unpruned ===
    #     ax1 = fig.add_subplot(2, 2, 1)
    #     ax1.scatter(self.func_1, self.func_2)
    #     ax1.set_xlabel(self.func_names[0])
    #     ax1.set_ylabel(self.func_names[1])
    #     ax1.set_title("Unpruned Objectives")

    #     if len(self.param_names) == 2:
    #         ax2 = fig.add_subplot(2, 2, 2)
    #         ax2.scatter(self.params[self.param_names[0]], self.params[self.param_names[1]])
    #         ax2.set_xlabel(self.param_names[0])
    #         ax2.set_ylabel(self.param_names[1])
    #         ax2.set_title("Unpruned Parameter Space")
    #     else:
    #         ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    #         ax2.scatter(self.params[self.param_names[0]],
    #                     self.params[self.param_names[1]],
    #                     self.params[self.param_names[2]])
    #         ax2.set_xlabel(self.param_names[0])
    #         ax2.set_ylabel(self.param_names[1])
    #         ax2.set_zlabel(self.param_names[2])
    #         ax2.set_title("Unpruned Parameter Space")
    #         ax2.grid(True)
    #         ax2.set_box_aspect([1, 1, 1])

    #     # === Bottom row: Pruned (Pareto) ===
    #     ax3 = fig.add_subplot(2, 2, 3)
    #     ax3.scatter(self.pareto_front[:, 0], self.pareto_front[:, 1])
    #     ax3.set_xlabel(self.func_names[0])
    #     ax3.set_ylabel(self.func_names[1])
    #     ax3.set_title("Pruned Objectives")

    #     if len(self.param_names) == 2:
    #         ax4 = fig.add_subplot(2, 2, 4)
    #         ax4.scatter(self.param_space[:, 0], self.param_space[:, 1])
    #         ax4.set_xlabel(self.param_names[0])
    #         ax4.set_ylabel(self.param_names[1])
    #         ax4.set_title("Pruned Parameter Space")
    #     else:
    #         ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    #         ax4.scatter(self.param_space[:, 0],
    #                     self.param_space[:, 1],
    #                     self.param_space[:, 2])
    #         ax4.set_xlabel(self.param_names[0])
    #         ax4.set_ylabel(self.param_names[1])
    #         ax4.set_zlabel(self.param_names[2])
    #         ax4.set_title("Pruned Parameter Space")
    #         ax4.grid(True)
    #         ax4.set_box_aspect([1, 1, 1])

    #     comment_text = f'GD = {self.gd}, IGD = {self.igd}\n Running Time = {time}'
    #     fig.text(0.5, 0.02, comment_text, ha='center', va='bottom', fontsize=14, wrap=True)

    #     # Adjust layout to prevent text from overlapping with the plot
    #     plt.tight_layout(rect=[0, 0.05, 1, 1])

    #     if self.circuit == 'posneg':
    #         plt.savefig(f'out/random_{self.circuit}_{self.choice1}{self.choice2}.jpg')
    #     else:
    #         plt.savefig(f'out/random_{self.circuit}.jpg')
