import numpy as np
from tqdm import tqdm
import random

class custom_mosa():
    def __init__(self, initial_temp=1000, final_temp=0.001, cooling_rate=0.95, num_iterations=1000, step_size=0.2):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.num_iterations = num_iterations
        self.step_size = step_size
        self.pareto_front = []

    def setup(self, params, bounds, objectives, alpha):
        self.params = params
        self.bounds = bounds
        self.objectives = objectives
        self.alpha = alpha

    def run(self, outer_pbar=None):
        temp = self.initial_temp
        # temps = [temp * (self.cooling_rate ** i) for i in range(self.num_temps)]

        pmax = 0

        last_percent = 0

        with tqdm(total=100, desc="Temperatures", unit='%', leave=False, position=1) as iter_pbar:
            while temp >= self.final_temp:
                # for temp in temps:

                vars_curr = {key: np.random.uniform(*self.bounds[key]) for key in self.params}
                # vars_curr = {key: vars_ini[key] for i, key in enumerate(self.params)}
                f_curr = self.objectives([vars_curr[param] for param in self.params])
                pareto = [{
                    'vars': {self.params[i]: vars_curr[self.params[i]] for i in range(len(self.params))},
                    'f': f_curr.copy()
                }]

                with tqdm(total=self.num_iterations, desc="Iterations", leave=False, position=2) as iter_pbar_2:
                    for _ in range(self.num_iterations):
                        gamma = 1

                        vars_new = {key: np.clip(vars_curr[key] + np.random.uniform(-self.step_size, self.step_size), *self.bounds[key]) for i, key in enumerate(self.params)}
                        f_new = self.objectives([vars_new[param] for param in self.params])

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

                            # pareto = self._update_pareto(
                            #     pareto,
                            #     {self.params[i]: vars_new[self.params[i]] for i in range(len(self.params))},
                            #     f_new.copy()
                            # )
                            pareto.append({
                                'vars': {self.params[i]: vars_new[self.params[i]] for i in range(len(self.params))},
                                'f': f_new.copy()
                            })

                        iter_pbar_2.update(1)
                
                temp *= self.cooling_rate

                percent_complete = (self.initial_temp - temp) / (self.initial_temp - self.final_temp) * 100
                iter_pbar.update(percent_complete - last_percent)
                last_percent = percent_complete

        self.pareto_front = pareto

    def _dominates(self, f1, f2):
        better_or_equal = all(f1[k] <= f2[k] for k in f1)
        strictly_better = any(f1[k] < f2[k] for k in f1)
        return better_or_equal and strictly_better

    def _update_pareto(self, front, vars_dict, f_dict):
        candidate = {
            'vars': vars_dict.copy(),
            'f': f_dict.copy()
        }
        new_front = []
        dominated = False

        for point in front:
            if self._dominates(candidate['f'], point['f']):
                continue
            elif self._dominates(point['f'], candidate['f']):
                dominated = True
                new_front.append(point)

        if not dominated:
            new_front.append(candidate)

        return new_front