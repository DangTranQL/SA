import numpy as np
from tqdm import tqdm
import random

class custom_mosa():
    def __init__(self, initial_temp=300, num_temps=100, cooling_rate=0.95, num_iterations=500, step_size=0.05):
        self.initial_temp = initial_temp
        self.num_temps = num_temps
        self.cooling_rate = cooling_rate
        self.num_iterations = num_iterations
        self.step_size = step_size
        self.pareto_front = []

    def setup(self, params, bounds, objectives, alpha):
        self.params = params
        self.bounds = bounds
        self.objectives = objectives
        self.alpha = alpha

    def run(self, vars_ini, outer_pbar=None):
        temp = self.initial_temp
        temps = [temp * (self.cooling_rate ** i) for i in range(self.num_temps)]

        vars_curr = {key: np.random.uniform(*self.bounds[key]) for key in self.params}
        # vars_curr = {key: vars_ini[key] for i, key in enumerate(self.params)}
        f_curr = self.objectives([vars_curr[param] for param in self.params])
        pareto = [{
            'vars': {self.params[i]: vars_curr[self.params[i]] for i in range(len(self.params))},
            'f': f_curr.copy()
        }]

        with tqdm(total=len(temps), desc="Temperatures", leave=False, position=1) as iter_pbar:
            for temp in temps:
                p_list = []

                with tqdm(total=self.num_iterations, desc="Iterations", leave=False, position=2) as iter_pbar_2:
                    for _ in range(self.num_iterations):
                        vars_new = {key: np.clip(vars_curr[key] + np.random.uniform(-self.step_size, self.step_size), *self.bounds[key]) for i, key in enumerate(self.params)}
                        f_new = self.objectives([vars_new[param] for param in self.params])

                        delta_f = np.sum([f_new[k] - f_curr[k] for k in f_new])

                        p = 1 if delta_f < 0 else np.exp(-delta_f / temp)

                        if p != 1:
                            p_list.append(p)

                        iter_pbar_2.update(1)

                p_comp = self.alpha * min(1, np.prod(p_list) if len(p_list) else 1) + (1-self.alpha) * min(1, max(p_list) if len(p_list) else 1)

                if p_comp == 1 or p_comp > random.random():
                    accept = True
                else:
                    accept = False

                if accept:
                    vars_curr = vars_new
                    f_curr = f_new.copy()

                    pareto = self._update_pareto(
                        pareto,
                        {self.params[i]: vars_new[self.params[i]] for i in range(len(self.params))},
                        f_new.copy()
                    )
                        # temp *= self.cooling_rate

                iter_pbar.update(1)
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