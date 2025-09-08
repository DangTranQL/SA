import random
from tqdm import tqdm
import numpy as np
from openmdao.core.driver import Driver

class MOSADriver(Driver):
    def __init__(self, params):
        super().__init__()
        self.num_iterations = 1000
        self.initial_temp = 1000.0
        # self.final_temp = 1e-6
        self.cooling_rate = 0.95
        self.step_size = 0.03
        self.pareto_front = []
        self.params = params

    def setup(self):
        pass

    def run(self, outer_pbar=None):
        temp = self.initial_temp

        vars_bound = []
        vars_current = []

        for param in self.params:
            vars_bound.append((self._designvars[param]['lower'], self._designvars[param]['upper']))
            vars_current.append(random.uniform(self._designvars[param]['lower'], self._designvars[param]['upper']))

        prob = self._problem()
        for i, param in enumerate(self.params):
            prob.set_val(param, vars_current[i])
        prob.run_model()

        f_current = self.get_objective_values()
        pareto = [{
            'vars': {self.params[i]: vars_current[i] for i in range(len(self.params))},
            'f': f_current.copy()
        }]

        # with tqdm(total=len(temps), desc="Temps", leave=False, position=1) as inner_pbar:
        #     for temp in temps:
        with tqdm(total=self.num_iterations, desc="Iterations", leave=False, position=1) as iter_pbar:
            for _ in range(self.num_iterations):
                vars_new = [np.clip(vars_current[i] + random.uniform(-self.step_size, self.step_size), *vars_bound[i]) for i in range(len(self.params))]

                for i, param in enumerate(self.params):
                    prob.set_val(param, vars_new[i])
                prob.run_model()

                f_new = self.get_objective_values()

                delta_f = np.sum([f_new[k] - f_current[k] for k in f_new])

                # if delta_f < 0 or np.exp(-delta_f / temp) > random.random():
                #     accept = True
                # if delta_f > 0:
                #     accept = False
                #     print('Probability: ', np.exp(-delta_f / temp))

                accept = delta_f < 0 or np.exp(-delta_f / temp) > random.random()
                if accept:
                    for i, param in enumerate(self.params):
                        vars_current[i] = vars_new[i]
                    f_current = f_new.copy()

                pareto = self._update_pareto(
                    pareto,
                    {self.params[i]: vars_new[i] for i in range(len(self.params))},
                    f_new.copy()
                )

                # temp = max(temp * self.cooling_rate, self.final_temp)
                temp = temp * self.cooling_rate

                iter_pbar.update(1)
                # inner_pbar.update(1)

        self.pareto_front = pareto
        return True

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