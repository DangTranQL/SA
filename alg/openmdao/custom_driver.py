import openmdao.api as om
from openmdao.core.driver import Driver
import numpy as np
import random
from paretoset import paretoset
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import time
from eq import *

class EqusComp(om.ExplicitComponent):
    def setup(self):
        if circuit=='neg':
            self.add_input('alpha', val=0.0)
            self.add_input('n', val=0.0)
            self.add_output('S_alpha', val=0.0)
            self.add_output('S_n', val=0.0)
            self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        if circuit=='neg':
            alpha = float(inputs['alpha'])
            n = float(inputs['n'])
            xss = ssfinder(alpha, n)
            if np.isnan(xss) or xss <= 0:
                outputs['S_alpha'] = 1e6
                outputs['S_n'] = 1e6

            else:
                S_alpha = S_alpha_xss_analytic(xss, alpha, n)
                S_n = S_n_xss_analytic(xss, alpha, n)
                outputs['S_alpha'] = S_alpha
                outputs['S_n'] = S_n

class MOSADriver(Driver):
    def __init__(self):
        super().__init__()
        self.num_iterations = 500
        self.initial_temp = 100.0
        self.final_temp = 1e-6
        self.cooling_rate = 0.95
        self.step_size = 0.1
        self.pareto_front = []

    def setup(self):
        pass

    def run(self, outer_pbar=None):
        temp = self.initial_temp

        alpha_bounds = self._designvars['alpha']['lower'], self._designvars['alpha']['upper']
        n_bounds = self._designvars['n']['lower'], self._designvars['n']['upper']
        alpha_current = random.uniform(*alpha_bounds)
        n_current = random.uniform(*n_bounds)

        prob = self._problem()
        prob.set_val('alpha', alpha_current)
        prob.set_val('n', n_current)
        prob.run_model()

        f_current = self.get_objective_values()
        pareto = [{'alpha': alpha_current, 'n': n_current, 'f': f_current.copy()}]

        with tqdm(total=self.num_iterations, desc="Iterations", leave=False, position=1) as inner_pbar:
            for _ in range(self.num_iterations):
                alpha_new = alpha_current + random.uniform(-self.step_size, self.step_size)
                n_new = n_current + random.uniform(-self.step_size, self.step_size)
                alpha_new = np.clip(alpha_new, *alpha_bounds)
                n_new = np.clip(n_new, *n_bounds)

                prob.set_val('alpha', alpha_new)
                prob.set_val('n', n_new)
                prob.run_model()

                f_new = self.get_objective_values()

                delta_f = np.sum([f_new[k] - f_current[k] for k in f_new])
                accept = delta_f < 0 or np.exp(-delta_f / temp) > random.random()
                if accept:
                    alpha_current = alpha_new
                    n_current = n_new
                    f_current = f_new.copy()

                pareto = self._update_pareto(pareto, {'alpha': alpha_new, 'n': n_new, 'f': f_new.copy()})

                temp = max(temp * self.cooling_rate, self.final_temp)

                inner_pbar.update(1)

        self.pareto_front = pareto
        return True

    def _dominates(self, f1, f2):
        better_or_equal = all(f1[k] <= f2[k] for k in f1)
        strictly_better = any(f1[k] < f2[k] for k in f1)
        return better_or_equal and strictly_better

    def _update_pareto(self, front, candidate):
        new_front = []
        dominated = False
        for point in front:
            if self._dominates(candidate['f'], point['f']):
                continue  
            elif self._dominates(point['f'], candidate['f']):
                dominated = True 
            else:
                new_front.append(point) 
        if not dominated:
            new_front.append(candidate)
        return new_front

def custom_run(runs=2):
    f1 = np.array([])
    f2 = np.array([])

    with tqdm(total=runs, desc="Runs", position=0) as outer_pbar:
        for _ in range(runs):
            prob = om.Problem(reports=False)
            model = prob.model

            model.add_subsystem('equs', EqusComp(), promotes=['*'])

            driver = MOSADriver()
            prob.driver = driver
            prob.driver.declare_coloring()

            prob.model.add_design_var('alpha', lower=0.01, upper=50.0)
            prob.model.add_design_var('n', lower=0.01, upper=10.0)

            prob.model.add_objective('S_alpha')
            prob.model.add_objective('S_n')

            prob.setup()
            prob.run_driver()

            driver.run(outer_pbar=outer_pbar)

            pareto = prob.driver.pareto_front
            pareto = sorted(pareto, key=lambda p: p['f']['S_alpha'])

            f1_vals = [p['f']['S_alpha'] for p in pareto]
            f2_vals = [p['f']['S_n'] for p in pareto]


            f1 = np.append(f1, f1_vals)
            f2 = np.append(f2, f2_vals)

            outer_pbar.update(1)

    mask = paretoset(np.column_stack((f1, f2)), sense=['min', 'min'])
    pareto_front = np.column_stack((f1[mask], f2[mask]))

    return pareto_front

if __name__ == "__main__":
    start_time = time.time()
    warnings.filterwarnings("ignore")
    pareto_front = custom_run(runs=200)
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")

    plt.figure(figsize=(10, 6))
    plt.scatter(pareto_front[:, 0], pareto_front[:, 1])
    plt.xlabel('S_alpha')
    plt.ylabel('S_n')
    plt.title('Custom MOSA Driver')
    plt.grid(True)
    plt.legend()
    plt.show()