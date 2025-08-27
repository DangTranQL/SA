import openmdao.api as om
import numpy as np
from tqdm import tqdm
from paretoset import paretoset
from eq import *
from mosa_driver import MOSADriver

class EqusComp(om.ExplicitComponent):
    def setup(self):
        if circuit=='neg':
            self.add_input('alpha', val=0.0)
            self.add_input('n', val=0.0)
            self.add_output('S_alpha', val=0.0)
            self.add_output('S_n', val=0.0)

        elif circuit=='posneg':
            self.add_input('betax', val=0.0)
            self.add_input('betay', val=0.0)
            self.add_input('n', val=0.0)
            self.add_output(labels[choice1], val=0.0)
            self.add_output(labels[choice2], val=0.0)

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

        elif circuit=='posneg':
            betax = float(inputs['betax'])
            betay = float(inputs['betay'])
            n = float(inputs['n'])
            xss, yss = ssfinder(betax, betay, n)

            if np.isnan(xss) or xss <= 0 or np.isnan(yss) or yss <= 0:
                outputs[labels[choice1]] = 1e6
                outputs[labels[choice2]] = 1e6

            else:
                S1, S2 = senpair(xss, yss, betax, betay, n, choice1, choice2)
                outputs[labels[choice1]] = S1
                outputs[labels[choice2]] = S2

def custom_run(runs=5):
    f1 = np.array([])
    f2 = np.array([])

    with tqdm(total=runs, desc="Runs", position=0) as outer_pbar:
        for _ in range(runs):
            prob = om.Problem(reports=False)
            model = prob.model

            model.add_subsystem('equs', EqusComp(), promotes=['*'])

            if circuit == 'neg':
                driver = MOSADriver(params=['alpha', 'n'])
            elif circuit == 'posneg':
                driver = MOSADriver(params=['betax', 'betay', 'n'])

            prob.driver = driver
            prob.driver.declare_coloring()

            if circuit == 'neg':
                prob.model.add_design_var('alpha', lower=0.01, upper=50.0)
                prob.model.add_design_var('n', lower=0.01, upper=10.0)
                prob.model.add_objective('S_alpha')
                prob.model.add_objective('S_n')
            
            elif circuit == 'posneg':
                prob.model.add_design_var('betax', lower=0.01, upper=50.0)
                prob.model.add_design_var('betay', lower=0.01, upper=50.0)
                prob.model.add_design_var('n', lower=0.01, upper=10.0)
                prob.model.add_objective(labels[choice1])
                prob.model.add_objective(labels[choice2])

            prob.setup()
            prob.run_driver()

            driver.run(outer_pbar=outer_pbar)

            pareto = prob.driver.pareto_front

            if circuit == 'neg':
                pareto = sorted(pareto, key=lambda p: p['f']['S_alpha'])
                f1_vals = [p['f']['S_alpha'] for p in pareto]
                f2_vals = [p['f']['S_n'] for p in pareto]
            
            elif circuit == 'posneg':
                pareto = sorted(pareto, key=lambda p: p['f'][labels[choice1]])
                f1_vals = [p['f'][labels[choice1]] for p in pareto]
                f2_vals = [p['f'][labels[choice2]] for p in pareto]

            f1 = np.append(f1, f1_vals)
            f2 = np.append(f2, f2_vals)

            outer_pbar.update(1)

    mask = paretoset(np.column_stack((f1, f2)), sense=['min', 'min'])
    pareto_front = np.column_stack((f1[mask], f2[mask]))

    return pareto_front