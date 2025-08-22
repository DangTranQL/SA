import openmdao.api as om
import numpy as np
from tqdm import tqdm
from eq import *
import random

class SenFunc(om.ExplicitComponent):

    def initialize(self):

        self.options.declare('w1', default=0.5, types=(float,))
        self.options.declare('w2', default=0.5, types=(float,))
        self.options.declare('circuit', default=circuit, types=(str,))

    def setup(self):

        if self.options['circuit'] == "neg":
            self.add_input('alpha', val=1.0)
            self.add_input('n', val=1.0)
            self.add_output('S_alpha', val=0.0)
            self.add_output('S_n', val=0.0)
            self.add_output('J', val=0.0)

        elif self.options['circuit'] == "posneg":
            self.add_input('beta_x', val=1.0)
            self.add_input('beta_y', val=1.0)
            self.add_input('n', val=1.0)
            self.add_output(f'{labels[choice1]}', val=0.0)
            self.add_output(f'{labels[choice2]}', val=0.0)
            self.add_output('J', val=0.0)

        else:
            self.add_input('x', val = 1.0)
            self.add_output('f1', val=0.0)
            self.add_output('f2', val=0.0)
            self.add_output('J', val=0.0)

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):

        if self.options['circuit'] == "neg":
            alpha = float(inputs['alpha'])
            n = float(inputs['n'])
            xss = ssfinder(alpha, n)

            if np.isnan(xss) or xss <= 0:
                outputs['J'] = 1e6
                outputs['S_alpha'] = 1e6
                outputs['S_n'] = 1e6

            else:
                S_alpha = S_alpha_xss_analytic(xss, alpha, n)
                S_n = S_n_xss_analytic(xss, alpha, n)
                w1 = self.options['w1']
                w2 = self.options['w2']
                outputs['S_alpha'] = S_alpha
                outputs['S_n'] = S_n
                outputs['J'] = w1 * S_alpha + w2 * S_n

        elif self.options['circuit'] == "posneg":
            beta_x = float(inputs['beta_x'])
            beta_y = float(inputs['beta_y'])
            n = float(inputs['n'])
            xss, yss = ssfinder(beta_x,beta_y,n)

            if np.isnan(xss) or xss <= 0 or np.isnan(yss) or yss <= 0:
                outputs['J'] = 1e6
                outputs[f'{labels[choice1]}'] = 1e6
                outputs[f'{labels[choice2]}'] = 1e6

            else:
                sens1, sens2 = senpair(xss, yss, beta_x, beta_y, n, choice1, choice2)
                w1 = self.options['w1']
                w2 = self.options['w2']
                outputs[f'{labels[choice1]}'] = sens1
                outputs[f'{labels[choice2]}'] = sens2
                outputs['J'] = w1 * sens1 + w2 * sens2

        else:
            x = float(inputs['x'])
            f1, f2 = Equs(x)
            w1 = self.options['w1']
            w2 = self.options['w2']
            outputs['f1'] = f1
            outputs['f2'] = f2
            outputs['J'] = w1 * f1 + w2 * f2

def weighted_optimization(w1, w2):

    prob = om.Problem(reports=False)
    model = prob.model
    model.add_subsystem('sens', SenFunc(w1=w1, w2=w2), promotes=['*'])

    if circuit == "neg":
        model.add_design_var('alpha', lower=0.01, upper=50.0)
        model.add_design_var('n', lower=0.01, upper=10.0)

    elif circuit == "posneg":
        model.add_design_var('beta_x', lower=0.01, upper=50.0)
        model.add_design_var('beta_y', lower=0.01, upper=50.0)
        model.add_design_var('n', lower=0.01, upper=10.0)

    else:
        model.add_design_var('x', lower=-1.0, upper=1.0)

    model.add_objective('J')
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1e-9
    prob.driver.options['disp'] = False

    prob.setup()

    if circuit == "neg":
        prob.set_val('alpha', random.uniform(0.01, 50.0))
        prob.set_val('n', random.uniform(0.01, 10.0))

    elif circuit == "posneg":
        prob.set_val('beta_x', random.uniform(0.01, 50.0))
        prob.set_val('beta_y', random.uniform(0.01, 50.0))
        prob.set_val('n', random.uniform(0.01, 10.0))

    else:
        prob.set_val('x', random.uniform(-1.0, 1.0))

    prob.run_driver()

    if circuit == "neg":
        return {
            'w1': w1,
            'w2': w2,
            'alpha': prob.get_val('alpha')[0],
            'n': prob.get_val('n')[0],
            'S_alpha': prob.get_val('S_alpha')[0],
            'S_n': prob.get_val('S_n')[0],
            'J': prob.get_val('J')[0]
        }
    
    elif circuit == "posneg":
        return {
            'w1': w1,
            'w2': w2,
            'beta_x': prob.get_val('beta_x')[0],
            'beta_y': prob.get_val('beta_y')[0],
            'n': prob.get_val('n')[0],
            f'{labels[choice1]}': prob.get_val(f'{labels[choice1]}')[0],
            f'{labels[choice2]}': prob.get_val(f'{labels[choice2]}')[0],
            'J': prob.get_val('J')[0]
        }
    
    else:
        return {
            'w1': w1,
            'w2': w2,
            'x': prob.get_val('x')[0],
            'f1': prob.get_val('f1')[0],
            'f2': prob.get_val('f2')[0],
            'J': prob.get_val('J')[0]
        }
    
def run(w_min, w_max, w_samples):

    results = []
    weights = np.linspace(w_min, w_max, w_samples) 

    with tqdm(total=len(weights), desc="Running optimizations") as pbar:
        for w1 in weights:
            w2 = w_max - w1
            result = weighted_optimization(w1, w2)
            results.append(result)
            pbar.update(1)

    return results