from eq import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from paretoset import paretoset
from tqdm import tqdm
import openmdao.api as om

t = 0

# DEFINE FUNCTION THAT SOLVES FOR STEADY STATES XSS GIVEN AN INITIAL GUESS
def ssfinder(alpha_val,n_val):

    # Load initial guesses for solving which can be a function of a choice of alpha and n values
    InitGuesses = generate_initial_guesses(alpha_val, n_val)

    # Define array of parameters
    params = np.array([alpha_val, n_val])

    # For each initial guess in the list of initial guesses we loaded
    for InitGuess in InitGuesses:

        # Get solution details
        output, infodict, intflag, _ = fsolve(Equs, InitGuess, args=(t, params), xtol=1e-12, full_output=True)
        xss = output
        fvec = infodict['fvec']

        # Check if stable attractor point
        delta = 1e-8
        dEqudx = (Equs(xss+delta, t, params)-Equs(xss, t, params))/delta
        jac = np.array([[dEqudx]])
        eig = jac
        instablility = np.real(eig) >= 0


        # Check if it is sufficiently large, has small residual, and successfully converges
        if xss > 0.04 and np.linalg.norm(fvec) < 1e-10 and intflag == 1 and instablility==False:
            # If so, it is a valid solution and we return it as a scalar
            return xss[0]

    # If no valid solutions are found after trying all initial guesses
    return float('nan')


class SenFunc(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('w1', default=0.5, types=(float,))
        self.options.declare('w2', default=0.5, types=(float,))

    def setup(self):
        self.add_input('alpha', val=1.0)
        self.add_input('n', val=2.0)

        self.add_output('J', val=0.0)
        self.add_output('S_alpha', val=0.0)
        self.add_output('S_n', val=0.0)

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
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

def weighted_optimization(w1, w2):
    prob = om.Problem(reports=False)
    model = prob.model

    model.add_subsystem('sens', SenFunc(w1=w1, w2=w2), promotes=['*'])

    model.add_design_var('alpha', lower=0.01, upper=50.0)
    model.add_design_var('n', lower=0.01, upper=10.0)

    model.add_objective('J')

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1e-6
    prob.driver.options['disp'] = False

    prob.setup()
    prob.set_val('alpha', 1.0)
    prob.set_val('n', 2.0)

    prob.run_driver()

    return {
        'w1': w1,
        'w2': w2,
        'alpha': prob.get_val('alpha')[0],
        'n': prob.get_val('n')[0],
        'S_alpha': prob.get_val('S_alpha')[0],
        'S_n': prob.get_val('S_n')[0],
        'J': prob.get_val('J')[0]
    }

def run_and_plot_pareto():
    results = []

    w_max = 10
    weights = np.linspace(0.01, w_max, 1000) 

    with tqdm(total=len(weights), desc="Running optimizations") as pbar:
        for w1 in weights:
            w2 = w_max - w1
            result = weighted_optimization(w1, w2)
            results.append(result)
            # print(f"w1={w1:.2f}, w2={w2:.2f}, S_alpha={result['S_alpha']:.4f}, S_n={result['S_n']:.4f}")
            pbar.update(1)

    S_alpha_vals = [r['S_alpha'] for r in results]
    S_n_vals = [r['S_n'] for r in results]
    S_alpha_vals = np.array(S_alpha_vals)
    S_n_vals = np.array(S_n_vals)

    mask = paretoset(np.column_stack((S_alpha_vals, S_n_vals)), sense=['min', 'min'])
    pareto_front = np.column_stack((S_alpha_vals[mask], S_n_vals[mask]))

    plt.figure(figsize=(10, 6))
    plt.plot(pareto_front[:, 0], pareto_front[:, 1], 'bo-')
    plt.xlabel('S_alpha')
    plt.ylabel('S_n')
    plt.title('Pareto Front')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_and_plot_pareto()