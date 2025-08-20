import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from paretoset import paretoset
from tqdm import tqdm
import importlib
import openmdao.api as om

# Choose circuit
circuit = input("Please enter name of the circuit: ")

# Import circuit config file
config = importlib.import_module(circuit)

t = 0

# Map indices to keys
labels = {
    0: "S_betax_xss",
    1: "S_betax_yss",
    2: "S_betay_xss",
    3: "S_betay_yss",
    4: "S_n_xss",
    5: "S_n_yss"}

if circuit == 'neg':
    S_alpha_xss_analytic = config.S_alpha_xss_analytic
    S_n_xss_analytic = config.S_n_xss_analytic
    Equs = config.Equs
    # DEFINE FUNCTION THAT SOLVES FOR STEADY STATES XSS GIVEN AN INITIAL GUESS
    def ssfinder(alpha_val,n_val):
        # Load initial guesses for solving which can be a function of a choice of alpha and n values
        InitGuesses = config.generate_initial_guesses(alpha_val, n_val)
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

elif circuit == 'posneg':
    # Define number of steady states expected
    numss = int(input("""
    Do you expect 1 or 2 stable steady states in your search space? 
    Please enter either 1 or 2: """))
    # dx/dt
    Equ1 = config.Equ1
    # dy/dt
    Equ2 = config.Equ2
    S_betax_xss_analytic = config.S_betax_xss_analytic
    S_betax_yss_analytic = config.S_betay_xss_analytic
    S_betay_xss_analytic = config.S_betay_xss_analytic
    S_betay_yss_analytic = config.S_betay_yss_analytic
    S_n_xss_analytic = config.S_n_xss_analytic
    S_n_yss_analytic = config.S_n_yss_analytic
    Equs = config.Equs
    # Choose pair of functions
    choice1 = int(input("Please select first option number:"))
    choice2 = int(input("Please select second option number:"))
    # List of sensitivity function names
    sensitivity_labels = [
        "|S_betax_xss|",
        "|S_betax_yss|",
        "|S_betay_xss|",
        "|S_betay_yss|",
        "|S_n_xss|",
        "|S_n_yss|"]
    # Save function names for later use
    label1 = sensitivity_labels[choice1]
    label2 = sensitivity_labels[choice2]
    # DEFINE FUNCTION THAT RETURNS PAIR OF SENSITIVITIES
    def senpair(xss_list, yss_list, beta_x_list, beta_y_list, n_list, choice1, choice2):
        # Evaluate sensitivities
        S_betax_xss = S_betax_xss_analytic(xss_list, yss_list, beta_x_list, beta_y_list, n_list)
        S_betax_yss = S_betax_yss_analytic(xss_list, yss_list, beta_x_list, beta_y_list, n_list)
        S_betay_xss = S_betay_xss_analytic(xss_list, yss_list, beta_x_list, beta_y_list, n_list)
        S_betay_yss = S_betay_yss_analytic(xss_list, yss_list, beta_x_list, beta_y_list, n_list)
        S_n_xss     =     S_n_xss_analytic(xss_list, yss_list, beta_x_list, beta_y_list, n_list)
        S_n_yss     =     S_n_yss_analytic(xss_list, yss_list, beta_x_list, beta_y_list, n_list)
        # Sensitivity dictionary
        sensitivities = {
            "S_betax_xss": S_betax_xss,
            "S_betax_yss": S_betax_yss,
            "S_betay_xss": S_betay_xss,
            "S_betay_yss": S_betay_yss,
            "S_n_xss": S_n_xss,
            "S_n_yss": S_n_yss}
        # Return values of the two sensitivities of interest
        return sensitivities[labels[choice1]], sensitivities[labels[choice2]]
    def ssfinder(beta_x_val,beta_y_val,n_val):
        # If we have one steady state
        if numss == 1: 
            # Define initial guesses
            InitGuesses = config.generate_initial_guesses(beta_x_val, beta_y_val)
            # Define array of parameters
            params = np.array([beta_x_val, beta_y_val, n_val])
            # For each until you get one that gives a solution or you exhaust the list
            for InitGuess in InitGuesses:
                # Get solution details
                output, infodict, intflag, _ = fsolve(Equs, InitGuess, args=(t, params), xtol=1e-12, full_output=True)
                xss, yss = output
                fvec = infodict['fvec'] 
                # Check if stable attractor point
                delta = 1e-8
                dEqudx = (Equs([xss+delta,yss], t, params)-Equs([xss,yss], t, params))/delta
                dEqudy = (Equs([xss,yss+delta], t, params)-Equs([xss,yss], t, params))/delta
                jac = np.transpose(np.vstack((dEqudx,dEqudy)))
                eig = np.linalg.eig(jac)[0]
                instablility = np.any(np.real(eig) >= 0)
                # Check if it is sufficiently large, has small residual, and successfully converges
                if xss > 0.04 and yss > 0.04 and np.linalg.norm(fvec) < 1e-10 and intflag == 1 and instablility==False:
                    # If so, it is a valid solution and we return it
                    return xss, yss
            # If no valid solutions are found after trying all initial guesses
            return float('nan'), float('nan')


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
            if np.isnan(xss) or xss <= 0:
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
    model.add_objective('J')
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1e-6
    prob.driver.options['disp'] = False
    prob.setup()
    if circuit == "neg":
        prob.set_val('alpha', 1.0)
        prob.set_val('n', 1.0)
    elif circuit == "posneg":
        prob.set_val('beta_x', 1.0)
        prob.set_val('beta_y', 1.0)
        prob.set_val('n', 1.0)
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

def run_and_plot_pareto():
    results = []
    w_max = 10
    weights = np.linspace(0.01, w_max, 500) 
    with tqdm(total=len(weights), desc="Running optimizations") as pbar:
        for w1 in weights:
            w2 = w_max - w1
            result = weighted_optimization(w1, w2)
            results.append(result)
            # print(f"w1={w1:.2f}, w2={w2:.2f}, S_alpha={result['S_alpha']:.4f}, S_n={result['S_n']:.4f}")
            pbar.update(1)
    if circuit == "neg":
        S_alpha_vals = [r['S_alpha'] for r in results]
        S_n_vals = [r['S_n'] for r in results]
        S_alpha_vals = np.array(S_alpha_vals)
        S_n_vals = np.array(S_n_vals)
        mask = paretoset(np.column_stack((S_alpha_vals, S_n_vals)), sense=['min', 'min'])
        pareto_front = np.column_stack((S_alpha_vals[mask], S_n_vals[mask]))
    elif circuit == "posneg":
        S_sens1_vals = [r[f'{labels[choice1]}'] for r in results]
        S_sens2_vals = [r[f'{labels[choice2]}'] for r in results]
        S_sens1_vals = np.array(S_sens1_vals)
        S_sens2_vals = np.array(S_sens2_vals)
        mask = paretoset(np.column_stack((S_sens1_vals, S_sens2_vals)), sense=['min', 'min'])
        pareto_front = np.column_stack((S_sens1_vals[mask], S_sens2_vals[mask]))

    plt.figure(figsize=(10, 6))
    plt.plot(pareto_front[:, 0], pareto_front[:, 1], 'bo-')
    if circuit == "neg":
        plt.xlabel('S_alpha')
        plt.ylabel('S_n')
    elif circuit == "posneg":
        plt.xlabel(f'{labels[choice1]}')
        plt.ylabel(f'{labels[choice2]}')
    plt.title('Pareto Front')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_and_plot_pareto()