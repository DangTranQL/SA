import numpy as np
from scipy.optimize import fsolve
import importlib

# Choose circuit
circuit = input("Please enter name of the circuit: ")

if circuit != "neg" and circuit != "posneg":
    raise NameError(f'{circuit} is not supported')

# Import circuit config file
config = importlib.import_module(circuit)

t = 0

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
    
    def compute(alpha_val, n_val):
        xss = ssfinder(alpha_val, n_val)
        return S_alpha_xss_analytic(xss, alpha_val, n_val), S_n_xss_analytic(xss, alpha_val, n_val)

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

    print('''
    0. S_betax_xss
    1. S_betax_yss
    2. S_betay_xss
    3. S_betay_yss
    4. S_n_xss
    5. S_n_yss\n''')

    # Choose pair of functions
    choice1 = int(input("Please select first option number: "))
    choice2 = int(input("Please select second option number: "))

    # Map indices to keys
    labels = {
        0: "S_betax_xss",
        1: "S_betax_yss",
        2: "S_betay_xss",
        3: "S_betay_yss",
        4: "S_n_xss",
        5: "S_n_yss"}

    # DEFINE FUNCTION THAT RETURNS PAIR OF SENSITIVITIES
    def senpair(xss_list, yss_list, beta_x_list, beta_y_list, n_list, choice1, choice2):

        # # Evaluate sensitivities
        # S_betax_xss = S_betax_xss_analytic(xss_list, yss_list, beta_x_list, beta_y_list, n_list)
        # S_betax_yss = S_betax_yss_analytic(xss_list, yss_list, beta_x_list, beta_y_list, n_list)
        # S_betay_xss = S_betay_xss_analytic(xss_list, yss_list, beta_x_list, beta_y_list, n_list)
        # S_betay_yss = S_betay_yss_analytic(xss_list, yss_list, beta_x_list, beta_y_list, n_list)
        # S_n_xss     =     S_n_xss_analytic(xss_list, yss_list, beta_x_list, beta_y_list, n_list)
        # S_n_yss     =     S_n_yss_analytic(xss_list, yss_list, beta_x_list, beta_y_list, n_list)

        # # Sensitivity dictionary
        # sensitivities = {
        #     "S_betax_xss": S_betax_xss,
        #     "S_betax_yss": S_betax_yss,
        #     "S_betay_xss": S_betay_xss,
        #     "S_betay_yss": S_betay_yss,
        #     "S_n_xss": S_n_xss,
        #     "S_n_yss": S_n_yss}
        
        # # Return values of the two sensitivities of interest
        # return sensitivities[labels[choice1]], sensitivities[labels[choice2]]

        sensitivity_funcs = {
            "S_betax_xss": lambda: S_betax_xss_analytic(xss_list, yss_list, beta_x_list, beta_y_list, n_list),
            "S_betax_yss": lambda: S_betax_yss_analytic(xss_list, yss_list, beta_x_list, beta_y_list, n_list),
            "S_betay_xss": lambda: S_betay_xss_analytic(xss_list, yss_list, beta_x_list, beta_y_list, n_list),
            "S_betay_yss": lambda: S_betay_yss_analytic(xss_list, yss_list, beta_x_list, beta_y_list, n_list),
            "S_n_xss":     lambda:     S_n_xss_analytic(xss_list, yss_list, beta_x_list, beta_y_list, n_list),
            "S_n_yss":     lambda:     S_n_yss_analytic(xss_list, yss_list, beta_x_list, beta_y_list, n_list)
        }

        result1 = sensitivity_funcs[labels[choice1]]()
        result2 = sensitivity_funcs[labels[choice2]]()

        return result1, result2
    
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
        
    def compute(beta_x_val, beta_y_val, n_val):
        xss, yss = ssfinder(beta_x_val, beta_y_val, n_val)
        return senpair(xss, yss, beta_x_val, beta_y_val, n_val, choice1, choice2)