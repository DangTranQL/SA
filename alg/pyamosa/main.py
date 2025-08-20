from eq import *
from scipy.optimize import fsolve
import numpy as np
import pyamosa

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

class sensitivity(pyamosa.Problem):
    n_var = 2

    def __init__(self):
        pyamosa.Problem.__init__(self, sensitivity.n_var, [pyamosa.Type.REAL]*sensitivity.n_var, [0.01]*sensitivity.n_var, [50.0, 20.0], 2, 0)
    
    def evaluate(self, x, out):
        alpha, n = x
        xss = ssfinder(alpha, n) # Call ssfinder here

        if xss == float('nan'):
            out['f'] = [float('inf'), float('inf')]

        else:
            s_alpha = S_alpha_xss_analytic(xss, alpha, n)
            s_n = S_n_xss_analytic(xss, alpha, n)
            out['f'] = [s_alpha, s_n]

if __name__ == "__main__":
    problem = sensitivity()

    config = pyamosa.Config()
    config.archive_hard_limit = 2000
    config.archive_soft_limit = 5000
    config.archive_gamma = 2
    config.clustering_max_iterations = 100
    config.hill_climbing_iterations = 300
    config.initial_temperature = 500
    config.cooling_factor = 0.9
    config.annealing_iterations = 1000
    config.annealing_strength = 1
    config.multiprocess_enabled = True

    optimizer = pyamosa.Optimizer(config)

    optimizer.run(problem, pyamosa.StopMaxTime("1:14"))