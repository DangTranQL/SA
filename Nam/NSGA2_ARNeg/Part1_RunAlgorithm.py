#IMPORT PACKAGES

import numpy as np
from scipy.optimize import fsolve
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from paretoset import paretoset
from tqdm import tqdm

# --------------------------------------------------------------------------------------------------

#IMPORT DATA

gridsearchdata = np.load('Data_GridSearchedParetoFront_ARneg.npy')

# --------------------------------------------------------------------------------------------------

#DEFINE SYSTEM

#Sensitivity wrt α
def S_alpha_xss_analytic(xss, alpha, n):
    numer = alpha * (1 + xss**n)
    denom = xss + alpha * n * xss**n + 2 * xss**(1+n) + xss**(1+2*n)
    sensitivity = numer/denom
    return abs(sensitivity)

#Sensitivity wrt n
def S_n_xss_analytic(xss, alpha, n):
    numer = alpha * n * np.log(xss) * xss**(n-1)
    denom = 1 + alpha * n * xss**(n-1) + 2 * xss**(n) + xss**(2*n)
    sensitivity = - numer/denom
    return abs(sensitivity)
    
#Dynamical system
def Equ1(x, alpha, n):
    return (alpha / (1 + x**n)) - x

#Wrapper function for the ODE system to be compatible with scipy's fsolve
def Equs(x, t, params):
    alpha, n = params
    return np.array([Equ1(x[0], alpha, n)])

# --------------------------------------------------------------------------------------------------

#SET UP STEADY STATE SOLVER

#Initial guesses
def generate_initial_guesses(alpha_val, n_val):
    return [np.array([2.0]), np.array([0.5]), np.array([4.627])]
    
#Steady state finder function given parameters (α, n)
def ssfinder(alpha_val, n_val):
    # Pack parameters into an array
    params = np.array([alpha_val, n_val])
    # Try multiple initial guesses to find steady states
    for guess in generate_initial_guesses(alpha_val, n_val):
        # Find root of the equation (where dx/dt = 0)
        xss, info, flag, _ = fsolve(Equs, guess, args=(0.0, params), full_output=True, xtol=1e-12)
        # Check if solution is valid, positive, and precise
        if (flag == 1 and xss[0] > 0.04 and np.linalg.norm(info["fvec"]) < 1e-10):
            # Check stability using numerical derivative
            d = 1e-8  # Small perturbation
            jac = (Equs(xss + d, 0.0, params) - Equs(xss, 0.0, params)) / d
            if np.real(jac)[0] < 0: # Negative Jacobian indicates stable steady state
                return xss[0]
    # Return NaN if no stable steady state found
    return np.nan

# --------------------------------------------------------------------------------------------------

#SET UP NSGA2 ALGORITHM

class Sensitivity(ElementwiseProblem):

    def __init__(self):
        super().__init__(
            n_var=2,                    # Number of decision variables (alpha, n)
            n_obj=2,                    # Number of objectives to minimize (S_alpha, S_n)
            n_constr=0,                 # Number of explicit constraints aside the bounds of alpha and n
            xl=np.array([0.01, 0.01]),  # Lower bounds for [alpha, n]
            xu=np.array([50.0, 10.0]))  # Upper bounds for [alpha, n]

    def _evaluate(self, x, out, *args, **kwargs):
        alpha, n = x              # Unpack decision variables (ie. parameters)
        xss = ssfinder(alpha, n)  # Find the steady state for these parameters
        if np.isnan(xss):         # Penalize parameter combinations that don't yield a stable steady state
            out["F"] = [1e5, 1e5]
        else:                     # Calculate both sensitivity measures for this parameter set
            out["F"] = [S_alpha_xss_analytic(xss, alpha, n), S_n_xss_analytic(xss, alpha, n)] 

# --------------------------------------------------------------------------------------------------

# RUN NSGA2 ALGORITHM FOR VARYING POPULATION SIZES AND NO. OF GENERATIONS

# Parameter sweeps
popsize_list = np.exp(np.linspace(np.log(1), np.log(5000), num=21)).astype(int)
gens_list = np.exp(np.linspace(np.log(1), np.log(500), num=11)).astype(int)

# Storage structure:
# ┌───────────────────────────────────────────────┐
# │                 Generation (j)                │
# │            gen_1     gen_2     gen_3   ...    │
# ├───────────────────────────────────────────────┤
# │ pop_1     [None]    [None]    [None]   ...    │
# │ pop_2     [None]    [None]    [None]   ...    │
# │ pop_3     [None]    [None]    [None]   ...    │
# │   ...       ...       ...       ...           │
# └───────────────────────────────────────────────┘
all_F = [[None for _ in gens_list] for _ in popsize_list]
all_X = [[None for _ in gens_list] for _ in popsize_list]
all_time = np.zeros((len(popsize_list), len(gens_list)))
all_size = np.zeros((len(popsize_list), len(gens_list)), dtype=int)

# Run loop
total_runs = len(popsize_list) * len(gens_list) # get total runs
with tqdm(total=total_runs, desc="Progress") as pbar: # set up progress bar
	for i, popsize in enumerate(popsize_list):
	    for j, gens in enumerate(gens_list):
	
	        algorithm = NSGA2(pop_size=int(popsize))
	
	        res = minimize(
	            Sensitivity(),        # your optimization problem
	            algorithm,            # algorithm instance
	            ('n_gen', int(gens)), # number of generations as stopping criterion
	            seed=1,               # reproducibility
	            verbose=False)
	
	        # Store results in grid form
	        all_F[i][j] = res.F
	        all_X[i][j] = res.X
	        all_time[i, j] = res.exec_time
	        all_size[i, j] = res.F.shape[0]

	        # Update bar
	        pbar.update(1)
	        pbar.set_postfix({
	            "Pop": f"{popsize}", "Gen": f"{gens}", "Time (s)": f"{res.exec_time:.1f}"})

# Our final all_F and all_X structures
# ┌───────────────────────────────────────────────┐
# │                 Generation (j)                │
# │         gen_1     gen_2     gen_3   ...       │
# ├───────────────────────────────────────────────┤
# │ pop_1  [F[0][0]] [F[0][1]] [F[0][2]] ...      │
# │ pop_2  [F[1][0]] [F[1][1]] [F[1][2]] ...      │
# │ pop_3  [F[2][0]] [F[2][1]] [F[2][2]] ...      │
# │  ...       ...       ...       ...            │
# └───────────────────────────────────────────────┘

# Conver to numpy array
all_F_np = np.array(all_F, dtype=object)
all_X_np = np.array(all_X, dtype=object)

# --------------------------------------------------------------------------------------------------

# SAVE DATA

np.savez(
    "Data_NSGA2_ArNeg.npz",
    all_F=all_F_np,
    all_X=all_X_np,
    all_time=all_time,
    all_size=all_size,
    popsize_list=popsize_list,
    gens_list=gens_list,
)