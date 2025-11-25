# ------------------------------ PRELIMINARY: SET UP PACKAGES --------------------------------------

# System packages
import os
import time
import json
import itertools
import importlib

# Scientific packages
import numpy as np
from scipy.optimize import fsolve
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from paretoset import paretoset
from tqdm import tqdm

# ------------------------------ PRELIMINARY: DEFINE FUNCTIONS -------------------------------------

# DEFINE FUNCTION THAT SOLVES FOR STEADY STATES XSS AND YSS GIVEN SOME INITIAL GUESS - ASSUMES ONLY ONE STABLE STEADY STATE EXISTS
def ssfinder(beta_x_val,beta_y_val,n_val):
        
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

# SET UP NSGA2 ALGORITHM
class Sensitivity(ElementwiseProblem):
    def __init__(self, choice1, choice2):
        super().__init__(
            n_var=3,  # beta_x, beta_y, n
            n_obj=2,  # two sensitivities
            n_constr=0,
            xl=np.array([0.01, 0.01, 0.01]),
            xu=np.array([50.0, 50.0, 10.0]),
        )

        # Select functions once
        _, f1, _, f2 = senpair(choice1, choice2)
        self.f1 = f1
        self.f2 = f2

    def _evaluate(self, x, out, *args, **kwargs):
        
        beta_x, beta_y, n = x
        xss, yss = ssfinder(beta_x, beta_y, n)

        if np.isnan(xss) or np.isnan(yss):
            out["F"] = [1e5, 1e5]  #### Penalize invalid solutions

        else:
            # Direct evaluation — assumes valid steady states
            S1 = self.f1(xss, yss, beta_x, beta_y, n)
            S2 = self.f2(xss, yss, beta_x, beta_y, n)

            # Ensure scalar floats
            S1 = float(np.asarray(S1))
            S2 = float(np.asarray(S2))

            # Output objective values
            out["F"] = [S1,S2]

# --------------------------------------------------------------------------------------------------

# -------------- PART 0: CHOOSE CIRCUIT AND SET UP FOLDER --------------

# Choose circuit
circuit = "posneg"

# Import circuit config file
config = importlib.import_module(circuit)

# Define the subfolder name
folder_name = f"NGSA2_{circuit}"

# Create folder if not yet exist
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Jump to folder
os.chdir(folder_name)

# Prompt new folder name
print(f"Current working directory: {os.getcwd()}")


# -------------- PART 0b: DEFINE DYNAMICAL SYSTEM --------------

# dx/dt
Equ1 = config.Equ1

# dy/dt
Equ2 = config.Equ2
    
# Define function to evaluate vector field
def Equs(P, t, params):
    x = P[0]
    y = P[1]
    beta_x = params[0]
    beta_y = params[1]
    n      = params[2]
    val0 = Equ1(x, y, beta_x, n)
    val1 = Equ2(x, y, beta_y, n)
    return np.array([val0, val1])

# Define values
t = 0.0

# -------------- PART 0c: DEFINE SENSITIVITY FUNCTIONS --------------

# Define analytical sensitivity expressions
S_betax_xss_analytic = config.S_betax_xss_analytic
S_betax_yss_analytic = config.S_betay_xss_analytic
S_betay_xss_analytic = config.S_betay_xss_analytic
S_betay_yss_analytic = config.S_betay_yss_analytic
S_n_xss_analytic = config.S_n_xss_analytic
S_n_yss_analytic = config.S_n_yss_analytic

# -------------- PART 0d: CHOOSE SENSITIVITY FUNCTIONS --------------

# Print prompt
print("""
We have the following sensitivity functions:
0. |S_betax_xss|
1. |S_betax_yss|
2. |S_betay_xss|
3. |S_betay_yss|
4. |S_n_xss|
5. |S_n_yss|
""")

# Choose pair of functions
choice1 = int(input("Please select first option number:"))
choice2 = int(input("Please select second option number:"))

def senpair(choice1, choice2):
    # Map each label to its corresponding analytical function
    sensitivity_funcs = {
        "|S_betax_xss|": S_betax_xss_analytic,
        "|S_betax_yss|": S_betax_yss_analytic,
        "|S_betay_xss|": S_betay_xss_analytic,
        "|S_betay_yss|": S_betay_yss_analytic,
        "|S_n_xss|":     S_n_xss_analytic,
        "|S_n_yss|":     S_n_yss_analytic
    }

    # Labels in consistent order
    sensitivity_labels = list(sensitivity_funcs.keys())

    # Select based on user choice
    label1, func1 = sensitivity_labels[choice1], sensitivity_funcs[sensitivity_labels[choice1]]
    label2, func2 = sensitivity_labels[choice2], sensitivity_funcs[sensitivity_labels[choice2]]

    # Return both the labels and the function handles
    return label1, func1, label2, func2

# -------------- PART 0e: CHANGING DIRECTORIES --------------

# Define the subfolder name
subfolder_name = f"NSGA2_sensfuncs_{choice1}_and_{choice2}"

# Create folder if not yet exist
if not os.path.exists(subfolder_name):
    os.makedirs(subfolder_name)

# Jump to folder
os.chdir(subfolder_name)

# Prompt new folder name
print(f"Current working directory: {os.getcwd()}")

# --------------------------------------------------------------------------------------------------

# RUN NSGA2 ALGORITHM FOR VARYING POPULATION SIZES AND NO. OF GENERATIONS

# --- Parameter sweeps ---
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
                Sensitivity(choice1,choice2),        # your optimization problem
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
    "Data_NSGA2_PosNeg.npz",
    all_F=all_F_np,
    all_X=all_X_np,
    all_time=all_time,
    all_size=all_size,
    popsize_list=popsize_list,
    gens_list=gens_list,
)