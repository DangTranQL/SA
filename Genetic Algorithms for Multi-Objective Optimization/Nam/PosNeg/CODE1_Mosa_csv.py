# -------------- PART 0: PYTHON PRELIM --------------

# Additional notes: 
# mosa.py evolve() function has been edited to return final stopping temperature

# Import packages
import importlib
import os
import time
from tqdm import tqdm
import itertools
import numpy as np
import json
import mosa
import matplotlib.pyplot as plt
import pyvista as pv
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy.optimize import fsolve
from numpy import random
import pandas as pd


# -------------- PART 0: CHOOSE CIRCUIT AND SET UP FOLDER --------------


# Choose circuit
circuit = input("Please enter name of the circuit: ")

# Import circuit config file
config = importlib.import_module(circuit)

# Define the subfolder name
folder_name = f"MOSA_{circuit}"

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

# Define number of steady states expected
numss = int(input("""
Do you expect 1 or 2 stable steady states in your search space? 
Please enter either 1 or 2: """))


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

# Name for text file to records stats
output_file = f"TwoSpecies_{choice1}_and_{choice2}.csv"

# -------------- PART 0e: CHANGING DIRECTORIES --------------

# Define the subfolder name
subfolder_name = f"MOSA_sensfuncs_{choice1}_and_{choice2}"

# Create folder if not yet exist
if not os.path.exists(subfolder_name):
    os.makedirs(subfolder_name)

# Jump to folder
os.chdir(subfolder_name)

# Prompt new folder name
print(f"Current working directory: {os.getcwd()}")

if not os.path.exists('data'):
    os.makedirs('data')

# -------------- PART 0f: DEFINE FUNCTIONS --------------

# DEFINE FUNCTION TO CALCULATE THE EUCLIDEAN DISTANCE BETWEEN TWO POINTS
def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2) 

# DEFINE FUNCTION THAT SOLVES FOR STEADY STATES XSS AND YSS GIVEN SOME INITIAL GUESS
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

    # Map indices to keys
    labels = {
        0: "S_betax_xss",
        1: "S_betax_yss",
        2: "S_betay_xss",
        3: "S_betay_yss",
        4: "S_n_xss",
        5: "S_n_yss"}

    # Return values of the two sensitivities of interest
    return sensitivities[labels[choice1]], sensitivities[labels[choice2]]
    """
    Note: 
    Both outputs are of type numpy.ndarray
    Eg:
    xss_list = np.array((1.0000000108275326, 2.8793852415718164))
    yss_list = np.array((2.8793852415718164, 1.0000000108275326))
    beta_x_list = np.array([2,3])
    beta_y_list = np.array([2,3])
    n_list = np.array([2,3])
    choice1 = 0
    choice2 = 1
    ans = senpair(xss_list, yss_list, beta_x_list, beta_y_list, n_list, choice1, choice2) == (array([4.61785728e+07, 1.01476269e+00]), array([ 0.53265904, 24.00004561]))
    type(ans) = tuple
    type(ans[0]) == type(ans[1]) == numpy.ndarray
    """
    
# DEFINE OBJECTIVE FUNCTION TO ANNEAL
def fobj(solution):
	
	# Update parameter set
    beta_x_val = solution["beta_x"]
    beta_y_val = solution["beta_y"]
    n_val = solution["n"]

    # Create an empty numpy array
    xss_collect = np.array([])
    yss_collect = np.array([])

    # Find steady states and store.   <--------------------------------------------------------------- dont need to make condition for if numss==1 or 2 because this just takes in the list given. condition comes later in main part of script
    xss, yss = ssfinder(beta_x_val,beta_y_val,n_val)
    xss_collect = np.append(xss_collect,xss)
    yss_collect = np.append(yss_collect,yss)
    
    # Get sensitivity pair
    sens1, sens2 = senpair(xss_collect, yss_collect, solution["beta_x"], solution["beta_y"], solution["n"], choice1, choice2)
    ans1 = float(sens1)
    ans2 = float(sens2)
    
    # Check for NaN values before returning
    if np.isnan(ans1) or np.isnan(ans2):
        return np.inf, np.inf
    return ans1, ans2
    

# -------------- PART 1: GAUGING MOSA PARAMETERS --------------
    
# Sample beta_x values
beta_x_min = float(input("Please enter minimum beta_x value: "))
beta_x_max = float(input("Please enter maximum beta_x value: "))
beta_x_sampsize = int(input("Please enter the number of beta_x samples: "))
beta_x_samps = np.linspace(beta_x_min, beta_x_max, beta_x_sampsize)

# Sample beta_y values
beta_y_min = float(input("Please enter minimum beta_y value: "))
beta_y_max = float(input("Please enter maximum beta_y value: "))
beta_y_sampsize = int(input("Please enter the number of beta_y samples: "))
beta_y_samps = np.linspace(beta_y_min, beta_y_max, beta_y_sampsize)

# Sample n values
n_min = float(input("Please enter minimum n value: "))
n_max = float(input("Please enter maximum n value: "))
n_sampsize = int(input("Please enter the number of n samples: "))
n_samps = np.linspace(n_min, n_max, n_sampsize)

# Create empty arrays to store corresponding values of xss and yss
xss_samps = np.array([])
yss_samps = np.array([])
sens1_samps = np.array([])
sens2_samps = np.array([])



# WITH LOADING BAR 
# Compute the total number of iterations for tqdm
total_iterations = len(beta_x_samps) * len(beta_x_samps) * len(n_samps)
# Loop over every combination of parameters with a progress bar
for i, j, k in tqdm(itertools.product(beta_x_samps, beta_x_samps, n_samps), total=total_iterations, desc="Gauging energies:"):
    # Get steady states and store
        xss, yss = ssfinder(i, j, k)
        xss_samps = np.append(xss_samps, xss)
        yss_samps = np.append(yss_samps, yss)
        # Get sensitivities and store
        sens1, sens2 = senpair(xss, yss, i, j, k, choice1, choice2)
        sens1_samps = np.append(sens1_samps, sens1)
        sens2_samps = np.append(sens2_samps, sens2)



# WITHOUT LOADING BAR 
## For each combination of parameters
#for i in beta_x_samps:
#    for j in beta_x_samps:
#        for k in n_samps:
#            # Get steady states and store
#            xss, yss = ssfinder(i,j,k)
#            xss_samps = np.append(xss_samps,xss)
#            yss_samps = np.append(yss_samps,yss)
#            # Get sensitivities and store
#            sens1, sens2 = senpair(xss, yss, i, j, k, choice1, choice2)
#            sens1_samps = np.append(sens1_samps,sens1)
#            sens2_samps = np.append(sens2_samps,sens2)



# Get min and max of each sensitivity and print
sens1_samps_min = np.nanmin(sens1_samps)
sens2_samps_min = np.nanmin(sens2_samps)
sens1_samps_max = np.nanmax(sens1_samps)
sens2_samps_max = np.nanmax(sens2_samps)

# Get MOSA energies
deltaE_sens1 = sens1_samps_max - sens1_samps_min
deltaE_sens2 = sens2_samps_max - sens2_samps_min
deltaE = np.linalg.norm([deltaE_sens1, deltaE_sens2])

# Get hot temperature
print("Now setting up hot run...")
probability_hot = float(input("Please enter probability of transitioning to a higher energy state (if in doubt enter 0.9): "))
temp_hot = deltaE / np.log(1/probability_hot)

# Get cold temperature
print("Now setting up cold run...")
probability_cold = float(input("Please enter probability of transitioning to a higher energy state (if in doubt enter 0.01): "))
temp_cold = deltaE / np.log(1/probability_cold)


# -------------- PART 2a: PREPPING MOSA --------------


# Print prompts
print("Now preparing to MOSA...")
runs = int(input("Please enter number of MOSA runs you would like to complete (if in doubt enter 5): "))
iterations = int(input("Please enter number of random walks per run (if in doubt enter 100): "))

hotrun_time = []
hotrun_stoptemp = []
temp_num = []
step_scale = []
coldrun_time = []
coldrun_stoptemp = []
prune_time = []
archive_cold_len = []
archive_prune_len = []

# For each run
for run in range(runs):
    print(f"MOSA run number: {run+1}")
    
    # Define lists to collect sensitivity and parameter values from each MOSA run before pruning
    annealed_sensfunc1 = []
    annealed_sensfunc2 = []
    annealed_betax     = []
    annealed_betay     = []
    annealed_n         = []
    # Define lists to collect sensitivity and parameter values from each MOSA run after pruning
    pareto_sensfunc1 = []
    pareto_sensfunc2 = []
    pareto_betax     = []
    pareto_betay     = []
    pareto_n         = []
    
    # Delete archive and checkpoint json files at the start of each new run
    files_to_delete = ["archive.json", "checkpoint.json"]
    for file in files_to_delete:
        if os.path.exists(file):
            os.remove(file)
            print(f"Deleted: {file}")
        else:
            print(f"File not found: {file}")

	# -------------- PART 2b: ANNEAL TO GET PARETO FRONT IN SENSITIVITY SPACE --------------
	
	# Set random seed for MOSA
    random.seed(run)
	
	# Initialisation of MOSA
    opt = mosa.Anneal()
    opt.archive_size = 10000
    opt.maximum_archive_rejections = opt.archive_size
    opt.population = {"beta_x": (beta_x_min, beta_x_max), "beta_y": (beta_y_min, beta_y_max), "n": (n_min, n_max)}
	
	# Hot run options
    opt.initial_temperature = temp_hot
    opt.number_of_iterations = iterations
    opt.temperature_decrease_factor = 0.95
    opt.number_of_temperatures = int(np.ceil(np.log(temp_cold / temp_hot) / np.log(opt.temperature_decrease_factor)))
    opt.number_of_solution_elements = {"beta_x":1, "beta_y":1, "n":1}
    step_scaling = 1/opt.number_of_iterations
    opt.mc_step_size= {"beta_x": (beta_x_max-beta_x_min)*step_scaling , "beta_y": (beta_y_max-beta_y_min)*step_scaling , "n": (n_max-n_min)*step_scaling}
	
    # Hot run
    start_time = time.time()
    hotrun_stoppingtemp = opt.evolve(fobj)
    
    hotrun_time.append(time.time() - start_time)
    hotrun_stoptemp.append(hotrun_stoppingtemp)
    temp_num.append(opt.number_of_temperatures)
    step_scale.append(step_scaling)
	
    # Cold run options
    opt.initial_temperature = hotrun_stoppingtemp
    opt.number_of_iterations = iterations
    opt.number_of_temperatures = 100
    opt.temperature_decrease_factor = 0.9
    opt.number_of_solution_elements = {"beta_x":1, "beta_y":1, "n":1}
    step_scaling = 1/opt.number_of_iterations
    opt.mc_step_size= {"beta_x": (beta_x_max-beta_x_min)*step_scaling , "beta_y": (beta_y_max-beta_y_min)*step_scaling , "n": (n_max-n_min)*step_scaling}
	
    # Cold run
    start_time = time.time()
    coldrun_stoppingtemp = opt.evolve(fobj)
    
    coldrun_time.append(time.time() - start_time)
    coldrun_stoptemp.append(coldrun_stoppingtemp)
        
    # Output 
    start_time = time.time()
    pruned = opt.prunedominated()
    
    prune_time.append(time.time() - start_time)
	
	# -------------- PART 2c: STORE AND PLOT UNPRUNED PARETO FRONT IN SENSITIVITY SPACE --------------
	
    # Read archive file
    with open('archive.json', 'r') as f:
        data = json.load(f)
        
    # Check archive length
    length = len([solution["beta_x"] for solution in data["Solution"]])

    archive_cold_len.append(length)
    
    # Extract the "Values" coordinates (pairs of values)
    values = data["Values"]
    
    # Split the values into two lists
    value_1 = [v[0] for v in values]
    value_2 = [v[1] for v in values]
    
    # Add parameter values to collections
    for dummy1, dummy2 in zip(value_1, value_2):
        annealed_sensfunc1.append(dummy1)
        annealed_sensfunc2.append(dummy2)
    
    # Create a 2D plot
    plt.figure()
    plt.scatter(value_1, value_2)
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.grid(True)
    plt.title(f'Unpruned MOSA Pareto Sensitivities - Run No. {run + 1}')
    plt.savefig(f'data/unpruned_pareto_sensitivities_run_{run + 1}.png', dpi=300)
    plt.close()
	
    # -------------- PART 2d: STORE AND PLOT CORRESPONDING POINTS IN PARAMETER SPACE --------------
    
    # Extract beta_x, beta_y and n values from the solutions
    beta_x_values = [solution["beta_x"] for solution in data["Solution"]]
    beta_y_values = [solution["beta_y"] for solution in data["Solution"]]
    n_values = [solution["n"] for solution in data["Solution"]]
    
    # Add parameter values to collections
    for dummy1, dummy2, dummy3 in zip(beta_x_values, beta_y_values, n_values):
        annealed_betax.append(dummy1)
        annealed_betay.append(dummy2)
        annealed_n.append(dummy3)
        
    # Create a 3D plot
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(beta_x_values, beta_y_values, n_values)
    ax.set_xlabel('beta_x')
    ax.set_ylabel('beta_y')
    ax.set_zlabel('n')
    ax.set_title(f'Unpruned MOSA Pareto Parameters - Run No. {run + 1}')
    fig.savefig(f'data/unpruned_pareto_parameters_run_{run + 1}.png', dpi=300)
    plt.close()
    
    # -------------- PART 2e: SAVE PARETO DATA FROM CURRENT RUN --------------

    # Save sensitivity function 1 annealed values
    filename = f"annealed_sensfunc1_run{run+1}.npy"
    np.save(f'data/{filename}',annealed_sensfunc1)
    # Save sensitivity function 2 annealed values
    filename = f"annealed_sensfunc2_run{run+1}.npy"
    np.save(f'data/{filename}',annealed_sensfunc2)
    # Save betax annealed values
    filename = f"annealed_betax_run{run+1}.npy"
    np.save(f'data/{filename}',annealed_betax)
    # Save betay annealed values
    filename = f"annealed_betay_run{run+1}.npy"
    np.save(f'data/{filename}',annealed_betay)
    # Save n annealed values
    filename = f"annealed_n_run{run+1}.npy"
    np.save(f'data/{filename}',annealed_n)
    
    # -------------- PART 2f: STORE AND PLOT PRUNED PARETO FRONT IN SENSITIVITY SPACE --------------
	
    data = pruned
        
    # Check archive length
    length = len([solution["beta_x"] for solution in data["Solution"]])

    archive_prune_len.append(length)
    
    # Extract the "Values" coordinates (pairs of values)
    values = data["Values"]
    
    # Split the values into two lists
    value_1 = [v[0] for v in values]
    value_2 = [v[1] for v in values]
    
    # Add parameter values to collections
    for dummy1, dummy2 in zip(value_1, value_2):
        pareto_sensfunc1.append(dummy1)
        pareto_sensfunc2.append(dummy2)
    
    # Create a 2D plot
    plt.figure()
    plt.scatter(value_1, value_2)
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.grid(True)
    plt.title(f'Pruned MOSA Pareto Sensitivities - Run No. {run + 1}')
    plt.savefig(f'data/Pruned_pareto_sensitivities_run_{run + 1}.png', dpi=300)
    plt.close()
    
    # -------------- PART 2g: STORE AND PLOT CORRESPONDING POINTS IN PARAMETER SPACE --------------
    
    # Extract beta_x, beta_y and n values from the solutions
    beta_x_values = [solution["beta_x"] for solution in data["Solution"]]
    beta_y_values = [solution["beta_y"] for solution in data["Solution"]]
    n_values = [solution["n"] for solution in data["Solution"]]
    
    # Add parameter values to collections
    for dummy1, dummy2, dummy3 in zip(beta_x_values, beta_y_values, n_values):
        pareto_betax.append(dummy1)
        pareto_betay.append(dummy2)
        pareto_n.append(dummy3)
        
    # Create a 3D plot
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(beta_x_values, beta_y_values, n_values)
    ax.set_xlabel('beta_x')
    ax.set_ylabel('beta_y')
    ax.set_zlabel('n')
    ax.set_title(f'Pruned MOSA Pareto Parameters - Run No. {run + 1}')
    fig.savefig(f'data/Pruned_pareto_parameters_run_{run + 1}.png', dpi=300)
    plt.close()

    # -------------- PART 2h: SAVE PARETO DATA FROM CURRENT RUN --------------

    # Save sensitivity function 1 pareto values
    filename = f"pareto_sensfunc1_run{run+1}.npy"
    np.save(f'data/{filename}',pareto_sensfunc1)
    # Save sensitivity function 2 pareto values
    filename = f"pareto_sensfunc2_run{run+1}.npy"
    np.save(f'data/{filename}',pareto_sensfunc2)
    # Save betax pareto values
    filename = f"pareto_betax_run{run+1}.npy"
    np.save(f'data/{filename}',pareto_betax)
    # Save betay pareto values
    filename = f"pareto_betay_run{run+1}.npy"
    np.save(f'data/{filename}',pareto_betay)
    # Save n pareto values
    filename = f"pareto_n_run{run+1}.npy"
    np.save(f'data/{filename}',pareto_n)
    
    # ---------------- PART 2i: SAVE MOSA DATA --------------------------------
    
    mosa_data = pd.DataFrame({
        'Circuit': [circuit],
        'Steady state': [numss],
        'Runs': [runs],
        'Random walks per run': [iterations],
        'beta_x (min, max, samples)': [[beta_x_min, beta_x_max, beta_x_sampsize]],
        'beta_y (min, max, samples)': [[beta_y_min, beta_y_max, beta_y_sampsize]],
        'n (min, max, samples)': [[n_min, n_max, n_sampsize]],
        f'(min, max) sampled value of {label1}': [[sens1_samps_min, sens1_samps_max]],
        f'(min, max) sampled value of {label2}': [[sens2_samps_min, sens2_samps_max]],  
        f'Sampled energy difference in {label1}': [deltaE_sens1],
        f'Sampled energy difference in {label2}': [deltaE_sens2],
        'Sampled cumulative energ difference': [deltaE],
        'Probability hot run': [probability_hot],
        'Hot run temperature': [temp_hot],
        'Probability cold run': [probability_cold],
        'Cold run temperature': [temp_cold],
        'Hot run time': [hotrun_time],
        'Hot run stopping temperature': [hotrun_stoptemp],
        'Number of temperatures': [temp_num],
        'Step scaling factor': [step_scale],
        'Cold run time': [coldrun_time],
        'Cold run stopping temperature': [coldrun_stoptemp],
        'Prune time': [prune_time],
        'Archive length after cold run': [archive_cold_len],
        'Archive length after prune': [archive_prune_len]
    })

file_exists = os.path.exists(output_file)
mosa_data.to_csv(output_file, mode='a', index=False, header=not file_exists)

print(f"Appended new data to {output_file}")

# -------------- PART 3a: VISUALISE EACH MOSA RECTANGULAR BOUNDS --------------

# Create a colormap
colors = plt.cm.viridis(np.linspace(0, 1, runs))
colors

# Create 3D plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Store legend handles and labels
legend_handles = []
legend_labels = []

# For each run in our total number of runs
for run in range(1, runs + 1):

    # Load the parameter values for that run's Pareto front
    pareto_betax = np.load(f"data/pareto_betax_run{run}.npy", allow_pickle=True)
    pareto_betay = np.load(f"data/pareto_betay_run{run}.npy", allow_pickle=True)
    pareto_n = np.load(f"data/pareto_n_run{run}.npy", allow_pickle=True)
    
    # Create 3D points from the loaded data
    points = np.array(list(zip(pareto_betax, pareto_betay, pareto_n)))
    
    # Compute the bounding box from the points
    min_x, min_y, min_z = np.min(points, axis=0)
    max_x, max_y, max_z = np.max(points, axis=0)
    
    # Assign a unique color for this run
    color = colors[run - 1]

    # If only one unique point exists, plot it as a single dot in the same color as the bounding box
    if np.array_equal([min_x, min_y, min_z], [max_x, max_y, max_z]):
        print(f"Plotting a single point for MOSA Run {run} (no bounding box).")
        scatter = ax.scatter(min_x, min_y, min_z, color=color, marker="o", s=10)
        legend_handles.append(mlines.Line2D([], [], color=color, marker="o", linestyle="None", markersize=6))
        legend_labels.append(f"Run {run} (Single Point)")
        # Skip bounding box creation
        continue

    # Define the vertices of the rectangular prism
    vertices = np.array([
        [min_x, min_y, min_z], [max_x, min_y, min_z], [max_x, max_y, min_z], [min_x, max_y, min_z],  # Bottom face
        [min_x, min_y, max_z], [max_x, min_y, max_z], [max_x, max_y, max_z], [min_x, max_y, max_z]   # Top face
    ])
    
    # Define the 3D bounding box as a set of six faces
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom face
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top face
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front face
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back face
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right face
        [vertices[0], vertices[3], vertices[7], vertices[4]]   # Left face
    ]
    
    # Create the 3D bounding box
    poly3d = Poly3DCollection(faces, alpha=0.3, facecolor=color, edgecolor='k', linewidths=1)
    ax.add_collection3d(poly3d)

    # Add to legend using a Patch with unique color
    legend_handles.append(mpatches.Patch(facecolor=color, edgecolor="black"))
    legend_labels.append(f"Run {run}")

# Set labels and limits
ax.set_xlabel(r"$\beta_x$")
ax.set_ylabel(r"$\beta_y$")
ax.set_zlabel(r"$n$")
ax.set_title("Bounding Rectangular Prisms for Each MOSA Run")
ax.set_xlim([0.01, 50])
ax.set_ylim([0.01, 50])
ax.set_zlim([0.01, 10])

# Add legend outside the plot
ax.legend(legend_handles, legend_labels, loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)

# Adjust layout to fit legend
plt.tight_layout()

# Show plot
plt.savefig('data/searchspaces_runs.png', dpi=300, bbox_inches="tight")
plt.show()