# ------------------------------ PRELIMINARY: SET UP PACKAGES --------------------------------------

# System packages
import os
import time
import json
import itertools
import importlib

# Scientific packages
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------ RELOCATE WORKING DIRECTORY ----------------------------------------

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
#
#if not os.path.exists('data'):
#    os.makedirs('data')


# ------------------------------------- PRELIMINARY: LOAD DATA -------------------------------------

# ----------- GRID SEARCH DATA -----------

gridsearchdata = np.load('../../Data_PositiveNegative_SensitivityPairsParetos.npz', allow_pickle=True)

# All possible unordered sensitivity function pairs (0–5)
all_pairs = list(itertools.combinations(range(6), 2))

def get_pareto_key(choice1, choice2):
    pair = tuple(sorted((choice1, choice2)))   # ensure order doesn’t matter
    idx = all_pairs.index(pair) + 1            # +1 since keys start from 1
    return f"paretoset_SensPair{idx}"

# Example: use your chosen sensitivity indices
key = get_pareto_key(choice1, choice2)

# Retrieve the dataset using the correct variable name
selected_griddata = gridsearchdata[key]
gridsearchdata = selected_griddata


# ------ PERFORMANCE INDICATOR DATA ------
data = np.load("Data_Part2_EvolvingParetosWithPopSize_PosNeg.npz", allow_pickle=True)

Storage_EvolvingParetos_Popsize = data["Storage_EvolvingParetos_Popsize"]
Storage_EvolvingParams_Popsize  = data["Storage_EvolvingParams_Popsize"]
popsize_list = data["popsize_list"]
gens_list    = data["gens_list"]
GDs          = data["GDs"]
IGDs         = data["IGDs"]
HV_grid      = data["HV_grid"]
HV_true      = data["HV_true"].item()  # convert 0-D array to scalar


# --------------------------------------------------------------------------------------------------

# PLOTTING GD and IGD VS POPULATION SIZE FOR EACH GENERATION

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ----- Left plot: GD -----
for col, gens in enumerate(gens_list):
    axes[0].plot(popsize_list, GDs[:, col], marker='.', linestyle='--', label=f'{gens}')

axes[0].set_xlabel("Population size")
axes[0].set_ylabel("Generational Distance (unitless)")
axes[0].set_yscale('log')
axes[0].set_xscale('log')
axes[0].set_title("PosNeg Feedback: GD vs Population Size")
axes[0].grid(alpha=0.3)

# ----- Right plot: IGD -----
for col, gens in enumerate(gens_list):
    axes[1].plot(popsize_list, IGDs[:, col], marker='.', linestyle='--')

axes[1].set_xlabel("Population size")
axes[1].set_ylabel("Inverted Generational Distance (unitless)")
axes[1].set_yscale('log')
axes[1].set_xscale('log')
axes[1].set_title("PosNeg Feedback: IGD vs Population Size")
axes[1].grid(alpha=0.3)

# ----- ONE shared legend, horizontal, below -----
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    title="Generations",
    loc='lower center',
    ncol=len(gens_list),           # horizontal legend
    bbox_to_anchor=(0.5, -0.02),   # slightly below the plots
    frameon=False
)

plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave space at bottom for the legend

plt.savefig("Part2a_GD_IGD_vs_PopSize.png", dpi=300, bbox_inches='tight')
plt.show()

# --------------------------------------------------------------------------------------------------

# PLOTTING HV VS POPULATION SIZE FOR EACH GENERATION

buffer = 0.10  # 10% buffer beyond true front bounds
pf_min = gridsearchdata.min(axis=0)
pf_max = gridsearchdata.max(axis=0)
ref_point = pf_max + buffer * (pf_max - pf_min)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ----- Left plot: True Pareto front and reference point -----
ax0 = axes[0]

ax0.scatter(gridsearchdata[:, 0], gridsearchdata[:, 1],
           color='black', label='True Pareto front', s=1, zorder=3)

ax0.scatter(ref_point[0], ref_point[1],
           color='red', marker='+', s=100, label='Reference point', zorder=4)

ax0.set_xlabel("S_alpha")
ax0.set_ylabel("S_n")
ax0.set_title("Grid searched Pareto front and Reference point")
ax0.grid(alpha=0.3)

# Keep this legend *inside* the left plot
ax0.legend(loc="best", frameon=False)

# ----- Right plot: Hypervolume vs Population size -----
ax1 = axes[1]

for col, gens in enumerate(gens_list):
    ax1.plot(popsize_list, HV_grid[:, col],
        marker='.', linestyle='--', label=f'{gens}', zorder=3)

ax1.axhline(y=HV_true, color='red',
    linestyle='--', linewidth=2, label='True Pareto HV',zorder=2)

ax1.set_xlabel("Population size")
ax1.set_ylabel("Hypervolume")
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_title("PosNeg Feedback: Hypervolume vs Population Size")
ax1.grid(alpha=0.3)

# ----- ONE shared horizontal legend (from right subplot only) -----
handles1, labels1 = ax1.get_legend_handles_labels()

fig.legend(handles1, labels1, title="Generations",
    loc='lower center',
    ncol=len(labels1),        # horizontal legend
    bbox_to_anchor=(0.5, -0.02), frameon=False)

plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave space at bottom for legend

fig.savefig("Fig2a_HV_vs_PopSize.png", dpi=300, bbox_inches='tight')
plt.show()