# IMPORT PACKAGES
import numpy as np
import matplotlib.pyplot as plt
from paretoset import paretoset
from pymoo.indicators.gd import GD
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD

# --------------------------------------------------------------------------------------------------

# LOAD DATA

# NSGA2 run data
data = np.load("Data_NSGA2_ArNeg.npz", allow_pickle=True)
all_F_np = data["all_F"]
all_X_np = data["all_X"]
all_time = data["all_time"]
all_size = data["all_size"]
popsize_list = data["popsize_list"]
gens_list = data["gens_list"]
data.close()

# Import grid search data
gridsearchdata = np.load('Data_GridSearchedParetoFront_ARneg.npy')

# --------------------------------------------------------------------------------------------------

# NOTES: 
# RECALL THAT OUR DATA LOOKED LIKE FOLLOWS
# ┌───────────────────────────────────────────────┐
# │                 Generation (j)                │
# │         gen_1     gen_2     gen_3   ...       │
# ├───────────────────────────────────────────────┤
# │ pop_1  [F[0][0]] [F[0][1]] [F[0][2]] ...      │
# │ pop_2  [F[1][0]] [F[1][1]] [F[1][2]] ...      │
# │ pop_3  [F[2][0]] [F[2][1]] [F[2][2]] ...      │
# │  ...       ...       ...       ...            │
# └───────────────────────────────────────────────┘

# --------------------------------------------------------------------------------------------------

# ANALYSIS: EVOLVING PARETO FRONTS WITH POPULATION SIZE (COLUMN WISE EVOLUTION)

# Create storage for evolving pareto fronts across population sizes for each generation
Storage_EvolvingParetos_Popsize = np.empty((len(popsize_list), len(gens_list)), dtype=object)
Storage_EvolvingParams_Popsize  = np.empty((len(popsize_list), len(gens_list)), dtype=object)

# Loop through generations column-wise
for columnnum, gens in enumerate(gens_list):

    # Get the data for this generation across all population sizes
    Fcolumn = all_F_np[:, columnnum] 
    Xcolumn = all_X_np[:, columnnum]

    # Column vectors to store sequence of Pareto front & parameter coordinates
    pareto_front_sequence = np.empty(Fcolumn.size, dtype=object)
    pareto_param_sequence = np.empty(Xcolumn.size, dtype=object)

    # Loop through populations
    for rownum, popsize in enumerate(popsize_list):
        
        # Initialize empty accumulators for this population size
        AllParetoFrontsUpToCurrentPop = np.empty((0, 2))  # Initialize an empty array
        AllParetoParamsUpToCurrentPop = np.empty((0, 2))  # Initialize an empty array

        # Accumulate data from the first run up to the current run
        for ind in range(rownum + 1):

            CurrentParetoFront = Fcolumn[ind]
            AllParetoFrontsUpToCurrentPop = np.vstack((AllParetoFrontsUpToCurrentPop, CurrentParetoFront))

            CurrentParetoParam = Xcolumn[ind]
            AllParetoParamsUpToCurrentPop = np.vstack((AllParetoParamsUpToCurrentPop, CurrentParetoParam))

        # Compute Pareto-optimal points
        mask = paretoset(AllParetoFrontsUpToCurrentPop, sense=["min", "min"])
        filteredfront = AllParetoFrontsUpToCurrentPop[mask]
        filteredparam = AllParetoParamsUpToCurrentPop[mask]

        # Store Pareto front and pareto parameters for this row
        pareto_front_sequence[rownum] = filteredfront
        pareto_param_sequence[rownum] = filteredparam

    # Store the full column sequence for this generation
    Storage_EvolvingParetos_Popsize[:,columnnum] = pareto_front_sequence
    Storage_EvolvingParams_Popsize[:,columnnum]  = pareto_param_sequence

# --------------------------------------------------------------------------------------------------

# COMPUTE GENERATIONAL DISTANCES

# Create storage
GDs = np.full(Storage_EvolvingParetos_Popsize.shape, np.nan)

# Define indicator once using the reference front
ind = GD(gridsearchdata)

# Evaluate generational distance of each Pareto front in data matrix
for col, gens in enumerate(gens_list):                  # loop over generations
    for row, popsize in enumerate(popsize_list):        # loop over population sizes
        pf = Storage_EvolvingParetos_Popsize[row, col]  # get Pareto front
        GDs[row, col] = ind(pf)                         # evaluate vs reference front

# --------------------------------------------------------------------------------------------------

# COMPUTE INVERTED GENERATIONAL DISTANCES

# Create storage
IGDs = np.full(Storage_EvolvingParetos_Popsize.shape, np.nan)

# Define indicator once using the reference front
ind = IGD(gridsearchdata)

# Evaluate inverted generational distance of each Pareto front in data matrix
for col, gens in enumerate(gens_list):                  # loop over generations
    for row, popsize in enumerate(popsize_list):        # loop over population sizes
        pf = Storage_EvolvingParetos_Popsize[row, col]  # get Pareto front
        IGDs[row, col] = ind(pf)                        # evaluate vs reference front
        

# --------------------------------------------------------------------------------------------------

# COMPUTE HYPERVOLUMES

# Reference point based on true Pareto front (t)
buffer = 0.10  # 10% buffer beyond true front bounds
t_min = gridsearchdata.min(axis=0)
t_max = gridsearchdata.max(axis=0)
ref_point = t_max + buffer * (t_max - t_min)

# Indicator
ind = HV(ref_point=ref_point)
# True hypervolume
HV_true = float(ind(gridsearchdata))
# Initialise storage
HV_grid = np.full(Storage_EvolvingParetos_Popsize.shape, np.nan)
# Compute HV for each cell
for i in range(Storage_EvolvingParetos_Popsize.shape[0]):      # population
    for j in range(Storage_EvolvingParetos_Popsize.shape[1]):  # generations
        pf = Storage_EvolvingParetos_Popsize[i, j]             # get Pareto front
        if pf is None or len(pf) == 0:                         # skip empty fronts
            continue
        pf = np.asarray(pf, dtype=float)                       # ensure correct type
        hv_val = float(ind(pf))                                # compute HV
        HV_grid[i, j] = hv_val                                 # store HV value

# --------------------------------------------------------------------------------------------------

# SAVE DATA

np.savez(
    "Data_Part2_EvolvingParetosWithPopSize_ArNeg.npz",
    Storage_EvolvingParetos_Popsize=Storage_EvolvingParetos_Popsize,
    Storage_EvolvingParams_Popsize=Storage_EvolvingParams_Popsize,
    popsize_list=popsize_list,
    gens_list=gens_list,
    GDs=GDs,
    IGDs=IGDs,
    HV_grid=HV_grid,
    HV_true=HV_true,
    )
    
print("Data saved!")
