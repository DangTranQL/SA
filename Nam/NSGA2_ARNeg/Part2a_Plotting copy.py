# IMPORT PACKAGES
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------------------

# LOAD DATA

# Grid search Pareto front
gridsearchdata = np.load('Data_GridSearchedParetoFront_ARneg.npy')

# Results from using algorithm
data = np.load("Data_Part2_EvolvingParetosWithPopSize_ArNeg.npz", allow_pickle=True)

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
axes[0].set_title("Negative Autoregulation: GD vs Population Size")
axes[0].grid(alpha=0.3)

# ----- Right plot: IGD -----
for col, gens in enumerate(gens_list):
    axes[1].plot(popsize_list, IGDs[:, col], marker='.', linestyle='--')

axes[1].set_xlabel("Population size")
axes[1].set_ylabel("Inverted Generational Distance (unitless)")
axes[1].set_yscale('log')
axes[1].set_xscale('log')
axes[1].set_title("Negative Autoregulation: IGD vs Population Size")
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


## PLOTTING GD and IGD VS POPULATION SIZE FOR EACH GENERATION
#
#fig, axes = plt.subplots(1, 2, figsize=(15, 5))
#
## ----- Left plot: GD -----
#ax = axes[0]
#
## For each column (number of generations)
#for col, gens in enumerate(gens_list):
#    # Plot IGD vs population size
#    plt.plot(popsize_list, GDs[:, col], marker='.', linestyle='--', label=f'{gens}')
#
#plt.xlabel("Population size")
#plt.ylabel("Inverted Generational Distance (unitless)")
#plt.yscale('log')
#plt.xscale('log')
#plt.title("Negative Autoregulation: GD vs Population Size")
#plt.grid(alpha=0.3)
#plt.legend(title="Generations", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
#plt.tight_layout(rect=[0, 0, 0.85, 1])  # leave space on the right for legend
#
## ----- Right plot: IGD -----
#ax = axes[1]
#
## For each column (number of generations)
#for col, gens in enumerate(gens_list):
#    # Plot IGD vs population size
#    plt.plot(popsize_list, IGDs[:, col], marker='.', linestyle='--', label=f'{gens}')
#
#plt.xlabel("Population size")
#plt.ylabel("Inverted Generational Distance (unitless)")
#plt.yscale('log')
#plt.xscale('log')
#plt.title("Negative Autoregulation: IGD vs Population Size")
#plt.grid(alpha=0.3)
#plt.legend(title="Generations", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
#plt.tight_layout(rect=[0, 0, 0.85, 1])  # leave space on the right for legend
#
## Save and show
#plt.savefig("Part2a_IGD_vs_PopSize.png", dpi=300, bbox_inches='tight')
#plt.show()

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
ax1.set_title("Negative Autoregulation: Hypervolume vs Population Size")
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