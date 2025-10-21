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
IGDs         = data["IGDs"]
HV_grid      = data["HV_grid"]
HV_true      = data["HV_true"].item()  # convert 0-D array to scalar

# --------------------------------------------------------------------------------------------------

# PLOTTING IGD VS POPULATION SIZE FOR EACH GENERATION

plt.figure(figsize=(8, 5))

# For each column (number of generations)
for col, gens in enumerate(gens_list):
    # Plot IGD vs population size
    plt.plot(popsize_list, IGDs[:, col], marker='.', linestyle='--', label=f'{gens}')

plt.xlabel("Population size")
plt.ylabel("Inverted Generational Distance (unitless)")
plt.yscale('log')
plt.xscale('log')
plt.title("Negative Autoregulation: IGD vs Population Size")
plt.grid(alpha=0.3)
plt.legend(title="Generations", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout(rect=[0, 0, 0.85, 1])  # leave space on the right for legend

# Save and show
plt.savefig("Part2a_IGD_vs_PopSize.png", dpi=300, bbox_inches='tight')
plt.show()

# --------------------------------------------------------------------------------------------------

# PLOTTING HV VS POPULATION SIZE FOR EACH GENERATION

# Get reference point for HV calculation

buffer = 0.10  # 10% buffer beyond true front bounds
pf_min = gridsearchdata.min(axis=0)
pf_max = gridsearchdata.max(axis=0)
ref_point = pf_max + buffer * (pf_max - pf_min)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# ----- Left plot: True Pareto front and reference point -----
ax = axes[0]

ax.scatter(gridsearchdata[:, 0], gridsearchdata[:, 1],
           color='black', label='True Pareto front', s=1, zorder=3)

ax.scatter(ref_point[0], ref_point[1],
           color='red', marker='+', s=100, label='Reference point', zorder=4)

ax.set_xlabel("S_alpha")
ax.set_ylabel("S_n")
ax.set_title("Grid searched Pareto front and Reference point")
ax.legend(loc='best')
ax.grid(alpha=0.3)

# ----- Right plot: Hypervolume vs Population size -----
ax = axes[1]

for col, gens in enumerate(gens_list):
    ax.plot(
        popsize_list,
        HV_grid[:, col],
        marker='.',
        linestyle='--',
        label=f'{gens}',
        zorder=3
    )

# Add horizontal line for HV of grid-searched Pareto front (below others)
ax.axhline(
    y=HV_true,
    color='red',
    linestyle='--',
    linewidth=2,
    label='True Pareto HV',
    zorder=2
)

ax.set_xlabel("Population size")
ax.set_ylabel("Hypervolume")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title("Negative Autoregulation: Hypervolume vs Population Size")
ax.grid(alpha=0.3)

# Move legend outside the plot (to the right)
ax.legend(
    title="Generations",
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    borderaxespad=0.
)

# --- Adjust layout and show both together ---
plt.tight_layout(rect=[0, 0, 0.85, 1])  # leave space on right for legend
fig.savefig("Fig2_HV_vs_PopSize.png", dpi=300, bbox_inches='tight')
plt.show()