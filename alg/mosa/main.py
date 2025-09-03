import numpy as np
from tqdm import tqdm
from paretoset import paretoset
import matplotlib.pyplot as plt
import warnings
import random
from itertools import product
from mosa import custom_mosa
from eq import *

def run(runs=200):
    f1 = np.array([])
    f2 = np.array([])

    if circuit == "neg":
        params = ['alpha', 'n']
        func_names = ['S_alpha', 'S_n']
        bounds = {'alpha': (0.01, 50), 'n': (0.01, 10)}
        objectives = compute
        
        num_n = 10
        num_alpha = 5*num_n
        choices_n = np.linspace(bounds['n'][0]+0.05, bounds['n'][1]-0.05, num_n)
        choices_alpha = np.linspace(bounds['alpha'][0]+0.05, bounds['alpha'][1]-0.05, num_alpha)
        pairs = list(product(choices_alpha, choices_n))

    elif circuit == "posneg":
        params = ['beta_x', 'beta_y', 'n']
        func_names = [labels[choice1], labels[choice2]]
        bounds = {'beta_x': (0.01, 50), 'beta_y': (0.01, 50), 'n': (0.01, 10)}
        objectives = compute

        num_n = 10
        num_beta_x = 5*num_n
        num_beta_y = 5*num_n
        choices_n = np.linspace(bounds['n'][0]+0.05, bounds['n'][1]-0.05, num_n)
        choices_beta_x = np.linspace(bounds['beta_x'][0]+0.05, bounds['beta_x'][1]-0.05, num_beta_x)
        choices_beta_y = np.linspace(bounds['beta_y'][0]+0.05, bounds['beta_y'][1]-0.05, num_beta_y)
        pairs = list(product(choices_beta_x, choices_beta_y, choices_n))

    with tqdm(total=runs, desc="Runs", position=0) as outer_pbar:
        for _ in range(runs):
            choice = random.choice(pairs)
            init = {key: choice[j] for j, key in enumerate(params)}
            optimizer = custom_mosa()
            optimizer.setup(params, bounds, objectives, alpha=0.5)
            optimizer.run(init, outer_pbar=outer_pbar)
            f1 = np.append(f1, optimizer.pareto_front[0]['f'][func_names[0]])
            f2 = np.append(f2, optimizer.pareto_front[0]['f'][func_names[1]])
            outer_pbar.update(1)
        
    mask = paretoset(np.column_stack((f1, f2)), sense=['min', 'min'])
    pareto_front = np.column_stack((f1[mask], f2[mask]))

    return pareto_front

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    runs = int(input("\nEnter number of runs: "))
    pareto_front = run(runs)

    plt.scatter(pareto_front[:, 0], pareto_front[:, 1])
    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.title("Pareto Front")
    plt.show()