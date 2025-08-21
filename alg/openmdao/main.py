import numpy as np
import matplotlib.pyplot as plt
from paretoset import paretoset
from model import *

def pareto_front(results):

    if circuit == "neg":
        S_alpha_vals = [r['S_alpha'] for r in results]
        S_n_vals = [r['S_n'] for r in results]
        S_alpha_vals = np.array(S_alpha_vals)
        S_n_vals = np.array(S_n_vals)
        mask = paretoset(np.column_stack((S_alpha_vals, S_n_vals)), sense=['min', 'min'])
        pareto_front = np.column_stack((S_alpha_vals[mask], S_n_vals[mask]))

    elif circuit == "posneg":
        S_sens1_vals = [r[f'{labels[choice1]}'] for r in results]
        S_sens2_vals = [r[f'{labels[choice2]}'] for r in results]
        S_sens1_vals = np.array(S_sens1_vals)
        S_sens2_vals = np.array(S_sens2_vals)
        mask = paretoset(np.column_stack((S_sens1_vals, S_sens2_vals)), sense=['min', 'min'])
        pareto_front = np.column_stack((S_sens1_vals[mask], S_sens2_vals[mask]))

    plt.figure(figsize=(10, 6))
    plt.scatter(pareto_front[:, 0], pareto_front[:, 1])

    if circuit == "neg":
        plt.xlabel('S_alpha')
        plt.ylabel('S_n')

    elif circuit == "posneg":
        plt.xlabel(f'{labels[choice1]}')
        plt.ylabel(f'{labels[choice2]}')

    plt.title('Pareto Front')
    plt.grid(True)
    plt.show()

# def params_space():



if __name__ == "__main__":
    results = run(0, 100, 50000)
    pareto_front(results)