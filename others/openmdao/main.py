import time
import warnings
import matplotlib.pyplot as plt
from comp import *

if __name__ == "__main__":
    start_time = time.time()
    runs = int(input("\nNumber of runs: "))
    print("\n")
    warnings.filterwarnings("ignore")

    pareto_front = custom_run(runs=runs)
    
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")

    plt.figure(figsize=(10, 6))
    plt.scatter(pareto_front[:, 0], pareto_front[:, 1])

    if circuit == 'neg':
        plt.xlabel('S_alpha')
        plt.ylabel('S_n')

    elif circuit == 'posneg':
        plt.xlabel(labels[choice1])
        plt.ylabel(labels[choice2])

    plt.title(f'Custom MOSA Driver - {runs} Runs')
    plt.grid(True)
    plt.legend()
    plt.show()