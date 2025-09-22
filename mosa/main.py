from mosa_random import custom_mosa
from mosa_old import custom_mosa_old
from sobol import custom_sobol
from eq import *
import numpy as np
import time

if __name__ == "__main__":

    choices = [[0,3], [0,4], [0,5], [1,2], [1,3], [1,4], [1,5], [2,3], [2,4], [2,5], [3,4], [3,4]]
    
    for choice in choices:
        print(f"\n\nRunning for choice: {choice}\n")
        choice1 = choice[0]
        choice2 = choice[1]

        if circuit == "neg":
            param_names = ['alpha', 'n']
            func_names = ['S_alpha', 'S_n']
            bounds = {'alpha': (0.01, 50), 'n': (0.01, 10)}
            objectives = compute
            data = np.load('data/ARneg_SensitivityPareto.npy')

        elif circuit == "posneg":
            param_names = ['beta_x', 'beta_y', 'n']
            func_names = [labels[choice1], labels[choice2]]
            bounds = {'beta_x': (0.01, 50), 'beta_y': (0.01, 50), 'n': (0.01, 10)}
            objectives = compute
            data = np.load('data/PosNegFbGridSearchData_renamed.npz')[f'paretoset_SensPair_{choice1}_{choice2}']

        # runs = int(input("\nEnter number of runs: "))
        runs = 200

        start_time_mosa_old = time.time()
        mosaold = custom_mosa_old()
        mosaold.setup(circuit=circuit, choice1=choice1, choice2=choice2, param_names=param_names, bounds=bounds, func_names=func_names, objectives=compute, alpha=0.5)
        mosaold.run(runs=runs)

        mosaold.prune()
        mosaold.gd_igd(ref=data)
        end_time_mosa_old = time.time()
        mosaold.plot(time=end_time_mosa_old-start_time_mosa_old)
        print("MOSA old done")

        start_time_mosa_sobol = time.time()
        mosasobol = custom_mosa()
        mosasobol.setup(circuit=circuit, choice1=choice1, choice2=choice2, param_names=param_names, bounds=bounds, func_names=func_names, objectives=compute, alpha=0.5)
        mosasobol.run_mosa_adaptive(runs=runs)

        mosasobol.prune()
        mosasobol.gd_igd(ref=data)
        end_time_mosa_sobol = time.time()
        mosasobol.plot(time=end_time_mosa_sobol-start_time_mosa_sobol)
        print("MOSA Sobol done")

        start_time_sobol = time.time()
        sobol = custom_sobol()
        sobol.setup(circuit=circuit, choice1=choice1, choice2=choice2, param_names=param_names, bounds=bounds, func_names=func_names, objectives=compute, alpha=0.5)
        sobol.run_adaptive(runs=runs)

        sobol.prune()
        sobol.gd_igd(ref=data)
        end_time_sobol = time.time()
        sobol.plot(time=end_time_sobol-start_time_sobol)
        print("Sobol done")