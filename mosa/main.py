from mosa_sobol import custom_mosa
from mosa_random import custom_mosa_old
from sobol import custom_sobol
from eq import *
import numpy as np
import time

if __name__ == "__main__":

    choice1 = 0
    choice2 = 1

    if circuit == "neg":
        param_names = ['alpha', 'n']
        func_names = ['S_alpha', 'S_n']
        bounds = {'alpha': (0.01, 50), 'n': (0.01, 10)}
        objectives = compute

    elif circuit == "posneg":
        param_names = ['beta_x', 'beta_y', 'n']
        func_names = [labels[choice1], labels[choice2]]
        bounds = {'beta_x': (0.01, 50), 'beta_y': (0.01, 50), 'n': (0.01, 10)}
        objectives = compute

    runs = 200

    # mosaold = custom_mosa_old()
    # mosaold.setup(circuit=circuit, choice1=choice1, choice2=choice2, param_names=param_names, bounds=bounds, func_names=func_names, objectives=compute, alpha=0.5)
    # mosaold.run(runs=runs)

    mosasobol = custom_mosa()
    mosasobol.setup(circuit=circuit, choice1=choice1, choice2=choice2, param_names=param_names, bounds=bounds, func_names=func_names, objectives=compute, alpha=0.5)

    start_time_sobol = time.time()
    mosasobol.run_mosa_adaptive(runs=runs, use_tqdm=True)
    end_time_sobol = time.time()
    print(f"----------------------------------\nTIME = {end_time_sobol-start_time_sobol}\n----------------------------------\n")

    # start_time_sobol = time.time()
    # sobol = custom_sobol()
    # sobol.setup(circuit=circuit, choice1=choice1, choice2=choice2, param_names=param_names, bounds=bounds, func_names=func_names, objectives=compute, alpha=0.5)
    # sobol.run_adaptive(runs=runs)

    # sobol.prune()
    # sobol.gd_igd(ref=data)
    # end_time_sobol = time.time()
    # sobol.plot(time=end_time_sobol-start_time_sobol)
    # print("Sobol done")