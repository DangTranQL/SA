from mosa import custom_mosa
from eq import *

if __name__ == "__main__":
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

    mosa = custom_mosa()
    mosa.setup(param_names=param_names, bounds=bounds, func_names=func_names, objectives=compute, alpha=0.5)

    runs = int(input("\nEnter number of runs: "))

    mosa.run(runs=runs, use_tqdm=True)
    mosa.prune()
    mosa.plot()
    mosa.gd_igd(ref=data)