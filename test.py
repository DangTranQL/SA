from sa import SimulatedAnnealing
import numpy as np
import math

anneal = SimulatedAnnealing(n_iterations=1000, step_size=0.1, temp=10)
def func(x):
    return 10 * len(x) + sum([(xi**2 - 10 * math.cos(2 * math.pi * xi)) for xi in x])
x = np.linspace(-5, 5, 100)
anneal.objective_function(func,x)
best_solution = anneal.run(sol = [0.111, 0.1424])
print(f'Solutions: {best_solution}')