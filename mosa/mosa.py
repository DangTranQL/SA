import random
import numpy as np

class MOSA:
    def __init__(self, hot_temp=100, hot_temp_factor=0.95, cold_temp=10, cold_temp_factor=0.95,
                n_iterations=1000, step_size=0.1):
        self.hot_temp = hot_temp
        self.hot_temp_factor = hot_temp_factor
        self.cold_temp = cold_temp
        self.cold_temp_factor = cold_temp_factor
        self.n_iterations = n_iterations
        self.step_size = step_size
        self.obj_funcs = None

    def get_neighbor(self, x):
        neighbor = x[:]
        index = random.randint(0, len(x) - 1)
        neighbor[index] += random.uniform(-self.step_size, self.step_size)
        return neighbor

    def objective_functions(self, funcs):
        for func in funcs:
            if not callable(func):
                raise ValueError("All objective functions must be callable.")
        self.obj_funcs = funcs
        return self.obj_funcs
    
    def run(self, sol=None):
        hot_temps = 