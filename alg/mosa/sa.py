import math
import random

class SimulatedAnnealing:
    def __init__(self, temp=10, n_iterations=1000, step_size=0.1):
        self.n_iterations = n_iterations
        self.step_size = step_size
        self.temp = temp
        self.obj_func = None

    def objective_function(self, obj_func, x):
        self.obj_func = obj_func
        return self.obj_func(x)

    def get_neighbor(self, x):
        neighbor = x[:]
        index = random.randint(0, len(x) - 1)
        neighbor[index] += random.uniform(-self.step_size, self.step_size)
        return neighbor

    def run(self, sol=None):
        '''
        sol: Keeping track of best solution
        current: Current solution
        '''
        sol_eval = self.objective_function(self.obj_func, sol)
        current, current_eval = sol, sol_eval
        p = 0

        for i in range(self.n_iterations):
            # Decrease temperature
            t = self.temp / float(i + 1)

            # Generate candidate solution
            candidate = self.get_neighbor(current)

            candidate_eval = self.objective_function(self.obj_func, candidate)
            # Check if we should keep the new solution
            if candidate_eval < sol_eval:
                p = 1
            else:
                p = math.exp((current_eval - candidate_eval) / t)

            # Equilibrium condition
            if p == 1 or p > random.random():
                current, current_eval = candidate, candidate_eval
                # Update best solution
                if candidate_eval < sol_eval:
                    sol, sol_eval = candidate, candidate_eval

            if i % 100 == 0:
                print(f"Iteration {i}, Temperature {t:.3f}")

        return sol, sol_eval