from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.factory import get_termination
import numpy as np


class DifferentialEvolution:
    def __init__(self, de):
        self.de = de

    def __call__(self, func):
        return self.de(func).x


class PSO:
    def __init__(self, pso):
        self.pso = pso

    def __call__(self, func):
        best_params, optimal_fitness = self.pso(func)
        return best_params


class CMAESWrapper:
    def __init__(self, cmaes):
        self.cmaes = cmaes

    def __call__(self, fitness):
        problem = TSProblem(fitness)
        termination = get_termination('n_eval', 20000)
        res = minimize(problem, self.cmaes, termination, seed=42, verbose=True, n_max_evals=1000)
        print(f"Best solution found: \nX = {res.X}\nF = {res.F}\nCV= {res.CV}")
        return res.X


class TSProblem(Problem):
    def __init__(self, fitness):
        super().__init__(n_var=10,
                         n_obj=1,
                         xl=-400.0,
                         xu=400.0)
        self.fitness = fitness

    def _evaluate(self, x, out, *args, **kwargs):
        f = []
        for lin_fun_params in x:
            f.append(self.fitness(lin_fun_params))
        out["F"] = np.asarray(f, dtype=float)
