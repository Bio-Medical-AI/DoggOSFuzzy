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
