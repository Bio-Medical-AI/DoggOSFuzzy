from functools import partial
import dill

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pyswarm import pso
from scipy.optimize import differential_evolution

from Logger import Logger
from TS_Experiments import TSExperiments
from metaheuristics_wrapper import DifferentialEvolution, PSO


def main():
    THRESHOLD = 0.5

    pso_logger = Logger("pso", "Haberman")
    de_logger = Logger("de", "Haberman")

    experiments = TSExperiments('data/Haberman.csv', ';', pso_logger)
    experiments.prepare_data([min_max_scale])
    experiments.prepare_fuzzy_system(n_mfs=9, fuzzy_set_type='it2')

    pso_partial = prepare_pso(experiments.n_params)
    de_partial = prepare_de(experiments.n_params)

    #experiments.select_optimal_parameters_kfold(threshold_classification(THRESHOLD),
    #                                            metaheuristic=pso_partial,
    #                                            debug=True,
    #                                            n_folds=6)

    experiments.logger = de_logger
    experiments.select_optimal_parameters_kfold(threshold_classification(THRESHOLD),
                                                metaheuristic=de_partial,
                                                debug=True,
                                                n_folds=6)


def threshold_classification(theta):
    def _classify(x):
        if x <= theta:
            return 0
        elif x > theta:
            return 1
        else:
            print('else')

    return _classify


def prepare_pso(n_params):
    pso_partial = partial(pso,
                          lb=[-250] * n_params,
                          ub=[250] * n_params,
                          debug=False,
                          maxiter=30,
                          swarmsize=30,
                          phig=1)
    return PSO(pso_partial)


def prepare_de(n_params):
    lb = [-250] * n_params
    ub = [250] * n_params
    bounds = []
    for l, u in zip(lb, ub):
        bounds.append((l, u))

    de_partial = partial(differential_evolution,
                         bounds=bounds,
                         maxiter=30,
                         popsize=15,
                         mutation=(0.5, 1),
                         seed=42,
                         updating="deferred",
                         workers=1,
                         disp=True)
    return DifferentialEvolution(de_partial)


def min_max_scale(data: np.ndarray):
    min_max_scaler = MinMaxScaler()
    return min_max_scaler.fit_transform(data)


if __name__ == '__main__':
    main()
