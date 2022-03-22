from functools import partial
import sys
import random

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.util.normalization import denormalize

from Logger import Logger
from TS_Experiments import TSExperiments
from metaheuristics_wrapper import PSO, CMAESWrapper

import warnings

warnings.filterwarnings("ignore")

THRESHOLD = 0.5
n_mfs = [3, 5, 7, 9, 11]
modes = ['equal', 'progressive']
adjustments = ['center', 'mean']
lower_scalings = np.flip(np.arange(0.5, 0.96, 0.05))
PARAM_LOWER_BOUND = -400
PARAM_UPPER_BOUND = 400
FUZZY_DEBUG = True
PSO_DEBUG = True

PARAMS_VALUES = {
    'Breast Cancer Data':
        {
            'MAXITER': 377,
            'SWARMSIZE': 53,
            'PHIG': 4.8976,
            'PHIP': -0.2746,
            'OMEGA': -0.3488
        },
    'Breast Cancer Data PCA':
        {
            'MAXITER': 377,
            'SWARMSIZE': 53,
            'PHIG': 4.8976,
            'PHIP': -0.2746,
            'OMEGA': -0.3488
        },
    'Breast Cancer Data StdPCA':
        {
            'MAXITER': 377,
            'SWARMSIZE': 53,
            'PHIG': 4.8976,
            'PHIP': -0.2746,
            'OMEGA': -0.3488
        },
    'Breast Cancer Wisconsin':
        {
            'MAXITER': 150,
            'SWARMSIZE': 69,
            'PHIG': 3.3950,
            'PHIP': -0.2699,
            'OMEGA': -0.4438
        },
    'Breast Cancer Wisconsin PCA':
        {
            'MAXITER': 377,
            'SWARMSIZE': 53,
            'PHIG': 4.8976,
            'PHIP': -0.2746,
            'OMEGA': -0.3488
        },
    'Breast Cancer Wisconsin StdPCA':
        {
            'MAXITER': 377,
            'SWARMSIZE': 53,
            'PHIG': 4.8976,
            'PHIP': -0.2746,
            'OMEGA': -0.3488
        },
    'Data Banknote Auth':
        {
            'MAXITER': 377,
            'SWARMSIZE': 53,
            'PHIG': 4.8976,
            'PHIP': -0.2746,
            'OMEGA': -0.3488
        },
    'Data Banknote Auth PCA':
        {
            'MAXITER': 45,
            'SWARMSIZE': 223,
            'PHIG': 3.3657,
            'PHIP': -0.1207,
            'OMEGA': -0.3699
        },
    'Data Banknote Auth StdPCA':
        {
            'MAXITER': 45,
            'SWARMSIZE': 223,
            'PHIG': 3.3657,
            'PHIP': -0.1207,
            'OMEGA': -0.3699
        },
    'HTRU':
        {
            'MAXITER': 377,
            'SWARMSIZE': 53,
            'PHIG': 4.8976,
            'PHIP': -0.2746,
            'OMEGA': -0.3488
        },
    'HTRU PCA':
        {
            'MAXITER': 377,
            'SWARMSIZE': 53,
            'PHIG': 4.8976,
            'PHIP': -0.2746,
            'OMEGA': -0.3488
        },
    'HTRU StdPCA':
        {
            'MAXITER': 377,
            'SWARMSIZE': 53,
            'PHIG': 4.8976,
            'PHIP': -0.2746,
            'OMEGA': -0.3488
        },
    'Immunotherapy':
        {
            'MAXITER': 377,
            'SWARMSIZE': 53,
            'PHIG': 4.8976,
            'PHIP': -0.2746,
            'OMEGA': -0.3488
        },
    'Immunotherapy PCA':
        {
            'MAXITER': 377,
            'SWARMSIZE': 53,
            'PHIG': 4.8976,
            'PHIP': -0.2746,
            'OMEGA': -0.3488
        },
    'Immunotherapy StdPCA':
        {
            'MAXITER': 377,
            'SWARMSIZE': 53,
            'PHIG': 4.8976,
            'PHIP': -0.2746,
            'OMEGA': -0.3488
        },
    'Ionosphere':
        {
            'MAXITER': 188,
            'SWARMSIZE': 106,
            'PHIG': 3.8876,
            'PHIP': -0.1564,
            'OMEGA': -0.2256
        },
    'Ionosphere PCA':
        {
            'MAXITER': 290,
            'SWARMSIZE': 69,
            'PHIG': 3.3950,
            'PHIP': -0.2699,
            'OMEGA': -0.4438
        },
    'Ionosphere StdPCA':
        {
            'MAXITER': 290,
            'SWARMSIZE': 69,
            'PHIG': 3.3950,
            'PHIP': -0.2699,
            'OMEGA': -0.4438
        },
    'Parkinson':
        {
            'MAXITER': 188,
            'SWARMSIZE': 106,
            'PHIG': 3.8876,
            'PHIP': -0.1564,
            'OMEGA': -0.2256
        },
    'Parkinson PCA':
        {
            'MAXITER': 377,
            'SWARMSIZE': 53,
            'PHIG': 4.8976,
            'PHIP': -0.2746,
            'OMEGA': -0.3488
        },
    'Parkinson StdPCA':
        {
            'MAXITER': 377,
            'SWARMSIZE': 53,
            'PHIG': 4.8976,
            'PHIP': -0.2746,
            'OMEGA': -0.3488
        },
    'Pima Indians Diabetes':
        {
            'MAXITER': 377,
            'SWARMSIZE': 53,
            'PHIG': 4.8976,
            'PHIP': -0.2746,
            'OMEGA': -0.3488
        },
    'Pima Indians Diabetes PCA':
        {
            'MAXITER': 377,
            'SWARMSIZE': 53,
            'PHIG': 4.8976,
            'PHIP': -0.2746,
            'OMEGA': -0.3488
        },
    'Pima Indians Diabetes StdPCA':
        {
            'MAXITER': 377,
            'SWARMSIZE': 53,
            'PHIG': 4.8976,
            'PHIP': -0.2746,
            'OMEGA': -0.3488
        },
    'Pima Indians Diabetes KernelPCA':
        {
            'MAXITER': 377,
            'SWARMSIZE': 53,
            'PHIG': 4.8976,
            'PHIP': -0.2746,
            'OMEGA': -0.3488
        }
}


# Parameters from article "Good parameters for particle swarm optimization"


def main():
    seed_libs(42)
    pso_logger = Logger("base_cmaes", sys.argv[1])

    experiments = TSExperiments('data/' + sys.argv[1] + '.csv', ';', pso_logger)
    experiments.prepare_data([min_max_scale])

    for ls in lower_scalings:
        for mode in modes:
            for adjustment in adjustments:
                for n_mf in n_mfs:
                    print(f'mode: {mode} adjustment: {adjustment} n_mfs {n_mf} lower_scaling {ls}')
                    experiments.prepare_fuzzy_system(n_mfs=n_mf, mode=mode, adjustment=adjustment, lower_scaling=ls,
                                                     fuzzy_set_type='it2')

                    cmaes = prepare_cmaes()

                    experiments.select_optimal_parameters_kfold(threshold_classification(THRESHOLD),
                                                                metaheuristic=cmaes,
                                                                debug=True,
                                                                ros=True)


def threshold_classification(theta):
    def _classify(x):
        if x <= theta:
            return 0
        elif x > theta:
            return 1
        else:
            print('else')

    return _classify


def prepare_cmaes():
    x0 = denormalize(np.random.random(10), PARAM_LOWER_BOUND, PARAM_UPPER_BOUND)
    kwargs = {
        'tolstagnation': 100,
        'popsize': 20
    }
    cmaes = CMAES(x0=x0,
                  maxfevals=20000,
                  sigma=0.99,
                  restarts=1,
                  incpopsize=2,
                  **kwargs)
    return CMAESWrapper(cmaes)


def min_max_scale(data: np.ndarray):
    min_max_scaler = MinMaxScaler()
    return min_max_scaler.fit_transform(data)


def seed_libs(seed=42):
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    main()
