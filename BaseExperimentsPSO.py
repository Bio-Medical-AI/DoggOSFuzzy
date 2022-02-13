from functools import partial
import sys
import random

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pyswarm import pso

from Logger import Logger
from TS_Experiments import TSExperiments
from metaheuristics_wrapper import PSO

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
    pso_logger = Logger("base_pso", sys.argv[1])

    experiments = TSExperiments('data/' + sys.argv[1] + '.csv', ';', pso_logger)
    experiments.prepare_data([min_max_scale])

    for ls in lower_scalings:
        for mode in modes:
            for adjustment in adjustments:
                for n_mf in n_mfs:
                    print(f'mode: {mode} adjustment: {adjustment} n_mfs {n_mf} lower_scaling {ls}')
                    experiments.prepare_fuzzy_system(n_mfs=n_mf, mode=mode, adjustment=adjustment, lower_scaling=ls,
                                                     fuzzy_set_type='it2')

                    pso_partial = prepare_pso(experiments.n_params, PARAMS_VALUES[sys.argv[1]])

                    experiments.select_optimal_parameters(threshold_classification(THRESHOLD),
                                                          metaheuristic=pso_partial)


def threshold_classification(theta):
    def _classify(x):
        if x <= theta:
            return 0
        elif x > theta:
            return 1
        else:
            print('else')

    return _classify


def prepare_pso(n_params, params_values):
    pso_partial = partial(pso,
                          lb=[PARAM_LOWER_BOUND] * n_params,
                          ub=[PARAM_UPPER_BOUND] * n_params,
                          debug=PSO_DEBUG,
                          maxiter=params_values['MAXITER'],
                          swarmsize=params_values['SWARMSIZE'],
                          phig=params_values['PHIG'],
                          omega=params_values['OMEGA'],
                          phip=params_values['PHIP'])
    return PSO(pso_partial)


def min_max_scale(data: np.ndarray):
    min_max_scaler = MinMaxScaler()
    return min_max_scaler.fit_transform(data)


def seed_libs(seed=42):
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    main()
