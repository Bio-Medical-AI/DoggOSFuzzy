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

THRESHOLD = 0.0
PARAM_LOWER_BOUND = -400
PARAM_UPPER_BOUND = 400
FUZZY_DEBUG = True
PSO_DEBUG = True

n_params = {
    'Breast Cancer Wisconsin': 20,
    'Breast Cancer Wisconsin StdPCA': 10,
    'Data Banknote Auth': 10,
    'Data Banknote Auth StdPCA': 6,
    'HTRU': 18,
    'HTRU StdPCA': 10,
    'Immunotherapy': 16,
    'Immunotherapy StdPCA': 14,
    'Ionosphere': 70,
    'Ionosphere StdPCA': 20,
    'Parkinson': 46,
    'Parkinson StdPCA': 20,
    'Pima Indians Diabetes': 18,
    'Pima Indians Diabetes StdPCA': 10,
    'wdbc': 62,
    'wdbc StdPCA': 10
}


def main():
    seed_libs(42)
    pso_logger = Logger("test_cmaes", sys.argv[1])

    experiments = TSExperiments('data/' + sys.argv[1] + '.csv', ';', pso_logger)
    experiments.prepare_data([min_max_scale])

    cmaes = prepare_cmaes(n_params[sys.argv[1]])

    experiments.prepare_fuzzy_system(n_mfs=int(sys.argv[2]),
                                     mode=sys.argv[3],
                                     adjustment=sys.argv[4],
                                     lower_scaling=int(sys.argv[5]),
                                     fuzzy_set_type='it2')

    experiments.select_optimal_parameters(threshold_classification(THRESHOLD),
                                          metaheuristic=cmaes,
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


def prepare_cmaes(n_param):
    x0 = denormalize(np.random.random(n_param), PARAM_LOWER_BOUND, PARAM_UPPER_BOUND)
    kwargs = {
        'tolstagnation': 100,
        'popsize': 20
    }
    cmaes = CMAES(x0=x0,
                  maxfevals=20000,
                  sigma=0.7,
                  restarts=1,
                  incpopsize=2,
                  **kwargs)
    return CMAESWrapper(cmaes, n_param)


def min_max_scale(data: np.ndarray):
    min_max_scaler = MinMaxScaler()
    return min_max_scaler.fit_transform(data)


def seed_libs(seed=42):
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    main()
