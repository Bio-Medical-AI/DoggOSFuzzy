from functools import partial
import sys
import random

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.util.normalization import denormalize

from Logger import Logger
from KvasirClassifier import KvasirExperiments
from metaheuristics_wrapper import CMAESWrapper

import warnings

warnings.filterwarnings("ignore")

THRESHOLD = 0.0
n_mfs = [3, 5, 7, 9, 11]
modes = ['equal', 'progressive']
adjustments = ['center', 'mean']
lower_scalings = np.flip(np.arange(0.5, 0.91, 0.05))
PARAM_LOWER_BOUND = -400
PARAM_UPPER_BOUND = 400
FUZZY_DEBUG = True
PSO_DEBUG = True

n_params = {
    'Kvasir': 10
}


def main():
    seed_libs(42)
    logger = Logger("kvasir_cmaes", 'kvasir')

    experiments = KvasirExperiments(logger)
    experiments.load_data('labelled_images/', ['angiectasia'])
    experiments.perform_pca()
    X_train_under_sampled = experiments.random_undersampling(experiments.X_train.join(experiments.y_train))
    print(X_train_under_sampled.value_counts('Label'))

    # for ls in lower_scalings:
    #     for mode in modes:
    #         for adjustment in adjustments:
    #             for n_mf in n_mfs:
    #                 print(f'mode: {mode} adjustment: {adjustment} n_mfs {n_mf} lower_scaling {ls}')
    #
    #                 cmaes = prepare_cmaes(n_params[sys.argv[1]])
    #
    #                 experiments.prepare_fuzzy_system(n_mfs=n_mf, mode=mode, adjustment=adjustment, lower_scaling=ls,
    #                                                  fuzzy_set_type='it2')
    #
    #                 experiments.select_optimal_parameters_kfold(threshold_classification(THRESHOLD),
    #                                                             metaheuristic=cmaes,
    #                                                             ros=True,
    #                                                             n_folds=int(sys.argv[2]))


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
