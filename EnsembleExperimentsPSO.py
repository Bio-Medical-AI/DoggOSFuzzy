import sys
from functools import partial

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pyswarm import pso

from Logger import Logger
from TS_Experiments import TSExperiments
from metaheuristics_wrapper import PSO


THRESHOLD = 0.5
n_mfs = [3, 5, 7, 9, 11]
modes = ['equal', 'progressive']
adjustments = ['center']
lower_scalings = np.arange(0.5, 1.01, 0.05)
PARAM_LOWER_BOUND = -250
PARAM_UPPER_BOUND = 250
FUZZY_DEBUG = True
PSO_DEBUG = True

PARAMS_VALUES = {
    'Haberman':
    {
        'MAXITER': 32,
        'SWARMSIZE': 63,
        'PHIG': 0.6239,
        'PHIP': 1.6319,
        'OMEGA': 0.6571
    },
    'Breast Cancer Data':
        {
            'MAXITER': 32,
            'SWARMSIZE': 63,
            'PHIG': 0.6239,
            'PHIP': 1.6319,
            'OMEGA': 0.6571
        },
    'Breast Cancer Wisconsin':
        {
            'MAXITER': 15,
            'SWARMSIZE': 69,
            'PHIG': 3.3950,
            'PHIP': -0.2699,
            'OMEGA': -0.4438
        },
    'diabetes':
        {
            'MAXITER': 32,
            'SWARMSIZE': 63,
            'PHIG': 0.6239,
            'PHIP': 1.6319,
            'OMEGA': 0.6571
        },
    'HTRU':
        {
            'MAXITER': 32,
            'SWARMSIZE': 63,
            'PHIG': 0.6239,
            'PHIP': 1.6319,
            'OMEGA': 0.6571
        },
    'Immunotherapy':
        {
            'MAXITER': 32,
            'SWARMSIZE': 63,
            'PHIG': 0.6239,
            'PHIP': 1.6319,
            'OMEGA': 0.6571
        },
    'Pima Indians Diabetes':
        {
            'MAXITER': 32,
            'SWARMSIZE': 63,
            'PHIG': 0.6239,
            'PHIP': 1.6319,
            'OMEGA': 0.6571
        },
    'Vertebral':
        {
            'MAXITER': 32,
            'SWARMSIZE': 63,
            'PHIG': 0.6239,
            'PHIP': 1.6319,
            'OMEGA': 0.6571
        }
}


def main():
    pso_logger = Logger("pso", sys.argv[1])

    experiments = TSExperiments('data/' + sys.argv[1] + '.csv', ';', pso_logger)
    experiments.prepare_data([min_max_scale])

    if len(sys.argv) >= 3:
        N_FOLDS = int(sys.argv[2])
    else:
        N_FOLDS = 10
    if len(sys.argv) >= 4:
        N_CLASSIFIERS = int(sys.argv[3])
    else:
        N_CLASSIFIERS = 5

    for n_mf in n_mfs:
        for mode in modes:
            for adjustment in adjustments:
                for ls in lower_scalings:
                    experiments.prepare_fuzzy_system(n_mfs=n_mf, mode=mode, adjustment=adjustment, lower_scaling=ls, fuzzy_set_type='it2')

                    pso_partial = prepare_pso(experiments.n_params, PARAMS_VALUES[sys.argv[1]])

                    experiments.select_optimal_parameters_kfold_ensemble(threshold_classification(THRESHOLD),
                                                                         metaheuristic=pso_partial,
                                                                         debug=FUZZY_DEBUG,
                                                                         n_folds=N_FOLDS,
                                                                         n_classifiers=N_CLASSIFIERS)


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
                          maxiter=MAXITER,
                          swarmsize=SWARMSIZE,
                          phig=PHIG,
                          omega=OMEGA,
                          phip=PHIP)
    return PSO(pso_partial)


def min_max_scale(data: np.ndarray):
    min_max_scaler = MinMaxScaler()
    return min_max_scaler.fit_transform(data)


if __name__ == '__main__':
    main()
