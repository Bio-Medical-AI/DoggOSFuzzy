import sys
from functools import partial

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import differential_evolution

from Logger import Logger
from TS_Experiments import TSExperiments
from metaheuristics_wrapper import DifferentialEvolution, PSO


THRESHOLD = 0.5
n_mfs = [3, 5, 7, 9, 11]
modes = ['equal', 'progressive']
adjustments = ['center', 'mean']
lower_scalings = np.arange(0.5, 0.96, 0.05)
PARAM_LOWER_BOUND = -250
PARAM_UPPER_BOUND = 250
STRATEGY = 'rand1bin'
UPDATING = 'immediate'
DE_DEBUG = True
FUZZY_DEBUG = True

PARAMS_VALUES = {
    'Haberman':
        {
          'NP': 28,
          'MAXITER': 72,
          'CR': 0.9426,
          'DIFFERENTIAL_WEIGHT': 0.6607
        },
    'Breast Cancer Data':
        {
          'NP': 28,
          'MAXITER': 72,
          'CR': 0.9426,
          'DIFFERENTIAL_WEIGHT': 0.6607
        },
    'Breast Cancer Wisconsin':
        {
          'NP': 28,
          'MAXITER': 72,
          'CR': 0.9426,
          'DIFFERENTIAL_WEIGHT': 0.6607
        },
    'diabetes':
        {
          'NP': 28,
          'MAXITER': 72,
          'CR': 0.9426,
          'DIFFERENTIAL_WEIGHT': 0.6607
        },
    'HTRU':
        {
          'NP': 28,
          'MAXITER': 72,
          'CR': 0.9426,
          'DIFFERENTIAL_WEIGHT': 0.6607
        },
    'Immunotherapy':
        {
          'NP': 28,
          'MAXITER': 72,
          'CR': 0.9426,
          'DIFFERENTIAL_WEIGHT': 0.6607
        },
    'Pima Indians Diabetes':
        {
          'NP': 28,
          'MAXITER': 72,
          'CR': 0.9426,
          'DIFFERENTIAL_WEIGHT': 0.6607
        },
    'Vertebral':
        {
          'NP': 28,
          'MAXITER': 72,
          'CR': 0.9426,
          'DIFFERENTIAL_WEIGHT': 0.6607
        },
}


def main():
    de_logger = Logger("de", sys.argv[1])

    experiments = TSExperiments('data/' + sys.argv[1] + '.csv', ';', de_logger)
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

                    de_partial = prepare_de(experiments.n_params, PARAMS_VALUES[sys.argv[1]])

                    experiments.select_optimal_parameters_kfold_ensemble(threshold_classification(THRESHOLD),
                                                                         metaheuristic=de_partial,
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


def prepare_de(n_params, params_values):
    lb = [PARAM_LOWER_BOUND] * n_params
    ub = [PARAM_UPPER_BOUND] * n_params
    bounds = []
    for l, u in zip(lb, ub):
        bounds.append((l, u))

    de_partial = partial(differential_evolution,
                         bounds=bounds,
                         maxiter=params_values['MAXITER'],
                         popsize=params_values['NP'],
                         mutation=params_values['DIFFERENTIAL_WEIGHT'],
                         recombination=params_values['CR'],
                         seed=42,
                         updating=UPDATING,
                         workers=1,
                         disp=DE_DEBUG)
    return DifferentialEvolution(de_partial)


def min_max_scale(data: np.ndarray):
    min_max_scaler = MinMaxScaler()
    return min_max_scaler.fit_transform(data)


if __name__ == '__main__':
    main()
