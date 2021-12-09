import numpy as np
from sklearn.preprocessing import MinMaxScaler

from TS_Experiments import TSExperiments


def main():
    THRESHOLD = 0.5

    experiments = TSExperiments('data/Haberman.csv', ';')
    experiments.prepare_data([min_max_scale])
    experiments.prepare_fuzzy_system(n_mfs=9, fuzzy_set_type='it2')
    experiments.select_optimal_parameters_kfold(threshold_classification(THRESHOLD),
                                                debug=True,
                                                n_folds=6,
                                                ga_swarmsize=60,
                                                ga_phip=1,
                                                ga_phig=1,
                                                ga_omega=1)


def threshold_classification(theta):
    def _classify(x):
        if x <= theta:
            return 0
        elif x > theta:
            return 1
        else:
            print('else')

    return _classify


def min_max_scale(data: np.ndarray):
    min_max_scaler = MinMaxScaler()
    return min_max_scaler.fit_transform(data)


if __name__ == '__main__':
    main()
