import pandas as pd
from pyswarm import pso
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold

from doggos.induction import InformationSystem
from doggos.inference import TakagiSugenoInferenceSystem
from doggos.inference.defuzzification_algorithms import takagi_sugeno_EIASC
from doggos.knowledge import LinguisticVariable, Domain, fuzzify, Rule
from doggos.knowledge.consequents import TakagiSugenoConsequent
from doggos.utils.grouping_functions import create_set_of_variables

import random
import time


class TSExperiments:
    def __init__(self, filepath, sep, test_size=0.3, params_lower_bound=-10, params_upper_bound=10):
        self.__filepath = filepath
        self.__sep = sep
        self.__data = None
        self.__n_classes = None
        self.__transformed_data = None
        self.__test_size = test_size
        self.__train = None
        self.__train_y = None
        self.__test = None
        self.__test_y = None
        self.__feature_names = None
        self.__decision_name = None
        self.__decision = None
        self.__ling_vars = None
        self.__fuzzy_sets = None
        self.__clauses = None
        self.__biases = None
        self.__consequents = None
        self.__n_params = None
        self.__params_lower_bound = params_lower_bound
        self.__params_upper_bound = params_upper_bound

    def prepare_data(self, transformations):
        self.__data = pd.read_csv(self.__filepath, sep=self.__sep)
        self.__decision_name = self.__data.columns[-1]
        self.__n_classes = len(self.__data[self.__decision_name].unique())

        transformed_data = self.__data.values
        for transform in transformations:
            transformed_data = transform(transformed_data)

        self.__transformed_data = pd.DataFrame(transformed_data, columns=self.__data.columns)
        self.__train, self.__test = train_test_split(self.__transformed_data,
                                                     stratify=self.__transformed_data['Decision'],
                                                     test_size=self.__test_size)
        self.__train_y = self.__train['Decision']
        self.__test_y = self.__test['Decision']
        self.__feature_names = list(self.__data.columns[:-1])

    def prepare_fuzzy_system(self,
                             fuzzy_domain=Domain(0, 1.001, 0.001),
                             mf_type='gaussian',
                             n_mfs=3,
                             fuzzy_set_type='t1',
                             mode='equal',
                             lower_scaling=0.8,
                             middle_vals=0.5):
        self.__decision = LinguisticVariable(self.__decision_name, Domain(0, 1.001, 0.001))
        self.__ling_vars, self.__fuzzy_sets, self.__clauses = create_set_of_variables(
            ling_var_names=self.__feature_names,
            domain=fuzzy_domain,
            mf_type=mf_type,
            n_mfs=n_mfs,
            fuzzy_set_type=fuzzy_set_type,
            mode=mode,
            lower_scaling=lower_scaling,
            middle_vals=middle_vals
        )

        self.__biases = []
        self.__consequents = []

        for i in range(self.__n_classes):
            self.__biases.append(random.uniform(-5, 5))
            parameters = {}
            for lv in self.__ling_vars:
                parameters[lv] = random.uniform(-5, 5)
            self.__consequents.append(TakagiSugenoConsequent(parameters, self.__biases[i], self.__decision))

        self.__n_params = len(self.__ling_vars) * len(self.__consequents) + len(self.__consequents)

    def select_optimal_parameters_kfold(self,
                                        classification,
                                        n_folds=10,
                                        shuffle=True,
                                        random_state=42,
                                        debug=False,
                                        ga_debug=False,
                                        ga_maxiter=30,
                                        ga_swarmsize=30,
                                        ga_phip=0.5,
                                        ga_phig=0.5,
                                        ga_omega=0.5):
        best_val_f1 = 0
        best_params = []
        best_rules = []
        skf = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
        n_fold = 0
        for train_idx, val_idx in skf.split(self.__train, self.__train_y):
            if debug:
                print(f'Fold {n_fold}')
                n_fold += 1

            rules, train_fitness = self.__fit_fitness(train_idx, classification)
            start = time.time()
            lin_fun_params_optimal, fitness_optimal = pso(train_fitness,
                                                          [self.__params_lower_bound] * self.__n_params,
                                                          [self.__params_upper_bound] * self.__n_params,
                                                          debug=ga_debug,
                                                          maxiter=ga_maxiter,
                                                          swarmsize=ga_swarmsize,
                                                          phig=ga_phig)
            end = time.time()
            print(f"pso: {end - start}")
            _, val_fitness = self.__fit_fitness(val_idx, classification, rules)
            val_f1 = 1 - val_fitness(lin_fun_params_optimal)
            if val_f1 > best_val_f1:
                if debug:
                    print(f"New best params in fold {n_fold} with f1 {val_f1}")
                best_val_f1 = val_f1
                best_params = lin_fun_params_optimal
                best_rules = rules

        test_fuzzified = fuzzify(self.__test, self.__clauses)
        test_measures = {}
        for idx, label in enumerate(self.__feature_names):
            test_measures[self.__ling_vars[idx]] = self.__test[label].values

        f1 = 1 - self.__fitness(best_params,
                                best_rules,
                                test_fuzzified,
                                self.__test_y,
                                test_measures,
                                classification)
        print(f'Final f1: {f1}')

    def __create_rules(self, train_X, train_y):
        train_X_fuzzified = fuzzify(train_X, self.__clauses)
        information_system = InformationSystem(train_X, train_y, self.__feature_names)
        antecedents, string_antecedents = information_system.induce_rules(self.__fuzzy_sets, self.__clauses)
        rules = []
        for idx, key in enumerate(antecedents.keys()):
            rules.append(Rule(antecedents[key], self.__consequents[idx]))
        return rules, string_antecedents, train_X_fuzzified

    def __fitness(self,
                  linear_fun_params,
                  rules,
                  fuzzified_data_X,
                  data_y,
                  measures,
                  classification):
        f_params1 = {}
        f_params2 = {}
        it = 0
        for idx, lv in enumerate(self.__ling_vars):
            f_params1[lv] = linear_fun_params[idx]
            it += 1

        for lv in self.__ling_vars:
            f_params2[lv] = linear_fun_params[it]
            it += 1

        rules[0].consequent.function_parameters = f_params1
        rules[1].consequent.function_parameters = f_params2
        rules[0].consequent.bias = linear_fun_params[it]
        it += 1
        rules[1].consequent.bias = linear_fun_params[it]

        ts = TakagiSugenoInferenceSystem(rules)
        result_eval = ts.infer(takagi_sugeno_EIASC, fuzzified_data_X, measures)
        y_pred_eval = list(map(lambda x: classification(x), result_eval[self.__decision]))
        f1 = f1_score(data_y.values, y_pred_eval)

        return 1 - f1

    def __fit_fitness(self, indexes, classification, rules=None):
        data_X = self.__train.iloc[indexes]
        data_y = data_X[self.__decision_name]
        if rules:
            train_X_fuzzified = fuzzify(data_X, self.__clauses)
        else:
            rules, string_antecedents, train_X_fuzzified = self.__create_rules(data_X, data_y)
        measures = {}
        for idx, label in enumerate(self.__feature_names):
            measures[self.__ling_vars[idx]] = data_X[label].values

        return rules, lambda parameters: self.__fitness(parameters,
                                                        rules,
                                                        fuzzified_data_X=train_X_fuzzified,
                                                        data_y=data_y,
                                                        measures=measures,
                                                        classification=classification)

    @property
    def data(self):
        return self.__data

    @property
    def transformed_data(self):
        return self.__transformed_data

    @property
    def train(self):
        return self.__train

    @property
    def test(self):
        return self.__test

    @property
    def train_y(self):
        return self.__train_y

    @property
    def test_y(self):
        return self.__test_y
