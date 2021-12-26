import math

import numpy as np
import pandas as pd
from scipy.stats import norm
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


def make_n_splits(data, n):
    batch_size = int(len(data) / n)
    batches = []
    for i in range(0, n - 1):
        batches.append(data[batch_size * i: batch_size * (i + 1)])
    batches.append(data[batch_size * (n - 1):])
    return batches


def vote_highest(y_pred):
    y_np = []
    for data in y_pred:
        y_np.append(np.array(data))

    y_sum = np.array(y_np[0])
    for data in y_np[1:]:
        y_sum += data

    y_voted = []
    for y in y_sum:
        if y < math.floor(len(y_pred) / 2.0) + 1:
            y_voted.append(0)
        else:
            y_voted.append(1)

    return y_voted


class TSExperiments:
    def __init__(self, filepath, sep, logger, test_size=0.2, params_lower_bound=-10, params_upper_bound=10):
        self.filepath = filepath
        self.sep = sep
        self.data = None
        self.n_classes = None
        self.transformed_data = None
        self.test_size = test_size
        self.train = None
        self.train_y = None
        self.test = None
        self.test_y = None
        self.feature_names = None
        self.decision_name = None
        self.decision = None
        self.ling_vars = None
        self.fuzzy_sets = None
        self.clauses = None
        self.biases = None
        self.consequents = None
        self.n_params = None
        self.n_mfs = None
        self.mode = None
        self.adjustment = None
        self.lower_scaling = None
        self.mid_evs = None
        self.params_lower_bound = params_lower_bound
        self.params_upper_bound = params_upper_bound
        self.logger = logger

    def prepare_data(self, transformations):
        self.data = pd.read_csv(self.filepath, sep=self.sep)
        self.decision_name = self.data.columns[-1]
        self.n_classes = len(self.data[self.decision_name].unique())

        transformed_data = self.data.values
        for transform in transformations:
            transformed_data = transform(transformed_data)

        self.transformed_data = pd.DataFrame(transformed_data, columns=self.data.columns)
        self.transformed_data = self.transformed_data.round(3)
        self.train, self.test = train_test_split(self.transformed_data,
                                                 stratify=self.transformed_data['Decision'],
                                                 test_size=self.test_size)
        self.train_y = self.train['Decision']
        self.test_y = self.test['Decision']
        self.feature_names = list(self.data.columns[:-1])

        self.mid_evs = []
        for column in self.train.columns:
            mean, _ = norm.fit(self.train[column].values)
            self.mid_evs.append(mean)

    def prepare_fuzzy_system(self,
                             fuzzy_domain=Domain(0, 1.001, 0.001),
                             mf_type='gaussian',
                             n_mfs=3,
                             fuzzy_set_type='t1',
                             mode='equal',
                             adjustment="center",
                             lower_scaling=0.8,
                             middle_vals=0.5):
        self.n_mfs = n_mfs
        self.mode = mode
        self.adjustment = adjustment
        self.lower_scaling = lower_scaling
        self.decision = LinguisticVariable(self.decision_name, Domain(0, 1.001, 0.001))
        if adjustment == 'mean':
            middle_vals = self.mid_evs
        self.ling_vars, self.fuzzy_sets, self.clauses = create_set_of_variables(
            ling_var_names=self.feature_names,
            domain=fuzzy_domain,
            mf_type=mf_type,
            n_mfs=n_mfs,
            fuzzy_set_type=fuzzy_set_type,
            mode=mode,
            lower_scaling=lower_scaling,
            middle_vals=middle_vals,
            adjustment=adjustment
        )

        self.biases = []
        self.consequents = []

        for i in range(self.n_classes):
            self.biases.append(random.uniform(-5, 5))
            parameters = {}
            for lv in self.ling_vars:
                parameters[lv] = random.uniform(-5, 5)
            self.consequents.append(TakagiSugenoConsequent(parameters, self.biases[i], self.decision))

        self.n_params = len(self.ling_vars) * len(self.consequents) + len(self.consequents)

    def select_optimal_parameters_kfold(self,
                                        classification,
                                        metaheuristic,
                                        n_folds=10,
                                        shuffle=True,
                                        random_state=42,
                                        debug=False):
        best_val_f1 = 0
        best_params = []
        best_rules = []
        skf = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
        n_fold = 0
        for train_idx, val_idx in skf.split(self.train, self.train_y):
            if debug:
                print(f'Fold {n_fold}')
                n_fold += 1

            start = time.time()
            rules, train_fitness = self.__fit_fitness(train_idx, classification)
            end = time.time()
            print(f"Induction: {end - start}")

            start = time.time()
            try:
                lin_fun_params_optimal = metaheuristic(train_fitness)
            except IndexError:
                continue
            end = time.time()
            print(f"Optimization: {end - start}")

            _, val_fitness = self.__fit_fitness(val_idx, classification, rules)
            val_f1 = 1 - val_fitness(lin_fun_params_optimal)
            if val_f1 > best_val_f1:
                if debug:
                    print(f"New best params in fold {n_fold} with f1 {val_f1}")
                best_val_f1 = val_f1
                best_params = lin_fun_params_optimal
                best_rules = rules

        test_fuzzified = fuzzify(self.test, self.clauses)
        test_measures = {}
        for idx, label in enumerate(self.feature_names):
            test_measures[self.ling_vars[idx]] = self.test[label].values

        f1 = 1 - self.__fitness(best_params,
                                best_rules,
                                test_fuzzified,
                                self.test_y,
                                test_measures,
                                classification)
        print(f'Final f1: {f1}')
        self.logger.log(best_val_f1, f1, n_folds, self.n_mfs, self.mode, self.adjustment, self.lower_scaling)

    def select_optimal_parameters_kfold_ensemble(self,
                                                 classification,
                                                 metaheuristic,
                                                 n_folds=10,
                                                 n_classifiers=5,
                                                 shuffle=True,
                                                 random_state=42,
                                                 debug=False):
        best_val_f1 = 0
        best_params = []
        best_rules = []
        skf = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
        n_fold = 0
        for train_idx, val_idx in skf.split(self.train, self.train_y):
            if debug:
                print(f'Fold {n_fold}')
                n_fold += 1

            indexes = make_n_splits(train_idx, n_classifiers)
            start = time.time()
            n_rules, train_fitness = self.__ensemble_fit_fitness_train(indexes, classification)
            end = time.time()
            print(f"Induction: {end - start}")

            start = time.time()
            try:
                lin_fun_params_optimal = metaheuristic(train_fitness)
            except IndexError:
                continue
            end = time.time()
            print(f"Ensemble optimization: {end - start}")

            _, val_fitness = self.__ensemble_fit_fitness_val(val_idx, classification, n_rules)
            val_f1 = 1 - val_fitness(lin_fun_params_optimal)
            if val_f1 > best_val_f1:
                if debug:
                    print(f"New best params in fold {n_fold} with f1 {val_f1}")
                best_val_f1 = val_f1
                best_params = lin_fun_params_optimal
                best_rules = n_rules
        if len(best_params) != 0:
            test_fuzzified = fuzzify(self.test, self.clauses)
            test_measures = {}
            for idx, label in enumerate(self.feature_names):
                test_measures[self.ling_vars[idx]] = self.test[label].values

            f1 = 1 - self.__ensemble_fitness(best_params,
                                             best_rules,
                                             test_fuzzified,
                                             self.test_y,
                                             test_measures,
                                             classification)
            print(f'Final f1: {f1}')
            self.logger.log(best_val_f1, f1, n_folds, self.n_mfs, self.mode, self.adjustment, self.lower_scaling)

    def __create_rules(self, train_X, train_y):
        train_X_fuzzified = fuzzify(train_X, self.clauses)
        information_system = InformationSystem(train_X, train_y, self.feature_names)
        antecedents, string_antecedents = information_system.induce_rules(self.fuzzy_sets, self.clauses)
        rules = []
        for idx, key in enumerate(antecedents.keys()):
            rules.append(Rule(antecedents[key], self.consequents[idx]))
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
        for idx, lv in enumerate(self.ling_vars):
            f_params1[lv] = linear_fun_params[idx]
            it += 1

        for lv in self.ling_vars:
            f_params2[lv] = linear_fun_params[it]
            it += 1

        rules[0].consequent.function_parameters = f_params1
        rules[1].consequent.function_parameters = f_params2
        rules[0].consequent.bias = linear_fun_params[it]
        it += 1
        rules[1].consequent.bias = linear_fun_params[it]

        ts = TakagiSugenoInferenceSystem(rules)
        result_eval = ts.infer(takagi_sugeno_EIASC, fuzzified_data_X, measures)
        y_pred_eval = [classification(x) for x in result_eval]
        f1 = f1_score(data_y.values, y_pred_eval)

        return 1 - f1

    def __fit_fitness(self, indexes, classification, rules=None):
        data_X = self.train.iloc[indexes]
        data_y = data_X[self.decision_name]
        if rules:
            train_X_fuzzified = fuzzify(data_X, self.clauses)
        else:
            rules, string_antecedents, train_X_fuzzified = self.__create_rules(data_X, data_y)
        measures = {}
        for idx, label in enumerate(self.feature_names):
            measures[self.ling_vars[idx]] = data_X[label].values

        return rules, lambda parameters: self.__fitness(parameters,
                                                        rules,
                                                        fuzzified_data_X=train_X_fuzzified,
                                                        data_y=data_y,
                                                        measures=measures,
                                                        classification=classification)

    def __ensemble_fitness(self,
                           linear_fun_params,
                           n_rules,
                           n_fuzzified_data_X,
                           data_y,
                           measures,
                           classification):
        models = []
        for rules in n_rules:
            f_params1 = {}
            f_params2 = {}
            it = 0
            for idx, lv in enumerate(self.ling_vars):
                f_params1[lv] = linear_fun_params[idx]
                it += 1

            for lv in self.ling_vars:
                f_params2[lv] = linear_fun_params[it]
                it += 1

            rules[0].consequent.function_parameters = f_params1
            rules[1].consequent.function_parameters = f_params2
            rules[0].consequent.bias = linear_fun_params[it]
            it += 1
            rules[1].consequent.bias = linear_fun_params[it]

            models.append(TakagiSugenoInferenceSystem(rules))

        results = []
        for model in models:
            results.append(model.infer(takagi_sugeno_EIASC, n_fuzzified_data_X, measures))
        y_pred = []
        for result in results:
            y_pred.append([classification(x) for x in result])
        # w tej chwili modele, dla któych reguły zostały utworzone z małych zbiorów danych przewidują etykiety dla
        # dużych zbiorów danych. Alternatywą jest, żeby każdy model przewidywał etykiety dla swoich danych,
        # i wtedy nie mamy głosowania większościowego, ale też ma to sens.
        # pomysł: optymalizacja metaheurystyką val_f1 a nie train_f1
        y_pred_ensemble = vote_highest(y_pred)
        f1 = f1_score(data_y, y_pred_ensemble)

        return 1 - f1

    def __ensemble_fit_fitness_train(self, indexes, classification):
        data_X = []
        data_y = []
        for i, idx in enumerate(indexes):
            data_X.append(self.train.iloc[idx])
            data_y.append(data_X[i][self.decision_name])

        n_rules = []
        n_train_X_fuzzified = {}
        for i in range(len(indexes)):
            rules, _, train_X_fuzzified = self.__create_rules(data_X[i], data_y[i])
            if i == 0:
                for key in train_X_fuzzified.keys():
                    n_train_X_fuzzified[key] = train_X_fuzzified[key]
            else:
                for key in train_X_fuzzified.keys():
                    n_train_X_fuzzified[key] = np.hstack((n_train_X_fuzzified[key], train_X_fuzzified[key]))
            n_rules.append(rules)

        measures = {}
        for n_model in range(len(indexes)):
            for idx, label in enumerate(self.feature_names):
                if n_model == 0:
                    measures[self.ling_vars[idx]] = []
                measures[self.ling_vars[idx]].extend(data_X[n_model][label].values)

        data_y_flattened = []
        for data in data_y:
            data_y_flattened.extend(data.values)

        return n_rules, lambda parameters: self.__ensemble_fitness(parameters,
                                                                   n_rules=n_rules,
                                                                   n_fuzzified_data_X=n_train_X_fuzzified,
                                                                   data_y=data_y_flattened,
                                                                   measures=measures,
                                                                   classification=classification)

    def __ensemble_fit_fitness_val(self, indexes, classification, n_rules):
        data_X = self.train.iloc[indexes]
        data_y = data_X[self.decision_name]

        measures = {}
        for idx, label in enumerate(self.feature_names):
            measures[self.ling_vars[idx]] = data_X[label].values

        val_X_fuzzified = fuzzify(data_X, self.clauses)

        return n_rules, lambda parameters: self.__ensemble_fitness(parameters,
                                                                   n_rules=n_rules,
                                                                   n_fuzzified_data_X=val_X_fuzzified,
                                                                   data_y=data_y,
                                                                   measures=measures,
                                                                   classification=classification)
