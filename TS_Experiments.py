import pandas as pd
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
    def __init__(self, filepath, sep, logger, test_size=0.3, params_lower_bound=-10, params_upper_bound=10):
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
        self.train, self.test = train_test_split(self.transformed_data,
                                                 stratify=self.transformed_data['Decision'],
                                                 test_size=self.test_size)
        self.train_y = self.train['Decision']
        self.test_y = self.test['Decision']
        self.feature_names = list(self.data.columns[:-1])

    def prepare_fuzzy_system(self,
                             fuzzy_domain=Domain(0, 1.001, 0.001),
                             mf_type='gaussian',
                             n_mfs=3,
                             fuzzy_set_type='t1',
                             mode='equal',
                             adjustment="Center",
                             lower_scaling=0.8,
                             middle_vals=0.5):
        self.n_mfs = n_mfs
        self.mode = mode
        self.adjustment = adjustment
        self.lower_scaling = lower_scaling
        self.decision = LinguisticVariable(self.decision_name, Domain(0, 1.001, 0.001))
        self.ling_vars, self.fuzzy_sets, self.clauses = create_set_of_variables(
            ling_var_names=self.feature_names,
            domain=fuzzy_domain,
            mf_type=mf_type,
            n_mfs=n_mfs,
            fuzzy_set_type=fuzzy_set_type,
            mode=mode,
            lower_scaling=lower_scaling,
            middle_vals=middle_vals
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

            rules, train_fitness = self.__fit_fitness(train_idx, classification)

            start = time.time()
            lin_fun_params_optimal = metaheuristic(train_fitness)
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
        self.logger.log(best_val_f1, f1, n_fold, self.n_mfs, self.mode, self.adjustment, self.lower_scaling)

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
        y_pred_eval = list(map(lambda x: classification(x), result_eval[self.decision]))
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
