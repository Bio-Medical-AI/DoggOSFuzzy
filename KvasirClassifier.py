import math
from copy import deepcopy

import numpy as np
import pandas as pd
import tqdm
from PIL import Image
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, balanced_accuracy_score, \
    roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from functools import partial

from sklearn.preprocessing import StandardScaler

from doggos.induction import InformationSystem
from doggos.inference import TakagiSugenoInferenceSystem
from doggos.inference.defuzzification_algorithms import takagi_sugeno_EIASC
from doggos.knowledge import LinguisticVariable, Domain, fuzzify, Rule
from doggos.knowledge.consequents import TakagiSugenoConsequent
from doggos.utils.grouping_functions import create_set_of_variables

from features_extraction import textural_features, red_prop_features, rgb_hsv_means, textural_features_mult_images, red_prop_features_mult_images, rgb_hsv_means_mult_images

import random
import os


def make_n_splits(data, n):
    batch_size = int(len(data.values) / n)
    batches = []
    for i in range(0, n - 1):
        batches.append(data.iloc[batch_size * i: batch_size * (i + 1)])
    batches.append(data[batch_size * (n - 1):])
    return batches


class KvasirExperiments:
    def __init__(self, logger, test_size=0.2, params_lower_bound=-10, params_upper_bound=10):
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
        self.val = None
        self.params_lower_bound = params_lower_bound
        self.params_upper_bound = params_upper_bound
        self.logger = logger

    def load_data(self, kvasir_dir_path, classes, pca_=False, standarize=False):
        labels = []
        images = []
        masks = []
        for idx, class_ in enumerate(classes):
            for path in tqdm.tqdm(os.listdir(os.path.join(kvasir_dir_path, class_))):
                path = os.path.join(kvasir_dir_path, class_, path)
                image = Image.open(path)
                images.append(np.array(image))
                masks.append(np.ones_like(images[-1][:, :, 0]))
                labels.append(idx)

        labels = np.array(labels)
        images = np.array(images)
        masks = np.array(masks)

        train_idxs, test_idxs = train_test_split(np.arange(0, len(labels), 1),
                                                 stratify=labels,
                                                 test_size=self.test_size,
                                                 random_state=42,
                                                 shuffle=True)

        train_undersampled = self.random_undersampling(pd.DataFrame({'Index': train_idxs, 'Label': labels[train_idxs]}))
        print(train_undersampled.value_counts('Label'))

        train_text_feats = textural_features_mult_images(images[train_undersampled['Index'].values],
                                                         masks[train_undersampled['Index'].values])
        test_text_feats = textural_features_mult_images(images[test_idxs],
                                                        masks[test_idxs])
        #red_feats = red_prop_features_mult_images(np.array(images), np.array(masks))
        #rgb_hsv_feats = rgb_hsv_means_mult_images(np.array(images), np.array(masks))
        #data = list(text_feats) + list(red_feats) + list(rgb_hsv_feats)
        data = []
        for train_feature, test_feature in zip(train_text_feats, test_text_feats):
            feature = train_feature.extend(test_feature)
            data.append(feature)

        df_dict = {}

        if pca_:
            pca = PCA()
            data = pca.fit_transform(data, labels)
            print(pca.explained_variance_ratio_)

        for i, features in enumerate(data):
            if standarize:
                scaler = StandardScaler()
                features = scaler.fit_transform(features, labels)
            df_dict[f'F{i}'] = features

        split = np.empty_like(data[0], dtype='object')
        split[:train_text_feats.shape[1]] = 'train'
        split[train_text_feats.shape[1]:] = 'test'
        df_dict['Split'] = split
        label = list(train_undersampled['Label'].values)
        label.extend(labels[test_idxs])
        df_dict['Label'] = label

        self.data = pd.DataFrame(df_dict)
        self.train, self.test = self.data.loc[self.data['split'] == 'train'], self.data.loc[self.data['split'] == 'test']
        self.train = self.train.drop(['Split'], axis=1)
        self.test = self.test.drop(['Split'], axis=1)
        self.train_y = self.train['Label']
        self.test_y = self.test['Label']

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

    def random_oversampling(self, df):
        new_df = pd.DataFrame(columns=df.columns)
        classes = df.value_counts('Label', sort=True)

        higher_class = 0
        for cls, val in classes.items():
            if classes[higher_class] < val:
                higher_class = cls

        for cls, _ in classes.items():
            cls_df = df[df['Label'] == cls]
            if cls != higher_class:
                resampled = cls_df.sample(classes[higher_class], replace=True, ignore_index=True)
            else:
                resampled = cls_df
            new_df = pd.concat([new_df, resampled], ignore_index=True)
        return new_df

    def random_undersampling(self, df):
        new_df = pd.DataFrame(columns=df.columns)
        classes = df.value_counts('Label', sort=True)
        print('df.value_counts w undersampling')
        print(classes)
        print(df.head())
        print(df.tail())

        lower_class = classes.keys()[0]
        print(classes[lower_class])

        for cls, _ in classes.items():
            cls_df = df[df['Label'] == cls]
            if cls != lower_class:
                resampled = cls_df.sample(classes[lower_class], replace=False, ignore_index=True, random_state=42)
            else:
                resampled = cls_df
            new_df = pd.concat([new_df, resampled], ignore_index=True)
            print(new_df.value_counts('Label', sort=True))
        return new_df

    def select_optimal_parameters(self,
                                  classification,
                                  metaheuristic,
                                  ros=False):
        train = self.train
        if ros:
            train = self.random_oversampling(train).astype('float')

        try:
            ts, rules, train_fitness = self.fit_fitness(train, classification)
        except ValueError:
            print("Induced only one rule")
            return

        lin_fun_params_optimal = metaheuristic(train_fitness)

        train_f1 = 1 - train_fitness(lin_fun_params_optimal)
        print(f'Train f1: {train_f1}')

        test_fuzzified = fuzzify(self.test, self.clauses)
        test_measures = {}
        for idx, label in enumerate(self.feature_names):
            test_measures[self.ling_vars[idx]] = self.test[label].values

        y_pred = self.predict(ts,
                              lin_fun_params_optimal,
                              rules,
                              test_fuzzified,
                              test_measures,
                              classification)
        f1, accuracy, recall, precision, balanced_accuracy, roc_auc = self.calc_metrics(self.test_y, y_pred)
        print(f'Test f1: {f1}')
        self.logger.log(train_f1, f1, accuracy, recall, precision, balanced_accuracy, roc_auc,
                        self.n_mfs, self.mode, self.adjustment, self.lower_scaling, lin_fun_params_optimal)

    def select_optimal_parameters_kfold(self,
                                        classification,
                                        metaheuristic,
                                        n_folds=10,
                                        shuffle=True,
                                        random_state=42,
                                        debug=False,
                                        ros=False):
        val_f1s = []
        val_accs = []
        val_precisions = []
        val_recalls = []
        val_balaccs = []
        val_roc_aucs = []
        skf = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
        n_fold = 0
        for train_idx, val_idx in skf.split(self.train, self.train_y):
            if debug:
                print(f'Fold {n_fold}')
                n_fold += 1

            train = self.train.iloc[train_idx]
            val = self.train.iloc[val_idx]
            val_y = val['Decision']
            if ros:
                train = self.random_oversampling(train).astype('float')

            try:
                ts, rules, train_fitness = self.fit_fitness(train, classification)
            except ValueError:
                print("Induced only one rule")
                return

            lin_fun_params_optimal = metaheuristic(train_fitness)

            val_fuzzified = fuzzify(val, self.clauses)
            val_measures = {}
            for idx, label in enumerate(self.feature_names):
                val_measures[self.ling_vars[idx]] = val[label].values

            y_pred = self.predict(ts,
                                  lin_fun_params_optimal,
                                  rules,
                                  val_fuzzified,
                                  val_measures,
                                  classification)

            f1, accuracy, recall, precision, balanced_accuracy, roc_auc = self.calc_metrics(val_y, y_pred)
            val_f1s.append(f1)
            val_accs.append(accuracy)
            val_recalls.append(recall)
            val_precisions.append(precision)
            val_balaccs.append(balanced_accuracy)
            val_roc_aucs.append(roc_auc)

        f1, accuracy, recall, precision, balanced_accuracy, roc_auc = np.mean(val_f1s), np.mean(val_accs), \
                                                                      np.mean(val_recalls), np.mean(val_precisions), \
                                                                      np.mean(val_balaccs), np.mean(val_roc_aucs)
        print(f'Mean val f1: {f1}')
        self.logger.log_kfold(f1, accuracy, recall, precision, balanced_accuracy, roc_auc,
                              self.n_mfs, self.mode, self.adjustment, self.lower_scaling, n_folds)

    def create_rules(self, train_X, train_y):
        train_X_fuzzified = fuzzify(train_X, self.clauses)
        information_system = InformationSystem(train_X, train_y, self.feature_names)
        antecedents, string_antecedents = information_system.induce_rules(self.fuzzy_sets, self.clauses)
        rules = []
        for idx, key in enumerate(antecedents.keys()):
            rules.append(Rule(antecedents[key], self.consequents[idx]))
        return rules, string_antecedents, train_X_fuzzified

    def fitness(self,
                linear_fun_params,
                ts,
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

        result_eval = ts.infer(takagi_sugeno_EIASC, fuzzified_data_X, measures)
        y_pred_eval = [classification(x) for x in result_eval]
        f1 = f1_score(data_y.values, y_pred_eval)

        return 1 - f1

    def fit_fitness(self, data_X, classification, rules=None):
        data_y = data_X[self.decision_name]
        if rules:
            train_X_fuzzified = fuzzify(data_X, self.clauses)
        else:
            rules, string_antecedents, train_X_fuzzified = self.create_rules(data_X, data_y)
        measures = {}
        for idx, label in enumerate(self.feature_names):
            measures[self.ling_vars[idx]] = data_X[label].values

        ts = TakagiSugenoInferenceSystem(rules)

        return ts, rules, partial(self.fitness,
                                  ts=ts,
                                  rules=rules,
                                  fuzzified_data_X=train_X_fuzzified,
                                  data_y=data_y,
                                  measures=measures,
                                  classification=classification)

    def predict(self,
                ts,
                linear_fun_params,
                rules,
                fuzzified_data_X,
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

        result_eval = ts.infer(takagi_sugeno_EIASC, fuzzified_data_X, measures)
        y_pred_eval = [classification(x) for x in result_eval]

        return y_pred_eval

    def calc_metrics(self, y_true, y_pred):
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)
        return f1, accuracy, recall, precision, balanced_accuracy, roc_auc
