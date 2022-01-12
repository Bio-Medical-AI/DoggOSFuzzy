from collections import defaultdict, OrderedDict
from typing import Dict, List, Any, Callable
import json

from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.feature_selection import GenericUnivariateSelect, RFE, \
    SelectFromModel, RFECV, SequentialFeatureSelector
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_validate, StratifiedKFold
from sklearn.neighbors import NeighborhoodComponentsAnalysis, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import KernelPCA, FactorAnalysis, FastICA, NMF, PCA

from pprint import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

from doggos.fuzzy_sets import Type1FuzzySet
from doggos.fuzzy_sets.fuzzy_set import FuzzySet
from doggos.induction.information_system import InformationSystem
from doggos.inference import MamdaniInferenceSystem
from doggos.inference.defuzzification_algorithms import center_of_gravity, karnik_mendel
from doggos.inference.inference_system import InferenceSystem
from doggos.knowledge import Rule, Clause, fuzzify, LinguisticVariable, Domain
from doggos.knowledge.consequents import MamdaniConsequent
from doggos.knowledge.consequents.consequent import Consequent
from doggos.utils.grouping_functions import create_set_of_variables
from doggos.utils.membership_functions.membership_functions import generate_equal_gausses, sigmoid, gaussian


def classify(data):
    preds = []
    for elem in data:
        if elem >= 0.5:
            preds.append(0.0)
        else:
            preds.append(1.0)
    return preds


def normalize(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)


class DataLoader:
    def __init__(self, X: np.ndarray, y: np.ndarray, feature_labels: List[str], target_label: str):
        self.X = X
        self.X_transformed = None
        self.X_train = None
        self.X_train_frame = None
        self.X_test = None
        self.X_test_frame = None
        self.y = y
        self.y_train = None
        self.y_train_frame = None
        self.y_test = None
        self.y_test_frame = None
        self.feature_labels = feature_labels
        self.target_label = target_label

    def prepare_data(self, transforms, test_size):
        self.X_transformed = self.X.copy()
        for transform in transforms:
            self.X_transformed = transform(self.X_transformed)
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X_transformed, self.y, test_size=test_size, stratify=self.y)
        self.X_train_frame = pd.DataFrame(data=self.X_train, columns=self.feature_labels)
        self.X_test_frame = pd.DataFrame(data=self.X_test, columns=self.feature_labels)
        self.y_train_frame = pd.Series(self.y_train)
        self.y_test_frame = pd.Series(self.y_test)


class FeatureSelector:
    def __init__(self, selector, **kwargs):
        self.__selector_class = selector
        self.__selector = selector(kwargs)

    def fit_transform(self, X, y=None):
        return self.__selector.fit_transform(X, y)

    def set_params(self, **kwargs):
        self.__selector = self.__selector_class(**kwargs)

    def search(self, linear_params, X_train, y=None, X_test=None):
        train_transformations = []
        test_transformations = []
        params_count = len(list(linear_params.values())[0])
        for i in range(params_count):
            params = {}
            for key in linear_params.keys():
                params[key] = linear_params[key][i]
            self.set_params(**params)
            train_transformations.append(self.fit_transform(X_train, y))
            test_transformations.append(self.__selector.transform(X_test))
        return train_transformations, test_transformations


class Evaluator(BaseEstimator):
    def __init__(self,
                 model: Any = None,
                 feature_labels: List[str] = None,
                 consequent_labels: List[str] = None,
                 inference_system: InferenceSystem.__class__ = None,
                 fuzzy_set_type: str = None,
                 n_mf: int = None,
                 mf_type: str = None,
                 con_mf_type: str = None,
                 con_n_mf: int = None,
                 domain: Domain = Domain(0, 1.001, 0.001),
                 fuzzy_sets: Dict[str, Dict[str, FuzzySet]] = None,
                 clauses: Dict[str, Dict[str, Clause]] = None,
                 consequents: Dict[str, Consequent] = None,
                 lower_scaling: float = 0.8,
                 mode: str = 'equal',
                 middle_vals: float = 0.5
                 ):
        self.model = model
        self.feature_labels = feature_labels
        self.consequent_labels = consequent_labels
        self.inference_system = inference_system
        if fuzzy_set_type == 't1':
            self.defuzz_method = center_of_gravity
        else:
            self.defuzz_method = karnik_mendel
        self.fuzzy_set_type = fuzzy_set_type
        self.n_mf = n_mf
        self.mf_type = mf_type
        self.con_mf_type = con_mf_type
        self.con_n_mf = con_n_mf
        self.domain = domain
        self.fuzzy_sets = fuzzy_sets
        self.clauses = clauses
        self.consequents = consequents
        self.rules = None
        self.antecedents = None
        self.lower_scaling = lower_scaling
        self.mode = mode
        self.middle_vals = middle_vals

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            return self
        valid_params = self.get_params(deep=True)
        sorted_keys = sorted(list(params.keys()))
        nested_params = defaultdict(dict)
        for key in sorted_keys:
            value = params[key]
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))
            if delim:
                sub_params = list(valid_params[key].get_params().keys())
                if sub_key in sub_params:
                    nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        if self.clauses is None:
            self._create_fuzzy_variables()

        if self.fuzzy_set_type == 't1':
            self.defuzz_method = center_of_gravity
        else:
            self.defuzz_method = karnik_mendel
        return self

    def _create_fuzzy_variables(self):
        _, fuzzy_sets, clauses = create_set_of_variables(self.feature_labels,
                                                         mf_type=self.mf_type,
                                                         n_mfs=self.n_mf,
                                                         fuzzy_set_type=self.fuzzy_set_type,
                                                         lower_scaling=self.lower_scaling,
                                                         mode=self.mode,
                                                         middle_vals=self.middle_vals)

        _, _, con_clauses = create_set_of_variables(self.consequent_labels,
                                                    mf_type=self.con_mf_type,
                                                    n_mfs=self.con_n_mf,
                                                    fuzzy_set_type=self.fuzzy_set_type,
                                                    lower_scaling=self.lower_scaling)
        self.fuzzy_sets = fuzzy_sets
        self.clauses = clauses
        self.consequents = {}
        for label in self.consequent_labels:
            clause = list(con_clauses[label].values())[0]
            self.consequents[label] = MamdaniConsequent(clause)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        information_system = InformationSystem(X, y, self.feature_labels)
        self.antecedents, _ = information_system.induce_rules(self.fuzzy_sets, self.clauses)
        self.rules = [Rule(self.antecedents[decision], self.consequents[decision])
                      for decision in self.consequent_labels]
        fuzzified_dataset = fuzzify(X, self.clauses)
        inference_system = self.inference_system(self.rules)
        inference = inference_system.infer(self.defuzz_method, fuzzified_dataset)
        self.model.fit(inference.reshape(-1, 1), y)

    def predict(self, X: pd.DataFrame):
        fuzzified_dataset = fuzzify(X, self.clauses)
        inference_system = self.inference_system(self.rules)
        inference = inference_system.infer(self.defuzz_method, fuzzified_dataset)
        y_pred = self.model.predict(inference.reshape(-1, 1))
        return y_pred


def find_best_selector(models, linear_params, X_train, y=None, X_test=None):
    train_transformed = []
    test_transformed = []
    for model, params in zip(models, linear_params):
        feature_selector = FeatureSelector(model)
        X_train_projected, X_test_projected = feature_selector.search(params, X_train, y, X_test)
        train_transformed.extend(X_train_projected)
        test_transformed.extend(X_test_projected)
        for projected in X_train_projected:
            plt.scatter(projected[:, 0], projected[:, 1], c=y)
            plt.title(str(model))
            plt.show()
    return train_transformed, test_transformed


full_dataset = pd.read_csv("../data/diabetes.csv")
target_label = 'Outcome'
feature_labels = list(full_dataset.columns)
feature_labels.remove(target_label)
X = full_dataset.values[:, :-1]
y = full_dataset.values[:, -1]
transforms = [normalize]
dataloader = DataLoader(X, y, feature_labels, target_label)
dataloader.prepare_data(transforms, test_size=0.2)

# Probably Best Selectors for Pima Indians:
# RFE, NCA
models = [NeighborhoodComponentsAnalysis, RFE]

linear_params = [
    # NCA
    {
        'init': ['pca', 'identity', 'random', 'pca', 'identity', 'random', 'pca', 'identity', 'random'],
        'n_components': [2, 2, 2, 4, 4, 4, 6, 6, 6]
    },
    # RFE
    {
        'estimator': [RandomForestClassifier(), ExtraTreesClassifier(), RandomForestClassifier(),
                      ExtraTreesClassifier(), RandomForestClassifier(), ExtraTreesClassifier()],
        'n_features_to_select': [2, 2, 4, 4, 6, 6]
    }
]

feature_labels_by_df = []
X_train_dfs, X_test_dfs = \
    find_best_selector(models, linear_params, dataloader.X_train, dataloader.y_train, dataloader.X_test)

for i, df in enumerate(X_train_dfs.copy()):
    columns = [feature_labels[column] for column in range(df.shape[1])]
    X_train_dfs[i] = pd.DataFrame(data=normalize(df), columns=columns)
    feature_labels_by_df.append(columns)

for i, df in enumerate(X_test_dfs.copy()):
    columns = [feature_labels[column] for column in range(df.shape[1])]
    X_test_dfs[i] = pd.DataFrame(data=normalize(df), columns=columns)

feature_labels_by_df.append(feature_labels)
X_train_dfs.append(dataloader.X_train_frame)
X_test_dfs.append(dataloader.X_test_frame)

# Trying to check if it is possible to use GaussianMixture instead of ProgressiveGauss
"""
for X_train in X_train_dfs:
    mix = GaussianMixture(2)
    mix.fit(X_train, dataloader.y_train)
    print(mix.weights_)
    print(mix.means_)
"""
con_labels = np.unique(dataloader.y)
con_labels = [str(label) for label in con_labels]
models = [RandomForestClassifier(), KNeighborsClassifier(), DecisionTreeClassifier()]
models_params = [
    # RandomForestClassifier
    {
        'model__n_estimators': [50, 100, 200, 400],
        'model__criterion': ['gini', 'entropy'],
        'model__max_features': [None, 'sqrt', 'log2']
    },
    # KNeighborsClassifier
    {
        'model__n_neighbors': [5, 10, 20, 50],
        'model__weights': ['uniform', 'distance']
    },
    # DecisionTreeClassifier
    {
        'model__criterion': ['gini', 'entropy'],
        'model__max_features': [None, 'sqrt', 'log2']
    }
]
means = []
i = 0
for X_train in X_train_dfs:
    means.append([])
    for column in X_train.columns:
        means[i].append(np.mean(X_train.loc[:, column]))
    i += 1

accuracies = []
f1_scores = []
best_parameters = []

with open('best_params.txt', mode='a') as f:
    for labels, train_df, test_df, df_means in zip(feature_labels_by_df, X_train_dfs, X_test_dfs, means):
        for model, params in zip(models, models_params):
            grid_params = {
                'model': [model],
                'feature_labels': [labels],
                'consequent_labels': [con_labels],
                'inference_system': [MamdaniInferenceSystem],
                'domain': [Domain(0, 1.001, 0.001), Domain(0, 1.0001, 0.0001)],
                'fuzzy_set_type': ['t1', 'it2'],
                'n_mf': [5, 7, 9, 11],
                'mf_type': ['gaussian'],#, 'triangular', 'trapezoidal'],
                'con_n_mf': [2],
                'con_mf_type': ['gaussian', 'triangular', 'trapezoidal', 'sigmoid'],
                'lower_scaling': [0.7, 0.8, 0.9],
                'mode': ['progressive'],
                'middle_vals': [df_means]
            }
            grid_params.update(params)
            # 4608 candidates for random forest
            evaluator = Evaluator()
            random_search = RandomizedSearchCV(evaluator, grid_params, n_iter=500, n_jobs=4,
                                               cv=StratifiedKFold(n_splits=5, shuffle=True), verbose=1, random_state=42,
                                               scoring='f1', error_score=np.nan)
            random_search.fit(train_df, dataloader.y_train_frame)
            y_pred = random_search.predict(test_df)

            acc = accuracy_score(dataloader.y_test, y_pred)
            f1 = f1_score(dataloader.y_test, y_pred)

            print(random_search.best_params_)
            print('\nAccuracy: ', acc)
            print('F1 Score: ', f1)
            accuracies.append(acc)
            f1_scores.append(f1)
            best_parameters.append(random_search.best_params_)
            #json.dump(random_search.best_params_, f)

for acc, f1, best in zip(accuracies, f1_scores, best_parameters):
    print(best)
    print('\nAccuracy: ', acc)
    print('F1 Score: ', f1)
