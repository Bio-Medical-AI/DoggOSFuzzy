from collections import defaultdict
from typing import Dict, List, Any, Callable

from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.feature_selection import GenericUnivariateSelect, RFE, \
    SelectFromModel, RFECV, SequentialFeatureSelector
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_validate, StratifiedKFold
from sklearn.neighbors import NeighborhoodComponentsAnalysis, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score
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

    def search(self, linear_params, X, y=None):
        transformations = []
        params_count = len(list(linear_params.values())[0])
        for i in range(params_count):
            params = {}
            for key in linear_params.keys():
                params[key] = linear_params[key][i]
            self.set_params(**params)
            transformations.append(self.fit_transform(X, y))
        return transformations


class Evaluator:
    def __init__(self,
                 model: BaseEstimator = None,
                 feature_labels: List[str] = None,
                 consequent_labels: List[str] = None,
                 fuzzy_sets: Dict[str, Dict[str, FuzzySet]] = None,
                 clauses: Any = None,
                 consequents: Any = None,
                 inference_system: InferenceSystem.__class__ = None,
                 defuzz_method: Callable = None,
                 domain: Domain = Domain(0, 1.001, 0.001),
                 fuzzy_set_type: str = None,
                 n_mf: int = None,
                 mf_type: str = None,
                 con_mf_type: str = None,
                 con_n_mf: int = None):
        self.model = model
        self.feature_labels = feature_labels
        self.consequent_labels = consequent_labels
        self.fuzzy_sets = fuzzy_sets
        self.inference_system = inference_system
        self.defuzz_method = defuzz_method
        self.domain = domain
        self.fuzzy_set_type = fuzzy_set_type
        self.n_mf = n_mf
        self.mf_type = mf_type
        self.con_mf_type = con_mf_type
        self.con_n_mf = con_n_mf
        self.consequents = consequents
        self.rules = []
        self.clauses = clauses
        self.antecedents = None
        if fuzzy_sets is None and feature_labels is not None:
            self.__create_fuzzy_variables(True, True)

    def get_params(self, deep=True):
        param_names = ['model', 'feature_labels', 'consequent_labels', 'consequents', 'fuzzy_sets', 'clauses',
                       'inference_system', 'defuzz_method', 'domain', 'fuzzy_set_type', 'n_mf', 'mf_type',
                       'con_mf_type', 'con_n_mf']
        out = dict()
        for key in param_names:
            value = getattr(self, key)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        if not params:
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        if self.clauses is None:
            self.__create_fuzzy_variables(True, True)
        return self

    def __create_fuzzy_variables(self, create_antecedent_features: bool, create_consequents: bool):
        if create_antecedent_features:
            _, fuzzy_sets_, clauses = create_set_of_variables(self.feature_labels,
                                                              mf_type=self.mf_type,
                                                              n_mfs=self.n_mf,
                                                              fuzzy_set_type=self.fuzzy_set_type)
            self.fuzzy_sets = fuzzy_sets_
            self.clauses = clauses
        if create_consequents:
            _, _, con_clauses = create_set_of_variables(self.consequent_labels,
                                                        mf_type=self.con_mf_type,
                                                        n_mfs=self.con_n_mf,
                                                        fuzzy_set_type=self.fuzzy_set_type)
            self.consequents = {}
            for label in self.consequent_labels:
                clause = list(con_clauses[label].values())[0]
                self.consequents[label] = MamdaniConsequent(clause)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        information_system = InformationSystem(X, y, self.feature_labels)
        self.antecedents, str_antecedents = information_system.induce_rules(self.fuzzy_sets, self.clauses)
        self.rules = [Rule(self.antecedents[decision], self.consequents[str(decision)])
                      for decision in self.antecedents.keys()]
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


def find_best_selector(models, linear_params, dataloader):
    for model, params in zip(models, linear_params):
        feature_selector = FeatureSelector(model)
        X_projected = feature_selector.search(params, dataloader.X_transformed, dataloader.y)
        for projected in X_projected:
            plt.scatter(projected[:, 0], projected[:, 1], c=dataloader.y)
            plt.title(str(model))
            plt.show()


full_dataset = pd.read_csv("../data/diabetes.csv")
target_label = 'Outcome'
feature_labels = list(full_dataset.columns)
feature_labels.remove(target_label)
X = full_dataset.values[:, :-1]
y = full_dataset.values[:, -1]
transforms = [normalize]
dataloader = DataLoader(X, y, feature_labels, target_label)
dataloader.prepare_data(transforms, test_size=0.2)
"""
models = [PCA, KernelPCA, NeighborhoodComponentsAnalysis, GenericUnivariateSelect, RFECV, RFE, SelectFromModel,
          SequentialFeatureSelector, FactorAnalysis]

linear_params = [
    # PCA
    {
        'n_components': [2, 2, 2],
        'svd_solver': ['full', 'arpack', 'randomized']
    },
    # KernelPCA
    {
        'kernel': ['linear', 'poly', 'rbf', 'cosine'],
        'n_components': [2, 2, 2, 2]
    },
    # NCA
    {
        'init': ['pca', 'identity', 'random'],
        'n_components': [2, 2, 2]
    },
    # GenericUnivariateSelect
    {
      'mode': ['k_best', 'fpr', 'fdr', 'fwe'],
      'param': [2, 5e-2, 5e-2, 5e-2]
    },
    # RFECV
    {
        'estimator': [RandomForestClassifier(), ExtraTreesClassifier()],
        'min_features_to_select': [2, 2],
        'n_jobs': [-1, -1]
    },
    # RFE
    {
        'estimator': [RandomForestClassifier(), ExtraTreesClassifier()],
        'n_features_to_select': [2, 2]
    },
    # SelectFromModel
    {
        'estimator': [RandomForestClassifier(), ExtraTreesClassifier()],
        'threshold': [-np.inf, -np.inf],
        'max_features': [2, 2]
    },
    # SequentialFeatureSelector
    {
        'estimator': [RandomForestClassifier(),  RandomForestClassifier(),
                      ExtraTreesClassifier(), ExtraTreesClassifier()],
        'n_features_to_select': [2, 2, 2, 2],
        'direction': ['forward', 'backward', 'forward', 'backward'],
        'n_jobs': [-1, -1, -1, -1]
    },
    # FactorAnalysis
    {
        'n_components': [2, 2, 2, 2],
        'svd_method': ['lapack', 'randomized', 'lapack', 'randomized'],
        'rotation': ['varimax', 'quartimax', 'quartimax', 'varimax']
    }
]

find_best_selector(models, linear_params, dataloader)

# Probably Best Selectors for Pima Indians:
# RFE, NCA
"""
con_labels = np.unique(dataloader.y)
con_labels = [str(label) for label in con_labels]

grid_params = {
    'model': [RandomForestClassifier(), KNeighborsClassifier(), DecisionTreeClassifier()],
    'feature_labels': [feature_labels],
    'consequent_labels': [con_labels],
    'inference_system': [MamdaniInferenceSystem],
    'defuzz_method': [center_of_gravity],
    'domain': [Domain(0, 1.001, 0.001)],
    'fuzzy_set_type': ['t1'],
    'n_mf': [3, 5, 7, 9, 11],
    'mf_type': ['gaussian', 'triangular', 'trapezoidal'],
    'con_n_mf': [2],
    'con_mf_type': ['gaussian', 'triangular', 'trapezoidal']
}

evaluator = Evaluator()
random_search = RandomizedSearchCV(evaluator, grid_params, n_iter=2, n_jobs=5,
                                   cv=StratifiedKFold(n_splits=5, shuffle=True), verbose=5, random_state=42,
                                   scoring='f1', error_score=1)
random_search.fit(dataloader.X_train_frame, dataloader.y_train_frame)
y_pred = random_search.predict(dataloader.X_test_frame)

acc = accuracy_score(dataloader.y_test, y_pred)
f1 = f1_score(dataloader.y_test, y_pred)

print('\nAccuracy: ', acc)
print('F1 Score: ', f1)

cv_results = pd.DataFrame(random_search.cv_results_)
print(cv_results['f1'])
