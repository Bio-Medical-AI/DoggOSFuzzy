import time
from collections import defaultdict
from typing import Dict, List, Any

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from doggos.fuzzy_sets.fuzzy_set import FuzzySet
from doggos.induction import InformationSystem
from doggos.inference import MamdaniInferenceSystem
from doggos.inference.defuzzification_algorithms import center_of_gravity, karnik_mendel
from doggos.inference.inference_system import InferenceSystem
from doggos.knowledge import Domain, fuzzify, Rule, Clause
from doggos.knowledge.consequents import MamdaniConsequent
from doggos.knowledge.consequents.consequent import Consequent
from doggos.utils.grouping_functions import create_set_of_variables


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


class EvaluatorTest(BaseEstimator):
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
        start = time.time()
        _, fuzzy_sets, clauses = create_set_of_variables(self.feature_labels,
                                                         mf_type=self.mf_type,
                                                         n_mfs=self.n_mf,
                                                         fuzzy_set_type=self.fuzzy_set_type,
                                                         lower_scaling=self.lower_scaling,
                                                         mode=self.mode,
                                                         middle_vals=self.middle_vals)
        end = time.time()
        print('Feature variables: ', end - start)
        start = time.time()
        _, _, con_clauses = create_set_of_variables(self.consequent_labels,
                                                    mf_type=self.con_mf_type,
                                                    n_mfs=self.con_n_mf,
                                                    fuzzy_set_type=self.fuzzy_set_type,
                                                    lower_scaling=self.lower_scaling)
        end = time.time()
        print('Consequent variables: ', end - start)
        self.fuzzy_sets = fuzzy_sets
        self.clauses = clauses
        self.consequents = {}
        for label in self.consequent_labels:
            clause = list(con_clauses[label].values())[0]
            self.consequents[label] = MamdaniConsequent(clause)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        information_system = InformationSystem(X, y, self.feature_labels)
        start = time.time()
        self.antecedents, _ = information_system.induce_rules(self.fuzzy_sets, self.clauses)
        end = time.time()
        print('InformationSystem_induce_rules: ', end - start)
        self.rules = [Rule(self.antecedents[decision], self.consequents[decision])
                      for decision in self.consequent_labels]
        start = time.time()
        fuzzified_dataset = fuzzify(X, self.clauses)
        end = time.time()
        print('Fuzzification: ', end - start)
        inference_system = self.inference_system(self.rules)
        start = time.time()
        inference = inference_system.infer(self.defuzz_method, fuzzified_dataset)
        end = time.time()
        print('Inference: ', end - start)
        start = time.time()
        self.model.fit(inference.reshape(-1, 1), y)
        end = time.time()
        print('Model_fit: ', end - start)

    def predict(self, X: pd.DataFrame):
        start = time.time()
        fuzzified_dataset = fuzzify(X, self.clauses)
        end = time.time()
        print('predict_fuzzify: ', end - start)
        inference_system = self.inference_system(self.rules)
        start = time.time()
        inference = inference_system.infer(self.defuzz_method, fuzzified_dataset)
        end = time.time()
        print('predict_infer: ', end - start)
        start = time.time()
        y_pred = self.model.predict(inference.reshape(-1, 1))
        end = time.time()
        print('model_predict: ', end - start)
        return y_pred


full_dataset = pd.read_csv("../data/diabetes.csv")
target_label = 'Outcome'
feature_labels = list(full_dataset.columns)
feature_labels.remove(target_label)
X = full_dataset.values[:, :-1]
y = full_dataset.values[:, -1]
transforms = [normalize]
dataloader = DataLoader(X, y, feature_labels, target_label)
dataloader.prepare_data(transforms, test_size=0.2)

evaluator = EvaluatorTest()
params = {
    'n_mf': 5,
    'model__n_estimators': 400,
    'model__max_features': 'log2',
    'model__criterion': 'entropy',
    'model': RandomForestClassifier(criterion='entropy', max_features='log2',n_estimators=400),
    'mf_type': 'triangular',
    'lower_scaling': 0.9,
    'inference_system': MamdaniInferenceSystem,
    'fuzzy_set_type': 't1',
    'feature_labels': feature_labels,
    'domain': Domain(0, 1.0001, 0.0001),
    'consequent_labels': ['0.0', '1.0'],
    'con_n_mf': 2,
    'con_mf_type': 'gaussian'
}
full_flow_start = time.time()
start = time.time()
evaluator.set_params(**params)
end = time.time()
print('set_params: ', end - start)
start = time.time()
evaluator.fit(dataloader.X_train_frame, dataloader.y_train)
end = time.time()
print('Fit time: ', end - start)
start = time.time()
y_pred = evaluator.predict(dataloader.X_test_frame)
end = time.time()
print('Test time: ', end - start)
full_flow_end = time.time()
print('Whole workflow time: ', full_flow_end - full_flow_start)