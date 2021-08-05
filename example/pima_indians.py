from typing import Dict

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

import pandas as pd
import numpy as np

from doggos.fuzzy_sets import Type1FuzzySet
from doggos.fuzzy_sets.fuzzy_set import FuzzySet
from doggos.induction.information_system import InformationSystem
from doggos.inference import MamdaniInferenceSystem
from doggos.inference.defuzzification_algorithms import center_of_gravity
from doggos.inference.inference_system import InferenceSystem
from doggos.knowledge import Rule, Clause, fuzzify, LinguisticVariable, Domain
from doggos.knowledge.consequents import MamdaniConsequent
from doggos.knowledge.consequents.consequent import Consequent
from doggos.utils.membership_functions.membership_functions import generate_equal_gausses, sigmoid, gaussian


def normalize(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)


class Evaluator:
    def __init__(self):
        self.feature_labels = []
        self.rules = []
        self.clauses = []

    def fit(self, X: pd.DataFrame, y, feature_labels, fuzzy_sets: Dict[str, Dict[str, FuzzySet]],
            consequents: Dict[str, Consequent]):
        induction_system = InductionSystem(X, y, feature_labels)
        antecedents, str_antecedents = induction_system.induce_rules(fuzzy_sets)
        for decision in antecedents.keys():
            self.rules.append(Rule(antecedents[decision], consequents[str(decision)]))
        self.clauses = induction_system.rule_builder.clauses

    def predict(self, X: pd.DataFrame, inference_system: InferenceSystem.__class__, defuzzification_method):
        fuzzified_dataset = fuzzify(X, self.clauses)
        inference_system = inference_system(self.rules)
        return inference_system.infer(defuzzification_method, fuzzified_dataset)

full_dataset = pd.read_csv("data/diabetes.csv")
feature_labels = list(full_dataset.columns[:-1])
X = full_dataset.values[:, :-1]
y = full_dataset.values[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
X_train_normal = normalize(X_train)
X_test_normal = normalize(X_test)
X_train_frame = pd.DataFrame(data=X_train_normal, columns=feature_labels)
X_test_frame = pd.DataFrame(data=X_test_normal, columns=feature_labels)
y_train_frame = pd.Series(y_train)
y_test_frame = pd.Series(y_test)

names = ['small', 'medium', 'high']
gausses = generate_equal_gausses(len(names), 0, 1)
classes = np.unique(y)

fuzzy_sets = {}
for column in feature_labels:
    fuzzy_sets[column] = {}
    for name, gauss in zip(names, gausses):
        fuzzy_sets[column].update({name: Type1FuzzySet(gauss)})

consequent_gausses = generate_equal_gausses(2, 0, 1)
consequents = {}
for class_, consequent_gauss in zip(classes, consequent_gausses):
    class_ = str(class_)
    print(class_)
    consequent_clause = Clause(LinguisticVariable(class_, Domain(0, 1.001, 0.001)),
                               class_, Type1FuzzySet(consequent_gauss))
    consequents[class_] = MamdaniConsequent(consequent_clause)

evaluator = Evaluator()
evaluator.fit(X_train_frame, y_train_frame, feature_labels, fuzzy_sets, consequents)
values_train = evaluator.predict(X_train_frame, MamdaniInferenceSystem, center_of_gravity)
values_test = evaluator.predict(X_test_frame, MamdaniInferenceSystem, center_of_gravity)
values_train = values_train.reshape(-1, 1)
values_test = values_test.reshape(-1, 1)

knn = KNeighborsClassifier()
knn.fit(values_train, y_train)
y_pred = knn.predict(values_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Accuracy: ', acc)
print('F1 Score: ', f1)

def classify(data):
    preds = []
    for elem in data:
        if elem >= 0.5:
            preds.append(1.0)
        else:
            preds.append(0.0)
    return preds

preds = classify(values_test)
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds)

print('Accuracy: ', acc)
print('F1 Score: ', f1)
