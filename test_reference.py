import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from doggos.fuzzy_sets import IntervalType2FuzzySet
from doggos.induction import InformationSystem
from doggos.inference import TakagiSugenoInferenceSystem, MamdaniInferenceSystem
from doggos.inference.defuzzification_algorithms import takagi_sugeno_EIASC, karnik_mendel
from doggos.knowledge import LinguisticVariable, Domain, fuzzify, Rule, Clause
from doggos.knowledge.consequents import MamdaniConsequent
from doggos.utils.grouping_functions import create_set_of_variables
from doggos.utils.membership_functions import sigmoid
from doggos.utils.membership_functions.membership_functions import sigmoid_reversed


def test_on_knn(x_train, x_test, y_train, y_test):
    for k in range(3, 20):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        print(f1_score(y_test, y_pred))


def test_on_mamdani(x_train, x_test, y_train, y_test, col_names, adjustment):
    if adjustment == 'mean':
        mid_evs = []
        for vals in x_train.T:
            mean, _ = norm.fit(vals)
            mid_evs.append(mean)
        middle_vals = mid_evs
    else:
        middle_vals = 0.5

    ling_vars, fuzzy_sets, clauses = create_set_of_variables(
        ling_var_names=col_names,
        domain=Domain(0, 1.001, 0.001),
        mf_type='gaussian',
        n_mfs=7,
        fuzzy_set_type='it2',
        mode='progressive',
        adjustment=adjustment,
        lower_scaling=0.975,
        middle_vals=middle_vals
    )

    decision = LinguisticVariable('Decision', Domain(0, 1.001, 0.001))

    decision_zero = IntervalType2FuzzySet(sigmoid(0.5, 0.71), sigmoid(0.5, 0.29))
    decision_zero_clause = Clause(decision, 'Zero', decision_zero)

    decision_one = IntervalType2FuzzySet(sigmoid_reversed(0.5, 0.29), sigmoid_reversed(0.5, 0.71))
    decision_one_clause = Clause(decision, 'One', decision_one)
    consequents = [
        MamdaniConsequent(decision_zero_clause),
        MamdaniConsequent(decision_one_clause)
    ]

    information_system = InformationSystem(x_train, y_train, col_names)
    print("Induction")
    antecedents, string_antecedents = information_system.induce_rules(fuzzy_sets, clauses)
    rule_base = []
    for idx, key in enumerate(antecedents.keys()):
        rule_base.append(Rule(antecedents[key], consequents[idx]))

    features = fuzzify(x_test, clauses)

    system = MamdaniInferenceSystem(rule_base)
    print("Infer")
    outputs = system.infer(karnik_mendel, features)
    y_pred = []
    for output in outputs:
        if output < 0.5:
            y_pred.append(0)
        else:
            y_pred.append(1)
    print(f1_score(y_test, y_pred))


def main():
    df = pd.read_csv('data/Breast Cancer Data.csv', sep=';')
    X = df.drop(columns=['Decision']).values
    y = df['Decision'].values
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    test_on_mamdani(x_train, x_test, y_train, y_test, df.drop(columns=['Decision']).columns, "mean")


if __name__ == '__main__':
    main()