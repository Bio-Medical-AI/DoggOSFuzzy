from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from doggos.knowledge import LinguisticVariable, Domain
from doggos.knowledge import Clause
from doggos.fuzzy_sets import IntervalType2FuzzySet, Type1FuzzySet
from doggos.utils.membership_functions import triangular, trapezoidal, gaussian
from doggos.knowledge.consequents import TakagiSugenoConsequent
from doggos.knowledge import Term
from doggos.algebras import GodelAlgebra
from doggos.knowledge import Rule
from doggos.knowledge import fuzzify
from doggos.inference import TakagiSugenoInferenceSystem
from doggos.inference.defuzzification_algorithms import takagi_sugeno_karnik_mendel, weighted_average


if __name__ == '__main__':
    species_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    df = pd.read_csv('example/data.csv')
    df_X = df.drop(columns='species')
    df_y = df['species'].map(species_map)
    x = df_X.values
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_X = pd.DataFrame(x_scaled, columns=df_X.columns)

    sepal_length = LinguisticVariable('sepal-length', Domain(0, 1.001, 0.001))
    sepal_width = LinguisticVariable('sepal-width', Domain(0, 1.001, 0.001))
    petal_length = LinguisticVariable('petal-length', Domain(0, 1.001, 0.001))
    petal_width = LinguisticVariable('petal-width', Domain(0, 1.001, 0.001))
    species = LinguisticVariable('species', Domain(0, 1.001, 0.001))

    clauses = [
        Clause(petal_length, 'small', IntervalType2FuzzySet(triangular(0, 0.25, 0.5, 0.8), triangular(0, 0.25, 0.5))),
        Clause(petal_length, 'medium',
               IntervalType2FuzzySet(triangular(0.25, 0.5, 0.75, 0.8), triangular(0.25, 0.5, 0.75))),
        Clause(petal_length, 'big', IntervalType2FuzzySet(triangular(0.5, 0.75, 1, 0.8), triangular(0.5, 0.75, 1))),
        Clause(petal_width, 'small', IntervalType2FuzzySet(gaussian(0.25, 0.1, 0.8), gaussian(0.25, 0.1))),
        Clause(petal_width, 'medium', IntervalType2FuzzySet(gaussian(0.5, 0.1, 0.8), gaussian(0.5, 0.1))),
        Clause(petal_width, 'big', IntervalType2FuzzySet(gaussian(0.75, 0.1, 0.8), gaussian(0.75, 0.1))),
        Clause(sepal_length, 'small', IntervalType2FuzzySet(triangular(0, 0.25, 0.5, 0.8), triangular(0, 0.25, 0.5))),
        Clause(sepal_length, 'medium',
               IntervalType2FuzzySet(triangular(0.25, 0.5, 0.75, 0.8), triangular(0.25, 0.5, 0.75))),
        Clause(sepal_length, 'big', IntervalType2FuzzySet(triangular(0.5, 0.75, 1, 0.8), triangular(0.5, 0.75, 1))),
        Clause(sepal_width, 'small', IntervalType2FuzzySet(gaussian(0.25, 0.1, 0.8), gaussian(0.25, 0.1))),
        Clause(sepal_width, 'medium', IntervalType2FuzzySet(gaussian(0.5, 0.1, 0.8), gaussian(0.5, 0.1))),
        Clause(sepal_width, 'big', IntervalType2FuzzySet(gaussian(0.75, 0.1, 0.8), gaussian(0.75, 0.1)))
    ]

    # clauses = [
    #     # clauses for petal length
    #     Clause(petal_length, 'small', Type1FuzzySet(triangular(0, 0.25, 0.5))),
    #     Clause(petal_length, 'medium', Type1FuzzySet(triangular(0.25, 0.5, 0.75))),
    #     Clause(petal_length, 'big', Type1FuzzySet(triangular(0.5, 0.75, 1))),
    #     # clauses for petal width
    #     Clause(petal_width, 'small', Type1FuzzySet(gaussian(0.25, 0.1))),
    #     Clause(petal_width, 'medium', Type1FuzzySet(gaussian(0.5, 0.1))),
    #     Clause(petal_width, 'big', Type1FuzzySet(gaussian(0.75, 0.1))),
    #     # claueses for sepal length
    #     Clause(sepal_length, 'small', Type1FuzzySet(triangular(0, 0.25, 0.5))),
    #     Clause(sepal_length, 'medium', Type1FuzzySet(triangular(0.25, 0.5, 0.75))),
    #     Clause(sepal_length, 'big', Type1FuzzySet(triangular(0.5, 0.75, 1))),
    #     # clauses for sepal width
    #     Clause(sepal_width, 'small', Type1FuzzySet(gaussian(0.25, 0.1))),
    #     Clause(sepal_width, 'medium', Type1FuzzySet(gaussian(0.5, 0.1))),
    #     Clause(sepal_width, 'big', Type1FuzzySet(gaussian(0.75, 0.1)))
    # ]

    # domain = np.arange(0, 1.001, 0.001)
    # for clause in clauses:
    #     print(clause)
    #     plt.plot(domain, clause.values[0])
    #     plt.plot(domain, clause.values[1])
    #     plt.grid()
    #     plt.show()

    parameters_1 = {sepal_length: 0.25, sepal_width: 0.25, petal_length: 0.5, petal_width: 0.5}
    parameters_2 = {sepal_length: 0.15, sepal_width: 0.35, petal_length: 0.15, petal_width: 0.35}
    parameters_3 = {sepal_length: 0.35, sepal_width: 0.15, petal_length: 0.35, petal_width: 0.15}
    consequent_1 = TakagiSugenoConsequent(parameters_1, 0, species)
    consequent_2 = TakagiSugenoConsequent(parameters_2, 1.5, species)
    consequent_3 = TakagiSugenoConsequent(parameters_3, 0.7, species)

    terms = list()
    algebra = GodelAlgebra()
    for clause in clauses:
        terms.append(Term(algebra, clause))

    # petal_length_small & petal_width_small
    antecedent1 = terms[0] & terms[3]
    # petal_width_small & sepal_length_big
    antecedent2 = terms[3] & terms[8]
    # sepal_width_medium | sepal-length_big | petal_width_big
    antecedent3 = terms[10] | terms[8] | terms[5]

    rules = [
        Rule(antecedent1, consequent_1),
        Rule(antecedent2, consequent_2),
        Rule(antecedent3, consequent_3)
    ]

    df_X_fuzzified = fuzzify(df_X, clauses)
    measures = {sepal_length: df_X['sepal-length'].values,
                sepal_width: df_X['sepal-width'].values,
                petal_length: df_X['petal-length'].values,
                petal_width: df_X['petal-width'].values}

    inference_system = TakagiSugenoInferenceSystem(rules)
    result = inference_system.infer(takagi_sugeno_karnik_mendel, df_X_fuzzified, measures)
    print(result)
    print(max(result[species]))
