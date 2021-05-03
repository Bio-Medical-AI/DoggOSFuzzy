import pytest
import numpy as np
import pandas as pd

from doggos.algebras import GodelAlgebra
from doggos.inference import TakagiSugenoInferenceSystem
from doggos.inference.defuzzification_algorithms import weighted_average, takagi_sugeno_karnik_mendel
from doggos.knowledge import Clause, LinguisticVariable, Domain, fuzzify, Term, Rule
from doggos.fuzzy_sets import Type1FuzzySet, IntervalType2FuzzySet
from doggos.knowledge.consequents import TakagiSugenoConsequent
from doggos.utils.membership_functions import triangular


class TestTakagiSugenoInferenceSystem:
    def test_inference_type_1(self):
        df = pd.DataFrame({'fire': [100.0, 112, 102.5, 100.1, 103.27],
                           'air': [1, 4, 3.021, 2.879, 3.55],
                           'earth': [0, 600, 58, 272, 384],
                           'water': [0, 13, 5.1, 7.9, 2.2]})

        lingustic_variable_1 = LinguisticVariable('fire', Domain(100, 112.01, 0.01))
        lingustic_variable_2 = LinguisticVariable('air', Domain(1, 4.001, 0.001))
        lingustic_variable_3 = LinguisticVariable('earth', Domain(0, 602, 2))
        lingustic_variable_4 = LinguisticVariable('water', Domain(0, 13.1, 0.1))

        measures = {lingustic_variable_1: [100.0, 112, 102.5, 100.1, 103.27],
                    lingustic_variable_2: [1, 4, 3.021, 2.879, 3.55],
                    lingustic_variable_3: [0, 600, 58, 272, 384],
                    lingustic_variable_4: [0, 13, 5.1, 7.9, 2.2]}

        clause1 = Clause(lingustic_variable_1, 'high', Type1FuzzySet(lambda x: triangular(106., 112., 112.1)(x)))
        clause2 = Clause(lingustic_variable_1, 'medium', Type1FuzzySet(lambda x: triangular(100, 106., 112)(x)))
        clause3 = Clause(lingustic_variable_1, 'low', Type1FuzzySet(lambda x: triangular(99.9, 100., 106.)(x)))
        clause4 = Clause(lingustic_variable_2, 'high', Type1FuzzySet(lambda x: triangular(2.5, 4, 4.1)(x)))
        clause5 = Clause(lingustic_variable_2, 'medium', Type1FuzzySet(lambda x: triangular(1, 2.5, 4)(x)))
        clause6 = Clause(lingustic_variable_2, 'low', Type1FuzzySet(lambda x: triangular(0.9, 1, 2.5)(x)))
        clause7 = Clause(lingustic_variable_3, 'high', Type1FuzzySet(lambda x: triangular(300, 600, 602)(x)))
        clause8 = Clause(lingustic_variable_3, 'medium', Type1FuzzySet(lambda x: triangular(0, 300, 600)(x)))
        clause9 = Clause(lingustic_variable_3, 'low', Type1FuzzySet(lambda x: triangular(-2, 0, 300)(x)))
        clause10 = Clause(lingustic_variable_4, 'high', Type1FuzzySet(lambda x: triangular(6.5, 13, 13.1)(x)))
        clause11 = Clause(lingustic_variable_4, 'medium', Type1FuzzySet(lambda x: triangular(0, 6.5, 13)(x)))
        clause12 = Clause(lingustic_variable_4, 'low', Type1FuzzySet(lambda x: triangular(-0.1, 0, 6.5)(x)))
        fuzzified = fuzzify(df, [clause1, clause2, clause3,
                                 clause4, clause5, clause6,
                                 clause7, clause8, clause9,
                                 clause10, clause11, clause12])
        algebra = GodelAlgebra()

        term1 = Term(algebra, clause1)
        term2 = Term(algebra, clause2)
        term3 = Term(algebra, clause3)
        term4 = Term(algebra, clause4)
        term5 = Term(algebra, clause5)
        term6 = Term(algebra, clause6)
        term7 = Term(algebra, clause7)
        term8 = Term(algebra, clause8)
        term9 = Term(algebra, clause9)
        term10 = Term(algebra, clause10)
        term11 = Term(algebra, clause11)
        term12 = Term(algebra, clause12)

        antecedent1 = term1 & term4 & term7 & term10
        antecedent2 = term2 | term4 & term9 | term11
        antecedent3 = (term3 | term5) & (term8 | term12)
        antecedent4 = term6

        lingustic_variable_for_consequent_1 = LinguisticVariable('avatar', Domain(0, 1000, 0.01))
        lingustic_variable_for_consequent_2 = LinguisticVariable('momo', Domain(0, 1000, 0.01))

        consequent1 = TakagiSugenoConsequent({lingustic_variable_1: 0.1,
                                              lingustic_variable_2: 10,
                                              lingustic_variable_3: 0.01,
                                              lingustic_variable_4: 0.3},
                                             -2.5, lingustic_variable_for_consequent_1)
        consequent2 = TakagiSugenoConsequent({lingustic_variable_1: 0.2,
                                              lingustic_variable_2: 7,
                                              lingustic_variable_3: 0.05,
                                              lingustic_variable_4: 3},
                                             -2.5, lingustic_variable_for_consequent_1)
        consequent3 = TakagiSugenoConsequent({lingustic_variable_1: 0.8,
                                              lingustic_variable_2: 12,
                                              lingustic_variable_3: 0.1,
                                              lingustic_variable_4: 1},
                                             -2.5, lingustic_variable_for_consequent_2)
        consequent4 = TakagiSugenoConsequent({lingustic_variable_1: 0.4,
                                              lingustic_variable_2: 5,
                                              lingustic_variable_3: 0.15,
                                              lingustic_variable_4: 5},
                                             -2.5, lingustic_variable_for_consequent_2)
        rules = [Rule(antecedent1, consequent1),
                 Rule(antecedent2, consequent2),
                 Rule(antecedent3, consequent3),
                 Rule(antecedent4, consequent4)]

        system = TakagiSugenoInferenceSystem(rules)

        print(system.infer(weighted_average, fuzzified, measures))

    def test_inference_type_I_2(self):
        df = pd.DataFrame({'fire': [100.0, 112, 102.5, 100.1, 103.27],
                           'air': [1, 4, 3.021, 2.879, 3.55],
                           'earth': [0, 600, 58, 272, 384],
                           'water': [0, 13, 5.1, 7.9, 2.2]})

        lingustic_variable_1 = LinguisticVariable('fire', Domain(100, 112.01, 0.01))
        lingustic_variable_2 = LinguisticVariable('air', Domain(1, 4.001, 0.001))
        lingustic_variable_3 = LinguisticVariable('earth', Domain(0, 602, 2))
        lingustic_variable_4 = LinguisticVariable('water', Domain(0, 13.1, 0.1))

        measures = {lingustic_variable_1: [100.0, 112, 102.5, 100.1, 103.27],
                    lingustic_variable_2: [1, 4, 3.021, 2.879, 3.55],
                    lingustic_variable_3: [0, 600, 58, 272, 384],
                    lingustic_variable_4: [0, 13, 5.1, 7.9, 2.2]}

        clause1 = Clause(lingustic_variable_1, 'high',
                         IntervalType2FuzzySet(lambda x: triangular(106., 112., 112.1)(x)/2,
                                               lambda x: triangular(106., 112., 112.1)(x)))
        clause2 = Clause(lingustic_variable_1, 'medium',
                         IntervalType2FuzzySet(lambda x: triangular(100, 106., 112)(x)/2,
                                               lambda x: triangular(100, 106., 112)(x)))
        clause3 = Clause(lingustic_variable_1, 'low',
                         IntervalType2FuzzySet(lambda x: triangular(99.9, 100., 106.)(x)/2,
                                               lambda x: triangular(99.9, 100., 106.)(x)))
        clause4 = Clause(lingustic_variable_2, 'high',
                         IntervalType2FuzzySet(lambda x: triangular(2.5, 4, 4.1)(x)/2,
                                               lambda x: triangular(2.5, 4, 4.1)(x)))
        clause5 = Clause(lingustic_variable_2, 'medium',
                         IntervalType2FuzzySet(lambda x: triangular(1, 2.5, 4)(x)/2,
                                               lambda x: triangular(1, 2.5, 4)(x)))
        clause6 = Clause(lingustic_variable_2, 'low',
                         IntervalType2FuzzySet(lambda x: triangular(0.9, 1, 2.5)(x)/2,
                                               lambda x: triangular(0.9, 1, 2.5)(x)))
        clause7 = Clause(lingustic_variable_3, 'high',
                         IntervalType2FuzzySet(lambda x: triangular(300, 600, 602)(x)/2,
                                               lambda x: triangular(300, 600, 602)(x)))
        clause8 = Clause(lingustic_variable_3, 'medium',
                         IntervalType2FuzzySet(lambda x: triangular(0, 300, 600)(x)/2,
                                               lambda x: triangular(0, 300, 600)(x)))
        clause9 = Clause(lingustic_variable_3, 'low',
                         IntervalType2FuzzySet(lambda x: triangular(-2, 0, 300)(x)/2,
                                               lambda x: triangular(-2, 0, 300)(x)))
        clause10 = Clause(lingustic_variable_4, 'high',
                          IntervalType2FuzzySet(lambda x: triangular(6.5, 13, 13.1)(x)/2,
                                                lambda x: triangular(6.5, 13, 13.1)(x)))
        clause11 = Clause(lingustic_variable_4, 'medium',
                          IntervalType2FuzzySet(lambda x: triangular(0, 6.5, 13)(x)/2,
                                                lambda x: triangular(0, 6.5, 13)(x)))
        clause12 = Clause(lingustic_variable_4, 'low',
                          IntervalType2FuzzySet(lambda x: triangular(-0.1, 0, 6.5)(x)/2,
                                                lambda x: triangular(-0.1, 0, 6.5)(x)))
        fuzzified = fuzzify(df, [clause1, clause2, clause3,
                                 clause4, clause5, clause6,
                                 clause7, clause8, clause9,
                                 clause10, clause11, clause12])
        algebra = GodelAlgebra()

        term1 = Term(algebra, clause1)
        term2 = Term(algebra, clause2)
        term3 = Term(algebra, clause3)
        term4 = Term(algebra, clause4)
        term5 = Term(algebra, clause5)
        term6 = Term(algebra, clause6)
        term7 = Term(algebra, clause7)
        term8 = Term(algebra, clause8)
        term9 = Term(algebra, clause9)
        term10 = Term(algebra, clause10)
        term11 = Term(algebra, clause11)
        term12 = Term(algebra, clause12)

        antecedent1 = term1 & term4 & term7 & term10
        antecedent2 = term2 | term4 & term9 | term11
        antecedent3 = (term3 | term5) & (term8 | term12)
        antecedent4 = term6

        lingustic_variable_for_consequent_1 = LinguisticVariable('avatar', Domain(0, 1000, 0.01))
        lingustic_variable_for_consequent_2 = LinguisticVariable('momo', Domain(0, 1000, 0.01))

        consequent1 = TakagiSugenoConsequent({lingustic_variable_1: 0.1,
                                              lingustic_variable_2: 10,
                                              lingustic_variable_3: 0.01,
                                              lingustic_variable_4: 0.3},
                                             -2.5, lingustic_variable_for_consequent_1)
        consequent2 = TakagiSugenoConsequent({lingustic_variable_1: 0.2,
                                              lingustic_variable_2: 7,
                                              lingustic_variable_3: 0.05,
                                              lingustic_variable_4: 3},
                                             -2.5, lingustic_variable_for_consequent_1)
        consequent3 = TakagiSugenoConsequent({lingustic_variable_1: 0.8,
                                              lingustic_variable_2: 12,
                                              lingustic_variable_3: 0.1,
                                              lingustic_variable_4: 1},
                                             -2.5, lingustic_variable_for_consequent_2)
        consequent4 = TakagiSugenoConsequent({lingustic_variable_1: 0.4,
                                              lingustic_variable_2: 5,
                                              lingustic_variable_3: 0.15,
                                              lingustic_variable_4: 5},
                                             -2.5, lingustic_variable_for_consequent_2)
        rules = [Rule(antecedent1, consequent1),
                 Rule(antecedent2, consequent2),
                 Rule(antecedent3, consequent3),
                 Rule(antecedent4, consequent4)]

        system = TakagiSugenoInferenceSystem(rules)

        print(system.infer(takagi_sugeno_karnik_mendel, fuzzified, measures))

