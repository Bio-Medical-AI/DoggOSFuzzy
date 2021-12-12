import pytest
import pandas as pd

from doggos.algebras import GodelAlgebra
from doggos.inference import TakagiSugenoInferenceSystem
from doggos.inference.defuzzification_algorithms import weighted_average, takagi_sugeno_karnik_mendel, \
    takagi_sugeno_EIASC
from doggos.knowledge import Clause, LinguisticVariable, Domain, fuzzify, Term, Rule
from doggos.fuzzy_sets import Type1FuzzySet, IntervalType2FuzzySet
from doggos.knowledge.consequents import TakagiSugenoConsequent
from doggos.utils.membership_functions import triangular
from tests.test_tools import approx


class TestTakagiSugenoInferenceSystem:
    def test_inference_type_1(self):
        df = pd.DataFrame({'fire': [1, 0.5, 0.2, 0.3, 0.7],
                           'air': [0.3, 0.4, 1, 0.7, 0.2],
                           'earth': [0, 0.3, 0.1, 0.9, 0.8],
                           'water': [0.6, 0.1, 0.4, 0.3, 0.5]})

        lingustic_variable_1 = LinguisticVariable('fire', Domain(0, 1.001, 0.001))
        lingustic_variable_2 = LinguisticVariable('air', Domain(0, 1.001, 0.001))
        lingustic_variable_3 = LinguisticVariable('earth', Domain(0, 1.001, 0.001))
        lingustic_variable_4 = LinguisticVariable('water', Domain(0, 1.001, 0.001))

        measures = {lingustic_variable_1: [1, 0.5, 0.2, 0.3, 0.7],
                    lingustic_variable_2: [0.3, 0.4, 1, 0.7, 0.2],
                    lingustic_variable_3: [0, 0.3, 0.1, 0.9, 0.8],
                    lingustic_variable_4: [0.6, 0.1, 0.4, 0.3, 0.5]}

        clause1 = Clause(lingustic_variable_1, 'low', Type1FuzzySet(triangular(0, 0.001, 0.5)))
        clause2 = Clause(lingustic_variable_1, 'medium', Type1FuzzySet(triangular(0.25, 0.5, 0.75)))
        clause3 = Clause(lingustic_variable_1, 'high', Type1FuzzySet(triangular(0.5, 0.999, 1)))
        clause4 = Clause(lingustic_variable_2, 'low', Type1FuzzySet(triangular(0, 0.001, 0.5)))
        clause5 = Clause(lingustic_variable_2, 'medium', Type1FuzzySet(triangular(0.25, 0.5, 0.75)))
        clause6 = Clause(lingustic_variable_2, 'high', Type1FuzzySet(triangular(0.5, 0.999, 1)))
        clause7 = Clause(lingustic_variable_3, 'low', Type1FuzzySet(triangular(0, 0.001, 0.5)))
        clause8 = Clause(lingustic_variable_3, 'medium', Type1FuzzySet(triangular(0.25, 0.5, 0.75)))
        clause9 = Clause(lingustic_variable_3, 'high', Type1FuzzySet(triangular(0.5, 0.999, 1)))
        clause10 = Clause(lingustic_variable_4, 'low', Type1FuzzySet(triangular(0, 0.001, 0.5)))
        clause11 = Clause(lingustic_variable_4, 'medium', Type1FuzzySet(triangular(0.25, 0.5, 0.75)))
        clause12 = Clause(lingustic_variable_4, 'high', Type1FuzzySet(triangular(0.5, 0.999, 1)))

        to_fuzz = {}
        to_fuzz[lingustic_variable_1.name] = {'Low': clause1,
                                              'Medium': clause2,
                                              'High': clause3}
        to_fuzz[lingustic_variable_2.name] = {'Low': clause4,
                                              'Medium': clause5,
                                              'High': clause6}
        to_fuzz[lingustic_variable_3.name] = {'Low': clause7,
                                              'Medium': clause8,
                                              'High': clause9}
        to_fuzz[lingustic_variable_4.name] = {'Low': clause10,
                                              'Medium': clause11,
                                              'High': clause12}

        fuzzified = fuzzify(df, to_fuzz)
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

        antecedent1 = term1 & term4 & term7 & term10 | term3 & term5 & term6 & term8 & term12
        antecedent2 = term2 | term4 & term9 | term11

        lingustic_variable_for_consequent_1 = LinguisticVariable('avatar', Domain(0, 1.001, 0.001))
        lingustic_variable_for_consequent_2 = LinguisticVariable('momo', Domain(0, 1.001, 0.001))

        consequent1 = TakagiSugenoConsequent({lingustic_variable_1: 1,
                                              lingustic_variable_2: 2,
                                              lingustic_variable_3: 2,
                                              lingustic_variable_4: 3},
                                             5,
                                             lingustic_variable_for_consequent_1)
        consequent2 = TakagiSugenoConsequent({lingustic_variable_1: 2,
                                              lingustic_variable_2: 3,
                                              lingustic_variable_3: 1,
                                              lingustic_variable_4: 4},
                                             -1, lingustic_variable_for_consequent_2)

        rules = [Rule(antecedent1, consequent1),
                 Rule(antecedent2, consequent2)]

        system = TakagiSugenoInferenceSystem(rules)

        print(system.infer(weighted_average, fuzzified, measures))


    def test_inference_type_I_2(self):
        df = pd.DataFrame({'fire': [1, 0.5, 0.2, 0.3, 0.7],
                           'air': [0.3, 0.4, 1, 0.7, 0.2],
                           'earth': [0, 0.3, 0.1, 0.9, 0.8],
                           'water': [0.6, 0.1, 0.4, 0.3, 0.5]})

        lingustic_variable_1 = LinguisticVariable('fire', Domain(0, 1.001, 0.001))
        lingustic_variable_2 = LinguisticVariable('air', Domain(0, 1.001, 0.001))
        lingustic_variable_3 = LinguisticVariable('earth', Domain(0, 1.001, 0.001))
        lingustic_variable_4 = LinguisticVariable('water', Domain(0, 1.001, 0.001))

        measures = {lingustic_variable_1: [1, 0.5, 0.2, 0.3, 0.7],
                    lingustic_variable_2: [0.3, 0.4, 1, 0.7, 0.2],
                    lingustic_variable_3: [0, 0.3, 0.1, 0.9, 0.8],
                    lingustic_variable_4: [0.6, 0.1, 0.4, 0.3, 0.5]}
        lower_scaling = 0.8
        clause1 = Clause(lingustic_variable_1, 'low', IntervalType2FuzzySet(triangular(0, 0.001, 0.5, lower_scaling),
                                                                            triangular(0, 0.001, 0.5)))
        clause2 = Clause(lingustic_variable_1, 'medium', IntervalType2FuzzySet(triangular(0.25, 0.5, 0.75, lower_scaling),
                                                                               triangular(0.25, 0.5, 0.75)))
        clause3 = Clause(lingustic_variable_1, 'high', IntervalType2FuzzySet(triangular(0.5, 0.999, 1, lower_scaling),
                                                                             triangular(0.5, 0.999, 1)))
        clause4 = Clause(lingustic_variable_2, 'low', IntervalType2FuzzySet(triangular(0, 0.001, 0.5, lower_scaling),
                                                                            triangular(0, 0.001, 0.5)))
        clause5 = Clause(lingustic_variable_2, 'medium', IntervalType2FuzzySet(triangular(0.25, 0.5, 0.75, lower_scaling),
                                                                               triangular(0.25, 0.5, 0.75)))
        clause6 = Clause(lingustic_variable_2, 'high', IntervalType2FuzzySet(triangular(0.5, 0.999, 1, lower_scaling),
                                                                             triangular(0.5, 0.999, 1)))
        clause7 = Clause(lingustic_variable_3, 'low', IntervalType2FuzzySet(triangular(0, 0.001, 0.5, lower_scaling),
                                                                            triangular(0, 0.001, 0.5)))
        clause8 = Clause(lingustic_variable_3, 'medium', IntervalType2FuzzySet(triangular(0.25, 0.5, 0.75, lower_scaling),
                                                                               triangular(0.25, 0.5, 0.75)))
        clause9 = Clause(lingustic_variable_3, 'high', IntervalType2FuzzySet(triangular(0.5, 0.999, 1, lower_scaling),
                                                                             triangular(0.5, 0.999, 1)))
        clause10 = Clause(lingustic_variable_4, 'low', IntervalType2FuzzySet(triangular(0, 0.001, 0.5, lower_scaling),
                                                                             triangular(0, 0.001, 0.5)))
        clause11 = Clause(lingustic_variable_4, 'medium', IntervalType2FuzzySet(triangular(0.25, 0.5, 0.75, lower_scaling),
                                                                                triangular(0.25, 0.5, 0.75)))
        clause12 = Clause(lingustic_variable_4, 'high', IntervalType2FuzzySet(triangular(0.5, 0.999, 1, lower_scaling),
                                                                              triangular(0.5, 0.999, 1)))

        to_fuzz = {}
        to_fuzz[lingustic_variable_1.name] = {'Low': clause1,
                                              'Medium': clause2,
                                              'High': clause3}
        to_fuzz[lingustic_variable_2.name] = {'Low': clause4,
                                              'Medium': clause5,
                                              'High': clause6}
        to_fuzz[lingustic_variable_3.name] = {'Low': clause7,
                                              'Medium': clause8,
                                              'High': clause9}
        to_fuzz[lingustic_variable_4.name] = {'Low': clause10,
                                              'Medium': clause11,
                                              'High': clause12}

        fuzzified = fuzzify(df, to_fuzz)
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

        antecedent1 = term1 & term4 & term7 & term10 & term3 & term5 & term6 & term8 & term12
        antecedent2 = term2 | term4 | term9 | term11

        lingustic_variable_for_consequent_1 = LinguisticVariable('avatar', Domain(0, 1.001, 0.001))
        lingustic_variable_for_consequent_2 = LinguisticVariable('momo', Domain(0, 1.001, 0.001))

        consequent1 = TakagiSugenoConsequent({lingustic_variable_1: 1,
                                              lingustic_variable_2: 2,
                                              lingustic_variable_3: 2,
                                              lingustic_variable_4: 3},
                                             5,
                                             lingustic_variable_for_consequent_1)
        consequent2 = TakagiSugenoConsequent({lingustic_variable_1: 2,
                                              lingustic_variable_2: 3,
                                              lingustic_variable_3: 1,
                                              lingustic_variable_4: 4},
                                             -1, lingustic_variable_for_consequent_2)

        rules = [Rule(antecedent1, consequent1),
                 Rule(antecedent2, consequent2)]

        system = TakagiSugenoInferenceSystem(rules)
        true_output = [4.3, 1.9, 4.1, 3.8, 3.8]
        for idx, val in enumerate(system.infer(takagi_sugeno_EIASC, fuzzified, measures)):
            assert val == approx(true_output[idx])
