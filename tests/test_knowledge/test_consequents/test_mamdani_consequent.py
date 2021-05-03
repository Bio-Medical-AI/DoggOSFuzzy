from doggos.knowledge.consequents.mamdani_consequent import MamdaniConsequent
from doggos.knowledge.clause import Clause
from doggos.knowledge.linguistic_variable import LinguisticVariable, Domain
from doggos.utils.membership_functions.membership_functions import triangular, gaussian, sigmoid
from doggos.fuzzy_sets.type1_fuzzy_set import Type1FuzzySet
from doggos.fuzzy_sets.interval_type2_fuzzy_set import IntervalType2FuzzySet

import pytest
import numpy as np


class TestMamdaniConsequent:

    def test_output_type1_firing(self):
        watering = LinguisticVariable('watering', Domain(0, 1, 0.001))

        low_watering = Type1FuzzySet(triangular(0, 0.25, 0.5))
        medium_watering = Type1FuzzySet(gaussian(0.5, 0.1))
        high_watering = Type1FuzzySet(sigmoid(0.75, 5))

        low_watering_clause = Clause(watering, 'Low', low_watering)
        medium_watering_clause = Clause(watering, 'Medium', medium_watering)
        high_watering_clause = Clause(watering, 'High', high_watering)

        low_watering_consequent = MamdaniConsequent(low_watering_clause)
        medium_watering_consequent = MamdaniConsequent(medium_watering_clause)
        high_watering_consequent = MamdaniConsequent(high_watering_clause)

        first_rule_firing = 0.3
        second_rule_firing = 0.5
        third_rule_firing = 0.65

        assert np.allclose(low_watering_consequent.output(first_rule_firing).values,
                           np.minimum(low_watering_clause.values, first_rule_firing))

        assert np.allclose(medium_watering_consequent.output(second_rule_firing).values,
                              np.minimum(medium_watering_clause.values, second_rule_firing))

        assert np.allclose(high_watering_consequent.output(third_rule_firing).values,
                              np.minimum(high_watering_clause.values, third_rule_firing))

    @pytest.mark.parametrize(['first_rule_firing', 'second_rule_firing', 'third_rule_firing'],
                             [[(0.3, 0.5), (0.2, 0.7), (0, 1)],
                              [[0.3, 0.5], [0.2, 0.7], [0, 1]],
                              [np.array([0.3, 0.5]), np.array([0.2, 0.7]), np.array([0, 1])],
                              [np.array([[0.3], [0.5]]), np.array([[0.2], [0.7]]), np.array([[0], [1]])]]
                             )
    def test_output_it2_firing_types(self, first_rule_firing, second_rule_firing, third_rule_firing):
        watering = LinguisticVariable('watering', Domain(0, 1, 0.001))

        low_watering = IntervalType2FuzzySet(triangular(0, 0.25, 0.5), triangular(-0.1, 0.25, 0.6))
        medium_watering = IntervalType2FuzzySet(gaussian(0.5, 0.1), gaussian(0.5, 0.15))
        high_watering = IntervalType2FuzzySet(sigmoid(0.75, 5), sigmoid(0.65, 5))

        low_watering_clause = Clause(watering, 'Low', low_watering)
        medium_watering_clause = Clause(watering, 'Medium', medium_watering)
        high_watering_clause = Clause(watering, 'High', high_watering)

        low_watering_consequent = MamdaniConsequent(low_watering_clause)
        medium_watering_consequent = MamdaniConsequent(medium_watering_clause)
        high_watering_consequent = MamdaniConsequent(high_watering_clause)

        first_consequent_output = low_watering_consequent.output(first_rule_firing).values
        second_consequent_output = medium_watering_consequent.output(second_rule_firing).values
        third_consequent_output = high_watering_consequent.output(third_rule_firing).values

        first_rule_firing = np.array(first_rule_firing).reshape(2, 1)
        second_rule_firing = np.array(second_rule_firing).reshape(2, 1)
        third_rule_firing = np.array(third_rule_firing).reshape(2, 1)

        assert np.allclose(first_consequent_output,
                              np.minimum(low_watering_clause.values, first_rule_firing))

        assert np.allclose(second_consequent_output,
                              np.minimum(medium_watering_clause.values, second_rule_firing))

        assert np.allclose(third_consequent_output,
                              np.minimum(high_watering_clause.values, third_rule_firing))

    @pytest.mark.parametrize('rule_firing',
                             [{0: 0.2, 1: 0.3},
                              "(0.2, 1)",
                              np.array([[0.1], [0.2], [0.3], [0.4]]),
                              np.array([[0.2, 0.3], [0.1, 0.2]])])
    def test_output_throws_exception_in_case_of_wrong_rule_firing_type(self, rule_firing):
        watering = LinguisticVariable('watering', Domain(0, 1, 0.001))
        low_watering = IntervalType2FuzzySet(triangular(0, 0.25, 0.5), triangular(-0.1, 0.25, 0.6))
        low_watering_clause = Clause(watering, 'Low', low_watering)
        low_watering_consequent = MamdaniConsequent(low_watering_clause)

        with pytest.raises(ValueError):
            low_watering_consequent.output(rule_firing)

    def test_clause_getter(self):
        watering = LinguisticVariable('watering', Domain(0, 1, 0.001))
        low_watering = IntervalType2FuzzySet(triangular(0, 0.25, 0.5), triangular(-0.1, 0.25, 0.6))
        low_watering_clause = Clause(watering, 'Low', low_watering)
        low_watering_consequent = MamdaniConsequent(low_watering_clause)

        low_watering_consequent.output((0.2, 0.3))

        assert low_watering_consequent.clause == low_watering_clause

    def test_cut_clause_getter(self):
        watering = LinguisticVariable('watering', Domain(0, 1, 0.001))
        low_watering = IntervalType2FuzzySet(triangular(0, 0.25, 0.5), triangular(-0.1, 0.25, 0.6))
        low_watering_clause = Clause(watering, 'Low', low_watering)
        low_watering_consequent = MamdaniConsequent(low_watering_clause)

        cut_clause = low_watering_consequent.output((0.2, 0.3))

        assert low_watering_consequent.cut_clause == cut_clause
