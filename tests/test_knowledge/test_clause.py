import pytest

from doggos.knowledge import Clause
from doggos.fuzzy_sets import Type1FuzzySet
from doggos.knowledge.linguistic_variable import LinguisticVariable, Domain

import numpy as np


class TestClause:

    def test_exception_typeerror_lingvar_init(self):
        fuzzy_set = Type1FuzzySet(lambda x: 0 if x < 0 else 1)
        with pytest.raises(TypeError) as e:
            clause = Clause([], 'High', fuzzy_set)
            assert 'Linguistic variable must be a LinguisticVariable type' in str(e.value)

    def test_exception_typeerror_fuzzyset_init(self):
        domain = Domain(0, 10, 0.01)
        ling_var = LinguisticVariable('Temperature', domain)
        with pytest.raises(TypeError) as e:
            clause = Clause(ling_var, 'High', [])
            assert 'Fuzzy set must be a FuzzySet type' in str(e.value)

    @pytest.mark.parametrize('x', np.arange(0, 10, 0.5))
    def test_get_value(self, x):
        domain = Domain(0, 10, 0.01)
        ling_var = LinguisticVariable('Temperature', domain)
        fuzzy_set = Type1FuzzySet(lambda x: 0.5 * x)
        clause = Clause(ling_var, 'High', fuzzy_set)
        values = fuzzy_set(domain())
        index = np.round((x - domain.min) / domain.precision).astype(int)
        assert values[index] == clause.get_value(x)

    @pytest.mark.parametrize('x, y, z', zip(np.arange(0, 10, 1), np.arange(2, 12, 1), np.arange(3, 13, 1)))
    def test_get_value_collection(self, x, y, z):
        collection = np.array([x, y, z])
        domain = Domain(0, 15, 0.01)
        ling_var = LinguisticVariable('Temperature', domain)
        fuzzy_set = Type1FuzzySet(lambda x: 0.5 * x)
        clause = Clause(ling_var, 'High', fuzzy_set)
        values = fuzzy_set(domain())
        index = np.round((collection - domain.min) / domain.precision).astype(int)
        assert np.array_equal(values[index], clause.get_value(collection))

    @pytest.mark.parametrize('x', np.arange(-5, -1, 1))
    def test_exception_value_error_get_value(self, x):
        domain = Domain(0, 10, 0.01)
        ling_var = LinguisticVariable('Temperature', domain)
        fuzzy_set = Type1FuzzySet(lambda x: 0.5 * x)
        clause = Clause(ling_var, 'High', fuzzy_set)
        with pytest.raises(ValueError) as e:
            clause.get_value(x)
            assert 'There is no such value in the domain' in str(e.value)

    def test_exception_typerror_setter_fuzzy_set(self):
        domain = Domain(0, 10, 0.01)
        ling_var = LinguisticVariable('Temperature', domain)
        fuzzy_set = Type1FuzzySet(lambda x: 0.5 * x)
        clause = Clause(ling_var, 'High', fuzzy_set)
        with pytest.raises(TypeError) as e:
            clause.fuzzy_set = []
            assert 'Fuzzy set must be a FuzzySet type' in str(e.value)

    def test_getter_fuzzy_set(self):
        domain = Domain(0, 10, 0.01)
        ling_var = LinguisticVariable('Temperature', domain)
        fuzzy_set = Type1FuzzySet(lambda x: 0.5 * x)
        clause = Clause(ling_var, 'High', fuzzy_set)
        assert clause.fuzzy_set == fuzzy_set

    def test_exception_typerror_setter_ling_var(self):
        domain = Domain(0, 10, 0.01)
        ling_var = LinguisticVariable('Temperature', domain)
        fuzzy_set = Type1FuzzySet(lambda x: 0.5 * x)
        clause = Clause(ling_var, 'High', fuzzy_set)
        with pytest.raises(TypeError) as e:
            clause.linguistic_variable = []
            assert 'Linguistic variable must be a LinguisticVariable type' in str(e.value)

    def test_getter_ling_var(self):
        domain = Domain(0, 10, 0.01)
        ling_var = LinguisticVariable('Temperature', domain)
        fuzzy_set = Type1FuzzySet(lambda x: 0.5 * x)
        clause = Clause(ling_var, 'High', fuzzy_set)
        assert clause.linguistic_variable == ling_var

    def test_gradation_adj_setter(self):
        domain = Domain(0, 10, 0.01)
        ling_var = LinguisticVariable('Temperature', domain)
        fuzzy_set = Type1FuzzySet(lambda x: 0.5 * x)
        clause = Clause(ling_var, 'High', fuzzy_set)
        clause.gradation_adjective = 'Low'
        assert clause.gradation_adjective == 'Low'

    def test_getter_gradation_adj(self):
        domain = Domain(0, 10, 0.01)
        ling_var = LinguisticVariable('Temperature', domain)
        fuzzy_set = Type1FuzzySet(lambda x: 0.5 * x)
        clause = Clause(ling_var, 'High', fuzzy_set)
        assert clause.gradation_adjective == 'High'

    def test_setter_exception_typeerror_values(self):
        domain = Domain(0, 10, 0.01)
        ling_var = LinguisticVariable('Temperature', domain)
        fuzzy_set = Type1FuzzySet(lambda x: 0.5 * x)
        clause = Clause(ling_var, 'High', fuzzy_set)
        func = lambda x: 0.7 * x
        new_domain = Domain(0, 11, 0.01)
        new_values = func(new_domain())
        with pytest.raises(ValueError) as e:
            clause.values = new_values
            assert 'Values length mismatches domain of linguistic variable' in str(e.value)

    def test_getter_values(self):
        domain = Domain(0, 10, 0.01)
        ling_var = LinguisticVariable('Temperature', domain)
        fuzzy_set = Type1FuzzySet(lambda x: 0.5 * x)
        clause = Clause(ling_var, 'High', fuzzy_set)
        new_values = fuzzy_set(domain())
        assert np.array_equal(clause.values, new_values)

    def test_setter_values(self):
        domain = Domain(0, 10, 0.01)
        ling_var = LinguisticVariable('Temperature', domain)
        fuzzy_set = Type1FuzzySet(lambda x: 0.5 * x)
        clause = Clause(ling_var, 'High', fuzzy_set)
        func = lambda x: 0.7 * x
        new_values = func(domain())
        clause.values = new_values
