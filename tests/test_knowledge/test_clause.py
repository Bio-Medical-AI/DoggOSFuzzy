import pytest

from doggos.knowledge import LinguisticVariable, Domain
from doggos.knowledge import Clause
from doggos.fuzzy_sets import Type1FuzzySet
from doggos.knowledge.linguistic_variable import LinguisticVariable, Domain

import numpy as np


class TestClause:

    def test_exception_typeerror_lingvar_init(self):
        fuzzy_set = Type1FuzzySet(lambda x: 0 if x < 0 else 1)
        with pytest.raises(TypeError) as e:
            clause = Clause([], 'High', fuzzy_set)
            assert 'Linguistic variable must be LinguisticVariable type' in str(e.value)

    def test_exception_typeerror_fuzzyset_init(self):
        domain = Domain(0, 10, 0.01)
        ling_var = LinguisticVariable('Temperature', domain)
        with pytest.raises(TypeError) as e:
            clause = Clause(ling_var, 'High', [])
            assert 'Fuzzy set must be FuzzySet type' in str(e.value)

    @pytest.mark.parametrize('x', np.arange(0, 10, 0.5))
    def test_get_value(self, x):
        domain = Domain(0, 10, 0.01)
        ling_var = LinguisticVariable('Temperature', domain)
        fuzzy_set = Type1FuzzySet(lambda x: 0.5*x)
        clause = Clause(ling_var, 'High', fuzzy_set)
        values = fuzzy_set(domain())
        index = (x - domain.min)/domain.precision
        assert values[index] == clause.get_value(x)

    @pytest.mark.parametrize('x', np.arange(-5, -1))
    def test_exception_valueerror_get_value(self, x):
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
        fuzzy_set = Type1FuzzySet(lambda x: 0.5*x)
        clause = Clause(ling_var, 'High', fuzzy_set)
        with pytest.raises(TypeError) as e:
            clause.fuzzy_set = []
            assert 'Fuzzy set must be FuzzySet type' in str(e.value)

    def test_getter_fuzzy_set(self):
        domain = Domain(0, 10, 0.01)
        ling_var = LinguisticVariable('Temperature', domain)
        fuzzy_set = Type1FuzzySet(lambda x: 0.5*x)
        clause = Clause(ling_var, 'High', fuzzy_set)
        assert clause.fuzzy_set == fuzzy_set

    def test_exveption_typerror_setter_ling_var(self):
        domain = Domain(0, 10, 0.01)
        ling_var = LinguisticVariable('Temperature', domain)
        fuzzy_set = Type1FuzzySet(lambda x: 0.5*x)
        clause = Clause(ling_var, 'High', fuzzy_set)
        with pytest.raises(TypeError) as e:
            clause.linguistic_variable = []
            assert 'Linguistic variable must be LinguisticVariable type' in str(e.value)

    def test_getter_ling_var(self):
        domain = Domain(0, 10, 0.01)
        ling_var = LinguisticVariable('Temperature', domain)
        fuzzy_set = Type1FuzzySet(lambda x: 0.5*x)
        clause = Clause(ling_var, 'High', fuzzy_set)
        assert  clause.linguistic_variable == ling_var

    def test_gradiation_adj_setter(self):
        domain = Domain(0, 10, 0.01)
        ling_var = LinguisticVariable('Temperature', domain)
        fuzzy_set = Type1FuzzySet(lambda x: 0.5*x)
        clause = Clause(ling_var, 'High', fuzzy_set)
        clause.gradiation_adjective = 'Low'
        assert clause.gradiation_adjective == 'Low'

    def test_getter_gradiation_adj(self):
        domain = Domain(0, 10, 0.01)
        ling_var = LinguisticVariable('Temperature', domain)
        fuzzy_set = Type1FuzzySet(lambda x: 0.5*x)
        clause = Clause(ling_var, 'High', fuzzy_set)
        assert clause.gradiation_adjective == 'High'
