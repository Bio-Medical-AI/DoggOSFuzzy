import pytest

from doggos.algebras import GodelAlgebra
from doggos.knowledge import LinguisticVariable, Domain
from doggos.knowledge import Clause
from doggos.knowledge import Term
from doggos.fuzzy_sets import Type1FuzzySet
from doggos.knowledge.linguistic_variable import LinguisticVariable, Domain

import numpy as np


class TestTermAntecedent:

    def test_exception_typerror_algebra_init(self):
        domain = Domain(0, 10, 0.5)
        ling_var = LinguisticVariable('Temperature', domain)
        fuzzy_set = Type1FuzzySet(lambda x: 0.5*x)
        clause = Clause(ling_var, 'Low', fuzzy_set)
        with pytest.raises(TypeError) as e:
            clause = Term(clause, [])
            assert 'algebra must be a Algebra type' in str(e.value)

    def test_exception_typerror_clause_init(self):
        with pytest.raises(TypeError) as e:
            clause = Term([], GodelAlgebra())
            assert 'clause must be a Clause type' in str(e.value)

    def test_exception_typerror_clause_setter(self):
        domain = Domain(0, 10, 0.5)
        ling_var = LinguisticVariable('Temperature', domain)
        fuzzy_set = Type1FuzzySet(lambda x: 0.5 * x)
        clause = Clause(ling_var, 'Low', fuzzy_set)
        antecedent = Term(clause, GodelAlgebra() )
        with pytest.raises(TypeError) as e:
            antecedent.clause = []
            assert 'clause must be a Clause type' in str(e.value)

    def test_exception_typerror_algebra_setter(self):
        domain = Domain(0, 10, 0.5)
        ling_var = LinguisticVariable('Temperature', domain)
        fuzzy_set = Type1FuzzySet(lambda x: 0.5 * x)
        clause = Clause(ling_var, 'Low', fuzzy_set)
        antecedent = Term( clause, GodelAlgebra())
        with pytest.raises(TypeError) as e:
            antecedent.algebra = []
            assert 'algebra must be a Algebra type' in str(e.value)

    def test_and(self):
        domain = Domain(0, 10, 0.5)
        ling_var = LinguisticVariable('Temperature', domain)
        fuzzy_set1 = Type1FuzzySet(lambda x: 0.5 * x)
        clause1 = Clause(ling_var, 'Low', fuzzy_set1)
        fuzzy_set2 = Type1FuzzySet(lambda x:2*x)
        clause2 = Clause(ling_var, 'High', fuzzy_set2)
        clause_dict = {
            clause1: clause1.get_value(2),
            clause2: clause2.get_value(2)
        }
        antecedent1 = Term(clause1, GodelAlgebra())
        antecedent2 = Term(clause2, GodelAlgebra())
        antecedent = antecedent1 & antecedent2
        assert antecedent.fire(clause_dict)

    def test_or(self):
        domain = Domain(0, 10, 0.5)
        ling_var = LinguisticVariable('Temperature', domain)
        fuzzy_set1 = Type1FuzzySet(lambda x: 0.5 * x)
        clause1 = Clause(ling_var, 'Low', fuzzy_set1)
        fuzzy_set2 = Type1FuzzySet(lambda x:2*x)
        clause2 = Clause(ling_var, 'High', fuzzy_set2)
        clause_dict = {
            clause1: clause1.get_value(2),
            clause2: clause2.get_value(2)
        }
        antecedent1 = Term(clause1, GodelAlgebra())
        antecedent2 = Term(clause2, GodelAlgebra())
        antecedent = antecedent1 | antecedent2
        assert antecedent.fire(clause_dict)
