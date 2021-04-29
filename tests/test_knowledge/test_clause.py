import pytest

from doggos.knowledge import LinguisticVariable, Domain
from doggos.knowledge import Clause
from doggos.fuzzy_sets import Type1FuzzySet
from doggos.knowledge.linguistic_variable import LinguisticVariable, Domain

import numpy as np


class TestClause:

    def test_exception_typeerror_ling_var_init(self):
        fuzzy_set = Type1FuzzySet(lambda x: 0 if x < 0 else 1)
        with pytest.raises(TypeError) as e:

            assert 'Linguistic variable must be LingusticVariable type' in str(e.value)

    def test_get_value(self, x):
        pass

    def test_exception_typerror_setter_fuzzy_set(self):
        pass

    def test_getter_fuzzy_set(self):
        pass

    def test_exveption_typerror_setter_ling_var(self):
        pass

    def test_getter_ling_var(self):
        pass

    def test_gradiation_adj_setter(self):
        pass

    def test_getter_gradiation_adj(self):
        pass
