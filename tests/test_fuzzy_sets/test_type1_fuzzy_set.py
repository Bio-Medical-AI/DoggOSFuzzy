import pytest


from doggos.fuzzy_sets import Type1FuzzySet
from doggos.utils.membership_functions import gaussian, sigmoid, triangular


class TestType1FuzzySet:

    def test_base_fuzzyset(self):
        membership_function = gaussian(0, 1)
        fuzzy_set = Type1FuzzySet(membership_function)
        assert fuzzy_set(1) == membership_function(1)
        assert fuzzy_set(0.3) == membership_function(0.3)
        assert fuzzy_set(-2) == membership_function(-2)

    def test_exception_uncallable_init(self):
        with pytest.raises(ValueError) as e:
            _ = Type1FuzzySet([])
            assert 'Membership function must be callable' in str(e.value)

    def test_getter_fuzzy_set(self):
        membership_function = sigmoid(0, 1)
        fuzzy_set = Type1FuzzySet(membership_function)
        assert fuzzy_set.membership_function == membership_function

    def test_setter_fuzzy_set(self):
        membership_function1 = sigmoid(0, 1)
        membership_function2 = triangular(2, 5, 2, 1)
        fuzzy_set = Type1FuzzySet(membership_function1)
        assert fuzzy_set(2) == membership_function1(2)
        fuzzy_set.membership_function = membership_function2
        assert fuzzy_set(3) == membership_function2(3)

    def test_exception_uncallable_setter(self):
        fuzzy_set = Type1FuzzySet(sigmoid(0, 1))
        with pytest.raises(ValueError) as e:
            fuzzy_set.membership_function = ()
            assert 'Membership function must be callable' in str(e.value)
