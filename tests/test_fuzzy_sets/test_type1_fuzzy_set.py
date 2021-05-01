import pytest


from tests.test_fuzzy_sets import _random_sample


from doggos.fuzzy_sets import Type1FuzzySet
from doggos.utils.membership_functions import gaussian, sigmoid, triangular


class TestType1FuzzySet:

    @pytest.mark.parametrize('x', _random_sample(-10, 10, 5))
    def test_base_fuzzyset(self, x):
        membership_function = gaussian(0, 1)
        fuzzy_set = Type1FuzzySet(membership_function)
        assert fuzzy_set(x) == membership_function(x)

        mf = gaussian(0, 1)
        fuzzy_set = Type1FuzzySet(mf)
        assert fuzzy_set(x) == (mf(x))

    def test_exception_uncallable_init(self):
        with pytest.raises(ValueError) as e:
            _ = Type1FuzzySet([])
            assert 'Membership function must be callable' in str(e.value)

    def test_getter_fuzzy_set(self):
        membership_function = sigmoid(0, 1)
        fuzzy_set = Type1FuzzySet(membership_function)
        assert fuzzy_set.membership_function == membership_function

    @pytest.mark.parametrize('x', _random_sample(-10, 10, 5))
    def test_setter_fuzzy_set(self, x):
        membership_function1 = sigmoid(0, 1)
        membership_function2 = triangular(2, 5, 2, 1)
        fuzzy_set = Type1FuzzySet(membership_function1)
        fuzzy_set.membership_function = membership_function2
        assert fuzzy_set(x) == membership_function2(x)

        mf1 = sigmoid(0, 1)
        mf2 = triangular(2, 5, 2, 1)
        fuzzy_set = Type1FuzzySet(mf1)
        fuzzy_set.membership_function = mf2
        assert fuzzy_set(x) == mf2(x)

    def test_exception_uncallable_setter(self):
        fuzzy_set = Type1FuzzySet(sigmoid(0, 1))
        with pytest.raises(ValueError) as e:
            fuzzy_set.membership_function = ()
            assert 'Membership function must be callable' in str(e.value)
