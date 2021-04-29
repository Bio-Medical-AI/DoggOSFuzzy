import pytest
import numpy as np

from doggos.knowledge import LinguisticVariable, Domain


class TestDomain:

    @pytest.mark.parametrize('min_, max_, precision', zip(np.arange(0, 10, 1),
                                                          np.arange(20, 30, 1),
                                                          np.arange(0.01, 0.1, 0.01)))
    def test_domain_values(self, min_, max_, precision):
        intervals = np.arange(min_, max_, precision)
        domain = Domain(min_, max_, precision)
        assert np.array_equal(domain(),intervals)
        assert domain.min == min_
        assert domain.max == max_


class TestLinguisticVariable:

    def test_exception_typeerror_init(self):
        domain = Domain(0, 10, 0.01)
        with pytest.raises(TypeError) as e:
            ling_var = LinguisticVariable('Temperature', [])
            assert 'Linguistic variable requires domain to be Domain type' in str(e.value)

    @pytest.mark.parametrize('min_, max_, precision', zip(np.arange(0, 10, 1),
                                                          np.arange(20, 30, 1),
                                                          np.arange(0.01, 0.1, 0.01)))
    def test_domain(self, min_, max_, precision):
        domain = Domain(0, 10, 0.01)
        ling_var = LinguisticVariable('Temperature', domain)
        assert ling_var.domain == domain
