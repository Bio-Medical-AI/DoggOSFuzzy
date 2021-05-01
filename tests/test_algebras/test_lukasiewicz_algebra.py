import pytest
import numpy as np

from tests.test_tools import approx
from doggos.algebras import LukasiewiczAlgebra


class TestLukasiewiczAlgebra:

    def test_t_norm(self):
        assert LukasiewiczAlgebra.t_norm(0.9, 0.2) == approx(0.1)
        assert LukasiewiczAlgebra.t_norm(1.0, 1.0) == approx(1.0)
        assert LukasiewiczAlgebra.t_norm(1.0, 0.0) == approx(0.0)
        assert LukasiewiczAlgebra.t_norm(0.2, 0.35) == approx(0.0)

    def test_t_norm_array(self):
        assert all(res == approx(exp) for res, exp in zip(
            LukasiewiczAlgebra.t_norm(
                np.array([0.9, 1.0, 1.0, 0.2]),
                np.array([0.2, 1.0, 0.0, 0.35])),
            [0.1, 1.0, 0.0, 0.0])
        )

    # @pytest.mark(
    #     'a, b',
    #     ([(0.9, 1.0), [0.1, 1.0]], [(0.2, 1.0), [0.2, 1.0]])
    # )
    # def test_t_norm_iterable(self, a, b):
    #     expected = [0.1, 1.0]
    #     assert all(res == approx(exp) for res, exp in zip(LukasiewiczAlgebra.t_norm(a, b), expected))

    def test_t_norm_incompatible_dimensions(self):
        with pytest.raises(ValueError):
            _ = LukasiewiczAlgebra.t_norm(np.array([0.2, 0.3]), 0)

    def test_s_norm(self):
        assert LukasiewiczAlgebra.s_norm(0.9, 0.2) == approx(1.0)
        assert LukasiewiczAlgebra.s_norm(1.0, 1.0) == approx(1.0)
        assert LukasiewiczAlgebra.s_norm(1.0, 0.0) == approx(1.0)
        assert LukasiewiczAlgebra.s_norm(0.2, 0.35) == approx(0.55)

    def test_s_norm_array(self):
        pass

    def test_s_norm_iterable(self):
        pass

    def test_s_norm_incompatible_dimensions(self):
        pass

    def test_negation(self):
        assert LukasiewiczAlgebra.negation(1.0) == approx(0.0)
        assert LukasiewiczAlgebra.negation(0.0) == approx(1.0)
        assert LukasiewiczAlgebra.negation(0.2) == approx(0.8)

    def test_negation_array(self):
        pass

    def test_negation_iterable(self):
        pass

    def test_negation_incompatible_dimensions(self):
        pass

    def test_implication(self):
        assert LukasiewiczAlgebra.implication(1.0, 0.0) == approx(0.0)
        assert LukasiewiczAlgebra.implication(0.0, 1.0) == approx(1.0)
        assert LukasiewiczAlgebra.implication(0.2, 0.35) == approx(1.0)
        assert LukasiewiczAlgebra.implication(0.95, 0.2) == approx(0.25)

    def test_implication_array(self):
        pass

    def test_implication_iterable(self):
        pass

    def test_implication_incompatible_dimensions(self):
        pass
