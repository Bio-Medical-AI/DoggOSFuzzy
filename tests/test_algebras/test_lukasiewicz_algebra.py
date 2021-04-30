import pytest


from tests.test_config import TestConfig
from doggos.algebras import LukasiewiczAlgebra


class TestLukasiewiczAlgebra:

    def test_t_norm(self):
        assert LukasiewiczAlgebra.t_norm(0.9, 0.2) == pytest.approx(0.1, TestConfig.FLOAT_COMPARISON_PRECISION)
        assert LukasiewiczAlgebra.t_norm(1.0, 1.0) == pytest.approx(1.0, TestConfig.FLOAT_COMPARISON_PRECISION)
        assert LukasiewiczAlgebra.t_norm(1.0, 0.0) == pytest.approx(0.0, TestConfig.FLOAT_COMPARISON_PRECISION)
        assert LukasiewiczAlgebra.t_norm(0.2, 0.35) == pytest.approx(0.0, TestConfig.FLOAT_COMPARISON_PRECISION)

    def test_s_norm(self):
        assert LukasiewiczAlgebra.s_norm(0.9, 0.2) == pytest.approx(1.0, TestConfig.FLOAT_COMPARISON_PRECISION)
        assert LukasiewiczAlgebra.s_norm(1.0, 1.0) == pytest.approx(1.0, TestConfig.FLOAT_COMPARISON_PRECISION)
        assert LukasiewiczAlgebra.s_norm(1.0, 0.0) == pytest.approx(1.0, TestConfig.FLOAT_COMPARISON_PRECISION)
        assert LukasiewiczAlgebra.s_norm(0.2, 0.35) == pytest.approx(0.55, TestConfig.FLOAT_COMPARISON_PRECISION)

    def test_negation(self):
        assert LukasiewiczAlgebra.negation(1.0) == pytest.approx(0.0, TestConfig.FLOAT_COMPARISON_PRECISION)
        assert LukasiewiczAlgebra.negation(0.0) == pytest.approx(1.0, TestConfig.FLOAT_COMPARISON_PRECISION)
        assert LukasiewiczAlgebra.negation(0.2) == pytest.approx(1.0, TestConfig.FLOAT_COMPARISON_PRECISION)

    def test_implication(self):
        assert LukasiewiczAlgebra.implication(1.0, 0.0) == pytest.approx(0.0, TestConfig.FLOAT_COMPARISON_PRECISION)
        assert LukasiewiczAlgebra.implication(0.0, 1.0) == pytest.approx(1.0, TestConfig.FLOAT_COMPARISON_PRECISION)
        assert LukasiewiczAlgebra.implication(0.2, 0.35) == pytest.approx(0.55, TestConfig.FLOAT_COMPARISON_PRECISION)
        assert LukasiewiczAlgebra.implication(0.95, 0.2) == pytest.approx(1.0, TestConfig.FLOAT_COMPARISON_PRECISION)
