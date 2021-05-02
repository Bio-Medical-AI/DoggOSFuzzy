import pytest


from tests.test_tools import TestConfig
from doggos.algebras import GodelAlgebra


class TestGodelAlgebra:

    def test_t_norm(self):
        assert GodelAlgebra.t_norm(0.9, 0.2) == pytest.approx(0.2, TestConfig.FLOAT_COMPARISON_PRECISION)
        assert GodelAlgebra.t_norm(1.0, 1.0) == pytest.approx(1.0, TestConfig.FLOAT_COMPARISON_PRECISION)
        assert GodelAlgebra.t_norm(1.0, 0.0) == pytest.approx(0.0, TestConfig.FLOAT_COMPARISON_PRECISION)
        assert GodelAlgebra.t_norm(0.2, 0.35) == pytest.approx(0.2, TestConfig.FLOAT_COMPARISON_PRECISION)

    def test_s_norm(self):
        assert GodelAlgebra.s_norm(0.9, 0.2) == pytest.approx(0.9, TestConfig.FLOAT_COMPARISON_PRECISION)
        assert GodelAlgebra.s_norm(1.0, 1.0) == pytest.approx(1.0, TestConfig.FLOAT_COMPARISON_PRECISION)
        assert GodelAlgebra.s_norm(1.0, 0.0) == pytest.approx(1.0, TestConfig.FLOAT_COMPARISON_PRECISION)
        assert GodelAlgebra.s_norm(0.2, 0.35) == pytest.approx(0.35, TestConfig.FLOAT_COMPARISON_PRECISION)

    def test_negation(self):
        assert GodelAlgebra.negation(1.0) == pytest.approx(0.0, TestConfig.FLOAT_COMPARISON_PRECISION)
        assert GodelAlgebra.negation(0.0) == pytest.approx(1.0, TestConfig.FLOAT_COMPARISON_PRECISION)
        assert GodelAlgebra.negation(0.2) == pytest.approx(0.8, TestConfig.FLOAT_COMPARISON_PRECISION)

    def test_implication(self):
        assert GodelAlgebra.implication(1.0, 0.0) == pytest.approx(0.0, TestConfig.FLOAT_COMPARISON_PRECISION)
        assert GodelAlgebra.implication(0.0, 1.0) == pytest.approx(1.0, TestConfig.FLOAT_COMPARISON_PRECISION)
        assert GodelAlgebra.implication(0.2, 0.35) == pytest.approx(0.8, TestConfig.FLOAT_COMPARISON_PRECISION)
        assert GodelAlgebra.implication(0.95, 0.2) == pytest.approx(0.2, TestConfig.FLOAT_COMPARISON_PRECISION)
