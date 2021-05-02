from tests.test_tools import approx
from doggos.algebras import LukasiewiczAlgebra


class TestLukasiewiczAlgebra:

    def test_t_norm(self):
        assert LukasiewiczAlgebra.t_norm(0.9, 0.2) == approx(0.1)
        assert LukasiewiczAlgebra.t_norm(1.0, 1.0) == approx(1.0)
        assert LukasiewiczAlgebra.t_norm(1.0, 0.0) == approx(0.0)
        assert LukasiewiczAlgebra.t_norm(0.2, 0.35) == approx(0.0)

    def test_s_norm(self):
        assert LukasiewiczAlgebra.s_norm(0.9, 0.2) == approx(1.0)
        assert LukasiewiczAlgebra.s_norm(1.0, 1.0) == approx(1.0)
        assert LukasiewiczAlgebra.s_norm(1.0, 0.0) == approx(1.0)
        assert LukasiewiczAlgebra.s_norm(0.2, 0.35) == approx(0.55)

    def test_negation(self):
        assert LukasiewiczAlgebra.negation(1.0) == approx(0.0)
        assert LukasiewiczAlgebra.negation(0.0) == approx(1.0)
        assert LukasiewiczAlgebra.negation(0.2) == approx(0.8)

    def test_implication(self):
        assert LukasiewiczAlgebra.implication(1.0, 0.0) == approx(0.0)
        assert LukasiewiczAlgebra.implication(0.0, 1.0) == approx(1.0)
        assert LukasiewiczAlgebra.implication(0.2, 0.35) == approx(1.0)
        assert LukasiewiczAlgebra.implication(0.95, 0.2) == approx(0.25)
