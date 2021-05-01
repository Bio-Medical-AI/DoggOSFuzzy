from tests.test_tools import approx
from doggos.algebras import GodelAlgebra


class TestGodelAlgebra:



    def test_t_norm(self):
        assert GodelAlgebra.t_norm(0.9, 0.2) == approx(0.2)
        assert GodelAlgebra.t_norm(1.0, 1.0) == approx(1.0)
        assert GodelAlgebra.t_norm(1.0, 0.0) == approx(0.0)
        assert GodelAlgebra.t_norm(0.2, 0.35) == approx(0.2)

    def test_s_norm(self):
        assert GodelAlgebra.s_norm(0.9, 0.2) == approx(0.9)
        assert GodelAlgebra.s_norm(1.0, 1.0) == approx(1.0)
        assert GodelAlgebra.s_norm(1.0, 0.0) == approx(1.0)
        assert GodelAlgebra.s_norm(0.2, 0.35) == approx(0.35)

    def test_negation(self):
        assert GodelAlgebra.negation(1.0) == approx(0.0)
        assert GodelAlgebra.negation(0.0) == approx(1.0)
        assert GodelAlgebra.negation(0.2) == approx(0.8)

    def test_implication(self):
        assert GodelAlgebra.implication(1.0, 0.0) == approx(0.0)
        assert GodelAlgebra.implication(0.0, 1.0) == approx(1.0)
        assert GodelAlgebra.implication(0.2, 0.35) == approx(0.8)
        assert GodelAlgebra.implication(0.95, 0.2) == approx(0.2)
