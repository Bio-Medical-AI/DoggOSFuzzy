import pytest
import numpy as np


from tests.test_tools import approx
from doggos.algebras import GodelAlgebra


class TestGodelAlgebra:

    @pytest.mark.parametrize('a, b, c', zip(
        [0.9, 1.0, 1.0, 0.2],
        [0.2, 1.0, 0.0, 0.35],
        [0.2, 1.0, 0.0, 0.2]
    ))
    def test_t_norm(self, a, b, c):
        assert GodelAlgebra.t_norm(a, b) == approx(c)

    @pytest.mark.parametrize('a, b, c', zip(
        [np.array([0.9, 1.0, 1.0, 0.2])],
        [np.array([0.2, 1.0, 0.0, 0.35])],
        [[0.2, 1.0, 0.0, 0.2]]
    ))
    def test_t_norm_array(self, a, b, c):
        assert all(res == approx(exp) for res, exp in zip(
            GodelAlgebra.t_norm(a, b),
            c
        ))

    @pytest.mark.parametrize('a, b, c', zip(
        [(0.9, 1.0)],
        [[0.2, 1.0]],
        [[0.2, 1.0]]
    ))
    def test_t_norm_iterable(self, a, b, c):
        assert all(res == approx(exp) for res, exp in zip(
            GodelAlgebra.t_norm(a, b),
            c
        ))

    @pytest.mark.parametrize('a, b', zip(
            [np.random.randn(2),
             np.random.randn(3),
             np.random.randn(2, 3),
             np.random.randn(5, 3),
             ],
            [np.random.randn(1),
             3,
             np.random.randn(3, 2),
             np.random.randn(5, 2),
             ],
    ))
    def test_t_norm_incompatible_dimensions(self, a, b):
        with pytest.raises(ValueError):
            _ = GodelAlgebra.t_norm(a, b)

    @pytest.mark.parametrize('a, b, c', zip(
        [0.9, 1.0, 1.0, 0.2],
        [0.2, 1.0, 0.0, 0.35],
        [0.9, 1.0, 1.0, 0.35]
    ))
    def test_s_norm(self, a, b, c):
        assert GodelAlgebra.s_norm(a, b) == approx(c)

    @pytest.mark.parametrize('a, b, c', zip(
        [np.array([0.9, 1.0, 1.0, 0.2])],
        [np.array([0.2, 1.0, 0.0, 0.35])],
        [[0.9, 1.0, 1.0, 0.35]]
    ))
    def test_s_norm_array(self, a, b, c):
        assert all(res == approx(exp) for res, exp in zip(
            GodelAlgebra.s_norm(a, b),
            c
        ))

    @pytest.mark.parametrize('a, b, c', zip(
        [(0.9, 1.0)],
        [[0.2, 1.0]],
        [[0.9, 1.0]]
    ))
    def test_s_norm_iterable(self, a, b, c):
        assert all(res == approx(exp) for res, exp in zip(
            GodelAlgebra.s_norm(a, b),
            c
        ))

    @pytest.mark.parametrize('a, b', zip(
            [np.random.randn(2),
             np.random.randn(3),
             np.random.randn(2, 3),
             np.random.randn(5, 3),
             ],
            [np.random.randn(1),
             3,
             np.random.randn(3, 2),
             np.random.randn(5, 2),
             ],
    ))
    def test_s_norm_incompatible_dimensions(self, a, b):
        with pytest.raises(ValueError):
            _ = GodelAlgebra.s_norm(a, b)

    @pytest.mark.parametrize('a, b', zip(
        [1.0, 0.0, 0.2],
        [0.0, 1.0, 0.8]
    ))
    def test_negation(self, a, b):
        assert GodelAlgebra.negation(a) == approx(b)

    @pytest.mark.parametrize('a, b', zip(
        [np.array([1.0, 0.0, 0.2])],
        [[0.0, 1.0, 0.8]]
    ))
    def test_negation_array(self, a, b):
        assert all(res == approx(exp) for res, exp in zip(
            GodelAlgebra.negation(a),
            b
        ))

    @pytest.mark.parametrize('a, b', zip(
        [[0.1, 0.2], (0.1, 0.9)],
        [[0.9, 0.8], (0.9, 0.1)]
    ))
    def test_negation_iterable(self, a, b):
        assert all(res == approx(exp) for res, exp in zip(
            GodelAlgebra.negation(a), b
        ))

    @pytest.mark.parametrize('a, b, c', zip(
        [1.0, 0.0, 0.2, 0.95],
        [0.0, 1.0, 0.35, 0.2],
        [0.0, 1.0, 0.8, 0.2]
    ))
    def test_implication(self, a, b, c):
        assert GodelAlgebra.implication(a, b) == approx(c)

    @pytest.mark.parametrize('a, b, c', zip(
        [np.array([1.0, 0.0, 0.2, 0.95])],
        [np.array([0.0, 1.0, 0.35, 0.2])],
        [[0.0, 1.0, 0.8, 0.2]]
    ))
    def test_implication_array(self, a, b, c):
        assert all(res == approx(exp) for res, exp in zip(
            GodelAlgebra.implication(a, b),
            c
        ))

    @pytest.mark.parametrize('a, b, c', zip(
        [(1.0, 0.0), [0.2, 0.95]],
        [(0.0, 1.0), np.array([0.35, 0.2])],
        [(0.0, 1.0), (0.8, 0.2)]
    ))
    def test_implication_iterable(self, a, b, c):
        assert all(res == approx(exp) for res, exp in zip(
            GodelAlgebra.implication(a, b),
            c
        ))

    @pytest.mark.parametrize('a, b', zip(
            [np.random.randn(2),
             np.random.randn(3),
             np.random.randn(2, 3),
             np.random.randn(5, 3),
             ],
            [np.random.randn(1),
             3,
             np.random.randn(3, 2),
             np.random.randn(5, 2),
             ],
    ))
    def test_implication_incompatible_dimensions(self, a, b):
        with pytest.raises(ValueError):
            _ = GodelAlgebra.implication(a, b)
