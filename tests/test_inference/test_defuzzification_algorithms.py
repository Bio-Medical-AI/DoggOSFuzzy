import pytest
import numpy as np

from doggos.inference.defuzzification_algorithms import (calculate_membership,
                                                         karnik_mendel,
                                                         takagi_sugeno_karnik_mendel,
                                                         weighted_average)


class TestDefuzzificationAlgorithms:

    @pytest.mark.parametrize('outputs_of_rules, expected_mf',
                             zip(np.array([[[3, 1], [6, 3], [9, 5], [13, 3], [19, 1], [21, 4]],
                                           [[3, 3], [6, 7], [9, 7], [13, 6], [19, 9], [21, 6]]]),
                                 [[0., 0., 1., 5 / 3, 7 / 3, 3., 11 / 3, 13 / 3, 5., 4.5, 4., 3.5, 3., 8 / 3,
                                   7 / 3, 2., 5 / 3, 4 / 3, 1., 2.5, 4., 0.],
                                  [0., 0., 3., 13 / 3, 17 / 3, 7., 7., 7., 7., 6.75, 6.5, 6.25, 6., 6.5, 7.,
                                   7.5, 8., 8.5, 9., 7.5, 6., 0.]]))
    def test_calculate_membership(self, outputs_of_rules, expected_mf):
        domain = np.arange(1, len(expected_mf) + 1, 1)
        mf = np.zeros(shape=domain.shape)
        for i in range(domain.shape[0]):
            mf[i] = calculate_membership(domain[i], outputs_of_rules)
        assert all(a == pytest.approx(b, 0.1) for a, b in zip(mf.tolist(), expected_mf))

    def test_weighted_average(self):
        outputs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        firings = np.array([0, 0.75, 0, 0.5, 3, -1, 0.1, 0.33, 2, 1])
        assert weighted_average(firings, outputs) == pytest.approx(43.84 / 6.68, 0.0000001)

    def test_takagi_sugeno_karnik_mendel(self):
        outputs = np.array([3, 6, 9, 13, 19, 21]).reshape((-1, 1))
        firings = np.array([[1, 3],
                            [3, 7],
                            [5, 7],
                            [3, 6],
                            [1, 9],
                            [4, 6]])
        lmf = np.array([1., 5 / 3, 7 / 3, 3., 11 / 3, 13 / 3, 5., 4.5, 4., 3.5, 3.,
                        8 / 3, 7 / 3, 2., 5 / 3, 4 / 3, 1., 2.5, 4.]).reshape(-1)
        umf = np.array([3., 13 / 3, 17 / 3, 7., 7., 7., 7., 6.75, 6.5, 6.25, 6., 6.5, 7., 7.5, 8., 8.5,
                        9., 7.5, 6.]).reshape(-1)
        domain = np.arange(3, 22, 1).reshape(-1)
        assert takagi_sugeno_karnik_mendel(firings, outputs, 1) == karnik_mendel(lmf, umf, domain)
