import pytest
import numpy as np


# from doggos.inference import MamdaniInferenceSystem


class TestMamdaniInferenceSystem:
    @pytest.mark.parametrize('c_prim, correct_k', zip(np.arange(0.0005, 1, 0.05),
                                                      np.arange(0, 1000, 50)))
    def test_karnik_mendel_find_k_with_c_between_universe_values(self, c_prim, correct_k):
        precision = 0.001
        start = 0
        stop = 1
        universe = np.arange(start, stop, precision)
        k = np.where(universe <= c_prim)[0][-1]
        assert correct_k == k

    @pytest.mark.parametrize('c_prim, correct_k', zip(np.arange(0, 1, 0.05),
                                                      np.arange(0, 1000, 50)))
    def test_karnik_mendel_find_k_with_c_equal_to_universe_values(self, c_prim, correct_k):
        precision = 0.001
        start = 0
        stop = 1
        universe = np.arange(start, stop, precision)
        k = np.where(universe <= c_prim)[0][-1]
        assert correct_k == k

