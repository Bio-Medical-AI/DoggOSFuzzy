import pytest
import numpy as np


from doggos.utils.membership_functions.membership_functions import triangular
from doggos.inference import MamdaniInferenceSystem


def defuzz(x, mfx):
    teta = (mfx[0] + mfx[1]) / 2.0
    yL, L = __find_y(x, teta.copy(), mfx[1], mfx[0])
    yR, R = __find_y(x, teta.copy(), mfx[0], mfx[1])
    return (yL + yR) / 2


def __find_y(universe, teta, first_mf, second_mf):
    def calc_c_second(new_teta, c):
        k = 0
        for i in universe:
            if i >= c:
                k -= 1
                break
            k += 1

        for i, l_val, r_val in zip(range(0, len(first_mf)), first_mf, second_mf):
            if i <= k:
                new_teta[i] = l_val
            else:
                new_teta[i] = r_val

        return np.average(universe, weights=new_teta), new_teta, k

    cprim = np.average(universe, weights=teta)
    csecond, teta, k = calc_c_second(teta, cprim)

    i = 0
    while abs(csecond - cprim) >= np.finfo(float).eps:
        cprim = csecond
        csecond, teta, k = calc_c_second(teta, cprim)
        i += 1
        if i > 1000:
            raise Exception("Probably endless loop in defuzz")

    return csecond, k


def __calc_y_val(universe, breakpoint, first_mf, second_mf):
    def sum_of_series(array, mf, start, end):
        acc = 0
        for i in range(start, end):
            acc += array[i] * mf[i]
        return acc

    return ((sum_of_series(universe, first_mf, 0, breakpoint + 1)
             + sum_of_series(universe, second_mf, breakpoint + 1, len(universe)))
            / (sum_of_series(np.ones_like(universe), first_mf, 0, breakpoint + 1)
               + sum_of_series(np.ones_like(universe), second_mf, breakpoint + 1, len(universe))))


class TestMamdaniInferenceSystem:
    # def test_karnik_mendel(self):
    #     precision = 0.01
    #     start = 0
    #     stop = 1
    #     universe = np.arange(start, stop, precision)
    #     lmf_func = triangular(0, 0.5, 1, max_value=0.8)
    #     lmf = np.array([lmf_func(x) for x in universe])
    #     umf_func = triangular(0, 0.5, 1)
    #     umf = np.array([umf_func(x) for x in universe])
    #     mamdani = MamdaniInferenceSystem([])
    #     output = mamdani.__karnik_mendel(lmf, umf, universe)
    #     skfuzzy = defuzz(universe, (lmf, umf))
    #     assert 0.5 == output
    @pytest.fixture
    def patch_rule_output(self, mocker):
        lower_func = triangular(0, 0.5, 1, 0.8)
        upper_func = triangular(0, 0.5, 1)
        universe = np.arange(0, 1, 0.01)
        lmf = np.array([lower_func(x) for x in universe])
        umf = np.array([upper_func(x) for x in universe])
        rule_output = (lmf, umf)
        mocker.patch('mamdani_inference_system.MamdaniInferenceSystem.__get_rule_outputs').return_value = rule_output

    @pytest.fixture
    def test_karnik_mendel_output(self, mocker):
        mocker
        NotImplemented()


