import pytest
import numpy as np

from tests.test_tools import approx
from doggos.utils.membership_functions.membership_functions import triangular, trapezoidal
from doggos.inference.mamdani_inference_system import MamdaniInferenceSystem
from doggos.inference.defuzzification_algorithms import *


class TestMamdaniInferenceSystem:
    @pytest.fixture
    def patch_domain_and_memberships_for_it2(self, mocker):
        lower_func = triangular(0, 0.5, 1, 0.8)
        upper_func = triangular(0, 0.5, 1)
        domain = np.arange(0, 1, 0.001)
        lmf = np.array([lower_func(x) for x in domain])
        umf = np.array([upper_func(x) for x in domain])
        domain_lmf_umf = (domain, lmf, umf)
        get_domain_and_memberships_path = 'doggos.inference.mamdani_inference_system.MamdaniInferenceSystem' \
                                          '._MamdaniInferenceSystem__get_domain_and_consequents_memberships_for_it2'
        mocker.patch(get_domain_and_memberships_path).return_value = domain_lmf_umf

    @pytest.fixture
    def patch_get_domain_and_membership_functions(self, mocker):
        membership_function = triangular(0, 0.5, 1)
        domain = np.arange(0, 1, 0.001)
        mf = np.array([membership_function(x) for x in domain])
        domain_mfs = (domain, mf)
        get_domain_and_membership_functions_path = 'doggos.inference.mamdani_inference_system.' \
                                                   'MamdaniInferenceSystem._MamdaniInferenceSystem' \
                                                   '__get_domain_and_consequents_membership_functions'
        mocker.patch(get_domain_and_membership_functions_path).return_value = domain_mfs

    @pytest.fixture
    def patch_domain_and_membership_functions_for_maxima(self, mocker):
        cut = trapezoidal(0, 0.25, 0.75, 1)
        domain = np.arange(0, 1, 0.001)
        cut_values = np.array([cut(x) for x in domain])
        domain_cut = (domain, cut_values)
        get_domain_and_cut_path = 'doggos.inference.mamdani_inference_system.MamdaniInferenceSystem' \
                                  '._MamdaniInferenceSystem__get_domain_and_consequents_membership_functions'
        mocker.patch(get_domain_and_cut_path).return_value = domain_cut

    @pytest.fixture
    def patch_get_degrees(self, mocker):
        get_degrees_path = 'doggos.inference.mamdani_inference_system.MamdaniInferenceSystem' \
                           '._MamdaniInferenceSystem__get_degrees'
        mocker.patch(get_degrees_path).return_value = [1]

    @pytest.fixture
    def patch_consequent_type1_true(self, mocker):
        is_consequent_type1_path = 'doggos.inference.mamdani_inference_system.MamdaniInferenceSystem' \
                                   '._MamdaniInferenceSystem__is_consequent_type1'
        mocker.patch(is_consequent_type1_path).return_value = True

    @pytest.fixture
    def patch_consequent_type1_false(self, mocker):
        is_consequent_type1_path = 'doggos.inference.mamdani_inference_system.MamdaniInferenceSystem' \
                                   '._MamdaniInferenceSystem__is_consequent_type1'
        mocker.patch(is_consequent_type1_path).return_value = False

    def test_karnik_mendel_output(self, patch_get_degrees, patch_consequent_type1_false,
                                  patch_domain_and_memberships_for_it2):
        mamdani = MamdaniInferenceSystem([])
        output = mamdani.infer(karnik_mendel, {})
        assert output == approx(0.5)

    @pytest.mark.parametrize('method', [center_of_gravity, largest_of_maximum, middle_of_maximum,
                                        smallest_of_maximum, mean_of_maxima, center_of_sums])
    def test_type1_defuzzification_methods(self, patch_get_degrees, patch_consequent_type1_true,
                                           patch_get_domain_and_membership_functions, method):
        mamdani = MamdaniInferenceSystem([])
        output = mamdani.infer(method, {})
        assert output == approx(0.5)

    @pytest.mark.parametrize('method, expected', zip([largest_of_maximum, middle_of_maximum, smallest_of_maximum,
                                                      mean_of_maxima], [0.75, 0.5, 0.25, 0.5]))
    def test_maxima(self, patch_get_degrees, patch_consequent_type1_true,
                    patch_domain_and_membership_functions_for_maxima, method, expected):
        mamdani = MamdaniInferenceSystem([])
        output = mamdani.infer(method, {})
        assert output == approx(expected)
