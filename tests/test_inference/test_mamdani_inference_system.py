import pytest
import numpy as np


from tests.test_tools import approx
from doggos.utils.membership_functions.membership_functions import triangular, trapezoidal
from doggos.inference.mamdani_inference_system import MamdaniInferenceSystem


class TestMamdaniInferenceSystem:
    @pytest.fixture
    def patch_domain_and_memberships_for_center_of_sums(self, mocker):
        first_function = trapezoidal(0, 1, 4, 5, max_value=0.3)
        second_function = trapezoidal(3, 4, 6, 7, max_value=0.5)
        third_function = trapezoidal(5, 6, 7, 8, max_value=1)
        domain = np.arange(0, 8, 0.1)
        first_mf = np.array([first_function(x) for x in domain])
        second_mf = np.array([second_function(x) for x in domain])
        third_mf = np.array([third_function(x) for x in domain])
        domain_mfs = (domain, (first_mf, second_mf, third_mf))
        get_domain_and_memberships_path = 'doggos.inference.mamdani_inference_system.MamdaniInferenceSystem' \
                                          '._MamdaniInferenceSystem__get_domain_and_memberships_for_type1'
        mocker.patch(get_domain_and_memberships_path).return_value = domain_mfs

    @pytest.fixture
    def patch_domain_and_memberships_for_it2(self, mocker):
        lower_func = triangular(0, 0.5, 1, 0.8)
        upper_func = triangular(0, 0.5, 1)
        domain = np.arange(0, 1, 0.1)
        lmf = np.array([lower_func(x) for x in domain])
        umf = np.array([upper_func(x) for x in domain])
        domain_lmf_umf = (domain, [lmf], [umf])
        get_domain_and_memberships_path = 'doggos.inference.mamdani_inference_system.MamdaniInferenceSystem' \
                                          '._MamdaniInferenceSystem__get_domain_and_memberships_for_it2'
        mocker.patch(get_domain_and_memberships_path).return_value = domain_lmf_umf

    @pytest.fixture
    def patch_domain_and_cut(self, mocker):
        cut = triangular(0, 0.5, 1)
        domain = np.arange(0, 1, 0.001)
        cut_values = np.array([cut(x) for x in domain])
        domain_cut = (domain, cut_values)
        get_domain_and_memberships_path = 'doggos.inference.mamdani_inference_system.MamdaniInferenceSystem' \
                                          '._MamdaniInferenceSystem__get_domain_and_cut'
        mocker.patch(get_domain_and_memberships_path).return_value = domain_cut

    @pytest.fixture
    def patch_domain_and_cut_for_maxima(self, mocker):
        cut = trapezoidal(0, 0.25, 0.75, 1)
        domain = np.arange(0, 1, 0.001)
        cut_values = np.array([cut(x) for x in domain])
        domain_cut = (domain, cut_values)
        get_domain_and_memberships_path = 'doggos.inference.mamdani_inference_system.MamdaniInferenceSystem' \
                                          '._MamdaniInferenceSystem__get_domain_and_cut'
        mocker.patch(get_domain_and_memberships_path).return_value = domain_cut

    def test_karnik_mendel_output(self, patch_domain_and_memberships_for_it2):
        mamdani = MamdaniInferenceSystem([])
        output = mamdani.output({}, 'KM')
        assert output == approx(0.5)

    def test_center_of_sums(self, patch_domain_and_memberships_for_center_of_sums):
        mamdani = MamdaniInferenceSystem([])
        output = mamdani.output({}, 'COS')
        assert output == approx(5)

    @pytest.mark.parametrize('method', ['COG', 'LOM', 'MOM', 'SOM', 'MeOM'])
    def test_type1_defuzzification_methods(self, patch_domain_and_cut, method):
        mamdani = MamdaniInferenceSystem([])
        output = mamdani.output({}, method)
        assert output == approx(0.5)

    @pytest.mark.parametrize('method, expected', zip(['LOM', 'MOM', 'SOM', 'MeOM'], [0.75, 0.5, 0.25, 0.5]))
    def test_maxima(self, patch_domain_and_cut_for_maxima, method, expected):
        mamdani = MamdaniInferenceSystem([])
        output = mamdani.output({}, method)
        assert output == approx(expected)

