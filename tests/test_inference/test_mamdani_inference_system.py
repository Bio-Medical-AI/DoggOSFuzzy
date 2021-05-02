import pytest
import numpy as np


from doggos.utils.membership_functions.membership_functions import triangular
from doggos.inference.mamdani_inference_system import MamdaniInferenceSystem


class TestMamdaniInferenceSystem:
    @pytest.fixture
    def patch_domain_and_memberships(self, mocker):
        lower_func = triangular(0, 0.5, 1, 0.8)
        upper_func = triangular(0, 0.5, 1)
        domain = np.arange(0, 1, 0.01)
        lmf = np.array([lower_func(x) for x in domain])
        umf = np.array([upper_func(x) for x in domain])
        domain_lmf_umf = (domain, [lmf], [umf])
        get_domain_and_memberships_path = 'doggos.inference.mamdani_inference_system.MamdaniInferenceSystem' \
                                          '._MamdaniInferenceSystem__get_domain_and_memberships_for_it2'
        mocker.patch(get_domain_and_memberships_path).return_value = domain_lmf_umf

    def test_karnik_mendel_output(self, patch_domain_and_memberships):
        mamdani = MamdaniInferenceSystem([])
        output = mamdani.output({})
        assert 0.5 == output
