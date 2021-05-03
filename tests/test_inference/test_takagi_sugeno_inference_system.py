import pytest
import numpy as np

from doggos.inference.takagi_sugeno_inference_system import calculate_membership


class TestTakagiSugenoInferenceSystem:
    def test_calculate_membership(self):
        lower_outputs_of_rules = np.array([[3, 1], [6, 3], [9, 5], [13, 3], [19, 1], [21, 4]])
        upper_outputs_of_rules = np.array([[3, 3], [6, 7], [9, 7], [13, 6], [19, 9], [21, 6]])
        domain = np.arange(1, 23, 1)
        lmf = np.zeros(shape=domain.shape)
        umf = np.zeros(shape=domain.shape)
        expected_lmf = [0., 0., 1., 5/3, 7/3, 3., 11/3, 13/3, 5., 4.5, 4., 3.5, 3.,
                        8/3, 7/3, 2., 5/3, 4/3, 1., 2.5, 4., 0.]
        expected_umf = [0., 0., 3., 13/3, 17/3, 7., 7., 7., 7., 6.75, 6.5, 6.25, 6., 6.5, 7., 7.5, 8., 8.5,
                        9., 7.5, 6., 0.]

        for i in range(domain.shape[0]):
            lmf[i] = calculate_membership(domain[i], lower_outputs_of_rules)
            umf[i] = calculate_membership(domain[i], upper_outputs_of_rules)
        assert all(a == pytest.approx(b, 0.1) for a, b in zip(lmf.tolist(), expected_lmf))
        assert all(a == pytest.approx(b, 0.1) for a, b in zip(umf.tolist(), expected_umf))
