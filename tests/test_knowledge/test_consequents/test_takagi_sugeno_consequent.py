from doggos.knowledge import Domain, LinguisticVariable
from doggos.knowledge.consequents import TakagiSugenoConsequent
import pytest


class TestTakagiSugenoConsequent:

    def test_output_1_input(self):
        domain = Domain(0, 10, 0.01)
        ling_var_f1 = LinguisticVariable('F1', domain)
        output_lv = LinguisticVariable('output', domain)

        ts1 = TakagiSugenoConsequent({ling_var_f1: 0.1}, -2.5, output_lv)
        assert (ts1.output({ling_var_f1: 1}) == -2.4)
        assert (ts1.output({ling_var_f1: 0}) == -2.5)

    def test_output_2_inputs(self):
        domain = Domain(0, 10, 0.01)
        ling_var_f1 = LinguisticVariable('F1', domain)
        ling_var_f2 = LinguisticVariable('F2', domain)
        output_lv = LinguisticVariable('output', domain)

        ts2 = TakagiSugenoConsequent({ling_var_f1: 2, ling_var_f2: 10}, 1, output_lv)
        assert (ts2.output({ling_var_f1: 1, ling_var_f2: 1}) == 13)
        assert (ts2.output({ling_var_f1: 0, ling_var_f2: 0}) == 1)

    def test_output_5_inputs(self):
        domain = Domain(0, 10, 0.01)
        lv_f1 = LinguisticVariable('F1', domain)
        lv_f2 = LinguisticVariable('F2', domain)
        lv_f3 = LinguisticVariable('F3', domain)
        lv_f4 = LinguisticVariable('F4', domain)
        lv_f5 = LinguisticVariable('F5', domain)
        output_lv = LinguisticVariable('output', domain)

        ts5 = TakagiSugenoConsequent({lv_f1: 1.5, lv_f2: 6.123, lv_f3: 2.5, lv_f4: -2, lv_f5: 0.15}, 1, output_lv)
        assert (ts5.output({lv_f1: 1, lv_f2: 0.1, lv_f3: 2, lv_f4: 0.2, lv_f5: 3}) ==
                8.1623)  # (1.5 * 1) + (6.123 * 0.1) + (2.5 * 2) + (-2 * 0.2) + (0.15 * 3) + 1
        assert (ts5.output({lv_f1: 0, lv_f2: 0, lv_f3: 0, lv_f4: 0, lv_f5: 1}) == 1.15)

    def test_wrong_inputs(self):
        domain = Domain(0, 10, 0.01)
        lv_f2 = LinguisticVariable('F2', domain)
        lv_f3 = LinguisticVariable('F3', domain)
        output_lv = LinguisticVariable('output', domain)

        ts = TakagiSugenoConsequent({lv_f3: 1}, 4, output_lv)
        with pytest.raises(KeyError):
            ts.output({lv_f2: 2})

    def test_setter(self):
        domain = Domain(0, 10, 0.01)
        ling_var_f1 = LinguisticVariable('F1', domain)
        ling_var_f2 = LinguisticVariable('F2', domain)
        output_lv = LinguisticVariable('output', domain)

        tss = TakagiSugenoConsequent({ling_var_f1: 1, ling_var_f2: 2}, 3, output_lv)
        assert (tss.output({ling_var_f1: 0.5, ling_var_f2: 0.2}) == 3.9)
        tss.function_parameters = {ling_var_f1: 0.1, ling_var_f2: 0.1}
        assert (tss.output({ling_var_f1: 0.99, ling_var_f2: 0.88}) == 3.1870000000000003)

    def test_setter_error(self):
        domain = Domain(0, 10, 0.01)
        ling_var_f1 = LinguisticVariable('F1', domain)
        ling_var_f2 = LinguisticVariable('F2', domain)
        output_lv = LinguisticVariable('output', domain)

        tse = TakagiSugenoConsequent({ling_var_f1: 0.2, ling_var_f2: -1}, 1, output_lv)
        assert (tse.output({ling_var_f1: 1, ling_var_f2: 2}) == -0.8)
        with pytest.raises(ValueError) as e:
            tse.function_parameters = {ling_var_f1: 1.0, ling_var_f2: 'a'}
            assert "Takagi-Sugeno consequent parameters must be Dict[LinguisticVariable, float]!" in str(e.value)

    def test_getter(self):
        domain = Domain(0, 10, 0.01)
        lv_f1 = LinguisticVariable('F1', domain)
        lv_f2 = LinguisticVariable('F2', domain)
        output_lv = LinguisticVariable('output', domain)

        tsg = TakagiSugenoConsequent({lv_f1: 1, lv_f2: 2}, 3, output_lv)
        assert (tsg.function_parameters == {lv_f1: 1, lv_f2: 2})
        assert (tsg.bias == 3)
        tsg.function_parameters = {lv_f1: 3, lv_f2: 5}
        assert (tsg.function_parameters == {lv_f1: 3, lv_f2: 5})
