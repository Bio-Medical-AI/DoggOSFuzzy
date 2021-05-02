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
        lv_f1 = LinguisticVariable('F1', domain)
        lv_f2 = LinguisticVariable('F2', domain)
        lv_f3 = LinguisticVariable('F3', domain)
        output_lv = LinguisticVariable('output', domain)

        ts = TakagiSugenoConsequent({lv_f1: 1, lv_f2: 2, lv_f3: 3}, 4, output_lv)
        with pytest.raises(ValueError) as e:
            ts.output({lv_f1: 1, lv_f2: 2})
            assert "Function parameters contain value for input which was not provided!" in str(e.value)

    def test_setter(self):
        domain = Domain(0, 10, 0.01)
        ling_var_f1 = LinguisticVariable('F1', domain)
        ling_var_f2 = LinguisticVariable('F2', domain)

        tss = TakagiSugenoConsequent({'F1': 1, 'F2': 2, 'const': 3})
        assert (tss.output({ling_var_f1: 0.5, ling_var_f2: 0.2}) == 3.9)
        tss.function_parameters = {'F1': 0.1, 'F2': 0.1, 'const': 1}
        assert (tss.output({ling_var_f1: 0.99, ling_var_f2: 0.88}) == 1.187)

    def test_setter_error(self):
        domain = Domain(0, 10, 0.01)
        ling_var_f1 = LinguisticVariable('F1', domain)
        ling_var_f2 = LinguisticVariable('F2', domain)

        tse = TakagiSugenoConsequent({'F1': 0.2, 'F2': -1, 'const': 0.5})
        assert (tse.output({ling_var_f1: 0.5, ling_var_f2: 0.2}) == 0.4)
        with pytest.raises(ValueError) as e:
            tse.function_parameters = {'F1': 1.0, 'F2': -5, 'const': 'a'}
            assert "Takagi-Sugeno consequent parameters must be Dict[str, float]!" in str(e.value)

    def test_getter(self):
        tsg = TakagiSugenoConsequent({'F1': 1, 'F2': 2, 'const': 3})
        assert (tsg.function_parameters == {'F1': 1, 'F2': 2, 'const': 3})
        tsg.function_parameters = {'F1': 0.1, 'F2': 0.1, 'const': 1}
        assert (tsg.function_parameters == {'F1': 0.1, 'F2': 0.1, 'const': 1})
