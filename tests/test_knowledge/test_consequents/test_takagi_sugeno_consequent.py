from doggos.knowledge import Domain, LinguisticVariable
from doggos.knowledge.consequents import TakagiSugenoConsequent
import pytest


class TestTakagiSugenoConsequent:

    def test_output_1_input(self):
        domain = Domain(0, 10, 0.01)
        ling_var_f1 = LinguisticVariable('F1', domain)
        output_ling_var = LinguisticVariable('output', domain)

        ts1 = TakagiSugenoConsequent({'F1': 0.1, 'const': -2.5}, output_ling_var)
        assert (ts1.output({ling_var_f1: 1}) == (output_ling_var, -2.4))
        assert (ts1.output({ling_var_f1: 0}) == (output_ling_var, -2.5))

    def test_output_2_inputs(self):
        domain = Domain(0, 10, 0.01)
        ling_var_f1 = LinguisticVariable('F1', domain)
        ling_var_f2 = LinguisticVariable('F2', domain)
        output_ling_var = LinguisticVariable('output', domain)

        ts2 = TakagiSugenoConsequent({'F1': 2, 'F2': 10, 'const': 1}, output_ling_var)
        assert (ts2.output({ling_var_f1: 1, ling_var_f2: 1}) == (output_ling_var, 13))
        assert (ts2.output({ling_var_f1: 0, ling_var_f2: 0}) == (output_ling_var, 1))

    def test_output_5_inputs(self):
        domain = Domain(0, 10, 0.01)
        ling_var_f1 = LinguisticVariable('F1', domain)
        ling_var_f2 = LinguisticVariable('F2', domain)
        ling_var_f3 = LinguisticVariable('F3', domain)
        ling_var_f4 = LinguisticVariable('F4', domain)
        ling_var_f5 = LinguisticVariable('F5', domain)
        output_ling_var = LinguisticVariable('output', domain)

        ts5 = TakagiSugenoConsequent({'F1': 1.5, 'F2': 6.123, 'F3': 2.5, 'F4': -2, 'F5': 0.15, 'const': 0.99},
                                     output_ling_var)
        assert (ts5.output({ling_var_f1: 1, ling_var_f2: 0.1, ling_var_f3: 2, ling_var_f4: 0.2, ling_var_f5: 3}) ==
                (output_ling_var, 8.1523))  # (1.5 * 1) + (6.123 * 0.1) + (2.5 * 2) + (-2 * 0.2) + (0.15 * 3) + 0.99
        assert (ts5.output({ling_var_f1: 0, ling_var_f2: 0, ling_var_f3: 0, ling_var_f4: 0, ling_var_f5: 1}) ==
                (output_ling_var, 1.14))

    def test_wrong_inputs(self):
        domain = Domain(0, 10, 0.01)
        ling_var_f1 = LinguisticVariable('F1', domain)
        ling_var_f2 = LinguisticVariable('F2', domain)
        output_ling_var = LinguisticVariable('output', domain)

        ts = TakagiSugenoConsequent({'F1': 1, 'F2': 2, 'F3': 3, 'const': 4}, output_ling_var)
        with pytest.raises(Exception) as e:
            ts.output({ling_var_f1: 1, ling_var_f2: 2})
            assert "Function parameters contain value for input which was not provided!" in str(e.value)

    def test_setter(self):
        domain = Domain(0, 10, 0.01)
        ling_var_f1 = LinguisticVariable('F1', domain)
        ling_var_f2 = LinguisticVariable('F2', domain)
        output_ling_var = LinguisticVariable('output', domain)

        tss = TakagiSugenoConsequent({'F1': 1, 'F2': 2, 'const': 3}, output_ling_var)
        assert (tss.output({ling_var_f1: 0.5, ling_var_f2: 0.2}) == (output_ling_var, 3.9))
        tss.function_parameters = {'F1': 0.1, 'F2': 0.1, 'const': 1}
        assert (tss.output({ling_var_f1: 0.99, ling_var_f2: 0.88}) == (output_ling_var, 1.187))

    def test_setter_error(self):
        domain = Domain(0, 10, 0.01)
        ling_var_f1 = LinguisticVariable('F1', domain)
        ling_var_f2 = LinguisticVariable('F2', domain)
        output_ling_var = LinguisticVariable('output', domain)

        tse = TakagiSugenoConsequent({'F1': 0.2, 'F2': -1, 'const': 0.5}, output_ling_var)
        assert (tse.output({ling_var_f1: 0.5, ling_var_f2: 0.2}) == (output_ling_var, 0.4))
        with pytest.raises(ValueError) as e:
            tse.function_parameters = {'F1': 1.0, 'F2': -5, 'const': 'a'}
            assert "Takagi-Sugeno consequent parameters must be Dict[str, float]!" in str(e.value)

    def test_getter(self):
        domain = Domain(0, 10, 0.01)
        output_ling_var = LinguisticVariable('output', domain)

        tsg = TakagiSugenoConsequent({'F1': 1, 'F2': 2, 'const': 3}, output_ling_var)
        assert (tsg.function_parameters == {'F1': 1, 'F2': 2, 'const': 3})
        tsg.function_parameters = {'F1': 0.1, 'F2': 0.1, 'const': 1}
        assert (tsg.function_parameters == {'F1': 0.1, 'F2': 0.1, 'const': 1})
