from doggos.knowledge.consequents import TakagiSugenoConsequent
import pytest


class TestTakagiSugenoConsequent:

    def test_output_1_input(self):
        ts1 = TakagiSugenoConsequent({'F1': 0.1, 'const': -2.5})
        assert (ts1.output({'F1': 1}) == -2.4)
        assert (ts1.output({'F1': 0}) == -2.5)

    def test_output_2_inputs(self):
        ts2 = TakagiSugenoConsequent({'F1': 2, 'F2': 10, 'const': 1})
        assert (ts2.output({'F1': 1, 'F2': 1}) == 13)
        assert (ts2.output({'F1': 0, 'F2': 0}) == 1)

    def test_output_5_inputs(self):
        ts5 = TakagiSugenoConsequent({'F1': 1.5, 'F2': 6.123, 'F3': 2.5, 'F4': -2, 'F5':  0.15, 'const': 0.99})
        assert (ts5.output({'F1': 1, 'F2': 0.1, 'F3': 2, 'F4': 0.2, 'F5': 3}) == 8.1523)  #(1.5 * 1) + (6.123 * 0.1) + (2.5 * 2) + (-2 * 0.2) + (0.15 * 3) + 0.99
        assert (ts5.output({'F1': 0, 'F2': 0, 'F3': 0, 'F4': 0, 'F5': 1}) == 1.14)

    def test_wrong_inputs(self):
        ts = TakagiSugenoConsequent({'F1': 1, 'F2': 2, 'F3': 3, 'const': 4})
        with pytest.raises(Exception) as e:
            ts.output({'F1': 1, 'F2': 2})
            assert "Number of inputs must be one less than number of consequent parameters!" in str(e.value)

        with pytest.raises(Exception) as e:
            ts.output({'F1': 1, 'F2': 2, 'F3': 3, 'F4': 4})
            assert "Number of inputs must be one less than number of consequent parameters!" in str(e.value)

    def test_setter(self):
        tss = TakagiSugenoConsequent({'F1': 1, 'F2': 2, 'const': 3})
        assert (tss.output({'F1': 0.5, 'F2': 0.2}) == 3.9)
        tss.function_parameters = {'F1': 0.1, 'F2': 0.1, 'const': 1}
        assert (tss.output({'F1': 0.99, 'F2': 0.88}) == 1.187)

    def test_setter_error(self):
        tse = TakagiSugenoConsequent({'F1': 0.2, 'F2': -1, 'const': 0.5})
        assert (tse.output({'F1': 0.5, 'F2': 0.2}) == 0.4)
        with pytest.raises(ValueError) as e:
            tse.function_parameters = {'F1': 1.0, 'F2': -5, 'const': 'a'}
            assert "Takagi-Sugeno consequent parameters must be Dict[str, float]!" in str(e.value)

    def test_getter(self):
        tsg = TakagiSugenoConsequent({'F1': 1, 'F2': 2, 'const': 3})
        assert (tsg.function_parameters == {'F1': 1, 'F2': 2, 'const': 3})
        tsg.function_parameters = {'F1': 0.1, 'F2': 0.1, 'const': 1}
        assert (tsg.function_parameters == {'F1': 0.1, 'F2': 0.1, 'const': 1})
