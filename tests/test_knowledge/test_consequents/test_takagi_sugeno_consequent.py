from doggos.knowledge.consequents import TakagiSugenoConsequent
import pytest


class TestTakagiSugenoConsequent:

    def test_output_1_input(self):
        ts1 = TakagiSugenoConsequent([0.1, -2.5])
        assert (ts1.output([1]) == -2.4)
        assert (ts1.output([0]) == -2.5)

    def test_output_2_inputs(self):
        ts2 = TakagiSugenoConsequent([2, 10, 1])
        assert (ts2.output([1, 1]) == 13)
        assert (ts2.output([0, 0]) == 1)

    def test_output_5_inputs(self):
        ts5 = TakagiSugenoConsequent([1.5, 6.123, 2.5, -2, 0.15, 0.99])
        assert (ts5.output([1, 0.1, 2, 0.2, 3]) == 8.1523)  #(1.5 * 1) + (6.123 * 0.1) + (2.5 * 2) + (-2 * 0.2) + (0.15 * 3) + 0.99
        assert (ts5.output([0, 0, 0, 0, 1]) == 1.14)

    def test_wrong_inputs(self):
        ts = TakagiSugenoConsequent([1, 2, 3, 4])
        with pytest.raises(Exception) as e:
            ts.output([1, 2])
            assert "Number of inputs must be one less than number of consequent parameters!" in str(e.value)

        with pytest.raises(Exception) as e:
            ts.output([1, 2, 3, 4])
            assert "Number of inputs must be one less than number of consequent parameters!" in str(e.value)

    def test_setter(self):
        tss = TakagiSugenoConsequent([1, 2, 3])
        assert (tss.output([0.5, 0.2]) == 3.9)
        tss.function_parameters = [0.1, 0.1, 1]
        assert (tss.output([0.99, 0.88]) == 1.187)

    def test_setter_error(self):
        tse = TakagiSugenoConsequent([0.2, -1, 0.5])
        assert (tse.output([0.5, 0.2]) == 0.4)
        with pytest.raises(ValueError) as e:
            tse.function_parameters = [1.0, -5, 'a']
            assert "Takagi-Sugeno consequent parameters must be list of floats!" in str(e.value)

    def test_getter(self):
        tsg = TakagiSugenoConsequent([1, 2, 3])
        assert (tsg.function_parameters == [1, 2, 3])
        tsg.function_parameters = [0.1, 0.1, 1]
        assert (tsg.function_parameters == [0.1, 0.1, 1])
