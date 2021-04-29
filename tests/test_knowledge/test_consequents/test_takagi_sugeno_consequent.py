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
        with pytest.raises(Exception):
            ts.output([1, 2])

        with pytest.raises(Exception):
            ts.output([1, 2, 3, 4])
