import pytest
import pandas as pd


from tests.test_tools import approx
from doggos.knowledge import Domain, LinguisticVariable, Clause, fuzzify
from doggos.fuzzy_sets import Type1FuzzySet, IntervalType2FuzzySet


class TestFuzzifier:

    @pytest.fixture
    def df(self):
        df = pd.DataFrame({
            'water': [-2, 3, 4, 1],
            'earth': [0.5, 2, 0.1, 2],
        })
        return df

    @pytest.fixture
    def proper_lw(self):
        lw1 = LinguisticVariable('water', Domain(-6, 6, 0.01))
        lw2 = LinguisticVariable('earth', Domain(-6, 6, 0.01))
        return lw1, lw2

    @pytest.fixture
    def t1_sets(self):
        f1 = Type1FuzzySet(lambda x: 0.1)
        f2 = Type1FuzzySet(lambda x: 0.2)
        f3 = Type1FuzzySet(lambda x: 0.3)
        f4 = Type1FuzzySet(lambda x: 0.4)
        return f1, f2, f3, f4

    @pytest.fixture
    def t2_sets(self):
        f1 = IntervalType2FuzzySet(lambda x: 0.11, lambda y: 0.12)
        f2 = IntervalType2FuzzySet(lambda x: 0.21, lambda y: 0.22)
        f3 = IntervalType2FuzzySet(lambda x: 0.31, lambda y: 0.32)
        f4 = IntervalType2FuzzySet(lambda x: 0.41, lambda y: 0.42)
        return f1, f2, f3, f4

    def test_fuzzify_type1_fuzzy_sets(self, df, proper_lw, t1_sets):
        lw1, lw2 = proper_lw
        f1, f2, f3, f4 = t1_sets
        clauses = list()
        clauses.append(Clause(lw1, 'high', f1))
        clauses.append(Clause(lw1, 'low', f2))
        clauses.append(Clause(lw2, 'small', f3))
        clauses.append(Clause(lw2, 'medium', f4))
        result = fuzzify(df, clauses)
        assert all(result[res] == approx(exp) for res, exp in zip(
            clauses,
            [0.1, 0.2, 0.3, 0.4]
        ))

    # def test_fuzzify_interval_type2_fuzzy_sets(self, df, proper_lw, t2_sets):
    #     lw1, lw2 = proper_lw
    #     f1, f2, f3, f4 = t2_sets
    #     clauses = list()
    #     clauses.append(Clause(lw1, 'high', f1))
    #     clauses.append(Clause(lw1, 'low', f2))
    #     clauses.append(Clause(lw2, 'small', f3))
    #     clauses.append(Clause(lw2, 'medium', f4))
    #     result = fuzzify(df, clauses)
    #     assert all(result[res] == approx(exp) for res, exp in zip(
    #         clauses,
    #         [(0.11, 0.12), (0.21, 0.22), (0.31, 32), (0.41, 0.42)]
    #     ))

    def test_fuzzify_key_error(self, df, t1_sets):
        lw1 = LinguisticVariable('water', Domain(-5,  5, 0.01))
        lw2 = LinguisticVariable('fire', Domain(-5, 5, 0.01))
        f1, f2, _, _ = t1_sets
        clauses = list()
        clauses.append(Clause(lw1, 'high', f1))
        clauses.append(Clause(lw2, 'low', f2))
        with pytest.raises(KeyError):
            _ = fuzzify(df, clauses)

    def test_fuzzify_empty_clauses(self, df):
        result = fuzzify(df, [])
        assert result == dict()
