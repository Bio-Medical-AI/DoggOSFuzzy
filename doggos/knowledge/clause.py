from doggos.fuzzy_sets.degree.membership_degree import MembershipDegree
from doggos.fuzzy_sets.fuzzy_set import FuzzySet
from doggos.knowledge.linguistic_variable import LinguisticVariable


class Clause:

    __lingustic_variable: LinguisticVariable
    __gradiation_adjectice: str
    __fuzzy_set: FuzzySet

    def get_value(self, x: float) -> MembershipDegree:
        pass
