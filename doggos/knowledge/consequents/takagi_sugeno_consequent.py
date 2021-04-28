from typing import Tuple, List

from doggos.knowledge.consequents.consequent import Consequent
from doggos.fuzzy_sets.degree.membership_degree import MembershipDegree


class TakagiSugenoConsequent(Consequent):
    def output(self, rule_firing: MembershipDegree) -> Tuple[List[float]] or List[float]:
        pass
