from typing import Tuple, List

from doggos.knowledge.consequents.consequent import Consequent
from doggos.fuzzy_sets.MembershipDegree.membership_degree import MembershipDegree


class MamdaniConsequent(Consequent):
    def output(self, rule_firing: MembershipDegree) -> Tuple[List[float]] or List[float]:
        pass
