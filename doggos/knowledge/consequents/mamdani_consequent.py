
from typing import List

from doggos.knowledge.consequents.consequent import Consequent
from doggos.fuzzy_sets.membership.membership_degree import MembershipDegree


class MamdaniConsequent(Consequent):
    def output(self, rule_firing: MembershipDegree) -> List[MembershipDegree]:
        pass
