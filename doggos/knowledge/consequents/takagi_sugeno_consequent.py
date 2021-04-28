from doggos.knowledge.consequents.consequent import Consequent
from doggos.fuzzy_sets.membership.membership_degree import MembershipDegree


class TakagiSugenoConsequent(Consequent):
    def output(self, rule_firing: MembershipDegree) -> float:
        pass
