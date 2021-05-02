from doggos.knowledge.consequents.consequent import Consequent
from doggos.fuzzy_sets import MembershipDegree


class TakagiSugenoConsequent(Consequent):
    def output(self, rule_firing: MembershipDegree) -> float:
        pass
