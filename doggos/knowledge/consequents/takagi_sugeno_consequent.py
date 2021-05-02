from doggos.knowledge.consequents.consequent import Consequent
from doggos.fuzzy_sets.fuzzy_set import MembershipDegree


class TakagiSugenoConsequent(Consequent):
    def output(self, rule_firing: MembershipDegree) -> float:
        pass
