from typing import Dict, Tuple, List

from doggos.fuzzy_sets.membership.membership_degree import MembershipDegree
from doggos.knowledge import Antecedent
from doggos.knowledge.consequents.consequent import Consequent


class Rule:
    def __init__(self, antecedent: Antecedent, consequent: Consequent):
        self.__firing_value: MembershipDegree = None
        self.__rule_output: List[MembershipDegree] or float = None
        self.__antecedent = antecedent
        self.__consequent = consequent

    @property
    def antecedent(self) -> Antecedent:
        pass

    @property
    def consequent(self) -> Consequent:
        pass

    def output(self, features: Dict[str, float]) -> List[MembershipDegree] or float:
        pass
