from typing import Dict, Tuple, List

import numpy as np

from doggos.fuzzy_sets.fuzzy_set import MembershipDegree
from doggos.knowledge.antecedent import Antecedent
from doggos.knowledge.consequents.consequent import Consequent


class Rule:
    def __init__(self, antecedent: Antecedent, consequent: Consequent):
        self.__firing_value: MembershipDegree = None
        self.__rule_output: np.ndarray or float = None
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
