from typing import Dict, Tuple
from doggos.knowledge import Antecedent
from doggos.knowledge.consequents.consequent import Consequent


class Rule:
    def __init__(self, antecedent: Antecedent, consequent: Consequent):
        self.__firing_value: Tuple[float, ...] = None
        self.__rule_output: float = None
        self.__antecedent = antecedent
        self.__consequent = consequent

    @property
    def antecedent(self) -> Antecedent:
        pass

    @property
    def consequent(self) -> Consequent:
        pass

    def calculate_rule(self, features: Dict[str, float]) -> Tuple[float, ...] or float:
        pass
