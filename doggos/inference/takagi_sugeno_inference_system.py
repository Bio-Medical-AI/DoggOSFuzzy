from typing import Dict, List

from doggos.inference.inference_system import InferenceSystem
from doggos.knowledge import Rule
from doggos.knowledge.consequents import TakagiSugenoConsequent


class TakagiSugenoInferenceSystem(InferenceSystem):
    def __init__(self, rules: List[Rule]):
        self.__type = "1"
        if rules[0].antecedent
        for rule in rules:
            if isinstance(rule.consequent, TakagiSugenoConsequent):
                self._rule_base.append(rule)

    def output(self, features: Dict[str, float], method: str) -> float:
        if len(self._rule_base) == 0:
            raise IndexError("Rule base of inference system is empty")
        return self.__type_1_output(features, method) if self.__type == "1" else self.__type_2_output(features, method)

    def __type_1_output(self, features: Dict[str, float], method: str) -> float:
        nominator = 0
        denominator = 0
        for rule in self._rule_base:
            out, memb = rule.output(features)
            nominator += out * memb
            denominator += memb
        return nominator / denominator

    def __type_2_output(self, features: Dict[str, float], method: str) -> float:

