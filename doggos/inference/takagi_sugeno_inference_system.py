from typing import Dict

from doggos.fuzzy_sets.membership import MembershipDegree
from doggos.inference.inference_system import InferenceSystem
from doggos.knowledge import Clause


class TakagiSugenoInferenceSystem(InferenceSystem):
    def output(self, features: Dict[Clause, MembershipDegree], measures: Dict[str, float], method: str = "base") -> float:

        if len(self._rule_base) == 0:
            raise IndexError("Rule base of inference system is empty")
        if method == "basic" or "type1 basic":
            return self.__type_1_basic_output(features, measures)
        elif method == "type2 basic":
            return self.__type_2_basic_output(features, measures)
        else:
            raise NotImplementedError("There is no method of that name")

    def __type_1_basic_output(self, features: Dict[Clause, MembershipDegree], measures: Dict[str, float]) -> float:
        nominator = 0
        denominator = 0
        for rule in self._rule_base:
            out, memb = rule.output(features, measures)
            nominator += out * memb
            denominator += memb
        return nominator / denominator

    def __type_2_basic_output(self, features: Dict[Clause, MembershipDegree]) -> float:
        # for rule in self._rule_base:
        pass
