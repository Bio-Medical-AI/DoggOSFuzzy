from typing import Dict, List, Callable
from doggos.fuzzy_sets.fuzzy_set import MembershipDegree
from doggos.inference.inference_system import InferenceSystem
from doggos.knowledge.linguistic_variable import LinguisticVariable
from doggos.knowledge.clause import Clause


class TakagiSugenoInferenceSystem(InferenceSystem):
    def infer(self,
              defuzzification_method: Callable,
              features: Dict[Clause, List[MembershipDegree]],
              measures: Dict[LinguisticVariable, List[float]],
              step: float = 0.01) -> list[float]:
        """
        Inferences output based on features of given object and measured values of them, using chosen method

        :param defuzzification_method: method of calculating final output,
        must match to the type of fuzzy sets used in rules and be callable
        :param features: a dictionary of clauses and list of their membership values calculated for measures
        :param measures: a dictionary of measures consisting of Linguistic variables, and list of measured float values
        for them
        :param step: size of step used in Karnik-Mendel algorithm
        :return: float that is output of whole inference system
        """
        if not isinstance(features, Dict):
            raise ValueError("Features must be dictionary")
        if not isinstance(measures, Dict):
            raise ValueError("Measures must be dictionary")
        if not isinstance(defuzzification_method, Callable):
            raise ValueError("Defuzzification_method must be Callable")

        outputs = []
        for i in range(len(list(features.values())[0])):
            single_features = {}
            single_measures = {}
            for key, value in features.items():
                single_features[key] = value[i]
            for key, value in measures.items():
                single_measures[key] = value[i]
            outputs.append(defuzzification_method(self._rule_base, single_features, single_measures))
        return outputs
