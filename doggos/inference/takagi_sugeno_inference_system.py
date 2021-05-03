from typing import Dict, List, Callable

import numpy as np

from doggos.fuzzy_sets.fuzzy_set import MembershipDegree
from doggos.inference.inference_system import InferenceSystem
from doggos.knowledge.linguistic_variable import LinguisticVariable
from doggos.knowledge.clause import Clause


class TakagiSugenoInferenceSystem(InferenceSystem):
    def infer(self,
              defuzzification_method: Callable,
              features: Dict[Clause, List[MembershipDegree]],
              measures: Dict[LinguisticVariable, List[float]]) -> list[float]:
        """
        Inferences output based on features of given object and measured values of them, using chosen method

        :param defuzzification_method: method of calculating inference system output.
        Must match to the type of fuzzy sets used in rules and be callable, and takes two ndarrays as parameters.
        Those arrays represent firing values of antecedents of all rules in _rule_base and outputs of their consequents
        :param features: a dictionary of clauses and list of their membership values calculated for measures
        :param measures: a dictionary of measures consisting of Linguistic variables, and list of measured float values
        for them
        :return: float that is output of whole inference system
        """
        if not isinstance(features, Dict):
            raise ValueError("Features must be dictionary")
        if not isinstance(measures, Dict):
            raise ValueError("Measures must be dictionary")
        if not isinstance(defuzzification_method, Callable):
            raise ValueError("Defuzzification_method must be Callable")

        conclusions = []
        length = len(list(self._rule_base))
        for i in range(len(list(features.values())[0])):
            single_features = {}
            single_measures = {}
            for key, value in features.items():
                single_features[key] = value[i]
            for key, value in measures.items():
                single_measures[key] = value[i]
            firings = np.zeros(shape=length)
            outputs = np.zeros(shape=length)
            # print(firings)
            # print(list(self._rule_base)[0].antecedent.fire(single_features))

            #for j in range(length):
            #    firings[j] = list(self._rule_base)[j].antecedent.fire(single_features)
            #    outputs[j] = list(self._rule_base)[j].consequent.output(single_measures)
            firings = np.array([rule.antecedent.fire(single_features) for rule in self._rule_base])
            outputs = np.array([rule.consequent.output(single_measures) for rule in self._rule_base])
            print(firings.shape)
            print(outputs.shape)
            print(firings)
            conclusions.append(defuzzification_method(firings, outputs))
        return conclusions
