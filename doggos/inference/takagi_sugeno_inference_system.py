from typing import Dict, List, Tuple
import numpy as np
from doggos.fuzzy_sets.membership import MembershipDegree
from doggos.inference.inference_system import InferenceSystem
from doggos.knowledge.linguistic_variable import LinguisticVariable
from doggos.knowledge.rule import Rule
from doggos.knowledge.clause import Clause


def calculate_membership(x: float,
                         outputs_of_rules: List[Tuple[float, Tuple[float, float]]]) -> Tuple[float, float]:
    """
    Calculates values of lower membership function and upper membership function for given element of domain,
    basing on outputs of the rules
    :param x: value from domain for which values are calculated
    :param outputs_of_rules: list of tuples of output values from rules and their firing values, sorted by outputs
    :return: returns value of both lower membership function and upper membership function for given x
    """
    if len(outputs_of_rules) == 1:
        if x == outputs_of_rules[0][0]:
            return outputs_of_rules[0][1]
    elif len(outputs_of_rules) > 1:
        if x >= outputs_of_rules[0][0]:
            for i in range(1, len(outputs_of_rules)):
                if x <= outputs_of_rules[i][0]:
                    distance_horizontal = outputs_of_rules[i][0] - outputs_of_rules[i - 1][0]
                    distance_vertical_lower = outputs_of_rules[i][1][0] - outputs_of_rules[i - 1][1][0]
                    distance_vertical_upper = outputs_of_rules[i][1][1] - outputs_of_rules[i - 1][1][1]
                    distance_of_x = x - outputs_of_rules[i - 1][0]
                    horizontal_proportion = distance_of_x / distance_horizontal
                    return (distance_vertical_lower * horizontal_proportion + outputs_of_rules[i - 1][1][0],
                            distance_vertical_upper * horizontal_proportion + outputs_of_rules[i - 1][1][1])
    return 0, 0


class TakagiSugenoInferenceSystem(InferenceSystem):

    def __init__(self, rules: List[Rule]):
        """
        Create Takagi-Sugeno inference system with given knowledge base
        All rules should have the same consequent type and consequents should be defined on the same domain

        :param rules: fuzzy knowledge base used for inference
        """
        if isinstance(rules, List):
            self.__rule_base = rules
        else:
            raise ValueError("Inference system must take rules as parameters")

    def output(self, features: Dict[Clause, MembershipDegree], measures: Dict[LinguisticVariable, float],
               method: str = "average", step: float = 0.0001) -> float:
        """
        Inferences output based on features of given object and measured values of them, using chosen method

        :param features: a dictionary of clauses and their membership value calculated for measures
        :param measures: a dictionary of measures consisting of Linguistic variables, and measured values for them
        :param method: method of calculating final output, must match type of fuzzy sets used in rules
        :param step: size of step used in Karnik-Mendel algorithm
        :return: float that is output of whole inference system
        """
        if not isinstance(features, Dict):
            raise ValueError("Features must be dictionary")
        if not isinstance(features, Dict):
            raise ValueError("Measures must be dictionary")

        if len(self._rule_base) == 0:
            raise IndexError("Rule base of inference system is empty")
        if method == "average":
            return self.__type_1_basic_output(features, measures)
        elif method == "karnik-mendel":
            return self.__type_2_basic_output(features, measures, step)
        else:
            raise NotImplementedError("There is no method of that name")

    def __type_1_basic_output(self, features: Dict[Clause, MembershipDegree], measures: Dict[LinguisticVariable, float]) -> float:
        """
        Method of calculating output of Takagi-Sugeno inference system for fuzzy sets of type 1

        :param features: a dictionary of clauses and their membership value calculated for measures
        :param measures: a dictionary of measures consisting of Linguistic variables, and measured values for them
        :return: float that is output of whole inference system
        """
        numerator = 0
        denominator = 0
        for rule in self._rule_base:
            out, memb = rule.output(features, measures)
            numerator += out * memb
            denominator += memb
        return numerator / denominator

    def __type_2_basic_output(self, features: Dict[Clause, MembershipDegree], measures: Dict[LinguisticVariable, float],
                              step: float) -> float:
        """
        Method of calculating output of Takagi-Sugeno inference system using Karnik-Mendel algorithm

        :param features: a dictionary of clauses and their membership value calculated for measures
        :param measures: a dictionary of measures consisting of Linguistic variables, and measured values for them
        :param step: size of step used in Karnik-Mendel algorithm
        :return: float that is output of whole inference system
        """
        outputs_of_rules = []
        for rule in self._rule_base:
            outputs_of_rules.append(rule.output(features, measures))
        outputs_of_rules.sort(key=lambda tup: tup[0])
        domain = np.arange(outputs_of_rules[0][0], outputs_of_rules[-1][0], step)
        lmf = np.zeros(shape=domain.shape)
        umf = np.zeros(shape=domain.shape)
        for i in range(domain.shape):
            lmf[i], umf[i] = calculate_membership(domain[i], outputs_of_rules)

        return self._karnik_mendel(self, lmf, umf, domain)
