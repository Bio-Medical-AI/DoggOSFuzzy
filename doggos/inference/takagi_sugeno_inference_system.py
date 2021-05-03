from typing import Dict, List
import numpy as np
from doggos.fuzzy_sets.fuzzy_set import MembershipDegree
from doggos.inference.inference_system import InferenceSystem
from doggos.knowledge.linguistic_variable import LinguisticVariable
from doggos.knowledge.rule import Rule
from doggos.knowledge.clause import Clause


def calculate_membership(x: float,
                         outputs_of_rules: np.ndarray) -> float:
    """
    Calculates values of lower membership function and upper membership function for given element of domain,
    basing on outputs of the rules
    :param x: value from domain for which values are calculated
    :param outputs_of_rules: ndarray with shape nx2, where n is number of records, where first column contains elements
    of domain sorted ascending and second one contains elements from their codomain. All elements are floats.
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
                    distance_vertical = outputs_of_rules[i][1] - outputs_of_rules[i - 1][1]
                    distance_of_x = x - outputs_of_rules[i - 1][0]
                    horizontal_proportion = distance_of_x / distance_horizontal
                    return distance_vertical * horizontal_proportion + outputs_of_rules[i - 1][1]
    return 0


class TakagiSugenoInferenceSystem(InferenceSystem):

    def __init__(self, rules: List[Rule]):
        """
        Create Takagi-Sugeno inference system with given knowledge base
        All rules should have the same consequent type and consequents should be defined on the same domain

        :param rules: fuzzy knowledge base used for inference
        """
        if isinstance(rules, List):
            self._rule_base = rules
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
        nominator = 0
        denominator = 0
        for rule in self._rule_base:
            memb = rule.antecedent.fire(features)
            out = rule.consequent.output(measures)
            nominator += out * memb
            denominator += memb
        return nominator / denominator

    def __type_2_basic_output(self, features: Dict[Clause, MembershipDegree], measures: Dict[LinguisticVariable, float],
                              step: float) -> float:
        """
        Method of calculating output of Takagi-Sugeno inference system using Karnik-Mendel algorithm

        :param features: a dictionary of clauses and their membership value calculated for measures
        :param measures: a dictionary of measures consisting of Linguistic variables, and measured values for them
        :param step: size of step used in Karnik-Mendel algorithm
        :return: float that is output of whole inference system
        """

        outputs_of_rules = np.zeros(shape=(len(self._rule_base), 3))
        for rule, outputs in zip(self._rule_base, outputs_of_rules):
            outputs[0] = rule.consequent.output(measures)
            firing = rule.antecedent.fire(features)
            outputs[1] = firing[0]
            outputs[2] = firing[1]

        outputs_of_rules = outputs_of_rules[np.argsort(outputs_of_rules[:, 0])]
        domain = np.arange(outputs_of_rules[0][0], outputs_of_rules[-1][0], step)
        lmf = np.zeros(shape=domain.shape)
        umf = np.zeros(shape=domain.shape)
        for i in range(domain.shape):
            lmf[i] = calculate_membership(domain[i], outputs_of_rules[:, :2])
            umf[i] = calculate_membership(domain[i],
                                          np.concatenate((outputs_of_rules[:, 0], outputs_of_rules[:, 2]), axis=1))

        return self._karnik_mendel(self, lmf, umf, domain)
