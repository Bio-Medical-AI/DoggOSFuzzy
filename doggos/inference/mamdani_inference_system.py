import numpy as np
from typing import List, Dict, Tuple, Callable, Iterable

from doggos.fuzzy_sets import MembershipDegree
from doggos.knowledge.clause import Clause
from doggos.knowledge.rule import Rule
from doggos.knowledge.consequents.mamdani_consequent import MamdaniConsequent
from doggos.inference.inference_system import InferenceSystem
from doggos.knowledge.linguistic_variable import LinguisticVariable
from doggos.fuzzy_sets.type1_fuzzy_set import Type1FuzzySet


class MamdaniInferenceSystem(InferenceSystem):
    """
    Class used to represent a mamdani inference system:

    https://www.mathworks.com/help/fuzzy/types-of-fuzzy-inference-systems.html

    Attributes
    --------------------------------------------
    __rule_base: List[Rule]
        fuzzy knowledge base used for inference

    Methods
    --------------------------------------------
    infer(self, features: Dict[Clause, List[MembershipDegree]], method: str = 'karnik_mendel') -> float:
        infer decision from knowledge base

    Examples:
    --------------------------------------------
    Creating simple mamdani inference system and infering decision
    >>> rules = [first_rule, second_rule, third_rule]
    >>> features: Dict[Clause, MembershipDegree] = fuzzifier.fuzzify(dataset)
    >>> mamdani = MamdaniInferenceSystem(rules)
    >>> mamdani.output(features, 'karnik_mendel')
    0.5
    """
    __rule_base: List[Rule]

    def __init__(self, rules: List[Rule]):
        """
        Create mamdani inference system with given knowledge base
        All rules should have the same consequent type and consequents should be defined on the same domain
        :param rules: fuzzy knowledge base used for inference
        """
        super().__init__(rules)
        self.__validate_consequents()

    def infer(self, defuzzification_method: Callable, features: Dict[Clause, List[MembershipDegree]]) \
            -> Iterable[float]:
        """
        Inferences output based on features of given object using chosen method
        :param defuzzification_method: 'KM', 'COG', 'LOM', 'MOM', 'SOM', 'MeOM', 'COS'
        :param features: dictionary of linguistic variables and their values
        :return: decision value
        """
        if not isinstance(features, Dict):
            raise ValueError("Features must be fuzzified dictionary")
        if not isinstance(defuzzification_method, Callable):
            raise ValueError("Defuzzifiaction method must be callable")

        degrees = self.__get_degrees(features)
        result = np.zeros(shape=(1, len(degrees)))
        is_type1 = self.__is_consequent_type1()
        for i in range(len(degrees)):
            single_features = {}
            for clause, memberships in features.items():
                single_features[clause] = memberships[i]

            if is_type1:
                domain, membership_functions = self.__get_domain_and_membership_functions(single_features)
                result[i] = defuzzification_method(domain, membership_functions)
            else:
                domain, lmfs, umfs = self.__get_domain_and_memberships_for_it2(single_features)
                result[i] = defuzzification_method(lmfs, umfs, domain)

        if result.shape[1] == 1:
            return result.item()

        return result

    def __get_degrees(self, features: Dict[Clause, List[MembershipDegree]]):
        values = np.array(features.values())
        return values[0]

    def __validate_consequents(self):
        for rule in self._rule_base:
            if not isinstance(rule.consequent, MamdaniConsequent):
                raise ValueError("All rule consequents must be mamdani consequents")

    def __is_consequent_type1(self):
        return isinstance(self.__rule_base[0].consequent.clause.fuzzy_set, Type1FuzzySet)

    def __get_domain_and_memberships_for_it2(self, features: Dict[Clause, List[MembershipDegree]]) \
            -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        Extracts domain and membership functions from rule base
        :param features: dictionary of linguistic variables and their values
        :return: domain, lower membership functions and upper membership functions extracted from rule base
        """
        domain, membership_functions = self.__get_domain_and_membership_functions(features)
        lmfs = [output[0] for output in membership_functions]
        umfs = [output[1] for output in membership_functions]
        return domain, lmfs, umfs

    def __get_domain_and_membership_functions(self, features: Dict[Clause, List[MembershipDegree]]) \
            -> Tuple[np.ndarray, List[np.ndarray]]:
        domain = self.__get_consequent_domain()
        membership_functions = self.__get_consequents_membership_functions(features)
        return domain, membership_functions

    def __get_consequents_membership_functions(self, features: Dict[Clause, List[MembershipDegree]]) -> np.ndarray:
        """
        Extracts rule outputs from rule base
        :param features: dictionary of linguistic variables and their values
        :return: cut membership functions from rule base
        """
        return np.array([rule.consequent.output(rule.antecedent.fire(features)).values for rule in self.__rule_base])

    def __get_consequent_domain(self) -> np.ndarray:
        return self.__rule_base[0].consequent.clause.linguistic_variable.domain()
