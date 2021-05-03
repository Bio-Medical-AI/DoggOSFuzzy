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

        values = np.array(features.values())
        degrees = values[0]

        for i in range(len(degrees)):
            single_features = {}
            for clause, memberships in features.items():
                single_features[clause] = memberships[i]

            if self.__is_consequent_type1():
                domain, membership_functions = self.__get_domain_and_membership_functions(single_features)
                cut = self.__membership_func_union(membership_functions)
                return defuzzification_method(domain, cut)
            else:
                domain, lmfs, umfs = self.__get_domain_and_memberships_for_it2(single_features)
                lower_cut = self.__membership_func_union(lmfs)
                upper_cut = self.__membership_func_union(umfs)
                return defuzzification_method(lower_cut, upper_cut, domain)

            domain, membership_functions = self.__get_domain_and_membership_functions(single_features)
            return self._center_of_sums(domain, membership_functions)

    def __validate_consequents(self):
        for rule in self._rule_base:
            if not isinstance(rule.consequent, MamdaniConsequent):
                raise ValueError("All rule consequents must be mamdani consequents")

    def __is_consequent_type1(self):
        return isinstance(self.__rule_base[0].consequent.clause.fuzzy_set, Type1FuzzySet)

    def __membership_func_union(self, mfs: List[np.ndarray]) -> np.ndarray:
        """
        Performs merge of given membership functions by choosing maximum of respective values
        :param mfs: membership functions to unify
        :return: unified membership functions
        """
        n_functions = len(mfs)
        universe_size = len(mfs[0])
        reshaped_mfs = np.zeros(shape=(n_functions, universe_size))
        for i, mf in enumerate(mfs):
            reshaped_mfs[i] = mf
        union = np.max(reshaped_mfs, axis=0)
        return union

    def __get_domain_and_cut(self, features: Dict[Clause, List[MembershipDegree]]):
        domain, membership_functions = self.__get_domain_and_membership_functions(features)
        cut = self.__membership_func_union(membership_functions)
        return domain, cut

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

    def __get_domain_and_membership_functions(self, features: Dict[Clause, List[MembershipDegree]]):
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

    def __get_consequent_domain(self):
        return self.__rule_base[0].consequent.clause.linguistic_variable.domain()
