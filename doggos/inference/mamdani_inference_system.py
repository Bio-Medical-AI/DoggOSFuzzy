from typing import List, Dict, Tuple
from doggos.knowledge.rule import Rule
import numpy as np

from doggos.inference.inference_system import InferenceSystem


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
    output(self, features: Dict[str, float], method: str = 'karnik_mendel') -> float:
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
        self.__rule_base = rules

    def output(self, features: Dict[str, float], method: str = 'karnik_mendel') -> float:
        """
        Inferences output based on features of given object using chosen method
        :param features: dictionary of linguistic variables and their values
        :param method: 'karnik_mendel' TODO: ...
        :return: decision value
        """
        if method == "karnik_mendel":
            domain, lmfs, umfs = self.__get_domain_and_memberships(features)
            lower_cut = self.__membership_func_union(lmfs)
            upper_cut = self.__membership_func_union(umfs)
            return self._karnik_mendel(lower_cut, upper_cut, domain)

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

    def __get_domain_and_memberships(self, features: Dict[str, float]) \
            -> Tuple[np.ndarray[float], List[np.ndarray], List[np.ndarray]]:
        """
        Extracts domain and membership functions from rule base
        :param features: dictionary of linguistic variables and their values
        :return: domain, lower membership functions and upper membership functions extracted from rule base
        """
        rule_outputs = self.__get_rule_outputs(features)
        domain = self.__rule_base[0].consequent.clause.linguistic_variable.domain()
        lmfs = [output[0] for output in rule_outputs]
        umfs = [output[1] for output in rule_outputs]
        return domain, lmfs, umfs

    def __get_rule_outputs(self, features: Dict[str, float]) -> np.ndarray[Tuple[np.ndarray, np.ndarray]]:
        """
        Extracts rule outputs from rule base
        :param features: dictionary of linguistic variables and their values
        :return: cut membership functions from rule base
        """
        return np.array([rule.output(features) for rule in self.__rule_base])
