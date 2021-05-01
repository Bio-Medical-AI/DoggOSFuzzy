from doggos.inference.inference_system import InferenceSystem
from typing import List, Dict, Tuple
from doggos.knowledge.rule import Rule
from functools import partial
import numpy as np


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
        Create mamdani inference system with given knowledge base.
        All rules should have the same consequent type and consequents should be defined on the same domain
        :param rules: fuzzy knowledge base used for inference
        """
        self.__rule_base = rules

    def output(self, features: Dict[str, float], method: str = 'karnik_mendel') -> float:
        if method == "karnik_mendel":
            domain, lmfs, umfs = self.__get_domain_and_memberships(features)
            lower_cut = self.__membership_func_union(lmfs)
            upper_cut = self.__membership_func_union(umfs)
            return self.__karnik_mendel(lower_cut, upper_cut, domain)

    def __karnik_mendel(self, lmf: np.ndarray[float], umf: np.ndarray[float], domain: np.ndarray[float]) -> float:
        thetas = (lmf + umf) / 2
        y_l = self.__find_y(partial(self.__find_c_minute, under_k_mf=umf, over_k_mf=lmf), domain, thetas)
        y_r = self.__find_y(partial(self.__find_c_minute, under_k_mf=lmf, over_k_mf=umf), domain, thetas)
        return (y_l + y_r) / 2

    def __find_y(self, partial_find_c_minute: partial, domain: np.ndarray[float], thetas: np.ndarray[float]) -> float:
        c_prim = np.average(domain, weights=thetas)
        c_minute = partial_find_c_minute(c=c_prim, domain=domain)
        while abs(c_minute - c_prim) > np.finfo(float).eps:
            c_prim = c_minute
            c_minute = partial_find_c_minute(c=c_prim, domain=domain)
        return c_minute

    def __find_c_minute(self, c: float, under_k_mf: np.ndarray[float], over_k_mf: np.ndarray[float],
                        domain: np.ndarray[float]) -> float:
        k = self.__find_k(c, domain)
        lower_thetas = under_k_mf[:(k + 1)]
        upper_thetas = over_k_mf[(k + 1):]
        weights = np.append(lower_thetas, upper_thetas)
        return np.average(domain, weights=weights)

    def __find_k(self, c: float, domain: np.ndarray[float]) -> float:
        return np.where(domain <= c)[0][-1]

    def __membership_func_union(self, mfs: List[np.ndarray[float]]) -> np.ndarray[float]:
        n_functions = len(mfs)
        universe_size = len(mfs[0])
        reshaped_mfs = np.zeros(shape=(n_functions, universe_size))
        for i, mf in enumerate(mfs):
            reshaped_mfs[i] = mf
        union = np.max(reshaped_mfs, axis=0)
        return union

    def __get_domain_and_memberships(self, features: Dict[str, float]) \
            -> Tuple[np.ndarray[float], List[np.ndarray[float]], List[np.ndarray[float]]]:
        rule_outputs = self.__get_rule_outputs(features)
        domain = self.__rule_base[0].consequent.clause.linguistic_variable.domain()
        lmfs = [output[0] for output in rule_outputs]
        umfs = [output[1] for output in rule_outputs]
        return domain, lmfs, umfs

    def __get_rule_outputs(self, features: Dict[str, float]) -> np.ndarray[Tuple[np.ndarray[float], np.ndarray[float]]]:
        return np.array([rule.output(features) for rule in self.__rule_base])
