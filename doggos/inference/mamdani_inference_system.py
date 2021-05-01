from doggos.inference.inference_system import InferenceSystem
from typing import List, Dict, Sequence
from doggos.knowledge.rule import Rule
from functools import partial
import numpy as np


class MamdaniInferenceSystem(InferenceSystem):
    def __init__(self, rules: List[Rule]):
        self.__rule_base = rules

    def output(self, features: Dict[str, float], method: str) -> float:
        if method == "karnik_mendel":
            domain, lmfs, umfs = self.__get_domain_and_memberships(features)
            lower_cut = self.__membership_func_union(lmfs)
            upper_cut = self.__membership_func_union(umfs)
            return self.__karnik_mendel(lower_cut, upper_cut, domain)

    def __karnik_mendel(self, lmf: np.ndarray, umf: np.ndarray, domain: np.ndarray) -> float:
        thetas = (lmf + umf) / 2
        y_l = self.__find_y(partial(self.__find_c_minute, under_k_mf=umf, over_k_mf=lmf), domain, thetas)
        y_r = self.__find_y(partial(self.__find_c_minute, under_k_mf=lmf, over_k_mf=umf), domain, thetas)
        return (y_l + y_r) / 2

    def __find_y(self, partial_find_c_minute: partial, domain: np.ndarray, thetas: np.ndarray) -> float:
        c_prim = np.average(domain, weights=thetas)
        c_minute = partial_find_c_minute(c_prim)
        while abs(c_minute - c_prim) > np.finfo(float).eps:
            c_prim = c_minute
            c_minute = partial_find_c_minute(c_prim)
        return c_minute

    def __find_c_minute(self, c: float, under_k_mf: np.ndarray, over_k_mf: np.ndarray, domain: np.ndarray) -> float:
        k = self.__find_k(c, domain)
        lower_thetas = under_k_mf[:(k + 1)]
        upper_thetas = over_k_mf[(k + 1):]
        weights = np.append(lower_thetas, upper_thetas)
        return np.average(domain, weights=weights)

    def __find_k(self, c: float, universe: np.ndarray) -> float:
        return np.where(universe <= c)[0][-1]

    def __membership_func_union(self, mfs) -> np.ndarray:
        n_functions = len(mfs)
        universe_size = len(mfs[0])
        reshaped_mfs = np.zeros(shape=(n_functions, universe_size))
        for i, mf in enumerate(mfs):
            reshaped_mfs[i] = mf
        union = np.max(reshaped_mfs, axis=0)
        return union

    def __get_rule_outputs(self, features: Dict[str, float]):
        return np.array([rule.output(features) for rule in self.__rule_base])

    def __get_domain_and_memberships(self, features: Dict[str, float]):
        rule_outputs = self.__get_rule_outputs(features)
        domain = self.__rule_base[0].consequent.clause.linguistic_variable.domain()
        lmfs = [output[0] for output in rule_outputs]
        umfs = [output[1] for output in rule_outputs]
        return domain, lmfs, umfs
