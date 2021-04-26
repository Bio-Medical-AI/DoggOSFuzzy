from doggos.inference.inference_system import InferenceSystem
from typing import List, Dict, Sequence
from doggos.knowledge import Rule
from functools import partial
import numpy as np


class MamdaniInferenceSystem(InferenceSystem):
    def __init__(self, rules: List[Rule]):
        self.__rule_base = rules

    def calculate_output(self, features: Dict[str, float], method: str) -> float:
        rule_outputs = [rule.calculate_rule(features) for rule in self.__rule_base]
        universe = self.__rule_base[0].domain.domain

        if method == "karnik_mendel":
            lmfs = [output[0] for output in rule_outputs]
            umfs = [output[0] for output in rule_outputs]
            lower_cut = self.__membership_func_union(lmfs)
            upper_cut = self.__membership_func_union(umfs)
            return self.__karnik_mendel(lower_cut, upper_cut, universe)

    def __membership_func_union(self, mfs) -> np.ndarray:
        n_functions = len(mfs)
        universe_size = len(mfs[0])
        reshaped_mfs = np.zeros(shape=(n_functions, universe_size))
        for i, mf in enumerate(mfs):
            reshaped_mfs[i] = mf
        union = np.max(reshaped_mfs, axis=0)
        return union

    def __karnik_mendel(self, lmf: np.ndarray, umf: np.ndarray, universe: Sequence[float]):
        def find_k(c: float):
            return np.where(universe_arr <= c <= universe_arr)

        def find_c_minute(c: float, under_k_mf: np.ndarray, over_k_mf: np.ndarray):
            k = find_k(c)
            lower_thetas = under_k_mf[:(k + 1)]
            upper_thetas = over_k_mf[(k + 1):]
            weights = np.append(lower_thetas, upper_thetas)
            return np.average(universe, weights=weights)

        def find_y(partial_find_c_minute: partial):
            c_prim = np.average(universe, weights=thetas)
            c_minute = partial_find_c_minute(c_prim)
            while abs(c_minute - c_prim) > np.finfo(float).eps:
                c_prim = c_minute
                c_minute, k = partial_find_c_minute(c_prim)
            return c_minute

        n_functions = 2
        universe_size = lmf.size
        universe_arr = np.array(universe, dtype=float)
        footprint = np.zeros(shape=(n_functions, universe_size))
        footprint[0] = lmf
        footprint[1] = umf
        thetas = np.sum(footprint, axis=0) / 2
        y_l = find_y(partial(find_c_minute, under_k_mf=umf, over_k_mf=lmf))
        y_r = find_y(partial(find_c_minute, under_k_mf=lmf, over_k_mf=umf))
        return (y_l + y_r) / 2


