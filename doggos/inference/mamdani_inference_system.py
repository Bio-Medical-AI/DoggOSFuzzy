from doggos.inference.inference_system import InferenceSystem
from typing import List, Dict
from doggos.knowledge import Rule
import numpy as np


class MamdaniInferenceSystem(InferenceSystem):
    def __init__(self, rules: List[Rule]):
        self.__rule_base = rules

    def calculate_output(self, features: Dict[str, float], method: str) -> float:
        rule_outputs = [rule.calculate_rule(features) for rule in self.__rule_base]

        if method == "karnik_mendel":
            lmfs = [output[0] for output in rule_outputs]
            umfs = [output[0] for output in rule_outputs]
            lower_cut = self.__membership_func_union(lmfs)
            upper_cut = self.__membership_func_union(umfs)


    def __membership_func_union(self, mfs) -> np.ndarray:
        n_functions = len(mfs)
        universe_size = len(mfs[0])
        reshaped_mfs = np.zeros(shape=(n_functions, universe_size))
        for i, mf in enumerate(mfs):
            reshaped_mfs[i] = mf
        union = np.max(reshaped_mfs, axis=0)
        return union


    def __karnik_mendel(self, lmf: np.ndarray, umf: np.ndarray):
        n_functions = 2
        universe_size = lmf.size
        footprint = np.zeros(shape=(n_functions, universe_size))
        footprint[0] = lmf
        footprint[1] = umf
        thetas = np.sum(footprint, axis=0) / 2
        universe = np.linspace(0, u)
