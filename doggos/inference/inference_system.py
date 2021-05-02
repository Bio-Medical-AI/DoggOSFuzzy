from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import numpy as np
from functools import partial

from doggos.knowledge import Rule


class InferenceSystem(ABC):
    __rule_base: List[Rule]

    @abstractmethod
    def output(self, features: Dict[str, float], method: str) -> float:
        pass

    def _center_of_gravity(self, domain, weights):
        return np.average(domain, weights=weights)

    def _largest_of_maximum(self, domain, cut):
        maximum = np.max(cut)
        return domain[np.where(cut == maximum)[-1]]

    def _smallest_of_maximum(self, domain, cut):
        maximum = np.max(cut)
        return domain[np.where(cut == maximum)[0]]

    def _middle_of_maximum(self, domain, cut):
        maximum = np.max(cut)
        indices = np.where(cut == maximum)[0]
        size = len(indices)
        middle = int(size / 2)
        return domain[[indices[middle]]]

    def _mean_of_maxima(self, domain, cut):
        maximum = np.max(cut)
        indices = np.where(cut == maximum)[0]
        size = len(indices)
        total = np.sum([domain[index] for index in indices])
        return total / size

    def _center_of_sums(self, domain, membership_functions):
        for x in domain:
            for membership_function in membership_functions:



    def _karnik_mendel(self, lmf: np.ndarray, umf: np.ndarray, domain: np.ndarray) -> float:
        """
        Karnik-Mendel algorithm for interval type II fuzzy sets
        :param lmf: lower membership function
        :param umf: upper membership function
        :param domain: universe on which rule consequents are defined
        :return: decision value
        """
        thetas = (lmf + umf) / 2
        y_l = self._find_y(partial(self._find_c_minute, under_k_mf=umf, over_k_mf=lmf), domain, thetas)
        y_r = self._find_y(partial(self._find_c_minute, under_k_mf=lmf, over_k_mf=umf), domain, thetas)
        return (y_l + y_r) / 2

    def _find_y(self, partial_find_c_minute: partial, domain: np.ndarray, thetas: np.ndarray) -> float:
        """
        Finds decision factor for specified part of algorithm
        :param partial_find_c_minute: _find_c_minute function with filled under_k_mf and over_k_mf arguments
        :param domain: universe on which rule consequents are defined
        :param thetas: weights for weighted average: (lmf + umf) / 2
        :return: Decision factor for specified part of algorithm
        """
        c_prim = np.average(domain, weights=thetas)
        c_minute = partial_find_c_minute(c=c_prim, domain=domain)
        while abs(c_minute - c_prim) > np.finfo(float).eps:
            c_prim = c_minute
            c_minute = partial_find_c_minute(c=c_prim, domain=domain)
        return c_minute

    def _find_c_minute(self, c: float, under_k_mf: np.ndarray, over_k_mf: np.ndarray,
                       domain: np.ndarray) -> float:
        """
        Finds weights and average for combined membership functions
        :param c: weighted average of domain values with previously defined thetas as weights
        :param under_k_mf: takes elements of under_k_mf with indices <= k as weights
        :param over_k_mf: takes elements of over_k_mf with indices >= k+1 as weights
        :param domain: universe on which rule consequents are defined
        :return: average for combined membership functions
        """
        k = self._find_k(c, domain)
        lower_thetas = under_k_mf[:(k + 1)]
        upper_thetas = over_k_mf[(k + 1):]
        weights = np.append(lower_thetas, upper_thetas)
        return np.average(domain, weights=weights)

    def _find_k(self, c: float, domain: np.ndarray) -> float:
        """
        Finds index for weighted average in given domain
        :param c: weighted average of combined membership functions
        :param domain: universe on which rule consequents are defined
        :return: index for weighted average in given domain
        """
        return np.where(domain <= c)[0][-1]

