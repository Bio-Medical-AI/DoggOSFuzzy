from collections import Iterable
from functools import partial
from typing import List

import numpy as np


def center_of_gravity(domain, membership_functions):
    cut = __membership_func_union(membership_functions)
    return np.average(domain, weights=cut)


def largest_of_maximum(domain, membership_functions):
    cut = __membership_func_union(membership_functions)
    maximum = np.max(cut)
    return domain[np.where(cut == maximum)[0][-1]]


def smallest_of_maximum(domain, membership_functions):
    cut = __membership_func_union(membership_functions)
    maximum = np.max(cut)
    return domain[np.where(cut == maximum)[0][0]]


def middle_of_maximum(domain, membership_functions):
    cut = __membership_func_union(membership_functions)
    maximum = np.max(cut)
    indices = np.where(cut == maximum)[0]
    size = len(indices)
    middle = int(size / 2)
    return domain[[indices[middle]]]


def mean_of_maxima(domain, membership_functions):
    cut = __membership_func_union(membership_functions)
    maximum = np.max(cut)
    indices = np.where(cut == maximum)[0]
    size = len(indices)
    total = np.sum([domain[index] for index in indices])
    return total / size


def center_of_sums(domain, membership_functions):
    if not isinstance(membership_functions[0], Iterable):
        sums_of_memberships = membership_functions
    else:
        universe = np.array(membership_functions)
        sums_of_memberships = np.sum(universe, axis=0)

    domain_memberships_sums = np.array((domain, sums_of_memberships))
    numerator = np.sum(np.prod(domain_memberships_sums, axis=0))
    denominator = np.sum(sums_of_memberships)

    return numerator / denominator


def karnik_mendel(lmfs: List[np.ndarray], umfs: List[np.ndarray], domain: np.ndarray) -> float:
    """
    Karnik-Mendel algorithm for interval type II fuzzy sets
    :param lmfs: lower membership functions
    :param umfs: upper membership functions
    :param domain: universe on which rule consequents are defined
    :return: decision value
    """
    lower_cut = __membership_func_union(lmfs)
    upper_cut = __membership_func_union(umfs)
    thetas = (lower_cut + upper_cut) / 2
    y_l = __find_y(partial(__find_c_minute, under_k_mf=upper_cut, over_k_mf=lower_cut), domain, thetas)
    y_r = __find_y(partial(__find_c_minute, under_k_mf=lower_cut, over_k_mf=upper_cut), domain, thetas)
    return (y_l + y_r) / 2


def __find_y(partial_find_c_minute: partial, domain: np.ndarray, thetas: np.ndarray) -> float:
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


def __find_c_minute(c: float, under_k_mf: np.ndarray, over_k_mf: np.ndarray,
                    domain: np.ndarray) -> float:
    """
    Finds weights and average for combined membership functions
    :param c: weighted average of domain values with previously defined thetas as weights
    :param under_k_mf: takes elements of under_k_mf with indices <= k as weights
    :param over_k_mf: takes elements of over_k_mf with indices >= k+1 as weights
    :param domain: universe on which rule consequents are defined
    :return: average for combined membership functions
    """
    k = __find_k(c, domain)
    lower_thetas = under_k_mf[:(k + 1)]
    upper_thetas = over_k_mf[(k + 1):]
    weights = np.append(lower_thetas, upper_thetas)
    return np.average(domain, weights=weights)


def __find_k(c: float, domain: np.ndarray) -> float:
    """
    Finds index for weighted average in given domain
    :param c: weighted average of combined membership functions
    :param domain: universe on which rule consequents are defined
    :return: index for weighted average in given domain
    """
    return np.where(domain <= c)[0][-1]


def __membership_func_union(mfs: List[np.ndarray]) -> np.ndarray:
    """
    Performs merge of given membership functions by choosing maximum of respective values
    :param mfs: membership functions to unify
    :return: unified membership functions
    """
    if not isinstance(mfs[0], Iterable):
        mfs = [mfs]
    n_functions = len(mfs)
    universe_size = len(mfs[0])
    reshaped_mfs = np.zeros(shape=(n_functions, universe_size))
    for i, mf in enumerate(mfs):
        reshaped_mfs[i] = mf
    union = np.max(reshaped_mfs, axis=0)
    return union
