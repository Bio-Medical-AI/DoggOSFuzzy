from functools import partial
from typing import List, Dict

import numpy as np

from doggos.fuzzy_sets import MembershipDegree
from doggos.knowledge import Rule, Clause, LinguisticVariable


def center_of_gravity(domain, cut):
    return np.average(domain, weights=cut)


def largest_of_maximum(domain, cut):
    maximum = np.max(cut)
    return domain[np.where(cut == maximum)[-1]]


def smallest_of_maximum(domain, cut):
    maximum = np.max(cut)
    return domain[np.where(cut == maximum)[0]]


def middle_of_maximum(domain, cut):
    maximum = np.max(cut)
    indices = np.where(cut == maximum)[0]
    size = len(indices)
    middle = int(size / 2)
    return domain[[indices[middle]]]


def mean_of_maxima(domain, cut):
    maximum = np.max(cut)
    indices = np.where(cut == maximum)[0]
    size = len(indices)
    total = np.sum([domain[index] for index in indices])
    return total / size


def center_of_sums(domain, membership_functions):
    nominator = 0
    denominator = 0
    domain_values = np.zeros(shape=(2, len(domain)))
    domain_values[0] = domain
    for membership_function in membership_functions:
        domain_values[1] = membership_function
        nominator += np.sum(np.prod(domain_values, axis=0))
        denominator += np.sum(domain_values[1])
    return nominator / denominator


def karnik_mendel(lmf: np.ndarray, umf: np.ndarray, domain: np.ndarray) -> float:
    """
    Karnik-Mendel algorithm for interval type II fuzzy sets
    :param lmf: lower membership function
    :param umf: upper membership function
    :param domain: universe on which rule consequents are defined
    :return: decision value
    """
    thetas = (lmf + umf) / 2
    y_l = __find_y(partial(__find_c_minute, under_k_mf=umf, over_k_mf=lmf), domain, thetas)
    y_r = __find_y(partial(__find_c_minute, under_k_mf=lmf, over_k_mf=umf), domain, thetas)
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


def takagi_sugeno_weighted_average(rules: List[Rule],
                                   features: Dict[Clause, MembershipDegree],
                                   measures: Dict[LinguisticVariable, float]) -> float:
    """
    Method of calculating output of Takagi-Sugeno inference system for fuzzy sets of type 1
    Used for fuzzy sets of type 1.

    :param rules: list of rules from rulebase
    :param features: a dictionary of clauses and their membership value calculated for measures
    :param measures: a dictionary of measures consisting of Linguistic variables, and measured values for them
    :return: float that is output of whole inference system
    """
    nominator = 0
    denominator = 0
    for rule in rules:
        memb = rule.antecedent.fire(features)
        out = rule.consequent.output(measures)
        nominator += out * memb
        denominator += memb
    return nominator / denominator


def takagi_sugeno_karnik_mendel(rules: List[Rule],
                                features: Dict[Clause, MembershipDegree],
                                measures: Dict[LinguisticVariable, float], step: float) -> float:
    """
    Method of calculating output of Takagi-Sugeno inference system using Karnik-Mendel algorithm.
    Used for fuzzy sets of type 2.

    :param features: a dictionary of clauses and their membership value calculated for measures
    :param measures: a dictionary of measures consisting of Linguistic variables, and measured values for them
    :param step: size of step used in Karnik-Mendel algorithm
    :return: float that is output of whole inference system
    """

    outputs_of_rules = np.zeros(shape=(len(rules), 3))
    for rule, outputs in zip(rules, outputs_of_rules):
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

    return karnik_mendel(lmf, umf, domain)


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
    return 0.
