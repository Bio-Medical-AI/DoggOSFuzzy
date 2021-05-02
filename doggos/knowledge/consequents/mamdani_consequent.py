from collections import Iterable
from typing import Tuple, List
import numpy as np
from copy import deepcopy

from doggos.knowledge.consequents.consequent import Consequent
from doggos.knowledge.clause import Clause
from doggos.fuzzy_sets.fuzzy_set import MembershipDegree
from doggos.knowledge.consequents.consequent import ConsequentOutput


class MamdaniConsequent(Consequent):
    """
    Class used to represent a fuzzy rule Mamdani consequent:
    https://en.wikipedia.org/wiki/Fuzzy_rule

    Attributes
    --------------------------------------------
    __clause : Clause
        supplies Consequent with universe and a fuzzy set. Provided fuzzy set type must match fuzzy sets used in Fuzzy
        Rule Antecedent, otherwise an exception will be raised during computations.

    __cut_clause : Clause
        fuzzy set provided by clause cut to the rule firing level

    Methods
    --------------------------------------------
    output(rule_firing: MembershipDegree) -> Clause
        cut fuzzy set provided by Clause to rule firing level

    Examples:
    --------------------------------------------

    """

    def __init__(self, clause: Clause):
        """
        Create Rules Consequent used in Mamdani Inference System. Provided Clause holds fuzzy set describing Consequent
        and Linguistic Variable which value user wants to compute.
        :param clause: Clause containing fuzzy set and linguistic variable
        """
        self.__clause = clause
        self.__cut_clause = None

    def output(self, rule_firing: MembershipDegree) -> Clause:
        """
        Cuts membership function to the level of rule firing. It is a minimum of membership function values
        and respecting rule firing. Rule firing should hold values from range [0, 1].
        IMPORTANT:
        Make sure type of fuzzy set used in Clause matches type of fuzzy sets used in Antecedent of Rule and therefore
        its firing type.
        :param rule_firing: firing value of a Rule in which Consequent is used
        :return: Clause with cut membership function
        """
        if isinstance(rule_firing, float):
            return self.__t1_cut(rule_firing)
        elif isinstance(rule_firing, Iterable) and len(rule_firing) == 2:
            if not isinstance(rule_firing, np.ndarray):
                rule_firing = np.array(rule_firing).reshape((2, 1))
            return self.__it2_cut(rule_firing)
        else:
            raise ValueError(f"Incorrect type of rule firing: {rule_firing}")

    def __cut(self, rule_firing: float) -> Clause:
        """
        Makes a cut for type one fuzzy sets. If Clause fuzzy set type mismatches rule_firing type, exception is raised.
        :param rule_firing: crisp value of rule firing
        :return: fuzzy set membership function, cut to the level of firing value
        """
        self.__cut_clause = deepcopy(self.__clause)
        self.__cut_clause.values = np.minimum(self.__cut_clause.values, rule_firing)
        return self.__cut_clause
