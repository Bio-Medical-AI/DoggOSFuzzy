import numpy as np
from collections.abc import Iterable

from doggos.algebras.algebra import Algebra
from doggos.fuzzy_sets.fuzzy_set import MembershipDegree


class GodelAlgebra(Algebra):

    @staticmethod
    def implication(a: MembershipDegree, b: MembershipDegree) -> MembershipDegree:
        """
        Calculate the Gödel implication
        :param a: first value
        :param b: second value
        :return: max(1 - a, b)
        """
        return np.maximum(1 - a, b)

    @staticmethod
    def negation(a: MembershipDegree) -> MembershipDegree:
        """
        Calculate the Gödel negation
        :param a: value
        :return: 1 - a
        """
        return 1 - a

    @staticmethod
    def s_norm(a: MembershipDegree, b: MembershipDegree) -> MembershipDegree:
        """
        Calculate the Gödel S-norm
        :param a: first value
        :param b: second value
        :return: max(a, b)
        """
        return max(a, b)

    @staticmethod
    def t_norm(a: MembershipDegree, b: MembershipDegree) -> MembershipDegree:
        """
        Calculate the Gödel T-norm
        :param a: first value
        :param b: second value
        :return: min(a, b)
        """
        return min(a, b)
