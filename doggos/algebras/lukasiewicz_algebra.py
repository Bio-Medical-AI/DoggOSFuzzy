import numpy as np
from collections.abc import Iterable


from doggos.algebras.algebra import Algebra
from doggos.fuzzy_sets.fuzzy_set import MembershipDegree


class LukasiewiczAlgebra(Algebra):

    @staticmethod
    def implication(a: MembershipDegree, b: MembershipDegree) -> MembershipDegree:
        """
        Calculate the Lukasiewicz implication
        :param a: first value
        :param b: second value
        :return: min(1., 1 - a + b)
        """
        return min(1., 1 - a + b)

    @staticmethod
    def negation(a: MembershipDegree) -> MembershipDegree:
        """
        Calculate the Lukasiewicz negation
        :param a: value
        :return: 1 - a
        """
        return 1 - a

    @staticmethod
    def s_norm(a: MembershipDegree, b: MembershipDegree) -> MembershipDegree:
        """
        Calculate the Lukasiewicz S-norm
        :param a: first value
        :param b: second value
        :return: min(1, a + b)
        """
        return min(1., a + b)

    @staticmethod
    def t_norm(a: MembershipDegree, b: MembershipDegree) -> MembershipDegree:
        """
        Calculate the Lukasiewicz T-norm
        :param a: first value
        :param b: second value
        :return: max(.0, a + b - 1)
        """
        if isinstance(a, Iterable):
            a = np.array(a)
        if isinstance(b, Iterable):
            b = np.array(b)
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape[0] != b.shape[0]:
            raise ValueError(f'Dimensions {a.shape[0]} and {b.shape[0]} are not compatible')
        return np.maximum(.0, a + b - 1.0)
