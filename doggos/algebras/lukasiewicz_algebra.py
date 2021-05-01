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
        return max(.0, a + b - 1.0)
