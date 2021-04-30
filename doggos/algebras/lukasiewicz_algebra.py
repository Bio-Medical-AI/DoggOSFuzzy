from doggos.algebras.algebra import Algebra


class LukasiewiczAlgebra(Algebra):

    @staticmethod
    def implication(a: float, b: float) -> float:
        """
        Calculate the Lukasiewicz implication
        :param a: first value
        :param b: second value
        :return:
        """
        return min(1., 1 - a + b)

    @staticmethod
    def negation(a: float) -> float:
        """
        Calculate the Lukasiewicz negation
        :param a: value
        :return: 1 - a
        """
        return 1 - a

    @staticmethod
    def s_norm(a: float, b: float) -> float:
        """
        Calculate the Lukasiewicz S-norm
        :param a: first value
        :param b: second value
        :return: min(1, a + b)
        """
        return min(1., a + b)

    @staticmethod
    def t_norm(a: float, b: float) -> float:
        """
        Calculate the Lukasiewicz T-norm
        :param a: first value
        :param b: second value
        :return: max(.0, a + b - 1)
        """
        return max(.0, a + b - 1.0)
