from doggos.algebras.algebra import Algebra


class LukasiewiczAlgebra(Algebra):
    def implication(self, a: float, b: float) -> float:
        """
        Calculate the Lukasiewicz implication
        :param a: first value
        :param b: second value
        :return:
        """
        return min(1., 1 - a + b)

    def negation(self, a: float) -> float:
        """
        Calculate the Lukasiewicz negation
        :param a: value
        :return: 1 - a
        """
        return 1 - a

    def s_norm(self, a: float, b: float) -> float:
        """
        Calculate the Lukasiewicz S-norm
        :param a: first value
        :param b: second value
        :return: min(1, a + b)
        """
        return min(1., a + b)

    def t_norm(self, a: float, b: float) -> float:
        """
        Calculate the Lukasiewicz T-norm
        :param a: first value
        :param b: second value
        :return: max(.0, a + b - 1)
        """
        return max(.0, a + b - 1)
