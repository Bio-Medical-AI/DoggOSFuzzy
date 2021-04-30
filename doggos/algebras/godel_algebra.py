from doggos.algebras.algebra import Algebra


class GodelAlgebra(Algebra):

    @staticmethod
    def implication(a: float, b: float) -> float:
        """
        Calculate the Gödel implication
        :param a: first value
        :param b: second value
        :return: max(1 - a, b)
        """
        return max(1 - a, b)

    @staticmethod
    def negation(a: float) -> float:
        """
        Calculate the Gödel negation
        :param a: value
        :return: 1 - a
        """
        return 1 - a

    @staticmethod
    def s_norm(a: float, b: float) -> float:
        """
        Calculate the Gödel T-norm
        :param a: first value
        :param b: second value
        :return:
        """
        return max(a, b)

    @staticmethod
    def t_norm(a: float, b: float) -> float:
        """
        Calculate the Gödel T-norm
        :param a: first value
        :param b: second value
        :return: min(a, b)
        """
        return min(a, b)
