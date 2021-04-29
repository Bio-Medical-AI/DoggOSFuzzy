from doggos.algebras.algebra import Algebra


class GodelAlgebra(Algebra):
    def implication(self, a: float, b: float) -> float:
        """
        Calculate the Gödel implication
        :param a: first value
        :param b: second value
        :return: max(1 - a, b)
        """
        return max(1 - a, b)

    def negation(self, a: float) -> float:
        """
        Calculate the Gödel negation
        :param a: value
        :return: 1 - a
        """
        return 1 - a

    def s_norm(self, a: float, b: float) -> float:
        """
        Calculate the Gödel T-norm
        :param a: first value
        :param b: second value
        :return:
        """
        return max(a, b)

    def t_norm(self, a: float, b: float) -> float:
        """
        Calculate the Gödel T-norm
        :param a: first value
        :param b: second value
        :return: min(a, b)
        """
        return min(a, b)
