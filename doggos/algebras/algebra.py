from abc import ABC, abstractmethod


class Algebra(ABC):
    """
    Class that represents algebra for specific fuzzy logic.
    For example: Lukasiewicz algebra, Gödel algebra.
    Each algebra contains following operations:
    - T-norm: generalized AND
    - S-norm: generalized OR
    - Negation
    - Implication
    """

    @staticmethod
    @abstractmethod
    def t_norm(self, a: float, b: float) -> float:
        pass

    @staticmethod
    @abstractmethod
    def s_norm(self, a: float, b: float) -> float:
        pass

    @staticmethod
    @abstractmethod
    def negation(self, a) -> float:
        pass

    @staticmethod
    @abstractmethod
    def implication(self, a, b) -> float:
        pass
