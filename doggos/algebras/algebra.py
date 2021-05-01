from abc import ABC, abstractmethod

from doggos.fuzzy_sets.fuzzy_set import MembershipDegree


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
    def t_norm(a: MembershipDegree, b: MembershipDegree) -> MembershipDegree:
        pass

    @staticmethod
    @abstractmethod
    def s_norm(a: MembershipDegree, b: MembershipDegree) -> MembershipDegree:
        pass

    @staticmethod
    @abstractmethod
    def negation(a: MembershipDegree) -> MembershipDegree:
        pass

    @staticmethod
    @abstractmethod
    def implication(a: MembershipDegree, b: MembershipDegree) -> MembershipDegree:
        pass
