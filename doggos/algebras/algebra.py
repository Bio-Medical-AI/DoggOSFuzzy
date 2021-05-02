from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np

from doggos.fuzzy_sets.fuzzy_set import MembershipDegree


def validate_input(function):
    """
    Decorator for algebra functions, to validate shape of the input
    :param function:
    :return:
    """
    def operation(a, b):
        if isinstance(a, Iterable):
            if not isinstance(a, np.ndarray):
                a = np.array(a)
            size_a = a.shape[0]
        else:
            size_a = 1
        if isinstance(b, Iterable):
            if not isinstance(b, np.ndarray):
                b = np.array(b)
            size_b = b.shape[0]
        else:
            size_b = 1
        if size_a != size_b:
            raise ValueError(f'Dimensions {size_a} and {size_b} are not compatible')
        return function(a, b)
    return operation


def expand_negation_argument(function):
    def operation(a):
        if isinstance(a, Iterable):
            a = np.array(a)
        return function(a)
    return operation


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
