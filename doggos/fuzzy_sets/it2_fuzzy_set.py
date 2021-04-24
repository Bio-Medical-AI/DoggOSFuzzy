from __future__ import annotations
from typing import Callable, Tuple, NoReturn


from doggos.fuzzy_sets.fuzzy_set import FuzzySet


class IT2FuzzySet(FuzzySet):
    """
    Class used to represent a fuzzy set type II :
    https://en.wikipedia.org/wiki/Fuzzy_set

    Attributes
    --------------------------------------------
    __umf : Callable[[float], float]
        upper membership function, determines the upper degree of belonging to a fuzzy set
    __lmf: Callable[[float], float]
        lower membership function, determines the lower degree of belonging to a fuzzy set

    Methods
    --------------------------------------------
    __call__(x: float) -> Tuple[float, float]
        calculate the degree of belonging to a fuzzy set of an element

    Examples:
    --------------------------------------------
    Creating simple fuzzy set type II and calculate degree of belonging
    >>> fuzzy_set = IT2FuzzySet(lambda x: 0 if x < 0 else 0.2, lambda x: 0 if x < 0 else 1)
    >>> fuzzy_set(2)
    (0.2, 1)

    Creating fuzzy set type II using numpy functions
    >>> import numpy as np
    >>> def f1(x):
    ...    return 1 / (1 + np.exp(-x))
    ...
    >>>def f2(x):
    ...    return 1
    ...
    >>> fuzzy_set = IT2FuzzySet(f1, f2)
    >>> fuzzy_set(2.5)
    (0.9241, 1)
    """
    __umf: Callable[[float], float]
    __lmf: Callable[[float], float]

    def __init__(self, lmf: Callable[[float], float], umf: Callable[[float], float]):
        """
        Create fuzzy set type II with given lower membership function and upper membership function.
        Both functions should return values from range [0, 1].
        IMPORTANT:
        Lower membership function should return  lesser values than upper membership function.
        This is not validated in constructor, but it will raise an exception if you try to call a fuzzy set with
        incorrect functions.
        :param umf: upper membership function of a set
        :param lmf: lower membership function of a set
        """
        if not callable(umf) or not callable(lmf):
            raise ValueError('Membership functions should be callable')
        self.__umf = umf
        self.__lmf = lmf

    def __call__(self, x: float) -> Tuple[float, float]:
        """
        Calculate the degree of belonging (a, b), raises an exception if a > b
        :param x: element of domain
        :return: degree of belonging of an element as tuple (lmf(x), umf(x))
        """
        a, b = self.__lmf(x), self.__umf(x)
        if a > b:
            raise ValueError('Lower membership function return higher value than upper membership function.')
        return a, b

    @property
    def umf(self) -> Callable[[float], float]:
        """
        Getter of the upper membership function
        :return: upper membership function
        """
        return self.__umf

    @umf.setter
    def umf(self, umf: Callable[[float], float]) -> NoReturn:
        """
        Setter of the upper membership function
        :param umf: new upper membership function, must be callable
        :return: NoReturn
        """
        if not callable(umf):
            raise ValueError('Membership function should be callable')
        self.__umf = umf

    @property
    def lmf(self) -> Callable[[float], float]:
        """
        Getter of the lower membership function
        :return: lower membership function
        """
        return self.__lmf

    @lmf.setter
    def lmf(self, lmf: Callable[[float], float]) -> NoReturn:
        """
        Setter of the lower membership function
        :param lmf: new lower membership function, must be callable
        :return: NoReturn
        """
        if not callable(lmf):
            raise ValueError('Membership function should be callable')
        self.__lmf = lmf
