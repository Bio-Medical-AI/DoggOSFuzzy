from typing import Callable, NoReturn


from doggos.fuzzy_sets.fuzzy_set import FuzzySet


class T1FuzzySet(FuzzySet):
    """
    Class used to represent a fuzzy set type:
    https://en.wikipedia.org/wiki/Fuzzy_set

    Attributes
    --------------------------------------------
    __mf : Callable[[float], float]
        membership function, determines the degree of belonging to a fuzzy set

    Methods
    --------------------------------------------
    __call__(x: float) -> float
        calculate the degree of belonging to a fuzzy set of an element

    Examples:
    --------------------------------------------
    Creating simple fuzzy set type I and calculate degree of belonging for 2
    >>> fuzzy_set = T1FuzzySet(lambda x: 0 if x < 0 else 1)
    >>> fuzzy_set(2)
    1

    Creating fuzzy set type I using numpy functions
    >>> def sigmoid(x):
    ...    return 1 / (1 + np.exp(-x))
    ...
    >>> fuzzy_set = T1FuzzySet(sigmoid)
    >>> fuzzy_set(2.5)
    0.9241
    """

    __mf: Callable[[float], float]

    def __init__(self, mf: Callable[[float], float]):
        """
        Create fuzzy set with given membership function
        :param mf: membership function of a set
        """
        if not callable(mf):
            raise ValueError('Membership function must be callable')
        self.__mf = mf

    def __call__(self, x: float) -> float:
        """
        Calculate the degree of belonging to a fuzzy set for of an element
        :param x: element of domain
        :return: degree of belonging of an element
        """
        return self.__mf(x)

    @property
    def mf(self) -> Callable[[float], float]:
        """
        Returns the membership function
        :return:
        """
        return self.__mf

    @mf.setter
    def mf(self, mf: Callable[[float], float]) -> NoReturn:
        """
        Sets new membership function
        :param mf: new membership function
        :return: NoReturn
        """
        if not callable(mf):
            raise ValueError('Membership function must be callable')
        self.__mf = mf
