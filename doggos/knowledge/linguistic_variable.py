from typing import Sequence, Tuple, NoReturn

import numpy as np


class Domain:
    """
    Class representing a domain:
    https://en.wikipedia.org/wiki/Domain_of_a_function
    
    Attributes
    --------------------------------------------
    __min_value : float
        minimum value in domain
    __max_value : float
        maximum value in domain
    __precision : float
        precision of the domain
            
    Methods
    --------------------------------------------
    domain -> Sequence[float]
        Returns sequence from assigned intervals and precision
    """
    def __init__(self, min_value: float, max_value: float, precision: float):
        """
        Creates a domain.

        :param min_value: minimum value in domain
        :param max_value: maximum value in domain
        :param precision: step
        """
        self.__min_value = min_value
        self.__max_value = max_value
        self.__precision = precision

    @property
    def precision(self) -> float:
        """
        Return domain's precision.

        :return: precision
        """
        return self.__precision

    def __call__(self) -> Sequence[float]:
        """
        Creates sequence matching given range and precision
        :return: domain as sequence of floats
        """
        return np.arange(self.min, self.max, self.precision)

    @property
    def min(self) -> float:
        """
        Return minimum value in domain.

        :return: minimum
        """
        return self.__min_value

    @property
    def max(self) -> float:
        """
        Returns maximum value in domain.

        :return: maximum
        """
        return self.__max_value


class LinguisticVariable:
    """
    Class representing a linguistic variable.
    Linguistic variable is a measurable fragment of reality:
    https://en.wikipedia.org/wiki/Fuzzy_set
    
    Attributes
    --------------------------------------------
    __name : str
        The name of reality fragment
    __domain : Domain
        The domain     
    
    """
    def __init__(self, name: str, domain: Domain):
        """
        Creates linguistic variable with given name and domain

        :param name: name of linguistic variable
        :param domain: domain 
        """
        if not isinstance(domain, Domain):
            raise TypeError('Linguistic variable requires the domain to be a Domain type')
        self.__name = name
        self.__domain = domain

    @property
    def domain(self) -> Domain:
        """
        Returns the domain

        :return: domain of the linguistic variable
        """
        return self.__domain
