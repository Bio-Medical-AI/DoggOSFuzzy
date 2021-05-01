from doggos.fuzzy_sets.fuzzy_set import FuzzySet, MembershipDegree
from doggos.knowledge.linguistic_variable import LinguisticVariable

import numpy as np
from typing import NoReturn, Sequence


class Clause:
    """
    Class representing a clause.
    Clause is a pair of a linguistic variable and a fuzzy set:
    https://en.wikipedia.org/wiki/Fuzzy_set
    
    Attributes
    --------------------------------------------
    __linguistic_variable : LinguisticVariable
        linguistic variable, provides a name for a feature
        and determines the domain of a fuzzy set
    __gradation_adjective : str
        gradation adjective, string representation of belonging level
    __fuzzy_set : FuzzySet
        fuzzy set provides its membership function
        
    Methods
    --------------------------------------------
    get_value(self, x: float) -> MembershipDegree
        returns a value representing membership degree
        
    Examples
    --------------------------------------------
    >>> domain = Domain(0,10,0.01)
    >>> ling_var = LinguisticVariable('Temperature', domain)
    >>> f_set = T1FuzzySet(lambda x: 0.05*x)
    >>> clause = Clause(ling_var, 'Medium', f_set)
    >>> clause.get_value(2)
    0.1
    
    """

    __linguistic_variable: LinguisticVariable
    __gradation_adjective: str
    __fuzzy_set: FuzzySet
    
    def __init__(self, linguistic_variable: LinguisticVariable, gradation_adjective: str, fuzzy_set: FuzzySet):
        """
        Creates clause with given linguistic variable, gradation adjective and fuzzy set.

        :param linguistic_variable: linguistic variable, provides a name for a feature
                                    and determines the domain of a fuzzy set
        :param gradation_adjective: gradation adjective, string representation of belonging level
        :param fuzzy_set: fuzzy set provides its memebership function
        """
        if not isinstance(linguistic_variable, LinguisticVariable):
            raise TypeError("Linguistic variable must be a LinguisticVariable type")
        if not isinstance(fuzzy_set, FuzzySet):
            raise TypeError("Fuzzy set must be a FuzzySet type")
        
        self.__linguistic_variable = linguistic_variable
        self.__gradation_adjective = gradation_adjective
        self.__fuzzy_set = fuzzy_set
        self.__values = self._calculate_values()

    def get_value(self, x: Sequence[float] or float) -> MembershipDegree or Sequence[MembershipDegree]:
        """
        returns a value representing membership degree
        :param x: degree of belonging
        """
        return self.__values[self._find_index(x)]
    
    def _calculate_values(self) -> Sequence[MembershipDegree]:
        """
        Calculates values for every element in the domain
        :return: array of membership degrees
        """
        return self.__fuzzy_set(self.linguistic_variable.domain())
    
    def _find_index(self, x: Sequence[float] or float) -> int or Sequence[int]:
        """
        Returns the index of given x in the values table 
        
        :param x: value in the domain
        :return: index
        """
        if np.any(x > self.linguistic_variable.domain.max) or np.any(x < self.linguistic_variable.domain.min):
            raise ValueError('There is no such value in the domain')
        return np.round((x - self.linguistic_variable.domain.min)/self.linguistic_variable.domain.precision).astype(int)
    
    @property
    def linguistic_variable(self) -> LinguisticVariable:
        """
        Returns the linguistic variable.

        :return: linguistic variable
        """
        return self.__linguistic_variable
    
    @linguistic_variable.setter
    def linguistic_variable(self, linguistic_variable: LinguisticVariable) -> NoReturn:
        """
        Sets new linguistic variable.

        :param linguistic_variable: linguistic variable
        :return: NoReturn
        """
        if not isinstance(linguistic_variable, LinguisticVariable):
            raise TypeError('Linguistic variable must be a LinguisticVariable type')
        
        self.__linguistic_variable = linguistic_variable

    @property
    def fuzzy_set(self) -> FuzzySet:
        """
        Returns the fuzzy set.

        :return: fuzzy set
        """
        return self.__fuzzy_set
    
    @fuzzy_set.setter
    def fuzzy_set(self, fuzzy_set: FuzzySet) -> NoReturn:
        """
        Sets new fuzzy set.

        :param fuzzy_set: fuzzy set
        :return: NoReturn
        """
        if not isinstance(fuzzy_set, FuzzySet):
            raise TypeError("Fuzzy set must be a FuzzySet type")
        self.__fuzzy_set = fuzzy_set
    
    @property
    def gradation_adjective(self) -> str:
        """
        Return the gradation adjective.

        :return: gradation adjective
        """
        return self.__gradation_adjective
    
    @gradation_adjective.setter
    def gradation_adjective(self, gradation_adjective: str) -> NoReturn:
        """
        Sets new gradation adjective.

        :param gradation_adjective: gradation adjective
        :return: NoReturn
        """
        self.__gradation_adjective = gradation_adjective
