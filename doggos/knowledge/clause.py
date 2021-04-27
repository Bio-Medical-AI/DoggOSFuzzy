from typing import Tuple, NoReturn


from doggos.fuzzy_sets.fuzzy_set import FuzzySet
from doggos.knowledge.linguistic_variable import LinguisticVariable


class Clause:
    """
    Class representing a clause.
    Clause is a pair of linguistic variable and fuzzy set.
    https://en.wikipedia.org/wiki/Fuzzy_set
    
    Attributes
    --------------------------------------------
    __lingustic_variable : LinguisticVariable
        linguistic variable, provides name and determines the domain of a fuzzy set
    __gradiation_adjective : str
        gradiation adjective, string representation of belonging level
    __fuzzy_set : FuzzySet
        fuzzy set 
        
    Methods
    --------------------------------------------
    get_value(self, x: float) -> Tuple[float, ...] or float
        returns a value representing degree of belonging to a fuzzy set
        
    Examples
    --------------------------------------------
    >>> domain = Domain(0,10,0.01)
    >>> ling_var = LinguisticVariable('Temperature', domain)
    >>> f_set = T1FuzzySet(lambda x: 0 if x < 0 else 1)
    >>> clause = Clause(ling_var, 'Medium', f_set)
    
    """

    __lingustic_variable: LinguisticVariable
    __gradiation_adjective: str
    __fuzzy_set: FuzzySet
    
    def __init__(self, linguistic_variable: LinguisticVariable, gradiation_adjective: str, fuzzy_set: FuzzySet):
        """
        Creates clause with given linguistic variable, gradiation adjective and fuzzy set.

        :param linguistic_variable:
        :param gradiation_adjective:
        :param fuzzy_set:
        """
        if not isinstance(linguistic_variable) is LinguisticVariable:
            raise TypeError("Linguistic variable must be LingusticVariable type")
        if not isinstance(fuzzy_set) is FuzzySet:
            raise TypeError("Fuzzy set must be FuzzySet type")
        
        self.__lingustic_variable = linguistic_variable
        self.__gradiation_adjective = gradiation_adjective
        self.__fuzzy_set = fuzzy_set

    def get_value(self, x: float) -> Tuple[float, ...] or float:
        """
        Returns 
        :param x:
        """
        pass
    
    @property
    def linguistic_variable(self) -> LinguisticVariable:
        """
        Returns linguistic variable.

        :return: linguistic variable
        """
        return self.__lingustic_variable
    
    @linguistic_variable.setter
    def linguistic_variable(self, linguistic_variable: LinguisticVariable) -> NoReturn:
        """
        Sets new linguistic variable.

        :param linguistic_variable: linguistic variable
        :return: NoReturn
        """
        if not isinstance(linguistic_variable) is LinguisticVariable:
            raise TypeError('Linguistic variable must be LinguisticVariable type')
        
        self.__lingustic_variable = linguistic_variable

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
        if not isinstance(fuzzy_set) is FuzzySet:
            raise TypeError("Fuzzy set must be FuzzySet type")
        self.__fuzzy_set = fuzzy_set
    
    @property
    def gradiation_adjective(self) -> str:
        """
        Return the gradiation adjective.

        :return: gradiation adjective
        """
        return self.__gradiation_adjective
    
    @gradiation_adjective.setter
    def gradiation_adjective(self, gradiation_adjective: str) -> NoReturn:
        """
        Sets new gradiation adjective.

        :param gradiation_adjective: gradiation adjective
        :return: NoReturn
        """
        self.__gradiation_adjective = gradiation_adjective