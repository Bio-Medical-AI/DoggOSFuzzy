from typing import Dict, Tuple, Sequence
from doggos.algebras.algebra import Algebra
from doggos.knowledge import Clause


class Antecedent:
    """
     Class representing antecedent:
     https://en.wikipedia.org/wiki/Fuzzy_set
     ``
     Attributes
    --------------------------------------------
    __clauses : Clause
        clause, linguistic variable with corresponding fuzzy set
    __algebta : Algebra
        algebra provides t-norm and s-norm
        
    Methods
    --------------------------------------------
    calculate_firing_interval(self, features: Dict[str, float]) -> Tuple[float, ...] or float
    
    Examples
    --------------------------------------------
    >>>
    >>>
    >>>
    """
    def __init__(self, clauses: Sequence[Clause], algebra: Algebra):        
        self.__clauses = clauses
        self.__algebra = algebra

    def calculate_firing_interval(self, features: Dict[str, float]) -> Tuple[float, ...] or float:
        """
        Method provides firing calculation.

        :param features:
        :return: firing interval
        """
        pass
