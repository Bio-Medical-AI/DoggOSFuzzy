from abc import ABC, abstractmethod
from typing import Tuple, List


class Consequent(ABC):
    """
    Base class for Consequents of Fuzzy Rules.
    https://en.wikipedia.org/wiki/Fuzzy_rule

    Methods
    --------------------------------------------
    calculate_cut(rule_firing: Tuple[float, ...] or float) -> Tuple[List[float]] or List[float]
        cut fuzzy set provided by Clause to rule firing level
    """
    @abstractmethod
    def calculate_cut(self, rule_firing: Tuple[float, ...] or float) -> Tuple[List[float]] or List[float]:
        pass
