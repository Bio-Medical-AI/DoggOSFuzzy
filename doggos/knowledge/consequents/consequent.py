from abc import ABC, abstractmethod
from typing import List

from doggos.fuzzy_sets.membership.membership_degree import MembershipDegree


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
    def output(self, rule_firing: MembershipDegree) -> List[MembershipDegree] or float:
        pass
