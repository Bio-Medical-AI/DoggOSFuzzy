from abc import ABC, abstractmethod
from typing import Tuple, List

from doggos.fuzzy_sets.degree.membership_degree import MembershipDegree


class Consequent(ABC):

    @abstractmethod
    def output(self, rule_firing: MembershipDegree) -> Tuple[List[float]] or List[float]:
        pass
