from abc import ABC, abstractmethod
from typing import List

from doggos.fuzzy_sets.membership.membership_degree import MembershipDegree



class Consequent(ABC):

    @abstractmethod
    def output(self, rule_firing: MembershipDegree) -> List[MembershipDegree] or float:
        pass
