from abc import ABC, abstractmethod
from typing import List

from doggos.fuzzy_sets.membership.membership_degree import MembershipDegree


class Consequent(ABC):

    @abstractmethod
    def output(self, consequent_input: MembershipDegree or List[float]) -> List[MembershipDegree] or float:
        pass
