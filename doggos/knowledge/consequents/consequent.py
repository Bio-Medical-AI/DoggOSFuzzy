from abc import ABC, abstractmethod
from typing import List, Dict

from doggos.fuzzy_sets.membership.membership_degree import MembershipDegree


class Consequent(ABC):

    @abstractmethod
    def output(self, consequent_input: MembershipDegree or Dict[str, float]) -> List[MembershipDegree] or float:
        pass
