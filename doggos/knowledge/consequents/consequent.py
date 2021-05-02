from abc import ABC, abstractmethod
from typing import List, Dict

from doggos.fuzzy_sets.fuzzy_set import MembershipDegree
from doggos.knowledge import LinguisticVariable


class Consequent(ABC):

    @abstractmethod
    def output(self, consequent_input: MembershipDegree or Dict[LinguisticVariable, float]) -> List[MembershipDegree] or float:
        pass
