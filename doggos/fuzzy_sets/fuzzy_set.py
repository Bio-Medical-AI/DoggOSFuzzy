from abc import ABC, abstractmethod
from typing import NewType


MembershipDegree = NewType('MembershipDegree', None)


class FuzzySet(ABC):

    @abstractmethod
    def __call__(self, x: float) -> MembershipDegree:
        pass

