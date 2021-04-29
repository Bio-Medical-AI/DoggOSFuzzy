from abc import ABC, abstractmethod

from doggos.fuzzy_sets.membership.membership_degree import MembershipDegree


class FuzzySet(ABC):

    @abstractmethod
    def __call__(self, x: float) -> MembershipDegree:
        pass

