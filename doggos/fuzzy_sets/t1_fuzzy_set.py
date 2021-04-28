from typing import Callable, NoReturn


from doggos.fuzzy_sets.fuzzy_set import FuzzySet
from doggos.fuzzy_sets.membership.membership_degree_t1 import MembershipDegreeT1


class T1FuzzySet(FuzzySet):

    __mf: Callable

    def __call__(self, x: float) -> MembershipDegreeT1:
        pass

    @property
    def mf(self) -> Callable:
        pass

    @mf.setter
    def mf(self, mf: Callable) -> NoReturn:
        pass
