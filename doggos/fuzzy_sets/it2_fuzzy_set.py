from typing import Callable, NoReturn


from doggos.fuzzy_sets.fuzzy_set import FuzzySet
from doggos.fuzzy_sets.membership.membership_degree_it2 import MembershipDegreeIT2


class IT2FuzzySet(FuzzySet):
    __umf: Callable
    __lmf: Callable

    def __call__(self, x: float) -> MembershipDegreeIT2:
        pass

    @property
    def umf(self) -> Callable:
        pass

    @umf.setter
    def umf(self, x: Callable) -> NoReturn:
        pass

    @property
    def lmf(self) -> Callable:
        pass

    @lmf.setter
    def lmf(self, x: Callable) -> NoReturn:
        pass
