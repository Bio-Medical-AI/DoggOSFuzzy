from typing import Callable, NoReturn


from doggos.fuzzy_sets.fuzzy_set import FuzzySet
from doggos.fuzzy_sets.membership.membership_degree_t2 import MembershipDegreeT2


class T2FuzzySet(FuzzySet):

    __umf: Callable
    __lmf: Callable
    __proba_distribution: Callable

    def __call__(self, x: float) -> MembershipDegreeT2:
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

    @property
    def proba_distribution(self) -> Callable:
        pass

    @proba_distribution.setter
    def proba_distribution(self, x: Callable) -> NoReturn:
        pass
