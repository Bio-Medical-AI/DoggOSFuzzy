from typing import Callable, NoReturn


from doggos.fuzzy_sets.fuzzy_set import FuzzySet


class T1FuzzySet(FuzzySet):

    __mf: Callable

    def __call__(self, x: float) -> float:
        pass

    @property
    def mf(self) -> Callable:
        pass

    @mf.setter
    def mf(self, mf: Callable) -> NoReturn:
        pass
