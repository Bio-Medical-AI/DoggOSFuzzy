from typing import Callable, Tuple, NoReturn


from doggos.fuzzy_sets.fuzzy_set import FuzzySet


class IT2FuzzySet(FuzzySet):
    __umf: Callable
    __lmf: Callable

    def __call__(self, x: float) -> Tuple[float, float]:
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
