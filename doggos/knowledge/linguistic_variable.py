from typing import Sequence, Tuple


class Domain:
    def __init__(self, domain: Sequence[Tuple[float, ...] or float], precision: float):
        self.__domain = domain
        self.__precision = precision

    @property
    def precision(self):
        pass

    @property
    def domain(self):
        pass

    def min(self):
        pass

    def max(self):
        pass


class LinguisticVariable:
    def __init__(self, name: str, domain: Domain):
        self.__name = name
        self.__domain = domain

    def __call__(self, x: float) -> Tuple[float, ...] or float:
        pass



