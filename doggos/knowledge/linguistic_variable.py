from typing import Sequence, Tuple


class Domain:
    def __init__(self, min_value: float, max_value: float, precision: float):
        self.__min_value = min_value
        self.__max_value = max_value
        self.__precision = precision

    @property
    def precision(self):
        pass

    @property
    def domain(self) -> Sequence[float]:
        """
        Creates sequence matching given range and precision
        :return:
        """
        pass

    @property
    def min(self):
        pass

    @property
    def max(self):
        pass


class LinguisticVariable:
    def __init__(self, name: str, domain: Domain):
        self.__name = name
        self.__domain = domain

    def __call__(self, x: float) -> Tuple[float, ...] or float:
        pass



