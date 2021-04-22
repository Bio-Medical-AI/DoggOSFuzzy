from abc import ABC, abstractmethod
from typing import Tuple


class FuzzySet(ABC):

    @abstractmethod
    def __call__(self, x: float) -> Tuple[float, ...] or float:
        pass

