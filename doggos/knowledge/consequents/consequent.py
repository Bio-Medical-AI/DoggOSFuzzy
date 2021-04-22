from abc import ABC, abstractmethod
from typing import Tuple


class Consequent(ABC):

    @abstractmethod
    def calculate_cut(self) -> Tuple[float, ...] or float:
        pass
