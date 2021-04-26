from abc import ABC, abstractmethod
from typing import Tuple, List


class Consequent(ABC):

    @abstractmethod
    def calculate_cut(self, rule_firing: Tuple[float, ...] or float) -> Tuple[List[float]] or List[float]:
        pass
