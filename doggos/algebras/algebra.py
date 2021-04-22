from abc import ABC, abstractmethod


class Algebra(ABC):

    @abstractmethod
    def t_norm(self, left: float, right: float) -> float:
        pass

    @abstractmethod
    def s_norm(self, left: float, right: float) -> float:
        pass
