from abc import ABC, abstractmethod


class Algebra(ABC):
    """
    Klasa reprezentująca algebrę dla konkretnej logiki rozmytej.
    Np. algebra Lukasiewicza, algebra Gödel.
    Każda algebra zawiera następujące działania:
    - T-norma: uogólnienie iloczynu (AND)
    - S-norma: uogólnienie sumy (OR)
    - Negacja: zanegowana wartość logiczna
    - Implikacja:
    """

    @abstractmethod
    def t_norm(self, a: float, b: float) -> float:
        pass

    @abstractmethod
    def s_norm(self, a: float, b: float) -> float:
        pass

    @abstractmethod
    def negation(self, a) -> float:
        pass

    @abstractmethod
    def implication(self, a, b) -> float:
        pass
