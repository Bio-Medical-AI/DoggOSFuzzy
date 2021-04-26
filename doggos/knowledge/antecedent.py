from typing import Dict, Tuple, Sequence
from doggos.algebras.algebra import Algebra
from doggos.knowledge import Clause


class Antecedent:
    def __init__(self, clauses: Sequence[Clause], algebra: Algebra):
        self.__clauses = clauses
        self.__algebra = algebra

    def calculate_firing_interval(self, features: Dict[str, float]) -> Tuple[float, ...] or float:
        pass
