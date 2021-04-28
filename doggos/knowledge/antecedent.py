from typing import Dict, Sequence
from doggos.algebras.algebra import Algebra
from doggos.fuzzy_sets.MembershipDegree.membership_degree import MembershipDegree
from doggos.knowledge import Clause


class Antecedent:
    def __init__(self, clauses: Sequence[Clause], algebra: Algebra):
        self.__clauses = clauses
        self.__algebra = algebra

    def firing(self, features: Dict[str, float]) -> MembershipDegree:
        pass
