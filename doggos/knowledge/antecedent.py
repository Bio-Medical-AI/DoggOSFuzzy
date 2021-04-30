from typing import Dict, Sequence
from doggos.algebras.algebra import Algebra
from doggos.fuzzy_sets.membership.membership_degree import MembershipDegree
from doggos.knowledge.clause import Clause


class Antecedent:
    def __init__(self, clauses: Sequence[Clause], algebra: Algebra):
        self.__clauses = clauses
        self.__algebra = algebra

    def firing(self, features: Dict[str, float]) -> MembershipDegree:
        pass
