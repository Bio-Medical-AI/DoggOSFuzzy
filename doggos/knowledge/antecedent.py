from typing import Dict, Sequence
from doggos.algebras.algebra import Algebra
from doggos.fuzzy_sets.membership.membership_degree import MembershipDegree
from doggos.knowledge import Clause
from abc import ABC


class Antecedent(ABC):
    """
    Class representing antecedent:
    https://en.wikipedia.org/wiki/Fuzzy_set

    Attributes
    --------------------------------------------
    __clauses : Sequence[Clause]
        sequence of clauses
    __algebra : Algebra
        algebra provides t-norm and s-norm

    Methods
    --------------------------------------------
    firing(self, features: Dict[str, float]) -> MembershipDegree:
        returns
    
    """
    def __init__(self, clauses: Sequence[Clause], algebra: Algebra):
        pass

    def firing(self, features: Dict[Clause, MembershipDegree]) -> MembershipDegree:
        pass


class TermAntecedent(Antecedent):
    
    def __init__(self, clause: Clause, algebra: Algebra):
        self.__clause = clause
        self.__algebra = algebra
        self.fire = lambda dict_: dict_[clause]

    def __and__(self, clause: Clause) -> TermAntecedent:
        return self.__class__()
    
    def __or__(self, clause: Clause) -> TermAntecedent:
        pass