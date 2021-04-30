from typing import Dict, Sequence, Callable, NoReturn
from doggos.algebras.algebra import Algebra
from doggos.fuzzy_sets.membership.membership_degree import MembershipDegree
from doggos.knowledge import Clause
from abc import ABC


class Antecedent(ABC):
    """
    Class representing antecedent:
    https://en.wikipedia.org/wiki/Fuzzy_set
    
    """
    __algebra: Algebra
    __fire: Callable[[Dict[Clause, MembershipDegree]], MembershipDegree]


class TermAntecedent(Antecedent):
    
    def __init__(self, algebra: Algebra, clause: Clause = None):
        self.__clause = clause
        self.__algebra = algebra
        self.__fire = lambda dict_: dict_[clause]

    def __and__(self, term: TermAntecedent) -> TermAntecedent:
        rterm = self.__class__(self.algebra)
        rterm.fire = lambda dict_: self.algebra.t_norm(self.fire(dict_), term.fire(dict_))
        return rterm
    
    def __or__(self, term: TermAntecedent) -> TermAntecedent:
        rterm = self.__class__(self.algebra)
        rterm.fire = lambda dict_: self.algebra.s_norm(self.fire(dict_), term.fire(dict_))
        return rterm

    @property
    def clause(self) -> Clause:
        return self.__clause

    @clause.setter
    def clause(self, clause: Clause) -> NoReturn:
        self.__clause = clause

    @property
    def fire(self) -> Callable[[Dict[Clause, MembershipDegree]], MembershipDegree]:
        return self.__fire

    @fire.setter
    def fire(self, fire: Callable[[Dict[Clause, MembershipDegree]], MembershipDegree]) -> NoReturn:
        self.__fire = fire

    @property
    def algebra(self) -> Algebra:
        return self.__algebra

    @algebra.setter
    def algebra(self, algebra: Algebra):
        self.__algebra = algebra

