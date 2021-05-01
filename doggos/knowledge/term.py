from __future__ import annotations

from typing import Dict, Sequence, Callable, NoReturn
from doggos.algebras.algebra import Algebra
from doggos.fuzzy_sets.membership.membership_degree import MembershipDegree
from doggos.knowledge.clause import Clause
from doggos.knowledge.antecedent import Antecedent


class Term(Antecedent):
    """
    Class representing an anteceden with recursive firing value computation:
    https://en.wikipedia.org/wiki/Fuzzy_set

    Attributes
    --------------------------------------------
    __clause : Clause
        clause which is stored in antecedent
    __algebra : Algebra
        algebra provides t-norm and s-norm

    Methods
    --------------------------------------------
    def fire(self, clause_dict: Dict[Clause, MembershipDegree]) -> MembershipDegree
        returns a firing value of the antecedent

    Examples
    --------------------------------------------
    TODO

    """

    def __init__(self, algebra: Algebra, clause: Clause or None = None):
        """
        Creates Term object with given algebra and clause.
        :param algebra: algebra provides t-norm and s-norm
        :param clause: provides a linguistic variable with corresponding fuzzy set
        """
        super().__init__(algebra)
        if clause is not None:
            self.__fire = lambda dict_: dict_[clause]


    @property
    def fire(self) -> Callable[[Dict[Clause, MembershipDegree]], MembershipDegree]:
        return self.__fire

    @fire.setter
    def fire(self, fire: Callable[[Dict[Clause, MembershipDegree]], MembershipDegree]):
        self.__fire = fire

    def __and__(self, term: Term) -> Term:
        new_term = self.__class__(self.algebra)
        new_term.fire = lambda dict_: self.algebra.t_norm(self.fire(dict_), term.fire(dict_))
        return new_term

    def __or__(self, term: Term) -> Term:
        new_term = self.__class__(self.algebra)
        new_term.fire = lambda dict_: self.algebra.s_norm(self.fire(dict_), term.fire(dict_))
        return new_term
