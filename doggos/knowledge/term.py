from __future__ import annotations

from typing import Dict, Sequence, Callable, NoReturn
from doggos.algebras.algebra import Algebra
from doggos.fuzzy_sets.membership.membership_degree import MembershipDegree
from doggos.knowledge.clause import Clause
from doggos.knowledge.antecedent import Antecedent


class Term(Antecedent):
    """
    Class representing an antecedent with recursive firing value computation:
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

    def __init__(self, clause: Clause, algebra: Algebra):
        """
        Creates Term object with given algebra and clause.
        :param algebra: algebra provides t-norm and s-norm
        :param clause: provides a linguistic variable with corresponding fuzzy set
        """
        if not isinstance(clause, Clause):
            raise TypeError('clause must be a Clause type')
        if not isinstance(algebra, Algebra):
            raise TypeError('algebra must be a Algebra type')
        self.__clause = clause
        self.__algebra = algebra

    def fire(self, clause_dict: Dict[Clause, MembershipDegree]) -> MembershipDegree:
        return clause_dict[self.clause]

    def __and__(self, term: Term) -> Term:
        new_term = self.__class__(None, self.algebra)
        new_term.fire = lambda dict_: self.algebra.t_norm(self.fire(dict_), term.fire(dict_))
        return new_term

    def __or__(self, term: Term) -> Term:
        new_term = self.__class__(None, self.algebra)
        new_term.fire = lambda dict_: self.algebra.s_norm(self.fire(dict_), term.fire(dict_))
        return new_term

    @property
    def clause(self) -> Clause:
        """
        Getter of the clause
        :return: clause
        """
        return self.__clause

    @clause.setter
    def clause(self, clause: Clause) -> NoReturn:
        """
        Sets new clause to the antecedent
        :param clause: new clause
        :return: NoReturn
        """
        if not isinstance(clause, Clause):
            raise TypeError('clause must be a Clause type')
        self.__clause = clause

    @property
    def algebra(self) -> Algebra:
        """
        Getter of the algebra
        :return: algebra
        """
        return self.__algebra

    @algebra.setter
    def algebra(self, algebra: Algebra):
        """
        Sets new algebra to the antecedent
        :param algebra: new algebra
        :return: NoReturn
        """
        if not isinstance(algebra, Algebra):
            raise TypeError('algebra must be a Algebra type')
        self.__algebra = algebra
