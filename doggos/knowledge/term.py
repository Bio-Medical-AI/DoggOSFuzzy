from __future__ import annotations

from typing import Dict, Sequence, Callable, NoReturn
from doggos.algebras.algebra import Algebra
from doggos.fuzzy_sets.fuzzy_set import MembershipDegree
from doggos.knowledge.clause import Clause
from doggos.knowledge.antecedent import Antecedent

from functools import partial


class Term(Antecedent):
    """
    Class representing an antecedent with recursive firing value computation:
    https://en.wikipedia.org/wiki/Fuzzy_set

    Attributes
    --------------------------------------------
    __algebra : Algebra
        algebra provides t-norm and s-norm

    Methods
    --------------------------------------------
    def fire(self) -> Callable[[Dict[Clause, MembershipDegree]], MembershipDegree]
        returns a firing value of the antecedent

    Examples
    --------------------------------------------
    TODO
    """

    def __init__(self, algebra: Algebra, clause: Clause = None, name: str = None):
        """
        Creates Term object with given algebra and clause.

        :param algebra: algebra provides t-norm and s-norm
        :param clause: provides a linguistic variable with corresponding fuzzy set
        :param name: name of the term
        """
        super().__init__(algebra)

        if not clause:
            self.__fire = None
            if name is None:
                self.name = ''
            else:
                self.name = name
        else:
            if name is None:
                self.name = clause.linguistic_variable.name + '_' + clause.gradation_adjective
            else:
                self.name = name
            self.__fire = partial(self.dict_clause, clause=clause)

    def dict_clause(self, dict_, clause):
        return dict_[clause]

    @property
    def fire(self) -> Callable[[Dict[Clause, MembershipDegree]], MembershipDegree]:
        """
        Returns the firing function.

        :return: firing function
        """
        return self.__fire

    @fire.setter
    def fire(self, fire: Callable[[Dict[Clause, MembershipDegree]], MembershipDegree]):
        """
        Sets new firing function to the antecedent.

        :param fire: firing function
        """
        self.__fire = fire

    def __and__(self, other: Term) -> Term:
        """
        Creates new antecedent object and sets new firing function which uses t-norm.

        :param other: other term
        :return: term
        """
        new_term = self.__class__(self.algebra, name=self.name + ' & ' + other.name)
        # Sprawdzić czy działa 01.02.2022
        new_term.fire = partial(self.apply_dict_t_norm, other=other)
        return new_term

    def __or__(self, other: Term) -> Term:
        """
        Creates new antecedent object and sets new firing function which uses s-norm.

        :param other: other term
        :return: term
        """
        new_term = self.__class__(self.algebra, name=self.name + ' | ' + other.name)
        # Sprawdzić czy działa 01.02.2022
        new_term.fire = partial(self.apply_dict_s_norm, other=other)
        return new_term

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    def apply_dict_t_norm(self, dict_, other):
        return self.algebra.t_norm(self.fire(dict_), other.fire(dict_))

    def apply_dict_s_norm(self, dict_, other):
        return self.algebra.s_norm(self.fire(dict_), other.fire(dict_))
