from typing import Dict, Sequence, Callable, NoReturn
from doggos.algebras.algebra import Algebra
from doggos.fuzzy_sets.membership.membership_degree import MembershipDegree
from doggos.knowledge.clause import Clause

from abc import ABC, abstractmethod


class Antecedent(ABC):
    """
    Base class for representing an antecedent:
    https://en.wikipedia.org/wiki/Fuzzy_set
    
    """
    @abstractmethod
    def fire(self, clause_dict: Dict[Clause, MembershipDegree]) -> MembershipDegree:
        pass


