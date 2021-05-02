from abc import ABC, abstractmethod
from typing import NewType


ConsequentOutput = NewType('ConsequentOutput', None)
"""
"""


class Consequent(ABC):
    """
    Base class for Consequents of Fuzzy Rules.
    https://en.wikipedia.org/wiki/Fuzzy_rule

    Methods
    --------------------------------------------
    output(*args) -> ConsequentOutput
        cut fuzzy set provided by Clause to rule firing level
    """
    @abstractmethod
    def output(self, *args) -> ConsequentOutput:
        pass
