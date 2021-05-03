from abc import ABC, abstractmethod
from collections import Iterable as Iter
from typing import Callable, Iterable

from doggos.knowledge import Rule


class InferenceSystem(ABC):
    __rule_base: Iterable[Rule]

    def __init__(self, rule_base: Iterable[Rule]):
        if not isinstance(rule_base, Iter) or any(not isinstance(rule, Rule) for rule in rule_base):
            raise TypeError('rule_base must be an iterable of type Rule')

        self.__rule_base = rule_base

    @abstractmethod
    def infer(self, defuzzification_method: Callable, *args) -> Iterable[float]:
        pass


