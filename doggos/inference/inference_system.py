from abc import ABC, abstractmethod
from typing import List


from doggos.knowledge import Rule


class InferenceSystem(ABC):

    __rule_base: List[Rule]

    @abstractmethod
    def output(self) -> float:
        pass
