from abc import ABC, abstractmethod
from typing import List, Dict


from doggos.knowledge import Rule


class InferenceSystem(ABC):

    __rule_base: List[Rule]

    @abstractmethod
    def calculate_output(self, features: Dict[str, float], method: str) -> float:
        pass
