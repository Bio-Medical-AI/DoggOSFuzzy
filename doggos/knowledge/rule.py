from typing import Dict, Tuple


class Rule:

    __fireing_value: Tuple[float, ...]
    __rule_output: float

    def calculate_rule(self, features: Dict[str, float]) -> Tuple[float, ...] or float:
        pass
