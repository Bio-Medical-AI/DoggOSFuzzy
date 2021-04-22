from doggos.fuzzy_sets.fuzzy_set import FuzzySet


from typing import Tuple


class Clause:

    __lingustic_variable: str
    __gradiation_adjectice: str
    __fuzzy_set: FuzzySet

    def get_value(self, x: float) -> Tuple[float, ...] or float:
        pass
