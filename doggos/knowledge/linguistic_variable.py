from typing import Sequence, Tuple


class LinguisticVariable:

    __name: str
    __domain: Sequence[Tuple[float, ...] or float]

    def __call__(self, x: float) -> Tuple[float, ...] or float:
        pass
