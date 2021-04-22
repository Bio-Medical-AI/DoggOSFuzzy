from typing import Tuple

from doggos.knowledge.consequents.consequent import Consequent


class TakagiSugenoConsequent(Consequent):
    def calculate_cut(self) -> Tuple[float, ...] or float:
        pass
