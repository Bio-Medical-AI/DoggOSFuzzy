from typing import Tuple

from doggos.fuzzy_sets.degree.membership_degree import MembershipDegree


class MembershipDegreeIT2(MembershipDegree):
    def __init__(self, value: Tuple[float, float]):
        self.value = value
