from typing import Tuple
from dataclasses import dataclass

from doggos.fuzzy_sets.membership.membership_degree import MembershipDegree


@dataclass
class MembershipDegreeIT2(MembershipDegree):
    value: Tuple[float, float]
