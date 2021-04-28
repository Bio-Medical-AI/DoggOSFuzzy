from dataclasses import dataclass

from doggos.fuzzy_sets.membership.membership_degree import MembershipDegree


@dataclass
class MembershipDegreeT1(MembershipDegree):
    value: float
