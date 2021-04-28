from abc import ABC
from dataclasses import dataclass
from typing import Any, Tuple


@dataclass
class MembershipDegree(ABC):
    value: Any


@dataclass
class MembershipDegreeIT2(MembershipDegree):
    value: Tuple[float, float]


@dataclass
class MembershipDegreeT1(MembershipDegree):
    value: float


@dataclass
class MembershipDegreeT2(MembershipDegree):
    pass
