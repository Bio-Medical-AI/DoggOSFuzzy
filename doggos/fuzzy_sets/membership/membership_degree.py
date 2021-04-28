from abc import ABC
from dataclasses import dataclass
from typing import Any


@dataclass
class MembershipDegree(ABC):
    value: Any
