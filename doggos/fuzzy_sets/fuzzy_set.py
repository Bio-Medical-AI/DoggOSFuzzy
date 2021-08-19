from __future__ import annotations
from abc import ABC, abstractmethod
from typing import NewType, Iterable, Tuple, Sequence, AnyStr, Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

MembershipDegree = NewType('MembershipDegree', np.ndarray)
"""
MembershipDegree is a type hint for membership degree of some crisp value to a certain fuzzy set.\n 
Membership degree type varies depending on type of fuzzy set used.\n
Therefore we use one type hint for all types for fuzzy sets, so treat it only as a hint, its not a type.\n
E.g.:\n
For type one fuzzy set membership degree is a float.\n
For type two fuzzy set membership degree is a Tuple[float, float]\n
For other types of fuzzy sets is a membership degree defined by user.
"""

DEFAULT_COLOR_SET = list(colors.TABLEAU_COLORS.keys())


def _parse_anystr(item: AnyStr or Sequence[AnyStr], takes: int, default: Callable[[], str]) -> Sequence[AnyStr]:
    if isinstance(item, Sequence) and not isinstance(item, str):
        shift = 0 if len(item) >= takes else abs(takes - len(item))
        res = list(item[:min(len(item), takes)])
        res.extend([default() for _ in range(shift)])
    else:
        res = [item]
        res.extend([default() for _ in range(takes - 1)])
    return res


def _mf_return(fuzzy_set: FuzzySet, domain: Iterable[float or int]) -> (Iterable[Iterable[float]], int):
    mf_res = fuzzy_set(domain)
    if mf_res.ndim > 1:
        takes = mf_res.shape[-1]
    else:
        takes = 1
        mf_res = [mf_res]
    return mf_res, takes



class FuzzySet(ABC):

    @abstractmethod
    def __call__(self, x: float or Iterable[float]) -> MembershipDegree:
        pass

    def plot(self,
             axis: plt.axis or None = None,
             domain: Iterable[float] = np.arange(0, 1.0001, 0.0001),
             y_range: Tuple[float] or None = None,
             grid: bool = True,
             title: str or None = None,
             color: AnyStr or Sequence[AnyStr] = DEFAULT_COLOR_SET,
             label: AnyStr or Sequence[AnyStr] = None,
             xlabel: str = 'Input value',
             ylabel: str = 'Membership function value',
             **kwargs):
        if not axis:
            fig, axis = plt.subplots(**kwargs)
        if title:
            axis.set_title(title)

        mf_return, takes = _mf_return(self, domain)
        label = _parse_anystr(label, takes, default=lambda: "")
        color = _parse_anystr(color, takes, default=lambda: np.random.choice(DEFAULT_COLOR_SET))
        for mf, c, lab in zip(mf_return, color, label):
            axis.plot(
                domain,
                mf,
                c=c if c else "blue",
                label=lab
            )
        if y_range:
            axis.set_ylim(y_range)
        axis.set_ylabel(xlabel)
        axis.set_xlabel(ylabel)
        axis.grid(grid, ls='--')
        axis.legend(loc=1)
