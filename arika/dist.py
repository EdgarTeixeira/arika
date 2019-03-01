import math
from typing import Callable

from scipy import stats


def uniform_float(min: float, max: float) -> Callable[[], float]:
    def sampler():
        return stats.uniform(min, max).rvs()

    return sampler


def loguniform_float(min: float, max: float) -> Callable[[], float]:
    def sampler():
        return math.exp(stats.uniform(math.log(min), math.log(max)).rvs())

    return sampler


def uniform_int(min: int, max: int) -> Callable[[], int]:
    def sampler():
        return stats.randint(min, max).rvs()

    return sampler
