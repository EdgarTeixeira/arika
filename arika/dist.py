import math
from typing import Callable

from numpy import random


def uniform_float(min_: float, max_: float) -> Callable[[], float]:
    def sampler():
        return random.uniform(min_, max_)
    return sampler


def loguniform_float(min_: float, max_: float) -> Callable[[], float]:
    def sampler():
        return math.exp(random.uniform(math.log(min_), math.log(max_)))
    return sampler


def uniform_int(min_: int, max_: int) -> Callable[[], int]:
    def sampler():
        return random.randint(min_, max_)
    return sampler
