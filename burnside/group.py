
import operator
from abc import ABC, abstractmethod
from functools import reduce
from itertools import product

from .utils import permutation_types


class Group(ABC):
    def __init__(self):
        pass

    def __mul__(self, other):
        if not isinstance(other, Group):
            raise ValueError
        return Product(self, other)

    @abstractmethod
    def __iter__(self):
        pass


class Z(Group):
    """
    Cyclic group on {0, 1, ..., n-1}.
    """
    def __init__(self, order):
        super().__init__()
        self.order = order

    def __iter__(self):
        return product(range(self.order), [1])


class S(Group):
    """
    Symmetric (permutation) group on {0, 1, ..., n-1}.
    """
    def __init__(self, order):
        super().__init__()
        self.order = order

    def __iter__(self):
        return permutation_types(self.order)


class Product(Group):
    """
    Direct product / direct sum of groups.
    """
    def __init__(self, *groups):
        super().__init__()
        self.groups = groups

    def __iter__(self):
        for ts in product(*self.groups):  # ts = ((g1, c1), (g2, c2), ...)
            gs, cs = zip(*ts)
            yield gs, reduce(operator.mul, cs)
