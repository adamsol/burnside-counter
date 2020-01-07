
from abc import ABC, abstractmethod
from itertools import permutations, product


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
        return iter(range(self.order))


class S(Group):
    """
    Symmetric (permutation) group on {0, 1, ..., n-1}.
    """
    def __init__(self, order):
        super().__init__()
        self.order = order

    def __iter__(self):
        return permutations(range(self.order))


class Product(Group):
    """
    Direct product / direct sum of groups.
    """
    def __init__(self, *groups):
        super().__init__()
        self.groups = groups

    def __iter__(self):
        return product(*self.groups)
