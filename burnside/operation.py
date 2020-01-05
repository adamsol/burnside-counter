
import operator
from abc import abstractmethod, ABC
from collections.abc import Iterable
from functools import reduce

from .group import S, Z

__all__ = [
    'Operation', 'ComplexOperation', 'Identity', 'EdgeColorSwap', 'EdgeReversal', 'VertexPermutation',
]


class Operation(ABC):

    def __init__(self, group):
        self.group = group

    def __iter__(self):
        return self.group.__iter__()

    def __mul__(self, other):
        if not isinstance(other, Operation):
            raise ValueError
        return ComplexOperation(self, other)

    @abstractmethod
    def apply(self, g, x):
        pass


class ComplexOperation(Operation):

    def __init__(self, *operations):
        if len(operations) == 1 and isinstance(operations[0], Iterable):
            operations = list(operations[0])
        super().__init__(reduce(operator.mul, (operation.group for operation in operations)))
        self.operations = operations

    def apply(self, g, x):
        if len(self.operations) == 1:
            self.operations[0].apply(g, x)
        else:
            for i, operation in enumerate(self.operations):
                operation.apply(g[i], x)


class Identity(Operation):

    def __init__(self):
        super().__init__(Z(1))

    def apply(self, g, x):
        pass


class EdgeColorSwap(Operation):

    def __init__(self):
        super().__init__(Z(2))

    def apply(self, g, x):
        if not g:
            return
        for e in x:
            e.change()


class EdgeReversal(Operation):

    def __init__(self):
        super().__init__(Z(2))

    def apply(self, g, x):
        if not g:
            return
        for e in x:
            e.reverse()


class VertexPermutation(Operation):

    def __init__(self, size):
        super().__init__(S(size))

    def apply(self, g, x):
        p = 0
        d = {}
        for i in g:
            for j in range(g[i]):
                for k in range(i):
                    d[p+k] = (p, i)
                p += i
        for e in x:
            if e.a in d:
                e.a = d[e.a][0] + (e.a - d[e.a][0] + 1) % d[e.a][1]
            if e.b in d:
                e.b = d[e.b][0] + (e.b - d[e.b][0] + 1) % d[e.b][1]
