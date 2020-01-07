
from abc import abstractmethod, ABC

from .group import S, Z, Product

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
        super().__init__(Product(*[operation.group for operation in operations]))
        self.operations = operations

    def apply(self, g, x):
        for operation, g_ in zip(self.operations, g):
            operation.apply(g_, x)


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

    def __init__(self, *sizes):
        super().__init__(Product(*[S(size) for size in sizes]))

    def apply(self, g, x):
        k = 0
        for p in g:
            k2 = k + len(p)
            for e in x:
                if k <= e.a < k2:
                    e.a = p[e.a-k] + k
                if k <= e.b < k2:
                    e.b = p[e.b-k] + k
            k = k2
