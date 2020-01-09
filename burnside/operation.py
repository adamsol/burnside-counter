
from abc import abstractmethod, ABC

from .group import S, Z, Product

__all__ = [
    'Operation', 'ComplexOperation', 'Identity', 'EdgeColorSwap', 'EdgeReversal', 'VertexPermutation', 'VertexCycle', 'Reflection',
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
        for e in x.edges:
            e.change()


class EdgeReversal(Operation):

    def __init__(self):
        super().__init__(Z(2))

    def apply(self, g, x):
        if not g:
            return
        for e in x.edges:
            e.reverse()


class VertexPermutation(Operation):

    def __init__(self, *sizes):
        super().__init__(Product(*[S(size) for size in sizes]))

    def apply(self, g, x):
        k = 0
        for p in g:
            k2 = k + len(p)
            for v in x.vertices:
                if k <= v.q < k2:
                    v.q = p[v.q - k] + k
            k = k2


class VertexCycle(Operation):

    def __init__(self, *sizes):
        super().__init__(Product(*[Z(size) for size in sizes]))
        self.sizes = sizes

    def apply(self, g, x):
        k = 0
        for size, p in zip(self.sizes, g):
            k2 = k + size
            for v in x.vertices:
                if k <= v.q < k2:
                    v.q = (v.q - k + p) % size + k
            k = k2


class Reflection(Operation):

    def __init__(self, size):
        super().__init__(Z(2))
        self.size = size

    def apply(self, g, x):
        if not g:
            return
        for v in x.vertices:
            if v.q < self.size:
                v.q = self.size - v.q - 1
