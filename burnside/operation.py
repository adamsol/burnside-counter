
from abc import abstractmethod, ABC

from .group import S, Z, Product

__all__ = [
    'Operation', 'ComplexOperation',
    'Identity', 'VertexColorSwap', 'EdgeColorSwap', 'EdgeReversal', 'FaceColorSwap',
    'VertexPermutation', 'VertexCycle', 'Reflection',
    'TetrahedronSymmetry', 'CubeSymmetry', 'OctahedronSymmetry',
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


class VertexColorSwap(Operation):

    def __init__(self):
        super().__init__(Z(2))

    def apply(self, g, x):
        if not g:
            return
        for v in x.vertices:
            v.change()


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


class FaceColorSwap(Operation):

    def __init__(self):
        super().__init__(Z(2))

    def apply(self, g, x):
        if not g:
            return
        for f in x.faces:
            f.change()


class VertexPermutation(Operation):

    def __init__(self, *sizes):
        super().__init__(Product(*[S(size) for size in sizes]))

    def apply(self, g, x):
        k = 0
        for p in g:
            k2 = k + len(p)
            for v in x.vertices:
                if k <= v.p < k2:
                    v.p = p[v.p - k] + k
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
                if k <= v.p < k2:
                    v.p = (v.p - k + p) % size + k
            k = k2


class Reflection(Operation):

    def __init__(self, size):
        super().__init__(Z(2))
        self.size = size

    def apply(self, g, x):
        if not g:
            return
        for v in x.vertices:
            if v.p < self.size:
                v.p = self.size - v.p - 1


class TetrahedronSymmetry(Operation):

    X = [1, 3, 2, 0]
    Y = [1, 2, 0, 3]
    PERMUTATIONS = [
        [], [X], [Y],
        [X, X], [X, Y], [Y, X], [Y, Y],
        [X, X, Y], [X, Y, Y], [Y, X, X], [Y, Y, X],
        [X, Y, Y, X],
    ]

    def __init__(self, reflections=False):
        size = 12 * (2 if reflections else 1)
        super().__init__(Z(size))

    def apply(self, g, x):
        for v in x.vertices:
            for p in self.PERMUTATIONS[g % 12]:
                v.p = p[v.p]
            if g >= 12 and v.p in {0, 1}:
                v.p = 1 - v.p


class CubeSymmetry(Operation):

    # https://www.euclideanspace.com/maths/discrete/groups/categorise/finite/cube/index.htm
    X = [1, 2, 3, 0, 5, 6, 7, 4]
    Y = [4, 0, 3, 7, 5, 1, 2, 6]
    PERMUTATIONS = [
        [], [X], [Y], [X, X],
        [X, Y], [Y, X], [Y, Y], [X, X, X],
        [X, X, Y], [X, Y, X], [X, Y, Y], [Y, X, X],
        [Y, Y, X], [Y, Y, Y], [X, X, X, Y], [X, X, Y, X],
        [X, X, Y, Y], [X, Y, X, X], [X, Y, Y, Y], [Y, X, X, X],
        [Y, Y, Y, X], [X, X, X, Y, X], [X, Y, X, X, X], [X, Y, Y, Y, X],
    ]

    def __init__(self, reflections=False):
        size = 24 * (2 if reflections else 1)
        super().__init__(Z(size))

    def apply(self, g, x):
        for v in x.vertices:
            for p in self.PERMUTATIONS[g % 24]:
                v.p = p[v.p]
            if g >= 24:
                v.p = (v.p + 4) % 8


class OctahedronSymmetry(Operation):

    X = [0, 2, 3, 4, 1, 5]
    Y = [4, 1, 0, 3, 5, 2]
    PERMUTATIONS = [
        [], [X], [Y], [X, X],
        [X, Y], [Y, X], [Y, Y], [X, X, X],
        [X, X, Y], [X, Y, X], [X, Y, Y], [Y, X, X],
        [Y, Y, X], [Y, Y, Y], [X, X, X, Y], [X, X, Y, X],
        [X, X, Y, Y], [X, Y, X, X], [X, Y, Y, Y], [Y, X, X, X],
        [Y, Y, Y, X], [X, X, X, Y, X], [X, Y, X, X, X], [X, Y, Y, Y, X],
    ]

    def __init__(self, reflections=False):
        size = 24 * (2 if reflections else 1)
        super().__init__(Z(size))

    def apply(self, g, x):
        for v in x.vertices:
            for p in self.PERMUTATIONS[g % 24]:
                v.p = p[v.p]
            if g >= 24 and v.p in {0, 5}:
                v.p = 5 - v.p
