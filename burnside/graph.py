
from abc import ABC, abstractmethod

from .group import S, Z
from .utils import permutation_representative

__all__ = [
    'Vertex', 'Edge', 'Face', 'Graph',
    'Node', 'Clique', 'Cycle',
    'Join', 'Biclique', 'Wheel',
    'Grid',
    'Tetrahedron', 'Cube', 'Octahedron',
]


class Vertex:
    def __init__(self, p):
        self.p = p
        self._offset = 0

    def translate(self, offset):
        self.p += offset
        self._offset += offset

    def update(self, callback):
        self.p = callback(self.p - self._offset) + self._offset


class Edge:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    @property
    def p(self):
        return frozenset({self.a.p, self.b.p})

    def reverse(self):
        self.a, self.b = self.b, self.a


class Face:
    def __init__(self, *vertices):
        self.vertices = vertices

    @property
    def p(self):
        return frozenset(v.p for v in self.vertices)


class Graph(ABC):
    def __init__(self, size, *, empty=False):
        self.size = size
        self.empty = empty
        self.vertices = []
        self.edges = []
        self.faces = []

    def build(self):
        self.vertices = [Vertex(x) for x in range(self.size)]

    @abstractmethod
    def operations(self):
        pass

    @abstractmethod
    def apply(self, op):
        pass


class Clique(Graph):
    def __init__(self, size, **kwargs):
        super().__init__(size, **kwargs)

    def build(self):
        super().build()
        if self.empty:
            return
        self.edges = [Edge(self.vertices[a], self.vertices[b]) for a in range(self.size) for b in range(a+1, self.size)]

    def operations(self):
        return S(self.size)

    def apply(self, op):
        permutation = permutation_representative(op)
        for v in self.vertices:
            v.update(lambda p: permutation[p])


class Node(Clique):
    def __init__(self):
        super().__init__(1)


class Cycle(Graph):
    def __init__(self, size, *, reflection=False, **kwargs):
        super().__init__(size, **kwargs)
        self.reflection = reflection

    def build(self):
        super().build()
        if self.empty:
            return
        self.edges = [Edge(self.vertices[a], self.vertices[a+1]) for a in range(self.size-1)]
        if self.size > 2:
            self.edges.append(Edge(self.vertices[0], self.vertices[self.size-1]))

    def operations(self):
        return Z(self.size) * Z(2 if self.reflection else 1)

    def apply(self, op):
        for v in self.vertices:
            v.update(lambda p: (p + op[0]) % self.size)
            if op[1]:
                v.update(lambda p: self.size - 1 - p)


class Join(Graph):
    def __init__(self, graph1, graph2, **kwargs):
        super().__init__(graph1.size + graph2.size, **kwargs)
        self.graphs = [graph1, graph2]

    def build(self):
        self.graphs[0].build()
        self.graphs[1].build()
        for vertex in self.graphs[1].vertices:
            vertex.translate(self.graphs[0].size)
        self.vertices = self.graphs[0].vertices + self.graphs[1].vertices
        self.edges = self.graphs[0].edges + self.graphs[1].edges
        if self.empty:
            return
        self.edges += [Edge(self.vertices[a], self.vertices[b]) for a in range(self.graphs[0].size) for b in range(self.graphs[0].size, self.size)]

    def operations(self):
        return self.graphs[0].operations() * self.graphs[1].operations()

    def apply(self, op):
        self.graphs[0].apply(op[0])
        self.graphs[1].apply(op[1])


class Biclique(Join):
    def __init__(self, size1, size2=None, *, reflection=False, **kwargs):
        if size2 is None:
            size2 = size1
        super().__init__(Clique(size1, empty=True), Clique(size2, empty=True), **kwargs)
        if reflection and size1 != size2:
            raise ValueError("Reflection operation requires the parts to be of equal size.")
        self.reflection = reflection

    def build(self):
        super().build()
        self._offset = self.vertices[0]._offset if self.vertices else 0

    def operations(self):
        return super().operations() * Z(2 if self.reflection else 1)

    def apply(self, op):
        if op[1]:
            # https://math.stackexchange.com/a/1165824/
            permutation = permutation_representative(op[0][0])
            s = self.graphs[0].size
            for v in self.vertices:
                if v.p < self._offset + s:
                    v.translate(s)
                else:
                    v.update(lambda p: permutation[p])
                    v.translate(-s)
        else:
            super().apply(op[0])


class Wheel(Join):
    def __init__(self, order, *, reflection=False, **kwargs):
        super().__init__(Cycle(order, reflection=reflection), Node(), **kwargs)
        self.reflection = reflection

    def operations(self):
        return self.graphs[0].operations()

    def apply(self, op):
        self.graphs[0].apply(op)


class Grid(Graph):
    def __init__(self, side, *, reflection=False, **kwargs):
        super().__init__(side*side, **kwargs)
        self.side = side
        self.reflection = reflection

    def operations(self):
        return Z(4) * Z(2 if self.reflection else 1)

    def apply(self, op):
        for i in range(op[0]):
            for v in self.vertices:
                v.update(lambda p: p % self.side * self.side + (self.side - 1 - p // self.side))
        if op[1]:
            for v in self.vertices:
                v.update(lambda p: p // self.side * self.side + (self.side - 1 - p % self.side))


class Tetrahedron(Graph):
    X = [1, 3, 2, 0]
    Y = [1, 2, 0, 3]
    PERMUTATIONS = [
        [], [X], [Y],
        [X, X], [X, Y], [Y, X], [Y, Y],
        [X, X, Y], [X, Y, Y], [Y, X, X], [Y, Y, X],
        [X, Y, Y, X],
    ]

    def __init__(self, *, reflection=False, **kwargs):
        super().__init__(4, **kwargs)
        self.reflection = reflection

    def build(self):
        super().build()
        if self.empty:
            return
        self.edges = [Edge(self.vertices[a], self.vertices[b]) for a, b in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]]
        self.faces = [Face(self.vertices[a], self.vertices[b], self.vertices[c]) for a, b, c in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]]

    def operations(self):
        return Z(12) * Z(2 if self.reflection else 1)

    def apply(self, op):
        for v in self.vertices:
            for permutation in self.PERMUTATIONS[op[0]]:
                v.update(lambda p: permutation[p])
            if op[1]:
                v.update(lambda p: 1 - p if p in {0, 1} else p)


class Cube(Graph):
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

    def __init__(self, *, reflection=False, **kwargs):
        super().__init__(8, **kwargs)
        self.reflection = reflection

    def build(self):
        super().build()
        if self.empty:
            return
        self.edges = [Edge(self.vertices[a], self.vertices[b]) for a, b in [(0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 7), (5, 6), (6, 7)]]
        self.faces = [Face(self.vertices[a], self.vertices[b], self.vertices[c], self.vertices[d]) for a, b, c, d in [(0, 1, 2, 3), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7)]]

    def operations(self):
        return Z(24) * Z(2 if self.reflection else 1)

    def apply(self, op):
        for v in self.vertices:
            for permutation in self.PERMUTATIONS[op[0]]:
                v.update(lambda p: permutation[p])
            if op[1]:
                v.update(lambda p: (p + 4) % 8)


class Octahedron(Graph):
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

    def __init__(self, *, reflection=False, **kwargs):
        super().__init__(6, **kwargs)
        self.reflection = reflection

    def build(self):
        super().build()
        if self.empty:
            return
        self.edges = [Edge(self.vertices[a], self.vertices[b]) for a, b in [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 4), (1, 5), (2, 3), (2, 5), (3, 4), (3, 5), (4, 5)]]
        self.faces = [Face(self.vertices[a], self.vertices[b], self.vertices[c]) for a, b, c in [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1), (5, 1, 2), (5, 2, 3), (5, 3, 4), (5, 4, 1)]]
        
    def operations(self):
        return Z(24) * Z(2 if self.reflection else 1)

    def apply(self, op):
        for v in self.vertices:
            for permutation in self.PERMUTATIONS[op[0]]:
                v.update(lambda p: permutation[p])
            if op[1]:
                v.update(lambda p: 5 - p if p in {0, 5} else p)


# TODO: dodecahedron and icosahedron
