
from .group import S, Z, Product

__all__ = [
    'Vertex', 'Edge', 'Face', 'Graph',
    'Clique', 'Empty', 'Node', 'Cycle',
    'Join', 'Biclique', 'Star', 'Wheel',
    'Grid',
    'Tetrahedron', 'Cube', 'Octahedron',
]


class Vertex:
    def __init__(self, p):
        self.p = p

    def translate(self, offset):
        self.p += offset


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


class Graph:
    def __init__(self, size):
        self.size = size
        self.vertices = []
        self.edges = []
        self.faces = []

    def build(self):
        self.vertices = [Vertex(x) for x in range(self.size)]


class Clique(Graph):
    def __init__(self, size):
        super().__init__(size)

    def build(self):
        super().build()
        self.edges = [Edge(self.vertices[a], self.vertices[b]) for a in range(self.size) for b in range(a+1, self.size)]


class Empty(Graph):
    def __init__(self, size):
        super().__init__(size)


class Node(Empty):
    def __init__(self):
        super().__init__(1)


class Cycle(Graph):
    def __init__(self, size):
        super().__init__(size)

    def build(self):
        super().build()
        self.edges = [Edge(self.vertices[a], self.vertices[a+1]) for a in range(self.size-1)]
        if self.size > 2:
            self.edges.append(Edge(self.vertices[0], self.vertices[self.size-1]))


class Join(Graph):
    def __init__(self, graph1, graph2):
        super().__init__(graph1.size + graph2.size)
        self.graphs = [graph1, graph2]

    def build(self):
        self.graphs[0].build()
        self.graphs[1].build()
        for vertex in self.graphs[1].vertices:
            vertex.translate(self.graphs[0].size)
        self.vertices = self.graphs[0].vertices + self.graphs[1].vertices
        self.edges = self.graphs[0].edges + self.graphs[1].edges
        self.edges += [Edge(self.vertices[a], self.vertices[b]) for a in range(self.graphs[0].size) for b in range(self.graphs[0].size, self.size)]


class Biclique(Join):
    def __init__(self, size1, size2=None):
        super().__init__(Empty(size1), Empty(size2 if size2 is not None else size1))


class Star(Join):
    def __init__(self, order):
        super().__init__(Empty(order), Node())


class Wheel(Join):
    def __init__(self, order):
        super().__init__(Cycle(order), Node())


class Grid(Graph):
    def __init__(self, width, height=None):
        if height is None:
            height = width
        super().__init__(width*height)
        self.width = width
        self.height = height


class Tetrahedron(Graph):
    def __init__(self):
        super().__init__(4)

    def build(self):
        super().build()
        self.edges = [Edge(self.vertices[a], self.vertices[b]) for a, b in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]]
        self.faces = [Face(self.vertices[a], self.vertices[b], self.vertices[c]) for a, b, c in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]]


class Cube(Graph):
    def __init__(self):
        super().__init__(8)

    def build(self):
        super().build()
        self.edges = [Edge(self.vertices[a], self.vertices[b]) for a, b in [(0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 7), (5, 6), (6, 7)]]
        self.faces = [Face(self.vertices[a], self.vertices[b], self.vertices[c], self.vertices[d]) for a, b, c, d in [(0, 1, 2, 3), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7)]]


class Octahedron(Graph):
    def __init__(self):
        super().__init__(6)

    def build(self):
        super().build()
        self.edges = [Edge(self.vertices[a], self.vertices[b]) for a, b in [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 4), (1, 5), (2, 3), (2, 5), (3, 4), (3, 5), (4, 5)]]
        self.faces = [Face(self.vertices[a], self.vertices[b], self.vertices[c]) for a, b, c in [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1), (5, 1, 2), (5, 2, 3), (5, 3, 4), (5, 4, 1)]]


# TODO: dodecahedron and icosahedron
