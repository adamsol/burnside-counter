
from abc import ABC, abstractmethod

__all__ = [
    'Edge', 'Graph', 'Clique', 'Empty', 'Node', 'Cycle', 'Join', 'Biclique', 'Star', 'Wheel',
]


class Vertex:

    def __init__(self, q):
        self.q = q
        self.c = 0

    def translate(self, offset):
        self.q += offset
        return self

    def change(self):
        self.c = 1 - self.c


class Edge:

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.c = 0

    def translate(self, offset):
        self.a.translate(offset)
        self.b.translate(offset)
        return self

    @property
    def v0(self):
        return min(self.a.q, self.b.q)

    @property
    def v1(self):
        return max(self.a.q, self.b.q)

    def reverse(self):
        self.a, self.b = self.b, self.a

    def change(self):
        self.c = 1 - self.c


class Graph(ABC):

    def __init__(self, size):
        self.size = size
        self.vertices = None
        self.edges = None

    @abstractmethod
    def build(self):
        self.vertices = [Vertex(x) for x in range(self.size)]
        self.edges = []


class Clique(Graph):

    def __init__(self, size):
        super().__init__(size)

    def build(self):
        super().build()
        self.edges = [Edge(self.vertices[a], self.vertices[b]) for a in range(self.size) for b in range(a+1, self.size)]


class Empty(Graph):

    def __init__(self, size):
        super().__init__(size)

    def build(self):
        super().build()


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

    def _translate_vertices(self, vertices, offset):
        for vertex in vertices:
            vertex.translate(offset)

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
