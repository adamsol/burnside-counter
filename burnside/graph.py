
from abc import ABC, abstractmethod

__all__ = [
    'Edge', 'Graph', 'Clique', 'Empty', 'Node', 'Join', 'Biclique', 'Star',
]


class Edge:

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.c = 0

    @property
    def v0(self):
        return min(self.a, self.b)

    @property
    def v1(self):
        return max(self.a, self.b)

    def reverse(self):
        self.a, self.b = self.b, self.a

    def change(self):
        self.c = 1 - self.c


class Graph(ABC):

    def __init__(self, size):
        self.size = size

    @abstractmethod
    def build(self):
        pass


class Clique(Graph):

    def __init__(self, size):
        super().__init__(size)

    def build(self):
        return {(a, b): Edge(a, b) for a in range(self.size) for b in range(a+1, self.size)}


class Empty(Graph):

    def __init__(self, size):
        super().__init__(size)

    def build(self):
        return {}


class Node(Empty):

    def __init__(self):
        super().__init__(1)


class Join(Graph):

    def __init__(self, graph1, graph2):
        super().__init__(graph1.size + graph2.size)
        self.graphs = [graph1, graph2]

    def _translate(self, graph, offset):
        return {(a+offset, b+offset): Edge(e.a+offset, e.b+offset) for (a, b), e in graph.items()}

    def build(self):
        result = {}
        result.update(self.graphs[0].build())
        result.update(self._translate(self.graphs[1].build(), self.graphs[0].size))
        result.update({(a, b): Edge(a, b) for a in range(self.graphs[0].size) for b in range(self.graphs[0].size, self.size)})
        return result


class Biclique(Join):

    def __init__(self, size1, size2=None):
        super().__init__(Empty(size1), Empty(size2 if size2 is not None else size1))


class Star(Join):

    def __init__(self, order):
        super().__init__(Empty(order), Node())
