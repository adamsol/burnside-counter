
from abc import ABC, abstractmethod

from .operation import VertexPermutation
from .utils import make_set, union, find

__all__ = [
    'Structure', 'EdgeColoring', 'EdgeOrientation',
]


class Structure(ABC):

    def __init__(self, graph, operation):
        self.graph = graph
        self.operation = operation

    @abstractmethod
    def fixed_point_count(self, g):
        pass

    def orbit_count(self):
        # https://en.wikipedia.org/wiki/Burnside%27s_lemma
        a = 0  # number of fixed points
        b = 0  # number of group elements

        for g, c in self.operation:
            x = self.fixed_point_count(g)
            a += c * x
            b += c

        assert a % b == 0
        return a // b


class EdgeColoring(Structure):

    def __init__(self, graph, operation=None, colors=2):
        if operation is None:
            operation = VertexPermutation(graph.size)
        super().__init__(graph, operation)
        self.colors = colors

    def _contradiction(self, e):
        return e.c != 0

    def fixed_point_count(self, g):
        x = self.graph.build()
        edges = list(x.values())

        for e in edges:
            make_set(e)

        while x:
            self.operation.apply(g, x.values())

            to_delete = []

            for p, e in x.items():
                union(e, x[e.v0, e.v1])

                if p[0] == e.v0 and p[1] == e.v1:
                    if self._contradiction(e):
                        return 0
                    to_delete.append(p)

            for p in to_delete:
                del x[p]

        s = set(find(e) for e in edges)
        return self.colors ** len(s)


class EdgeOrientation(EdgeColoring):

    def __init__(self, graph, operation=None):
        if operation is None:
            operation = VertexPermutation(graph.size)
        super().__init__(graph, operation, colors=2)

    def _contradiction(self, e):
        if super()._contradiction(e):
            return True
        return e.a > e.b
