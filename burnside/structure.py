
from abc import ABC, abstractmethod

from .operation import VertexPermutation
from .utils import make_set, union, find

__all__ = [
    'Structure', 'VertexColoring', 'EdgeColoring', 'EdgeOrientation',
]


class NonAutomorphism(Exception):
    pass


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

        for g in self.operation:
            try:
                c = self.fixed_point_count(g)
            except NonAutomorphism:
                continue
            a += c
            b += 1

        if a == b == 0:
            return 1

        assert a % b == 0
        return a // b


class VertexColoring(Structure):

    def __init__(self, graph, operation=None, colors=2):
        if operation is None:
            operation = VertexPermutation(graph.size)
        super().__init__(graph, operation)
        self.colors = colors

    def _contradiction(self, v):
        return v.c != 0

    def fixed_point_count(self, g):
        self.graph.build()
        vertices = {v.q: v for v in self.graph.vertices}
        edges = {(e.a.q, e.b.q): e for e in self.graph.edges}

        self.operation.apply(g, self.graph)

        for e in edges.values():
            if (e.v0, e.v1) not in edges:
                raise NonAutomorphism()

        for v in vertices.values():
            make_set(v)

        while vertices:
            to_delete = []

            for p, v in vertices.items():
                union(v, vertices[v.q])

                if p == v.q:
                    if self._contradiction(v):
                        return 0
                    to_delete.append(p)

            for p in to_delete:
                del vertices[p]

            self.operation.apply(g, self.graph)

        s = set(find(v) for v in self.graph.vertices)
        return self.colors ** len(s)


class EdgeColoring(Structure):

    def __init__(self, graph, operation=None, colors=2):
        if operation is None:
            operation = VertexPermutation(graph.size)
        super().__init__(graph, operation)
        self.colors = colors

    def _contradiction(self, e):
        return e.c != 0

    def fixed_point_count(self, g):
        self.graph.build()
        edges = {(e.a.q, e.b.q): e for e in self.graph.edges}

        self.operation.apply(g, self.graph)

        for e in edges.values():
            if (e.v0, e.v1) not in edges:
                raise NonAutomorphism()

        for e in edges.values():
            make_set(e)

        while edges:
            to_delete = []

            for p, e in edges.items():
                union(e, edges[e.v0, e.v1])

                if p[0] == e.v0 and p[1] == e.v1:
                    if self._contradiction(e):
                        return 0
                    to_delete.append(p)

            for p in to_delete:
                del edges[p]

            self.operation.apply(g, self.graph)

        s = set(find(e) for e in self.graph.edges)
        return self.colors ** len(s)


class EdgeOrientation(EdgeColoring):

    def __init__(self, graph, operation=None):
        if operation is None:
            operation = VertexPermutation(graph.size)
        super().__init__(graph, operation, colors=2)

    def _contradiction(self, e):
        if super()._contradiction(e):
            return True
        return e.a.q > e.b.q
