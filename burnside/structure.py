
from .operation import Identity
from .utils import make_set, union, find

__all__ = [
    'Structure',
]


class NonAutomorphism(Exception):
    pass


class Structure:

    def __init__(self, graph, operation=Identity(), vertex_colors=1, edge_colors=1, edge_direction=False):
        self.graph = graph
        self.operation = operation

        self.vertex_colors = vertex_colors
        self.edge_colors = edge_colors
        self.edge_direction = edge_direction

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
        for e in edges.values():
            make_set(e)

        while vertices or edges:
            vertices_to_delete = []

            for p, v in vertices.items():
                union(v, vertices[v.q])

                if p == v.q:
                    if v.c != 0:
                        return 0
                    vertices_to_delete.append(p)

            for p in vertices_to_delete:
                del vertices[p]

            edges_to_delete = []

            for p, e in edges.items():
                union(e, edges[e.v0, e.v1])

                if p[0] == e.v0 and p[1] == e.v1:
                    if e.c != 0:
                        return 0
                    if self.edge_direction and e.a.q > e.b.q:
                        return 0
                    edges_to_delete.append(p)

            for p in edges_to_delete:
                del edges[p]

            self.operation.apply(g, self.graph)

        vertex_cycles = set(find(v) for v in self.graph.vertices)
        edge_cycles = set(find(e) for e in self.graph.edges)
        return self.vertex_colors ** len(vertex_cycles) * (self.edge_colors * (2 if self.edge_direction else 1)) ** len(edge_cycles)

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

        return a // b
