
import operator
from collections import Counter
from functools import reduce

from .operation import Identity
from .polynomial import Polynomial, Term, Variable
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

        graph.build()
        self.vertex_variables = [Variable('v_{}'.format(i)) for i in range(len(graph.vertices) + 1)]
        self.edge_variables = [Variable('e_{}'.format(i)) for i in range(len(graph.edges) + 1)]

    def _cycle_index_monomial(self, g):
        self.graph.build()
        vertices = {v.q: v for v in self.graph.vertices}
        edges = {(e.a.q, e.b.q): e for e in self.graph.edges}

        self.operation.apply(g, self.graph)

        for e in edges.values():
            if (e.v0, e.v1) not in edges:
                raise NonAutomorphism()

        for v in vertices.values():
            v.cycle_length = 0
            make_set(v)
        for e in edges.values():
            e.cycle_length = 0
            make_set(e)

        while vertices or edges:
            vertices_to_delete = []

            for p, v in vertices.items():
                v.cycle_length += 1
                union(v, vertices[v.q])

                if p == v.q:
                    if v.c != 0:
                        return 0
                    vertices_to_delete.append(p)

            for p in vertices_to_delete:
                del vertices[p]

            edges_to_delete = []

            for p, e in edges.items():
                e.cycle_length += 1
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
        vertex_cycle_lengths = Counter(v.cycle_length for v in vertex_cycles)

        edge_cycles = set(find(e) for e in self.graph.edges)
        edge_cycle_lengths = Counter(e.cycle_length for e in edge_cycles)

        result = 1
        if self.vertex_colors != 1:
            result *= reduce(operator.mul, (self.vertex_variables[length] ** count for length, count in vertex_cycle_lengths.items()), 1)
        if self.edge_colors != 1 or self.edge_direction:
            result *= reduce(operator.mul, (self.edge_variables[length] ** count for length, count in edge_cycle_lengths.items()), 1)
        return result

    def cycle_index(self):
        a = Polynomial(Term(0))
        b = 0

        for g in self.operation:
            try:
                c = self._cycle_index_monomial(g)
            except NonAutomorphism:
                continue
            a += c
            b += 1

        if a == b == 0:
            return Polynomial(Term(1))

        return a // b

    def orbit_count(self):
        return self.cycle_index().substitute({
            **{var: self.vertex_colors for var in self.vertex_variables},
            **{var: self.edge_colors * (2 if self.edge_direction else 1) for var in self.edge_variables},
        })

    def generating_function(self, full=False, color_names=None):
        if color_names is None:
            vertex_color_names = [chr(ord('x') + i if i < 3 else ord('z') - i) for i in range(26)]
            edge_color_names = [chr(ord('a') + i) for i in range(26)]
        else:
            vertex_color_names = edge_color_names = color_names

        vertex_color_variables = [Variable(vertex_color_names[i]) for i in range(self.vertex_colors)]
        if not full and vertex_color_variables:
            vertex_color_variables[-1] = 1
        edge_color_variables = [Variable(edge_color_names[i]) for i in range(self.edge_colors * (2 if self.edge_direction else 1))]
        if not full and edge_color_variables:
            edge_color_variables[-1] = 1

        return self.cycle_index().substitute({
            **{var: sum(color ** i for color in vertex_color_variables) for i, var in enumerate(self.vertex_variables) if i > 0},
            **{var: sum(color ** i for color in edge_color_variables) for i, var in enumerate(self.edge_variables) if i > 0},
        })
