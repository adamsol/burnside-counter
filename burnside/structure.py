
import operator
from collections import Counter
from functools import reduce

from .operation import Identity
from .polynomial import Polynomial, Term, Variable
from .utils import fact, permutation_types, make_set, union, find

__all__ = [
    'Structure',
]


class NonAutomorphism(Exception):
    pass


class Structure:

    def __init__(self, graph, operation=Identity(), vertex_colors=1, edge_colors=1, edge_direction=False, face_colors=1):
        self.graph = graph
        self.operation = operation

        self.vertex_colors = vertex_colors
        self.edge_colors = edge_colors
        self.edge_direction = edge_direction
        self.face_colors = face_colors

        graph.build()
        self.vertex_variables = [Variable('v_{}'.format(i)) for i in range(len(graph.vertices) + 1)]
        self.edge_variables = [Variable('e_{}'.format(i)) for i in range(len(graph.edges) + 1)]
        self.face_variables = [Variable('f_{}'.format(i)) for i in range(len(graph.faces) + 1)]

    def _cycle_index_monomial(self, g):
        self.graph.build()
        vertices = {v.p: v for v in self.graph.vertices}
        edges = {e.p: e for e in self.graph.edges}
        faces = {f.p: f for f in self.graph.faces}

        self.operation.apply(g, self.graph)

        for e in edges.values():
            if e.p not in edges:
                raise NonAutomorphism()

        for v in vertices.values():
            v.cycle_length = 0
            make_set(v)
        for e in edges.values():
            e.cycle_length = 0
            make_set(e)
        for f in faces.values():
            f.cycle_length = 0
            make_set(f)

        while vertices or edges or faces:
            vertices_to_delete = []

            for p, v in vertices.items():
                v.cycle_length += 1
                union(v, vertices[v.p])

                if p == v.p:
                    vertices_to_delete.append(p)

            for p in vertices_to_delete:
                del vertices[p]

            edges_to_delete = []

            for p, e in edges.items():
                e.cycle_length += 1
                union(e, edges[e.p])

                if p == e.p:
                    if self.edge_direction and e.a.p > e.b.p:
                        return 0
                    edges_to_delete.append(p)

            for p in edges_to_delete:
                del edges[p]

            faces_to_delete = []

            for p, f in faces.items():
                f.cycle_length += 1
                union(f, faces[f.p])

                if p == f.p:
                    faces_to_delete.append(p)

            for p in faces_to_delete:
                del faces[p]

            self.operation.apply(g, self.graph)

        vertex_cycles = set(find(v) for v in self.graph.vertices)
        vertex_cycle_lengths = Counter(v.cycle_length for v in vertex_cycles)
        edge_cycles = set(find(e) for e in self.graph.edges)
        edge_cycle_lengths = Counter(e.cycle_length for e in edge_cycles)
        face_cycles = set(find(f) for f in self.graph.faces)
        face_cycle_lengths = Counter(f.cycle_length for f in face_cycles)

        result = 1
        if self.vertex_colors != 1:
            result *= reduce(operator.mul, (self.vertex_variables[length] ** count for length, count in vertex_cycle_lengths.items()), 1)
        if self.edge_colors != 1 or self.edge_direction:
            result *= reduce(operator.mul, (self.edge_variables[length] ** count for length, count in edge_cycle_lengths.items()), 1)
        if self.face_colors != 1:
            result *= reduce(operator.mul, (self.face_variables[length] ** count for length, count in face_cycle_lengths.items()), 1)
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

    def orbit_count(self, permutable_colors=False):
        result = self.cycle_index()

        for variables, color_count in [
            (self.vertex_variables, self.vertex_colors),
            (self.edge_variables, self.edge_colors * (2 if self.edge_direction else 1)),  # FIXME: edge direction won't work properly with permutable colors
            (self.face_variables, self.face_colors),
        ]:
            if permutable_colors:
                tmp = 0
                for p, k in permutation_types(color_count):
                    tmp += result.substitute({var: sum(c * m for c, m in p.items() if i % c == 0) for i, var in enumerate(variables)}) * k
                result = tmp // fact[color_count]
            else:
                result = result.substitute({var: color_count for var in variables})
            
        return result

    def generating_function(self, full=False, color_names=None):
        if color_names is None:
            vertex_color_names = [chr(ord('x') + i if i < 3 else ord('z') - i) for i in range(26)]
            edge_color_names = [chr(ord('a') + i) for i in range(26)]
            face_color_names = [chr(ord('A') + i) for i in range(26)]
        else:
            vertex_color_names = edge_color_names = face_color_names = color_names

        vertex_color_variables = [Variable(vertex_color_names[i]) for i in range(self.vertex_colors)]
        if not full and vertex_color_variables:
            vertex_color_variables[-1] = 1
        edge_color_variables = [Variable(edge_color_names[i]) for i in range(self.edge_colors * (2 if self.edge_direction else 1))]
        if not full and edge_color_variables:
            edge_color_variables[-1] = 1
        face_color_variables = [Variable(face_color_names[i]) for i in range(self.face_colors)]
        if not full and face_color_variables:
            face_color_variables[-1] = 1

        # TODO: handle permutable colors like in orbit_count
        return self.cycle_index().substitute({
            **{var: sum(color ** i for color in vertex_color_variables) for i, var in enumerate(self.vertex_variables) if i > 0},
            **{var: sum(color ** i for color in edge_color_variables) for i, var in enumerate(self.edge_variables) if i > 0},
            **{var: sum(color ** i for color in face_color_variables) for i, var in enumerate(self.face_variables) if i > 0},
        })
