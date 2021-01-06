
import operator
from abc import ABC, abstractmethod
from collections import Counter
from functools import reduce

from .group import S, Z
from .polynomial import Polynomial, Term, Variable
from .utils import DisjointSets, fact, KeyDefaultDict, permutation_representative, permutation_types

__all__ = [
    'Vertex', 'Edge', 'Face', 'Graph',
    'Node', 'Clique', 'Cycle',
    'Join', 'Biclique', 'Wheel',
    'Grid', 'Prism',
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
        self.reset()

    def reset(self):
        self.vertices = []
        self.edges = []
        self.faces = []
        self.vertex_variables = KeyDefaultDict(lambda l: Variable('v_{}'.format(l)))
        self.edge_variables = KeyDefaultDict(lambda l: Variable('e_{}'.format(l)))
        self.face_variables = KeyDefaultDict(lambda l: Variable('f_{}'.format(l)))

    def build(self):
        self.vertices = [Vertex(x) for x in range(self.size)]

    @abstractmethod
    def operations(self):
        pass

    @abstractmethod
    def apply(self, op):
        pass

    def cycle_index_monomial(self, op=None, skip_vertices=False, skip_edges=False, skip_faces=False, edge_direction=False):
        self.build()
        vertices = {v.p: v for v in self.vertices}
        edges = {e.p: e for e in self.edges}
        faces = {f.p: f for f in self.faces}

        for v in vertices.values():
            v.cycle_length = 0
            DisjointSets.make_set(v)
        for e in edges.values():
            e.cycle_length = 0
            DisjointSets.make_set(e)
        for f in faces.values():
            f.cycle_length = 0
            DisjointSets.make_set(f)

        while vertices or edges or faces:
            if op is not None:
                self.apply(op[0])
                if op[1]:
                    for e in self.edges:
                        e.reverse()

            vertices_to_delete = []

            for p, v in vertices.items():
                v.cycle_length += 1
                DisjointSets.union(v, vertices[v.p])

                if p == v.p:
                    vertices_to_delete.append(p)

            for p in vertices_to_delete:
                del vertices[p]

            edges_to_delete = []

            for p, e in edges.items():
                e.cycle_length += 1
                DisjointSets.union(e, edges[e.p])

                if p == e.p:
                    if edge_direction and e.a.p > e.b.p:
                        return 0
                    edges_to_delete.append(p)

            for p in edges_to_delete:
                del edges[p]

            faces_to_delete = []

            for p, f in faces.items():
                f.cycle_length += 1
                DisjointSets.union(f, faces[f.p])

                if p == f.p:
                    faces_to_delete.append(p)

            for p in faces_to_delete:
                del faces[p]

        result = 1
        # The `skip_*` variables are for optimization.
        if not skip_vertices:
            vertex_cycles = set(DisjointSets.find(v) for v in self.vertices)
            vertex_cycle_lengths = Counter(v.cycle_length for v in vertex_cycles)
            result *= reduce(operator.mul, (self.vertex_variables[length] ** count for length, count in vertex_cycle_lengths.items()), 1)
        if not skip_edges:
            edge_cycles = set(DisjointSets.find(e) for e in self.edges)
            edge_cycle_lengths = Counter(e.cycle_length for e in edge_cycles)
            result *= reduce(operator.mul, (self.edge_variables[length] ** count for length, count in edge_cycle_lengths.items()), 1)
        if not skip_faces:
            face_cycles = set(DisjointSets.find(f) for f in self.faces)
            face_cycle_lengths = Counter(f.cycle_length for f in face_cycles)
            result *= reduce(operator.mul, (self.face_variables[length] ** count for length, count in face_cycle_lengths.items()), 1)
        return result

    def cycle_index(self, *, reversible_edges=False, **kwargs):
        self.reset()

        a = Polynomial(Term(0))
        b = 0

        for op, c in self.operations() * Z(2 if reversible_edges else 1):
            a += c * self.cycle_index_monomial(op, **kwargs)
            b += c

        if a == b == 0:
            return Polynomial(Term(1))

        return a // b

    def orbit_count(self, *, vertex_colors=1, edge_colors=1, face_colors=1, permutable_colors=False, edge_direction=False, reversible_edges=False):
        edge_color_count_multiplier = 2 if edge_direction else 1
        edge_colors *= edge_color_count_multiplier

        result = self.cycle_index(skip_vertices=(vertex_colors == 1), skip_edges=(edge_colors == 1), skip_faces=(face_colors == 1), edge_direction=edge_direction, reversible_edges=reversible_edges)

        for variables, color_count, color_count_multiplier in [
            (self.vertex_variables, vertex_colors, 1),
            (self.edge_variables, edge_colors, edge_color_count_multiplier),
            (self.face_variables, face_colors, 1),
        ]:
            if permutable_colors:
                tmp = 0
                color_count //= color_count_multiplier
                for p, k in permutation_types(color_count):
                    # The number of ways to color one cycle of graph elements (of length `l`) under a given color permutation (`p`)
                    # is the sum of color cycle lengths (`c`) that divide the length of the given cycle in the graph (`l`)
                    # multiplied by the number of additional non-permutable colors (like edge direction).
                    # https://math.stackexchange.com/a/834494/
                    tmp += result.substitute({var: sum(c for c in p if l % c == 0) * color_count_multiplier for l, var in variables.items()}) * k
                result = tmp // fact[color_count]
            else:
                result = result.substitute({var: color_count for var in variables.values()})

        return result

    def generating_function(self, *, vertex_colors=1, edge_colors=1, face_colors=1):
        def color_variables(t):
            colors, prefix = t
            result = list(map(Variable, colors if isinstance(colors, (str, tuple, list)) else ['{}_{}'.format(prefix, chr(ord('a') + i)) for i in range(colors)]))
            if 1 <= len(result) <= 2:
                result[-1] = 1
            return result

        vertex_colors, edge_colors, face_colors = map(color_variables, zip([vertex_colors, edge_colors, face_colors], 'vef'))

        # TODO: handle permutable colors like in orbit_count
        return self.cycle_index(skip_vertices=(vertex_colors == 1), skip_edges=(edge_colors == 1), skip_faces=(face_colors == 1)).substitute({
            **{var: sum(color ** l for color in vertex_colors) for l, var in self.vertex_variables.items()},
            **{var: sum(color ** l for color in edge_colors) for l, var in self.edge_variables.items()},
            **{var: sum(color ** l for color in face_colors) for l, var in self.face_variables.items()},
        })


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


class Prism(Graph):
    def __init__(self, base, *, reflection=False, **kwargs):
        super().__init__(base*2, **kwargs)
        if base < 3:
            raise ValueError("The base must have at least 3 vertices.")
        self.base = base
        self.reflection = reflection

    def build(self):
        super().build()
        if self.empty:
            return
        self.edges = [
            *(Edge(self.vertices[i], self.vertices[(i+1) % self.base]) for i in range(self.base)),
            *(Edge(self.vertices[i + self.base], self.vertices[(i+1) % self.base + self.base]) for i in range(self.base)),
            *(Edge(self.vertices[i], self.vertices[i + self.base]) for i in range(self.base)),
        ]
        self.faces = [
            Face(*(self.vertices[i] for i in range(self.base))),
            Face(*(self.vertices[i + self.base] for i in range(self.base))),
            *(Face(self.vertices[i], self.vertices[(i+1) % self.base], self.vertices[(i+1) % self.base + self.base], self.vertices[i + self.base]) for i in range(self.base)),
        ]

    def operations(self):
        return Z(self.base) * Z(2) * Z(2 if self.reflection else 1)

    def apply(self, op):
        for v in self.vertices:
            v.update(lambda p: (p + op[0][0]) % self.base + p // self.base * self.base)
            if op[0][1]:
                v.update(lambda p: self.size - 1 - p)
            if op[1]:
                v.update(lambda p: self.base - 1 - p % self.base + p // self.base * self.base)


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
