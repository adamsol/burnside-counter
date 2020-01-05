
from abc import ABC, abstractmethod
from collections.abc import Iterable

__all__ = [
    'Edge', 'Graph', 'Clique', 'Empty', 'Node',
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
