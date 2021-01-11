burnside-counter
================

Count graph colorings (and possibly other mathematical objects) using
[Burnside's lemma](https://en.wikipedia.org/wiki/Burnside%27s_lemma)
and [Pólya enumeration theorem](https://en.wikipedia.org/wiki/P%C3%B3lya_enumeration_theorem).


Requirements
------------

* Python 3.5+


Usage
-----

``` python
from burnside import *

# Number of different unlabeled 4-vertex graphs (i.e. edge colorings of a 4-clique using 2 colors).
print(Clique(4).orbit_count(edge_colors=2))

# Generating function for the number of 4-vertex graphs with a given number of edges.
print(Clique(4).generating_function(edge_colors=2))

# Number of different unlabeled 4-vertex tournaments (i.e. ways of directing edges of a 4-clique).
print(Clique(4).orbit_count(edge_direction=True))

# Number of bracelets with 8 beads -- 3 white and 5 black.
print(Cycle(8, reflection=True).generating_function(vertex_colors=2).extract(3))

# Number of ways to color faces of a cube using exactly 3 colors (each color has to be used at least once).
print(Cube().generating_function(face_colors=3).extract(lambda vars: len(vars) == 3))

# Generating function for the number of ways of coloring a cube using 3 exchangeable colors.
print(Cube().generating_function(face_colors='xyz', permutable_colors=True))

# Number of ways to place an arrow (pointing at one of the 3 vertices) on each face of a tetrahedron.
print(Tetrahedron().orbit_count(face_arrows=3))
```

See ``test.py`` for more examples.
