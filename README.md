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

for n in range(2, 8):
    c = Structure(Clique(n), edge_colors=2).orbit_count()
    print("There are {} different graphs on {} unlabeled nodes.".format(c, n))
    
triangle = Structure(Clique(3), edge_colors=2)

# Cycle index of K_3: (e_1^3 + 3 e_1 e_2 + 2 e_3) / 6.
print(triangle.cycle_index())

# Generating function for the number of 3-vertex graphs with a given number of edges: a^3 + a^2 + a + 1.
print(triangle.generating_function())

# Number of necklaces with 8 beads -- 3 white and 5 black.
print(Structure(Cycle(8), vertex_colors=2).generating_function().extract(3))

# Number of ways to color faces of a cube using exactly 3 colors (each color has to be used at least once).
print(Structure(Cube(), face_colors=3).generating_function(full=True).extract(lambda vars: len(vars) == 3))
```

See ``test.py`` for more examples.
