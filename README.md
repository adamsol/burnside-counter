burnside-counter
================

Count graph colorings (and possibly other mathematical objects) using Burnside's lemma.


Requirements
------------

* Python 3.5+


Usage
-----

```
from burnside import *

for n in range(2, 10):
    c = EdgeColoring(Clique(n)).orbit_count()
    print("There are {} different graphs on {} unlabeled nodes.".format(c, n))
```

See ``test.py`` for more examples.
