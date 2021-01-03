
from collections import defaultdict

__all__ = [
    'gcd', 'fact', 'ffact', 'choose', 'power',
    'permutation_types', 'permutation_representative', 'DisjointSets',
]

N = 100


# Greatest common divisors

gcd = {(i, j): 0 for i in range(N+1) for j in range(N+1)}

for i in range(N+1):
    for j in range(N+1):
        a, b = i, j
        while b != 0:
            a, b = b, a % b
        gcd[i, j] = a


# Factorials and falling factorials

fact = {i: 0 for i in range(N+1)}
ffact = {(i, j): 0 for i in range(N+1) for j in range(N+1)}

for i in range(N+1):
    for j in range(i+1):
        if j == 0:
            ffact[i, j] = 1
        else:
            ffact[i, j] = ffact[i, j-1] * (i - j + 1)
    fact[i] = ffact[i, i]


# Binomial coefficients

choose = {(i, j): 0 for i in range(N+1) for j in range(N+1)}

for i in range(N+1):
    for j in range(i+1):
        if j == 0:
            choose[i, j] = 1
        else:
            choose[i, j] = choose[i-1, j-1] + choose[i-1, j]


# Integer powers

power = {(i, j): 0 for i in range(N+1) for j in range(N+1)}

for i in range(N+1):
    for j in range(N+1):
        if j == 0:
            power[i, j] = 1
        else:
            power[i, j] = power[i, j-1] * i


# Permutation types (conjugacy classes)
# http://jeromekelleher.net/generating-integer-partitions.html
# https://groupprops.subwiki.org/wiki/Cycle_type_determines_conjugacy_class

def _permutation_type(partition):
    p = defaultdict(lambda: 0)
    n = 0
    for c in partition:
        p[c] += 1
        n += c
    l = n
    x = 1
    for i in p:
        x *= ffact[l, i*p[i]] // power[i, p[i]] // fact[p[i]]
        l -= i*p[i]
    return partition, x  # cycle lengths, number of permutations

def permutation_types(n):
    if n < 1:
        yield _permutation_type([])
        return
    a = [0] * (n+1)
    k = 1
    y = n - 1
    while k != 0:
        x = a[k-1] + 1
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield _permutation_type(a[:k+2])
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield _permutation_type(a[:k+1])

def permutation_representative(partition):
    permutation = []
    s = 0
    for c in partition:
        for i in range(c):
            permutation.append((i+1) % c + s)
        s += c
    return permutation


# Union-Find (disjoint set forests)
# http://code.activestate.com/recipes/577225-union-find/

class DisjointSets:

    @staticmethod
    def make_set(x):
        x.djs_parent = x
        x.djs_rank = 0

    @staticmethod
    def union(x, y):
        x_root = DisjointSets.find(x)
        y_root = DisjointSets.find(y)
        if x_root.djs_rank > y_root.djs_rank:
            y_root.djs_parent = x_root
        elif x_root.djs_rank < y_root.djs_rank:
            x_root.djs_parent = y_root
        elif x_root != y_root:
            y_root.djs_parent = x_root
            x_root.djs_rank = x_root.djs_rank + 1

    @staticmethod
    def find(x):
        if x.djs_parent == x:
            return x
        else:
            x.djs_parent = DisjointSets.find(x.djs_parent)
            return x.djs_parent


# https://stackoverflow.com/a/2912455/

class KeyDefaultDict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            result = self[key] = self.default_factory(key)
            return result
