
from collections import defaultdict

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
    return p, x  # cycle type, number of permutations

def permutation_types(n):
    if n < 1:
        yield [], 1
        return
    a = [0 for _ in range(n+1)]
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
            b = a[:k+2]
            yield _permutation_type(b)
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        b = a[:k+1]
        yield _permutation_type(b)


# Union-Find (disjoint set forests)
# http://code.activestate.com/recipes/577225-union-find/

def make_set(x):
    x.parent = x
    x.rank = 0

def union(x, y):
    xRoot = find(x)
    yRoot = find(y)
    if xRoot.rank > yRoot.rank:
        yRoot.parent = xRoot
    elif xRoot.rank < yRoot.rank:
        xRoot.parent = yRoot
    elif xRoot != yRoot:
        yRoot.parent = xRoot
        xRoot.rank = xRoot.rank+1

def find(x):
    if x.parent == x:
        return x
    else:
        x.parent = find(x.parent)
        return x.parent
