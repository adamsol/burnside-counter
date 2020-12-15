
import math
import operator
from collections import defaultdict
from functools import reduce, total_ordering
from itertools import chain

__all__ = ['Variable']


@total_ordering
class Variable:
    def __init__(self, name):
        self.name = name

    def __add__(self, other):
        if isinstance(other, (int, Variable, Term, Polynomial)):
            return Term(1, {self: 1}) + other
        raise TypeError()

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, Variable, Term, Polynomial)):
            return Term(1, {self: 1}) - other
        raise TypeError()

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __neg__(self):
        return Term(-1, {self: 1})

    def __mul__(self, other):
        if isinstance(other, (int, Variable, Term, Polynomial)):
            return Term(1, {self: 1}) * other
        raise TypeError()

    def __rmul__(self, other):
        return self.__mul__(other)

    def __floordiv__(self, other):
        if isinstance(other, int):
            return Term(1, {self: 1}) // other
        raise TypeError()

    def __pow__(self, exp):
        if isinstance(exp, int):
            return Term(1, {self: 1}) ** exp
        raise TypeError()

    def __eq__(self, other):
        if isinstance(other, int):
            return False
        if isinstance(other, str):
            return self.name == other
        if isinstance(other, Variable):
            return self.name == other.name
        if isinstance(other, (Term, Polynomial)):
            return Term(1, {self: 1}) == other
        raise TypeError()

    def __lt__(self, other):
        if isinstance(other, Variable):
            # Variable with an empty name is always greater, so that term sorting for polynomials works correctly.
            if not self.name:
                return False
            if not other.name:
                return True
            return self.name < other.name
        raise TypeError()

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def substitute(self, variables):
        return Term(1, {self: 1}).substitute(variables)

    def extract(self, *exponents):
        return Term(1, {self: 1}).extract(*exponents)


@total_ordering
class Term:
    def __init__(self, coef=1, vars=None):
        self.coef = coef
        self.vars = defaultdict(lambda: 0)
        if coef and vars:
            self.vars.update(vars)

    def __add__(self, other):
        if isinstance(other, int):
            return self + Term(other)
        if isinstance(other, Variable):
            return self + Term(1, {other: 1})
        if isinstance(other, Term):
            return self + Polynomial(other)
        if isinstance(other, Polynomial):
            return Polynomial(self) + other
        raise TypeError()

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, int):
            return self - Term(other)
        if isinstance(other, Variable):
            return self - Term(1, {other: 1})
        if isinstance(other, Term):
            return self - Polynomial(other)
        if isinstance(other, Polynomial):
            return Polynomial(self) - other
        raise TypeError()

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __neg__(self):
        return self * (-1)

    def __mul__(self, other):
        if isinstance(other, int):
            if other == 0:
                return Term(0)
            return Term(self.coef*other, self.vars)
        if isinstance(other, Variable):
            return self * Term(1, {other: 1})
        if isinstance(other, Term):
            vars = self.vars.copy()
            for var, exp in other.vars.items():
                vars[var] += exp
            return Term(self.coef*other.coef, vars)
        if isinstance(other, Polynomial):
            return Polynomial(self) * other
        raise TypeError()

    def __rmul__(self, other):
        return self.__mul__(other)

    def __floordiv__(self, other):
        if isinstance(other, int):
            if self.coef % other != 0:
                return Polynomial(self) // other
            return Term(self.coef // other, self.vars)
        raise TypeError()

    def __pow__(self, other):
        if isinstance(other, int):
            if other == 0:
                return Term()
            if other > 0:
                return Term(self.coef**other, {var: exp*other for var, exp in self.vars.items()})
            raise ValueError()
        raise TypeError()

    def __eq__(self, other):
        if isinstance(other, int):
            return self == Term(other)
        if isinstance(other, Variable):
            return self == Term(1, {other: 1})
        if isinstance(other, Term):
            return (self.coef, self.vars) == (other.coef, other.vars)
        if isinstance(other, Polynomial):
            return Polynomial(self) == other
        raise TypeError()

    def __lt__(self, other):
        if isinstance(other, Term):
            a = sorted((var, -coef) for var, coef in self.vars.items()) + [(Variable(None), 0)]
            b = sorted((var, -coef) for var, coef in other.vars.items()) + [(Variable(None), 0)]
            return a < b
        raise TypeError()

    def __hash__(self):
        return hash((self.coef, frozenset(self.vars.items())))

    def __str__(self):
        return '{}{}'.format(
            str(self.coef) if not self.vars else '' if self.coef == 1 else '-' if self.coef == -1 else '{} '.format(self.coef),
            ' '.join('{}{}'.format(var, '^{}'.format(exp) if exp > 1 else '') for var, exp in sorted(self.vars.items())),
        ).strip()

    def __repr__(self):
        return str(self)

    def substitute(self, variables):
        return Polynomial(self).substitute(variables)

    def extract(self, *exponents):
        return Polynomial(self).extract(*exponents)


class Polynomial:
    def __init__(self, *terms, denominator=1):
        self.terms = {}
        for term in terms:
            self._add(term)
        self.denominator = denominator

    def _key(self, term):
        return frozenset(term.vars.items())

    def _add(self, term):
        key = self._key(term)
        coef = self.terms.get(key, Term(0)).coef + term.coef
        if coef:
            self.terms[key] = Term(coef, term.vars)
        else:
            self.terms.pop(key, None)

    def __add__(self, other):
        if isinstance(other, int):
            return self + Term(other)
        if isinstance(other, Variable):
            return self + Term(1, {other: 1})
        if isinstance(other, Term):
            return self + Polynomial(other)
        if isinstance(other, Polynomial):
            if self.denominator != 1 or other.denominator != 1:
                return (Polynomial(*self.terms.values()) * other.denominator + Polynomial(*other.terms.values()) * self.denominator) // (self.denominator * other.denominator)
            return Polynomial(*chain(self.terms.values(), other.terms.values()))
        raise TypeError()

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, int):
            return self - Term(other)
        if isinstance(other, Variable):
            return self - Term(1, {other: 1})
        if isinstance(other, Term):
            return self - Polynomial(other)
        if isinstance(other, Polynomial):
            return self + (-other)
        raise TypeError()

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __neg__(self):
        return self * (-1)

    def __mul__(self, other):
        if isinstance(other, int):
            return self * Term(other)
        if isinstance(other, Variable):
            return self * Term(1, {other: 1})
        if isinstance(other, Term):
            return self * Polynomial(other)
        if isinstance(other, Polynomial):
            result = Polynomial()
            for term1 in self.terms.values():
                for term2 in other.terms.values():
                    result._add(term1 * term2)
            return result // (self.denominator * other.denominator)
        raise TypeError()

    def __rmul__(self, other):
        return self.__mul__(other)

    def __floordiv__(self, other):
        if isinstance(other, int):
            if other != 0:
                gcd = reduce(math.gcd, [term.coef for term in self.terms.values()], other)
                if other < 0:
                    gcd *= -1
                denominator = other // gcd
                return Polynomial(*[term // gcd for term in self.terms.values()], denominator=(denominator*self.denominator))
            raise ZeroDivisionError()
        raise TypeError()

    def __pow__(self, other):
        if isinstance(other, int):
            if other == 0:
                return Term(1)
            if other > 0:
                return reduce(operator.mul, [self] * other)
            raise ValueError()
        raise TypeError()

    def __eq__(self, other):
        if isinstance(other, int):
            return self == Term(other)
        if isinstance(other, Variable):
            return self == Term(1, {other: 1})
        if isinstance(other, Term):
            return self == Polynomial(other)
        if isinstance(other, Polynomial):
            return set(self.terms.values()) == set(other.terms.values()) and self.denominator == other.denominator
        raise TypeError()

    def __hash__(self):
        return hash(self.terms)

    def __str__(self):
        if not self.terms:
            return '0'
        parts = []
        for i, term in enumerate(sorted(self.terms.values())):
            s = str(term)
            if i > 0:
                if term.coef < 0:
                    s = '- ' + s[1:]
                else:
                    s = '+ ' + s
            parts.append(s)
        if self.denominator != 1:
            numerator = ' '.join(parts)
            if len(parts) > 1:
                numerator = '({})'.format(numerator)
            return '{} / {}'.format(numerator, self.denominator)
        else:
            return ' '.join(parts)

    def __repr__(self):
        return str(self)

    def substitute(self, variables):
        result = Polynomial()
        for term in self.terms.values():
            part = Polynomial(Term(term.coef))
            for var, exp in term.vars.items():
                part *= (variables[var] if var in variables else var) ** exp
            result += part
        return result // self.denominator

    def extract(self, *args):
        if not args or args == (0,):
            f = lambda vars: not vars
        elif callable(args[0]):
            f = args[0]
        else:
            f = lambda vars: vars and list(zip(*sorted(vars.items())))[1] == args

        result = 0
        for term in self.terms.values():
            if f(term.vars):
                result += term.coef
        return result
