
import unittest

from burnside import *


class PolynomialTests(unittest.TestCase):

    def test_variables(self):
        k = Variable('k')
        n = Variable('n')

        self.assertEqual(k, k)
        self.assertNotEqual(k, 5)
        self.assertNotEqual(0, k)
        self.assertNotEqual(k, n)

        self.assertEqual(str(k), 'k')

    def test_terms(self):
        k = Variable('k')
        n = Variable('n')

        self.assertEqual(k**0, 1)
        self.assertEqual(k**1, k)
        self.assertEqual(k*k, k**2)
        self.assertEqual(k**3, k * k * k)
        self.assertEqual(k**5 * k**3, k**2 * k**4 * k**2)
        self.assertNotEqual(k**3, k**2)
        self.assertRaises(ValueError, lambda: k**-1)
        self.assertRaises(TypeError, lambda: k**1.5)

        self.assertEqual(k*n, n*k)
        self.assertEqual(5*k*n*k*k, n*k*k**2*5)
        self.assertEqual(n*0*k, 0)
        self.assertEqual(0*n**5, k*k*0)
        self.assertEqual(k*-1, -k)
        self.assertEqual(-n*6, -6*n**1)
        self.assertNotEqual(k*n, n)
        self.assertNotEqual(k**2*n, n**2*k)

        self.assertEqual(k // 1, k)
        self.assertEqual(4*k**3 // 2, 2*k**3)
        self.assertEqual(5*n // -10, -n // 2)
        self.assertRaises(ZeroDivisionError, lambda: n // 0)

        self.assertEqual(str(n**0), '1')
        self.assertEqual(str(-n**0), '-1')
        self.assertEqual(str(n**1), 'n')
        self.assertEqual(str(n**4 * n**2), 'n^6')
        self.assertEqual(str(1 * k**16 * n**0 * k), 'k^17')
        self.assertEqual(str(n * 5), '5 n')
        self.assertEqual(str(0 * k**10), '0')
        self.assertEqual(str(-n), '-n')
        self.assertEqual(str(k * n * k), 'k^2 n')
        self.assertEqual(str(n**5 * n**2 * -4 * k**3), '-4 k^3 n^7')
        self.assertEqual(str(k // 5), 'k / 5')
        self.assertEqual(str(3*n**2 // 6), 'n^2 / 2')

    def test_polynomials(self):
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')

        self.assertEqual(x + y, y + x)
        self.assertEqual(x + 6, x + 4 + 2)
        self.assertEqual(x + 2*x, x*3)
        self.assertEqual(x + y**5 - x**2 + x, 2*x - x*x + y**5)
        self.assertEqual(x*y + y*x, 2*x*y)
        self.assertNotEqual(x + y, x*y)
        self.assertNotEqual(x + y, x + 2*y)

        self.assertEqual(0 + x, x)
        self.assertEqual(x + y*x - x, y*x)
        self.assertEqual(x - x, 0)
        self.assertEqual(0, x*y**3 + x*y**3 - 2*x*y**3)
        self.assertNotEqual(x + x - x**2, 0)

        self.assertEqual(2 * (x*y + z - 5), 2*x*y + 2*z - 10)
        self.assertEqual(-(x - y), y - x)
        self.assertEqual(-x * (-x**5 + y + 8), x**6 - y*x - 8*x)
        self.assertEqual((x*y**2 + 3*x) * 5*y**2*z, 15*x*y**2*z + 5*x*y**4*z)
        self.assertEqual((x + y + 3*z) * (x - y), x**2 - y**2 + 3*x*z - 3*y*z)
        self.assertEqual((x - 1) ** 4, x**4 - 4*x**3 + 6*x**2 - 4*x + 1)
        self.assertEqual((x + y + z + 3) ** 1, x + y + z + 3)
        self.assertEqual((x + y**2) ** 0, 1)
        self.assertNotEqual(x - y, y - x)
        self.assertRaises(ValueError, lambda: (x + y) ** -1)
        self.assertRaises(TypeError, lambda: (x + y) * 2.5)

        self.assertEqual(x + y, (x + y) // 1)
        self.assertEqual((9*x + 6*y) // 4, (9*x + 6*y) // 4)
        self.assertEqual((5*x + 15*y) // 5, x + 3*y)
        self.assertEqual((8*x + 16*y**2) // 24, (x + 2*y**2) // 3)
        self.assertEqual((2*x - 4*y) // -2, 2*y - x)
        self.assertEqual(x // 2 + x // 2, x)
        self.assertEqual((9*x) // 4 - x // 4, 2*x)
        self.assertEqual((x // 2) * (x**2 // 5), x**3 // 10)
        self.assertEqual((x + y**4) // -2 + (y**4 + z) // 3, (-3*x - y**4 + 2*z) // 6)
        self.assertEqual(((3*x + y**5) // -3) ** 2, (9*x**2 + 6*x*y**5 + y**10) // 9)
        self.assertNotEqual(x + y, (x + y) // 2)
        self.assertRaises(ZeroDivisionError, lambda: (x + y) // 0)

        self.assertEqual(x.substitute({x: y}), y)
        self.assertEqual((x**6 + y**2).substitute({y: x**3}), 2*x**6)
        self.assertEqual((x**2 - y**2 - y).substitute({y: 1 - x}), 3*x - 2)

        self.assertEqual(str(x - x), '0')
        self.assertEqual(str(x**0 + 2), '3')
        self.assertEqual(str(y + 7 - y - 8), '-1')
        self.assertEqual(str(y**0 - 5*y**0), '-4')
        self.assertEqual(str(x + y), 'x + y')
        self.assertEqual(str(x + y*y), 'x + y^2')
        self.assertEqual(str(2*x - z**0 - y**4), '2 x - y^4 - 1')
        self.assertEqual(str(-z + x * (1 + x) * (x + 1) + y*y), 'x^3 + 2 x^2 + x + y^2 - z')
        self.assertEqual(str((x + 5 - 2*y**4) ** 3), 'x^3 - 6 x^2 y^4 + 15 x^2 + 12 x y^8 - 60 x y^4 + 75 x - 8 y^12 + 60 y^8 - 150 y^4 + 125')
        self.assertEqual(str((x**2 + z) // 1), 'x^2 + z')
        self.assertEqual(str((4*x**2 - 6*y) // 8), '(2 x^2 - 3 y) / 4')
        self.assertEqual(str((3*x // 2) ** 2 - 6*x**2 // 8), '3 x^2 / 2')
        self.assertEqual(str((x // 7 + x**0).substitute({x: 7*z**2 + 21*x - 7*y})), '3 x - y + z^2 + 1')


class CountingTests(unittest.TestCase):

    def test_labeled_graphs(self):
        # http://oeis.org/A006125
        for n in range(100):
            self.assertEqual(Structure(Clique(n), edge_colors=2).orbit_count(), 2**(n*(n-1)//2))

    def test_unlabeled_graphs(self):
        # http://oeis.org/A000088
        for n, a_n in enumerate([1, 1, 2, 4, 11, 34, 156]):
            self.assertEqual(Structure(Clique(n), VertexPermutation(n), edge_colors=2).orbit_count(), a_n)

    def test_complementary_unlabeled_graphs(self):
        # http://oeis.org/A007869
        for n, a_n in enumerate([1, 1, 1, 2, 6, 18, 78]):
            self.assertEqual(Structure(Clique(n), VertexPermutation(n) * EdgeColorSwap(), edge_colors=2).orbit_count(), a_n)

    def test_symmetric_relations(self):
        # http://oeis.org/A000666
        for n, a_n in enumerate([1, 2, 6, 20, 90, 544, 5096]):
            self.assertEqual(Structure(Clique(n), VertexPermutation(n), vertex_colors=2, edge_colors=2).orbit_count(), a_n)

    def test_multigraphs_on_five_nodes(self):
        # http://oeis.org/A063843
        for n, a_n in enumerate([0, 1, 34, 792, 10688, 90005, 533358, 2437848, 9156288, 29522961, 84293770]):
            self.assertEqual(Structure(Clique(5), VertexPermutation(5), edge_colors=n).orbit_count(), a_n)

    def test_unlabeled_tournaments(self):
        # http://oeis.org/A000568
        for n, a_n in enumerate([1, 1, 1, 2, 4, 12, 56, 456]):
            self.assertEqual(Structure(Clique(n), VertexPermutation(n), edge_direction=True).orbit_count(), a_n)

    def test_unlabeled_tournaments_with_signed_nodes(self):
        # http://oeis.org/A093934
        for n, a_n in enumerate([1, 2, 4, 12, 48, 296, 3040]):
            self.assertEqual(Structure(Clique(n), VertexPermutation(n), vertex_colors=2, edge_direction=True).orbit_count(), a_n)

    def test_complementary_unlabeled_tournaments(self):
        # http://oeis.org/A059735
        for n, a_n in enumerate([1, 1, 1, 2, 3, 10, 34]):
            self.assertEqual(Structure(Clique(n), VertexPermutation(n) * EdgeReversal(), edge_direction=True).orbit_count(), a_n)

    def test_bicliques(self):
        # http://oeis.org/A007139
        for n, a_n in enumerate([1, 2, 6, 26, 192]):
            self.assertEqual(Structure(Biclique(n), VertexPermutation(n*2), edge_colors=2).orbit_count(), a_n)

    def test_matrices_with_two_symbols(self):
        # http://oeis.org/A091059
        for n, a_n in enumerate([1, 1, 5, 18, 173]):
            self.assertEqual(Structure(Biclique(n), VertexPermutation(n, n) * EdgeReversal(), edge_direction=True).orbit_count(), a_n)

    def test_necklaces(self):
        # https://oeis.org/A000031
        for n, a_n in enumerate([1, 2, 3, 4, 6, 8, 14, 20, 36, 60, 108, 188, 352, 632, 1182, 2192]):
            self.assertEqual(Structure(Cycle(n), VertexCycle(n), vertex_colors=2).orbit_count(), a_n)

    def test_bracelets(self):
        # https://oeis.org/A000029
        for n, a_n in enumerate([1, 2, 3, 4, 6, 8, 13, 18, 30, 46, 78, 126, 224, 380, 687, 1224]):
            self.assertEqual(Structure(Cycle(n), VertexCycle(n) * Reflection(n), vertex_colors=2).orbit_count(), a_n)

    def test_bracelets_with_n_colors(self):
        # https://oeis.org/A081721
        for n, a_n in enumerate([1, 1, 3, 10, 55, 377, 4291, 60028, 1058058, 21552969, 500280022]):
            self.assertEqual(Structure(Cycle(n), VertexCycle(n) * Reflection(n), vertex_colors=n).orbit_count(), a_n)

    def test_0_vertex_colors(self):
        for n, a_n in enumerate([1, 0, 0, 0, 0, 0]):
            self.assertEqual(Structure(Clique(n), VertexPermutation(n), vertex_colors=0).orbit_count(), a_n)

    def test_0_edge_colors(self):
        for n, a_n in enumerate([1, 1, 0, 0, 0, 0]):
            self.assertEqual(Structure(Clique(n), VertexPermutation(n), edge_colors=0).orbit_count(), a_n)

    def test_1_color(self):
        for n in range(20):
            self.assertEqual(Structure(Cycle(n), VertexCycle(n)).orbit_count(), 1)

    def test_parameterized_number_of_colors(self):
        a = Variable('a')
        b = Variable('b')
        for n in range(20):
            self.assertEqual(Structure(Clique(n), vertex_colors=a, edge_colors=b).orbit_count(), a**n * b**(n*(n-1)//2))

    def test_wheel_graph(self):
        k = Variable('k')
        self.assertEqual(Structure(Wheel(6), VertexCycle(6) * Reflection(6), vertex_colors=k, edge_direction=True).orbit_count(), (1024*k**7 + 96*k**5 + 16*k**4 + 8*k**3 + 2*k**2) // 3)

    def test_cycle_index(self):
        self.assertEqual(str(Structure(Clique(3), VertexPermutation(3), vertex_colors=2).cycle_index()), '(v_1^3 + 3 v_1 v_2 + 2 v_3) / 6')
        self.assertEqual(str(Structure(Clique(4), VertexPermutation(4), edge_colors=2).cycle_index()), '(e_1^6 + 9 e_1^2 e_2^2 + 6 e_2 e_4 + 8 e_3^2) / 24')


if __name__ == '__main__':
    unittest.main()
