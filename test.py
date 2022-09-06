#!/usr/bin/env python

import time
import unittest

from burnside import *


class TestCase(unittest.TestCase):

    def setUp(self):
        self.start_time = time.time()

    def tearDown(self):
        print(f'\n{time.time()-self.start_time:.3f}s {type(self).__name__}.{self._testMethodName} ', end='')


class PolynomialTests(TestCase):

    def test_variables(self):
        k = Variable('k')
        n = Variable('n')

        self.assertEqual(k, k)
        self.assertEqual(k, 'k')
        self.assertEqual(Variable(''), '')
        self.assertNotEqual(k, 5)
        self.assertNotEqual(0, k)
        self.assertNotEqual(k, n)
        self.assertNotEqual(k, 'K')

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
        self.assertEqual(k // 2 // 5, k // 10)
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
        self.assertEqual((3*x + 9*y*y) // 2 // 3, (x + 3*y**2) // 2)
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

        self.assertEqual(x.extract(1), 1)
        self.assertEqual(x.extract(2), 0)
        self.assertEqual(x.extract(0), 0)
        self.assertEqual((3*x**5).extract(5), 3)
        self.assertEqual((y**2).extract(1), 0)
        self.assertEqual((10*y**0).extract(0), 10)
        self.assertEqual((2*x + 3*x**2 + 2*y).extract(lambda vars: vars[x] == 2), 3)
        self.assertEqual((x*y - y).extract(lambda vars: vars['x'] == vars['y'] == 1), 1)
        self.assertEqual((x*y - 2*y).extract(lambda vars: vars[y] == 1), -1)
        self.assertEqual((7*y**2*x**3 - 8*x**2*y**3).extract(lambda vars: vars[x] == 2 and vars[y] == 3), -8)
        self.assertEqual(((x+1)**6).extract(lambda vars: vars[x] >= 4), 22)
        self.assertEqual((y**0).extract(lambda vars: vars[y] == 0 and vars[z] == 0), 1)

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
        self.assertEqual(str((2*x + 8*y) // -4 // 3), '(-x - 4 y) / 6')
        self.assertEqual(str((3*x // 2) ** 2 - 6*x**2 // 8), '3 x^2 / 2')
        self.assertEqual(str((x // 7 + x**0).substitute({x: 7*z**2 + 21*x - 7*y})), '3 x - y + z^2 + 1')


class GraphTests(TestCase):

    def test_cycle_index(self):
        self.assertEqual(str(Clique(3).cycle_index()), '(e_1^3 v_1^3 + 3 e_1 e_2 v_1 v_2 + 2 e_3 v_3) / 6')
        self.assertEqual(str(Clique(4, empty=True).cycle_index()), '(v_1^4 + 6 v_1^2 v_2 + 8 v_1 v_3 + 3 v_2^2 + 6 v_4) / 24')
        self.assertEqual(str(Clique(4).cycle_index(edge_direction=True)), '(e_1^6 v_1^4 + 8 e_3^2 v_1 v_3) / 24')
        self.assertEqual(str(Tetrahedron().cycle_index()), '(e_1^6 f_1^4 v_1^4 + 3 e_1^2 e_2^2 f_2^2 v_2^2 + 8 e_3^2 f_1 f_3 v_1 v_3) / 12')

    def test_generating_function(self):
        self.assertEqual(str(Clique(3).generating_function(vertex_colors=2)), 'v_a^3 + v_a^2 + v_a + 1')
        self.assertEqual(str(Clique(4).generating_function(edge_colors=2)), 'e_a^6 + e_a^5 + 2 e_a^4 + 3 e_a^3 + 2 e_a^2 + e_a + 1')
        self.assertEqual(str(Clique(2).generating_function(vertex_colors='xy')), 'x^2 + x y + y^2')
        self.assertEqual(str(Node().generating_function(vertex_colors='xyz')), 'x + y + z')
        self.assertEqual(str(Cube().generating_function(face_colors='AB')), 'A^6 + A^5 B + 2 A^4 B^2 + 2 A^3 B^3 + 2 A^2 B^4 + A B^5 + B^6')

    def test_permutable_colors(self):
        self.assertEqual(Clique(4).orbit_count(edge_colors=2), 11)
        self.assertEqual(Clique(4).orbit_count(edge_colors=2, permutable_colors=True), 6)
        self.assertEqual(Clique(4).orbit_count(edge_direction=True), 4)
        self.assertEqual(Clique(4).orbit_count(permutable_colors=True, edge_direction=True), 4)
        self.assertEqual(Clique(4).orbit_count(edge_direction=True, reversible_edges=True), 3)
        self.assertEqual(Clique(4).orbit_count(permutable_colors=True, edge_direction=True, reversible_edges=True), 3)
        self.assertEqual(Clique(4).orbit_count(edge_colors=2, permutable_colors=True, edge_direction=True), 88)
        self.assertEqual(Clique(4).orbit_count(edge_colors=2, permutable_colors=True, edge_direction=True, reversible_edges=True), 52)
        self.assertEqual(Cube().orbit_count(face_colors=2, permutable_colors=True), 6)

    def test_generating_function_with_permutable_colors(self):
        self.assertEqual(str(Clique(3).generating_function(vertex_colors=2, permutable_colors=True)), 'v_a^3 + v_a^2')
        self.assertEqual(str(Clique(4).generating_function(edge_colors=2, permutable_colors=True)), 'e_a^6 + e_a^5 + 2 e_a^4 + 2 e_a^3')
        self.assertEqual(str(Node().generating_function(vertex_colors=5, permutable_colors=True)), 'v_a')
        self.assertEqual(str(Cube().generating_function(edge_colors='xy', permutable_colors=True)), 'x^12 + x^11 y + 5 x^10 y^2 + 13 x^9 y^3 + 27 x^8 y^4 + 38 x^7 y^5 + 29 x^6 y^6')
        self.assertEqual(Cube().generating_function(edge_colors=3, permutable_colors=True).extract(lambda vars: set(vars.values()) == {4}), 282)
        self.assertEqual(Cube().generating_function(face_colors='rgb', permutable_colors=True).extract(lambda vars: vars['r'] == 4 and vars['g'] == 1), 2)

    def test_0_vertex_colors(self):
        for n in range(5):
            self.assertEqual(Cycle(n).orbit_count(vertex_colors=0), int(n == 0))
            self.assertEqual(Clique(n).orbit_count(vertex_colors=0, permutable_colors=True), int(n == 0))
            self.assertEqual(Clique(n).generating_function(vertex_colors=0), int(n == 0))
            self.assertEqual(Cycle(n).generating_function(vertex_colors=0, permutable_colors=True), int(n == 0))

    def test_0_edge_colors(self):
        for n in range(5):
            self.assertEqual(Cycle(n).orbit_count(edge_colors=0), int(n <= 1))
            self.assertEqual(Clique(n).orbit_count(edge_colors=0, permutable_colors=True), int(n <= 1))
            self.assertEqual(Clique(n).generating_function(edge_colors=0), int(n <= 1))
            self.assertEqual(Cycle(n).generating_function(edge_colors=0, permutable_colors=True), int(n <= 1))

    def test_1_color(self):
        for n in range(5):
            self.assertEqual(Cycle(n).orbit_count(), 1)
            self.assertEqual(Clique(n).orbit_count(permutable_colors=True), 1)
            self.assertEqual(Clique(n).generating_function(), 1)
            self.assertEqual(Cycle(n).generating_function(permutable_colors=True), 1)

    def test_parameterized_number_of_colors(self):
        a = Variable('a')
        b = Variable('b')
        self.assertEqual(Clique(2).orbit_count(vertex_colors=a, edge_colors=b), a*(a+1)//2 * b)

    def test_face_arrows(self):
        # http://www.baxterweb.com/puzzles/burnside5.pdf (page 11)
        self.assertEqual(Cube().orbit_count(face_arrows=4), 192)


class OeisTests(TestCase):

    def test_unlabeled_graphs(self):
        # http://oeis.org/A000088
        for n, a_n in enumerate([1, 1, 2, 4, 11, 34, 156, 1044, 12346, 274668]):
            self.assertEqual(Clique(n).orbit_count(edge_colors=2), a_n)

    def test_complementary_unlabeled_graphs(self):
        # http://oeis.org/A007869
        for n, a_n in enumerate([1, 1, 1, 2, 6, 18, 78, 522, 6178, 137352]):
            self.assertEqual(Clique(n).orbit_count(edge_colors=2, permutable_colors=True), a_n)

    def test_symmetric_relations(self):
        # http://oeis.org/A000666
        for n, a_n in enumerate([1, 2, 6, 20, 90, 544, 5096, 79264, 2208612, 113743760]):
            self.assertEqual(Clique(n).orbit_count(vertex_colors=2, edge_colors=2), a_n)

    def test_multigraphs_on_five_nodes(self):
        # http://oeis.org/A063843
        for n, a_n in enumerate([0, 1, 34, 792, 10688, 90005, 533358, 2437848, 9156288, 29522961]):
            self.assertEqual(Clique(5).orbit_count(edge_colors=n), a_n)

    def test_unlabeled_tournaments(self):
        # http://oeis.org/A000568
        for n, a_n in enumerate([1, 1, 1, 2, 4, 12, 56, 456, 6880, 191536]):
            self.assertEqual(Clique(n).orbit_count(edge_direction=True), a_n)

    def test_unlabeled_tournaments_with_signed_nodes(self):
        # http://oeis.org/A093934
        for n, a_n in enumerate([1, 2, 4, 12, 48, 296, 3040, 54256, 1716608, 97213472]):
            self.assertEqual(Clique(n).orbit_count(vertex_colors=2, edge_direction=True), a_n)

    def test_complementary_unlabeled_tournaments(self):
        # http://oeis.org/A059735
        for n, a_n in enumerate([1, 1, 1, 2, 3, 10, 34, 272, 3528, 97144]):
            self.assertEqual(Clique(n).orbit_count(edge_direction=True, reversible_edges=True), a_n)

    def test_bicliques(self):
        # http://oeis.org/A007139
        for n, a_n in enumerate([1, 2, 6, 26, 192, 3014]):
            self.assertEqual(Biclique(n, reflection=True).orbit_count(edge_colors=2), a_n)

    def test_matrices_with_two_symbols(self):
        # http://oeis.org/A091059
        for n, a_n in enumerate([1, 1, 5, 18, 173, 2812]):
            self.assertEqual(Biclique(n).orbit_count(edge_direction=True, reversible_edges=True), a_n)

    def test_matrices_with_five_symbols(self):
        # http://oeis.org/A091062
        for n, a_n in enumerate([1, 1, 9, 649, 2283123, 173636442196]):
            self.assertEqual(Biclique(n).orbit_count(edge_colors=5, permutable_colors=True), a_n)

    def test_necklaces(self):
        # https://oeis.org/A000031
        for n, a_n in enumerate([1, 2, 3, 4, 6, 8, 14, 20, 36, 60]):
            self.assertEqual(Cycle(n).orbit_count(vertex_colors=2), a_n)

    def test_complementary_necklaces(self):
        # https://oeis.org/A000013
        for n, a_n in enumerate([1, 1, 2, 2, 4, 4, 8, 10, 20, 30]):
            self.assertEqual(Cycle(n).orbit_count(vertex_colors=2, permutable_colors=True), a_n)

    def test_bracelets(self):
        # https://oeis.org/A000029
        for n, a_n in enumerate([1, 2, 3, 4, 6, 8, 13, 18, 30, 46]):
            self.assertEqual(Cycle(n, reflection=True).orbit_count(vertex_colors=2), a_n)

    def test_complementary_bracelets(self):
        # https://oeis.org/A000011
        for n, a_n in enumerate([1, 1, 2, 2, 4, 4, 8, 9, 18, 23]):
            self.assertEqual(Cycle(n, reflection=True).orbit_count(vertex_colors=2, permutable_colors=True), a_n)

    def test_bracelets_with_n_colors(self):
        # https://oeis.org/A081721
        for n, a_n in enumerate([1, 1, 3, 10, 55, 377, 4291, 60028, 1058058, 21552969]):
            self.assertEqual(Cycle(n, reflection=True).orbit_count(vertex_colors=n), a_n)

    def test_rooted_plane_trees(self):
        # https://oeis.org/A003239
        for n, a_n in enumerate([1, 1, 2, 4, 10, 26, 80, 246, 810, 2704]):
            self.assertEqual(Cycle(n*2).generating_function(vertex_colors=2).extract(n), a_n)

    def test_prisms(self):
        # https://oeis.org/A222187
        for n, a_n in enumerate([13, 34, 78, 237, 687, 2299, 7685], 3):
            self.assertEqual(Prism(n, reflection=True).orbit_count(vertex_colors=2), a_n)

    def test_tetrahedral_symmetry(self):
        # https://oeis.org/A006008
        n = Variable('n')
        formula = (n**4 + 11*n**2) // 12
        self.assertEqual(Tetrahedron().orbit_count(face_colors=n), formula)
        self.assertEqual(Tetrahedron().orbit_count(vertex_colors=n), formula)

    def test_tetrahedral_symmetry_with_reflections(self):
        # https://oeis.org/A000332
        # https://oeis.org/A063842
        n = Variable('n')
        formula = (n**6 + 9*n**4 + 14*n**2) // 24
        self.assertEqual(Tetrahedron(reflection=True).orbit_count(edge_colors=n), formula)

    def test_octahedral_symmetry(self):
        # https://oeis.org/A047780
        n = Variable('n')
        formula = (n**6 + 3*n**4 + 12*n**3 + 8*n**2) // 24
        self.assertEqual(Cube().orbit_count(face_colors=n), formula)
        self.assertEqual(Octahedron().orbit_count(vertex_colors=n), formula)

    def test_octahedral_symmetry_with_reflection(self):
        # https://oeis.org/A199406
        n = Variable('n')
        formula = (n**12 + 3*n**8 + 12*n**7 + 4*n**6 + 8*n**4 + 12*n**3 + 8*n**2) // 48
        self.assertEqual(Cube(reflection=True).orbit_count(edge_colors=n), formula)
        self.assertEqual(Octahedron(reflection=True).orbit_count(edge_colors=n), formula)

    def test_icosahedral_symmetry(self):
        # https://oeis.org/A000545
        n = Variable('n')
        formula = (n**12 + 15*n**6 + 44*n**4) // 60
        self.assertEqual(Dodecahedron().orbit_count(face_colors=n), formula)
        self.assertEqual(Icosahedron().orbit_count(vertex_colors=n), formula)

    def test_icosahedral_symmetry_with_reflection(self):
        # https://oeis.org/A337963
        n = Variable('n')
        formula = (n**30 + 15*n**17 + 15*n**16 + n**15 + 20*n**10 + 24*n**6 + 20*n**5 + 24*n**3) // 120
        self.assertEqual(Dodecahedron(reflection=True).orbit_count(edge_colors=n), formula)
        self.assertEqual(Icosahedron(reflection=True).orbit_count(edge_colors=n), formula)

    def test_partitions_into_four_parts(self):
        # http://oeis.org/A001400
        for n, a_n in enumerate([1, 1, 2, 3, 5, 6, 9, 11, 15, 18]):
            self.assertEqual(Clique(n).orbit_count(vertex_colors=4, permutable_colors=True), a_n)

    def test_binary_grids(self):
        # http://oeis.org/A047937
        for n, a_n in enumerate([1, 2, 6, 140, 16456, 8390720, 17179934976, 140737496748032, 4611686019501162496, 604462909807864344215552]):
            self.assertEqual(Grid(n).orbit_count(vertex_colors=2), a_n)

    def test_grids_with_interchangeable_colors(self):
        # http://oeis.org/A264787
        for n, a_n in enumerate([1, 1, 7, 2966, 1310397193]):
            self.assertEqual(Grid(n, reflection=True).orbit_count(vertex_colors=n*n, permutable_colors=True), a_n)

    def test_grids_with_n_fields_selected(self):
        # http://oeis.org/A019318
        for n, a_n in enumerate([1, 1, 2, 16, 252, 6814, 244344, 10746377]):
            self.assertEqual(Grid(n, reflection=True).generating_function(vertex_colors=2).extract(n), a_n)


class ExamTests(TestCase):

    def test_prisms(self):
        # 2013-06
        self.assertEqual(Prism(3, reflection=True).generating_function(vertex_colors=2, edge_colors=2).extract(lambda vars: vars['v_a'] == 3 and vars['e_a'] == 5), 222)

    def test_joined_cycles(self):
        # 2014-09
        self.assertEqual(Join(Cycle(5), Cycle(4)).generating_function(vertex_colors=3).extract(lambda vars: set(vars.values()) == {3}), 90)

    def test_tournaments(self):
        # 2015-06
        self.assertEqual(Clique(5).orbit_count(edge_direction=True, reversible_edges=True), 10)

    def test_rectangles(self):
        # 2016-06
        self.assertEqual(Join(Clique(3, empty=True), Cycle(4, empty=True)).generating_function(edge_colors=2).extract(6), 48)

    def test_matrices(self):
        # 2017-06
        self.assertEqual(Biclique(2, 3).generating_function(edge_colors=3).extract(lambda vars: len(vars) == 3), 56)

    def test_congruences(self):
        # 2017-09
        self.assertEqual(Cycle(6).generating_function(vertex_colors='210').extract(lambda vars: (2*vars['2'] + vars['1']) % 3 == 0), 46)

    def test_wheel_graphs(self):
        # 2018-06
        k = Variable('k')
        self.assertEqual(Join(Cycle(6, reflection=True), Node()).orbit_count(vertex_colors=k, edge_direction=True), (1024*k**7 + 96*k**5 + 16*k**4 + 8*k**3 + 2*k**2) // 3)

    def test_bicliques(self):
        # 2019-06
        self.assertEqual(Biclique(3, reflection=True).orbit_count(edge_direction=True), 18)

    def test_cubes(self):
        # 2022-06
        self.assertEqual(Cube().generating_function(edge_colors='rg', face_colors='RG').extract(lambda vars: vars['r'] + 2*vars['g'] + 3*vars['R'] + 5*vars['G'] == 50), 47)


if __name__ == '__main__':
    unittest.main()
