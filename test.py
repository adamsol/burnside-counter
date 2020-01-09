
import unittest

from burnside import *


class BurnsideTests(unittest.TestCase):

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


if __name__ == '__main__':
    unittest.main()
