
import unittest

from burnside import *


class BurnsideTests(unittest.TestCase):

    def test_labeled_graphs(self):
        # http://oeis.org/A006125
        for n in range(100):
            self.assertEqual(EdgeColoring(Clique(n), Identity()).orbit_count(), 2**(n*(n-1)//2))

    def test_unlabeled_graphs(self):
        # http://oeis.org/A000088
        for n, a_n in enumerate([1, 1, 2, 4, 11, 34, 156]):
            self.assertEqual(EdgeColoring(Clique(n)).orbit_count(), a_n)

    def test_complementary_unlabeled_graphs(self):
        # http://oeis.org/A007869
        for n, a_n in enumerate([1, 1, 1, 2, 6, 18, 78]):
            self.assertEqual(EdgeColoring(Clique(n), VertexPermutation(n) * EdgeColorSwap()).orbit_count(), a_n)

    def test_multigraphs_on_five_nodes(self):
        # http://oeis.org/A063843
        for n, a_n in enumerate([0, 1, 34, 792, 10688, 90005, 533358, 2437848, 9156288, 29522961, 84293770]):
            self.assertEqual(EdgeColoring(Clique(5), colors=n).orbit_count(), a_n)

    def test_unlabeled_tournaments(self):
        # http://oeis.org/A000568
        for n, a_n in enumerate([1, 1, 1, 2, 4, 12, 56, 456]):
            self.assertEqual(EdgeOrientation(Clique(n)).orbit_count(), a_n)

    def test_complementary_unlabeled_tournaments(self):
        # http://oeis.org/A059735
        for n, a_n in enumerate([1, 1, 1, 2, 3, 10, 34]):
            self.assertEqual(EdgeOrientation(Clique(n), VertexPermutation(n) * EdgeReversal()).orbit_count(), a_n)

    def test_bicliques(self):
        # http://oeis.org/A007139
        for n, a_n in enumerate([1, 2, 6, 26, 192]):
            self.assertEqual(EdgeColoring(Biclique(n)).orbit_count(), a_n)

    def test_matrices_with_two_symbols(self):
        # http://oeis.org/A091059
        for n, a_n in enumerate([1, 1, 5, 18, 173]):
            self.assertEqual(EdgeOrientation(Biclique(n), VertexPermutation(n, n) * EdgeReversal()).orbit_count(), a_n)

    def test_necklaces(self):
        # https://oeis.org/A000031
        for n, a_n in enumerate([1, 2, 3, 4, 6, 8, 14, 20, 36, 60, 108, 188, 352, 632, 1182, 2192]):
            self.assertEqual(VertexColoring(Cycle(n), VertexCycle(n)).orbit_count(), a_n)

    def test_bracelets(self):
        # https://oeis.org/A000029
        for n, a_n in enumerate([1, 2, 3, 4, 6, 8, 13, 18, 30, 46, 78, 126, 224, 380, 687, 1224]):
            self.assertEqual(VertexColoring(Cycle(n), VertexCycle(n) * Reflection(n), colors=2).orbit_count(), a_n)

    def test_bracelets_with_n_colors(self):
        # https://oeis.org/A081721
        for n, a_n in enumerate([1, 1, 3, 10, 55, 377, 4291, 60028, 1058058, 21552969, 500280022]):
            self.assertEqual(VertexColoring(Cycle(n), VertexCycle(n) * Reflection(n), colors=n).orbit_count(), a_n)


if __name__ == '__main__':
    unittest.main()
