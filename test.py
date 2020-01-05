
import unittest

from burnside import *


class BurnsideTests(unittest.TestCase):

    def test_labeled_graphs(self):
        # http://oeis.org/A006125
        for n in range(100):
            self.assertEqual(EdgeColoring(Clique(n), Identity()).orbit_count(), 2**(n*(n-1)//2))

    def test_unlabeled_graphs(self):
        # http://oeis.org/A000088
        a = [*enumerate([1, 1, 2, 4, 11, 34, 156, 1044, 12346, 274668, 12005168]), (15, 31426485969804308768)]
        for n, a_n in a:
            self.assertEqual(EdgeColoring(Clique(n)).orbit_count(), a_n)

    def test_complementary_unlabeled_graphs(self):
        # http://oeis.org/A007869
        a = [*enumerate([1, 1, 1, 2, 6, 18, 78, 522, 6178, 137352, 6002584]), (13, 25251015686776)]
        for n, a_n in a:
            self.assertEqual(EdgeColoring(Clique(n), VertexPermutation(n) * EdgeColorSwap()).orbit_count(), a_n)

    def test_multigraphs_on_five_nodes(self):
        # http://oeis.org/A063843
        a = [*enumerate([0, 1, 34, 792, 10688, 90005, 533358, 2437848, 9156288, 29522961, 84293770]), (500, 8138021486328135468800000)]
        for n, a_n in a:
            self.assertEqual(EdgeColoring(Clique(5), colors=n).orbit_count(), a_n)

    def test_unlabeled_tournaments(self):
        # http://oeis.org/A000568
        a = [*enumerate([1, 1, 1, 2, 4, 12, 56, 456, 6880, 191536, 9733056]), (19, 24605641171260376770598003978281472)]
        for n, a_n in a:
            self.assertEqual(EdgeOrientation(Clique(n)).orbit_count(), a_n)

    def test_complementary_unlabeled_tournaments(self):
        # http://oeis.org/A059735
        a = [*enumerate([1, 1, 1, 2, 3, 10, 34, 272, 3528, 97144, 4870920]), (16, 31765207922047709885696)]
        for n, a_n in a:
            self.assertEqual(EdgeOrientation(Clique(n), VertexPermutation(n) * EdgeReversal()).orbit_count(), a_n)


if __name__ == '__main__':
    unittest.main()
