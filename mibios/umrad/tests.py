from django.test import TestCase

from .utils import compile_ranges


class CompileRanges(TestCase):
    """ test the compile_ranges function """
    def test_empty(self):
        ranges, singles = compile_ranges([])
        with self.subTest(return_value='singles'):
            self.assertEqual(singles, [])
        with self.subTest(return_value='ranges'):
            self.assertEqual(ranges, [])

    def test_one_range(self):
        int_list = [4, 5, 6, 7, 8, 9]
        ranges, singles = compile_ranges(int_list)
        with self.subTest(return_value='singles'):
            self.assertEqual(singles, [])
        with self.subTest(return_value='ranges'):
            self.assertEqual(ranges, [(4, 9)])

    def test_just_singles(self):
        int_list = [-3, 88, -5, 0, 10, -22]
        ranges, singles = compile_ranges(int_list)
        with self.subTest(return_value='singles'):
            self.assertEqual(singles, [-22, -5, -3, 0, 10, 88])
        with self.subTest(return_value='ranges'):
            self.assertEqual(ranges, [])

    def test_general(self):
        int_list = [72, 3, 11, 12, 10, 4, 2, 10, 0, 1, 5, 55]
        ranges, singles = compile_ranges(int_list)
        with self.subTest(return_value='singles'):
            self.assertEqual(singles, [55, 72])
        with self.subTest(return_value='ranges'):
            self.assertEqual(ranges, [(0, 5), (10, 12)])

    def test_max_num_ranges(self):
        int_list = [72, 3, 11, 12, 10, 4, 2, 10, 0, 1, 5, 55]
        ranges, singles = compile_ranges(int_list, max_num_ranges=1)
        with self.subTest(return_value='singles'):
            self.assertEqual(singles, [10, 11, 12, 55, 72])
        with self.subTest(return_value='ranges'):
            self.assertEqual(ranges, [(0, 5)])

    def test_min_range_size(self):
        int_list = [25, 26, 22, 24, 4, 5, 6, 7, 8, 9, 23, 99]
        ranges, singles = compile_ranges(int_list, min_range_size=6)
        with self.subTest(return_value='singles'):
            self.assertEqual(singles, [22, 23, 24, 25, 26, 99])
        with self.subTest(return_value='ranges'):
            self.assertEqual(ranges, [(4, 9)])
