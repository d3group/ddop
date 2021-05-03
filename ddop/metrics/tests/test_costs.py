import unittest
from ddop.metrics._costs import pairwise_costs
from numpy.testing import assert_array_equal


class TestCosts(unittest.TestCase):

    def test_pairwise_costs(self):
        self.assertEqual(pairwise_costs([1], [10], 1, 2), 18)
        self.assertEqual(pairwise_costs([10], [1], 2, 1), 18)
        assert_array_equal(pairwise_costs([1, 10], [2, 5], 1, 1), [[1], [5]])
        assert_array_equal(pairwise_costs([1, 10], [2, 5], 2, 1), [[1], [10]])
        assert_array_equal(pairwise_costs([1, 10], [2, 5], 1, 2), [[2], [5]])


if __name__ == '__main__':
    unittest.main()
