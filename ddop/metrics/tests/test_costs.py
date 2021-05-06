import unittest
from ddop.metrics._costs import pairwise_costs, average_costs, total_costs
from numpy.testing import assert_array_equal


class TestCosts(unittest.TestCase):

    def test_pairwise_costs(self):
        assert_array_equal(pairwise_costs([1], [10], 1, 2), [[18]])
        assert_array_equal(pairwise_costs([10], [1], 2, 1), [[18]])
        assert_array_equal(pairwise_costs([1, 10], [2, 5], 1, 1), [[1], [5]])
        assert_array_equal(pairwise_costs([1, 10], [2, 5], 2, 1), [[1], [10]])
        assert_array_equal(pairwise_costs([1, 10], [2, 5], 1, 2), [[2], [5]])
        # multioutput
        assert_array_equal(pairwise_costs([[1, 10], [2, 5]], [[2, 4], [1, 8]], 1, 1), [[1, 6], [1, 3]])
        # multioutput and differend cost coefficients
        assert_array_equal(pairwise_costs([[1, 10], [2, 5]], [[2, 4], [1, 8]], [1, 1], [1, 2]), [[1, 6], [1, 6]])

    def test_total_costs(self):
        assert_array_equal(total_costs([1], [10], 1, 2), 18)
        assert_array_equal(total_costs([10], [1], 2, 1), 18)
        assert_array_equal(total_costs([1, 10], [2, 5], 1, 1), 6)
        assert_array_equal(total_costs([1, 10], [2, 5], 2, 1), 11)
        assert_array_equal(total_costs([1, 10], [2, 5], 1, 2), 7)
        # multioutput
        assert_array_equal(total_costs([[1, 10], [2, 5]], [[2, 4], [1, 8]], 1, 1, multioutput="cumulated"), 11)
        assert_array_equal(total_costs([[1, 10], [2, 5]], [[2, 4], [1, 8]], 1, 1, multioutput="raw_values"), [2, 9])
        # multioutput and differend cost coefficients
        assert_array_equal(total_costs([[1, 10], [2, 5]], [[2, 4], [1, 8]], [1, 1], [1, 2], multioutput="cumulated"),
                           14)
        assert_array_equal(total_costs([[1, 10], [2, 5]], [[2, 4], [1, 8]], [1, 1], [1, 2], multioutput="raw_values"),
                           [2, 12])

    def test_average_costs(self):
        assert_array_equal(average_costs([1], [10], 1, 2), 18)
        assert_array_equal(average_costs([10], [1], 2, 1), 18)
        assert_array_equal(average_costs([1, 10], [2, 5], 1, 1), 3)
        assert_array_equal(average_costs([1, 10], [2, 5], 2, 1), 5.5)
        assert_array_equal(average_costs([1, 10], [2, 5], 1, 2), 3.5)
        # multioutput
        assert_array_equal(average_costs([[1, 10], [2, 5]], [[2, 4], [1, 8]], 1, 1, multioutput="uniform_average"), 2.75)
        assert_array_equal(average_costs([[1, 10], [2, 5]], [[2, 4], [1, 8]], 1, 1, multioutput="raw_values"), [1, 4.5])
        # multioutput and differend cost coefficients
        assert_array_equal(average_costs([[1, 10], [2, 5]], [[2, 4], [1, 8]], [1, 1], [1, 2], multioutput="uniform_average"),
                           3.5)
        assert_array_equal(average_costs([[1, 10], [2, 5]], [[2, 4], [1, 8]], [1, 1], [1, 2], multioutput="raw_values"),
                           [1, 6])


if __name__ == '__main__':
    unittest.main()
