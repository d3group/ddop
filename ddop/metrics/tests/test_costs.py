import unittest
import numpy as np
from ddop.metrics.costs import calc_costs
from numpy.testing import assert_array_equal


class TestCosts(unittest.TestCase):

    def setUp(self):
        self.y_true_single = np.array([0, 0, 10, 13, 8])
        self.y_pred_single = np.array([0, 5, 6, 20, 0])
        self.y_true_multi = np.array([[0, 0, 0], [0, 0, 0], [10, 10, 10], [13, 13, 13], [8, 9, 10]])
        self.y_pred_multi = np.array([[0, 0, 0], [5, 6, 7], [6, 7, 8], [20, 21, 22], [0, 0, 0]])
        # y_true-y_pred
        # 0     -5      4       -7      8
        # 0     -6      3       -8      9
        # 0     -7      2       -9      10

    def tearDown(self):
        pass

    def test_costs(self):
        result_single_equal = np.array([[0], [5], [4], [7], [8]])
        result_single_cu_greater = np.array([[0], [5], [8], [7], [16]])
        result_single_co_greater = np.array([[0], [10], [4], [14], [8]])
        result_multi = np.array([[0, 0, 0], [5, 6, 14], [4, 6, 2], [7, 8, 18], [8, 18, 10]])
        assert_array_equal(calc_costs(self.y_true_single, self.y_pred_single, 1, 1), result_single_equal)
        assert_array_equal(calc_costs(self.y_true_single, self.y_pred_single, cu=2, co=1), result_single_cu_greater)
        assert_array_equal(calc_costs(self.y_true_single, self.y_pred_single, cu=1, co=2), result_single_co_greater)
        assert_array_equal(calc_costs(self.y_true_multi, self.y_pred_multi, cu=[1, 2, 1], co=[1, 1, 2]), result_multi)


if __name__ == '__main__':
    unittest.main()
