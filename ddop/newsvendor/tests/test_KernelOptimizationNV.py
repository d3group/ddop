import unittest
import numpy as np
from numpy.testing import assert_array_equal
from ddop.newsvendor import KernelOptimizationNewsvendor


class TestKernelOptimizationNV(unittest.TestCase):

    def setUp(self):
        self.X_train = np.array([[0, 1], [0, 0], [1, 1], [1, 0], [1, 0], [0, 1]])
        self.y_train = np.array([[6, 5, 9], [1, 14, 3], [3, 8, 9], [7, 5, 1], [6, 10, 4], [1, 2, 9]])

    def tearDown(self):
        pass

    def test_predict(self):
        mdl = KernelOptimizationNewsvendor(1, 1, "uniform", 1)
        mdl.fit(self.X_train, self.y_train)
        assert_array_equal(mdl.predict([[1, 1]]), [[6, 5, 9]])


if __name__ == '__main__':
    unittest.main()
