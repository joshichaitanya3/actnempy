import numpy as np
from numpy.testing import assert_allclose, assert_equal
import sys
import os
from pathlib import Path
import unittest
from io import StringIO
from unittest.mock import patch
from unittest import skip

test_dir = Path(__file__).parent.absolute()
# The modules are two directories up
module_dir = Path(__file__).resolve().parents[1].absolute()
# sys.path.append(module_dir.as_posix())

from actnempy.utils.grid import Grid

class TestGrid(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    # Actual tests begin

    # @skip("This test still needs to be written.")
    # def test_skipdemo(self):
    #     pass
    
    def test__validate_grid_incorrectshape(self):
        
        self.assertRaises(ValueError, Grid, 2, (1,1,1))
        self.assertRaises(ValueError, Grid, 2, (1,))
        self.assertRaises(ValueError, Grid, 1, (1,1))
        self.assertRaises(ValueError, Grid, 3, (1,1))
    
    def test__validate_grid_negative_h(self):

        self.assertRaises(ValueError, Grid, 2, -1)
        self.assertRaises(ValueError, Grid, 2, (-1,1))
        self.assertRaises(ValueError, Grid, 2, (-1,-1))
    
    def test__validate_grid_zero_h(self):

        self.assertRaises(ValueError, Grid, 2, 0)
        self.assertRaises(ValueError, Grid, 2, (0,1))

    def test__validate_grid_no_h(self):

        grid = Grid(ndims=2)
        self.assertIsNone(assert_equal(grid.h(), np.array([1,1])))

    def test_get_ndims(self):

        grid = Grid(ndims=2)
        self.assertEqual(grid.ndims(), 2)

    def test_get_g(self):

        grid = Grid(ndims=2, h=0.1)
        self.assertIsNone(assert_equal(grid.h(), np.array([0.1,0.1])))

    def test_grad(self):

        x = np.linspace(0,1,11)

        g1 = Grid(ndims=1, h=0.1, boundary="regular")
        self.assertIsNone(assert_allclose(g1.grad(x), np.ones([1,11])))
        y = np.linspace(0,1,11)
        X, Y = np.meshgrid(x, y, indexing="ij")
        
        g2 = Grid(ndims=2, h=0.1, boundary="regular")
        gradX = g2.grad(X)
        self.assertTrue(gradX.shape==(2,11,11))
        self.assertIsNone(assert_allclose(gradX[0], np.ones([11,11])))
        self.assertIsNone(assert_allclose(gradX[1], np.zeros([11,11])))



if __name__ == "__main__":
    unittest.main()
