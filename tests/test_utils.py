import numpy as np
from numpy.testing import assert_allclose
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

import actnempy.utils as ut

class TestUtils(unittest.TestCase):

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

    @skip("This test still needs to be written.")
    def test_denoise(self):
        pass
    
    def test_add_noise(self):

        a = np.random.default_rng().standard_normal(tuple([100, 100]))
        b = ut.add_noise(a.copy(), noise_strength=0.01)
        noise = b - a
        self.assertIsNone(assert_allclose(noise.mean(), 0, atol=0.02))
        self.assertIsNone(assert_allclose(noise.std(), 0.01, rtol=0.02))

    @skip("This test still needs to be written.")
    def test_compute_Q(self):
        pass

    def test_compute_n(self):
        rng = np.random.default_rng(seed=42)
        S = rng.random((5,5))
        nx = 1.0 - 2*rng.random((5,5))
        ny = np.sqrt(1-nx**2)
        Qxx = S * (nx**2 - 0.5)
        Qxy = S * nx * ny

        (S0, nx0, ny0) = ut.compute_n(Qxx, Qxy)
        self.assertIsNone(assert_allclose(S, S0))
        self.assertIsNone(assert_allclose(nx, nx0))
        self.assertIsNone(assert_allclose(ny, ny0))
        
        pass

    def test_remove_NaNs(self):

        # Setup the array x such that it is pi/2 almost everywhere.
        # It has a NaN in one location, and two points around it have values -pi/2.
        # Note that the values around the NaN now average to zero, but not if they are director angles, since pi/2 and -pi/2 are equivalent.
        
        x = 0.5* np.pi * np.ones([10,10])
        x[4,4] = -0.5 * np.pi
        x[5,4] = np.nan
        x[6,4] = -0.5 * np.pi

        # Remove NaNs
        ut.remove_NaNs(x)
        
        # Expect average value to be zero
        self.assertAlmostEqual(abs(x[5,4]), 0.0)

        # Now putting the NaN back
        x[5,4] = np.nan
        
        # Removing it assuming the data being a nematic director
        ut.remove_NaNs(x, nematic=True)
        
        # Expect the value to be +- pi/2, since they are equivalent for a nematic field
        self.assertAlmostEqual(abs(x[5,4]), 0.5 * np.pi)

    def test_count_NaNs(self):
        a = np.zeros([5,5])
        a[1,2] = np.nan 
        a[2,3] = np.nan
        self.assertEqual(ut.count_NaNs(a),2)

    def test_get_random_sample(self):

        a = np.zeros([64,64,64])
        shp = a.shape
        num_points = 5
        box_size = tuple([15,15,15])
        diff_order = 2
        points, views = ut.get_random_sample(shp, num_points, box_size, diff_order, seed=1)
        self.assertEqual(len(points),5)
        for _, view in views.items():
            # The returned size is slightly larger than 15 so that we
            # can ignore the 1 edge point in all directions to
            # accomodate the 2nd order derivatives
            self.assertEqual(a[view].shape, tuple([17,17,17])) 
    
    # Same as above but with diff_order 3
    def test_get_random_sample3(self):

        a = np.zeros([64,64,64])
        shp = a.shape
        num_points = 5
        box_size = tuple([15,15,15])
        diff_order = 3
        points, views = ut.get_random_sample(shp, num_points, box_size, diff_order, seed=5)
        self.assertEqual(len(points),5)
        for _, view in views.items():
            # The returned size is slightly larger than 15 so that we
            # can ignore the 2 edges point in all directions to
            # accomodate the 3rd order derivatives
            self.assertEqual(a[view].shape, tuple([19,19,19])) 

if __name__ == "__main__":
    unittest.main()

