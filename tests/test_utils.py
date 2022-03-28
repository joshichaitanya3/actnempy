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
    
    @skip("This test still needs to be written.")
    def test_add_noise(self):
        pass

    @skip("This test still needs to be written.")
    def test_compute_Q(self):
        pass

    @skip("This test still needs to be written.")
    def test_compute_n(self):
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

    @skip("This test still needs to be written.")
    def test__circular_shifts(self):
        pass

    @skip("This test still needs to be written.")
    def test_set_boundary(self):
        pass

    @skip("This test still needs to be written.")
    def test_set_boundary_region(self):
        pass

    @skip("This test still needs to be written.")
    def test_count_NaNs(self):
        pass

    @skip("This test still needs to be written.")
    def test_get_random_sample(self):
        pass


if __name__ == "__main__":
    unittest.main()

