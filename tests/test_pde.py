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

from actnempy.SINDy import HRidge, print_pde, kfold_cv

class TestPDE(unittest.TestCase):

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

    def test_HRidge(self):

        x = np.linspace(0,1,100)
        f1 = np.cos(x)
        f2 = np.exp(-x)
        f3 = x**2

        y = f1 + 0.1*f2 + 0.01*f3

        y = (y + 0.001*(0.5-np.random.random(x.shape)))[:, np.newaxis]

        X0 = np.array([f1, f2, f3]).T
        (w_all, r2) = HRidge(X0, y, 10**(-5))
        w0exp = np.array([1.0+0.j, 0.1+0.j, 0.01+0.j], dtype=np.complex64)
        w1exp = np.array([0.973+0.j, 0.07+0.j, 0.0+0.j], dtype=np.complex64)
        w2exp = np.array([1.131871+0.j, 0, 0], dtype=np.complex64)
        self.assertIsNone(assert_allclose(w_all[0], w0exp, rtol=0.1))
        self.assertTrue(w_all[1,-1]==0) # The first term to be removed has to be the one with the smallest strength, which is the f3 one, and so on.
        self.assertIsNone(assert_allclose(w_all[1], w1exp, rtol=0.1))
        self.assertTrue(w_all[2,-1]==0) 
        self.assertTrue(w_all[2,-2]==0)
        self.assertIsNone(assert_allclose(w_all[2], w2exp, rtol=0.1))
    
if __name__ == "__main__":
    unittest.main()

