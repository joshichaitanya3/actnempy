import numpy as np
from numpy.testing import assert_allclose
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt 
import gdown
import unittest
from io import StringIO
from unittest.mock import patch

test_dir = Path(__file__).parent.absolute() 
# The modules are two directories up
module_dir = Path(__file__).resolve().parents[1].absolute()
# sys.path.append(module_dir.as_posix())
from actnempy import ActNem
import actnempy.utils as ut

class TestActNem(unittest.TestCase):
    
    # Download the testing data from Google Drive
    @classmethod
    def setUpClass(cls):
        print('Initializing the test sequence...')
    
        cls.data_dir =  module_dir / "TestData"

        url = "https://drive.google.com/uc?id=1BYS1iVh9rCR_aNSnPk2qodJzDuh_2mI6"

        cls.output = cls.data_dir / "processed_data.npz"

        gdown.download(url, cls.output.as_posix(), quiet=False)

        cls.an = ActNem(cls.data_dir) 

    # Clean up the testing data after all the tests are done
    @classmethod
    def tearDownClass(cls):
        os.remove(cls.output)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @patch('sys.stdout', new_callable=StringIO)
    def test__qprint(self, mock_stdout):

        # Expect no output when quiet==True
        self.an.quiet = True
        self.an._qprint('foo')
        assert mock_stdout.getvalue() == ""

        # Expect normal print behavior when quiet==False
        self.an.quiet = False
        self.an._qprint('foo')
        assert mock_stdout.getvalue() == "foo\n"

    def test_reset_data(self):
        self.an.u_all = None
        self.an.reset_data()

        self.assertIsNotNone(self.an.u_all)

    def test_compute_divergence(self):

        divu = self.an.compute_divergence(frame=0, plot=False)
        data = np.load(test_dir / "divu.npz")
        self.assertIsNone(assert_allclose(divu, data["divu"]))

    def test_vcorr(self):
        print('test_vcorr')
        
        vcorr, tc = self.an.velocity_autocorr()

        data = np.load(test_dir / "vcorr.npz")
        self.assertIsNone(assert_allclose(tc, 39.0))
        self.assertIsNone(assert_allclose(vcorr, data["vcorr"]))

    def test_ocorr(self):
        print('test_ocorr')

        ocorr, tc = self.an.orientation_autocorr()

        data = np.load(test_dir / "ocorr.npz")
        self.assertIsNone(assert_allclose(tc, 34.0))
        self.assertIsNone(assert_allclose(ocorr, data["ocorr"]))
    
    def test_find_defects(self):

        [cp, cm, phi_p, phi_m] = self.an.find_defects()

        data = np.load(test_dir / "defects.npz")
        self.assertIsNone(assert_allclose(cp, data["cp"]))
        self.assertIsNone(assert_allclose(cm, data["cm"]))
        self.assertIsNone(assert_allclose(phi_p, data["phi_p"]))
        self.assertIsNone(assert_allclose(phi_m, data["phi_m"]))
    
    def test_num_defects_all(self):

        [num_p, num_m] = self.an.num_defects_all()

        data = np.load(test_dir / "num_defects.npz")

        self.assertIsNone(assert_allclose(num_p, data["num_p"]))
        self.assertIsNone(assert_allclose(num_m, data["num_m"]))


if __name__ == "__main__":
    unittest.main()
