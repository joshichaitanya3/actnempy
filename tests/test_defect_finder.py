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

from actnempy.utils import defect_finder as df 


def defectpair(X, Y, dist, varphi1, varphi2, rcore=0.1):
    '''
    Function to generate the orientation profile for a pair
    of +-1/2 defects centered at (-dist/2,0) and (dist/2,0) respectively.
    varphi1 and varphi2 set the orientation of the minus and the plus half defects.
    rcore sets the defect core size.
    Refer to Eq. (33) of X. Tang and J. V. Selinger, Soft Matter 13, 5481 (2017).
    '''
    dth = varphi2-varphi1 + np.pi/2
    Th = varphi1 - np.pi/2
    th = ( -0.5*np.arctan2(X+0.5*dist, Y) 
          + 0.5*np.arctan2(X-0.5*dist, Y) 
          + 0.5*dth*(1+(np.log((X+0.5*dist)**2+Y**2)-np.log((X-0.5*dist)**2+Y**2))/(2*np.log(dist/rcore)))
          + Th)

    th[((X+0.5*dist)**2+Y**2<rcore**2)] = 0
    th[((X-0.5*dist)**2+Y**2<rcore**2)] = 0
    
    return th.T


class TestUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        '''
        This will generate an orientation profile with a +1/2 defect at (-5,0) and a -1/2 defect at (5,0). With varphi1=pi/3 and varphi2=0, we should get phi_p = 5.35589009 and phi_m=-0.19739556
        '''

        cls.x = np.linspace(-10, 10, 100)
        cls.y = np.linspace(-10, 10, 100)

        cls.dx = cls.x[1]-cls.x[0]
        cls.X, cls.Y = np.meshgrid(cls.x, cls.y, indexing="ij")

        cls.th = defectpair(cls.X, cls.Y, 10, np.pi/3, 0)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    # Actual tests begin

    def test_defectpos(self):
        # create charge density map
        nx = np.cos(self.th)
        ny = np.sin(self.th)
        _, map_p, map_m = df.func_defectfind(nx, ny, filter_radius=5, switchsign=0)

        # search map and identify circular regions of positive and negative charge
        centroids_p = df.func_defectpos(map_p, areathresh=60)
        centroids_m = df.func_defectpos(map_m, areathresh=60)

        cp = centroids_p*self.dx-10
        cm = centroids_m*self.dx-10
        self.assertEqual(len(cp), 1)
        self.assertEqual(len(cm), 1)
        self.assertTrue(np.abs(cp[0][0]+5)<0.25)
        self.assertTrue(np.abs(cp[0][1])<0.25)
        self.assertTrue(np.abs(cm[0][0]-5)<0.25)
        self.assertTrue(np.abs(cm[0][1])<0.25)
    
    def test_defectorient(self):
        
        nx = np.cos(self.th)
        ny = np.sin(self.th)
        _, map_p, map_m = df.func_defectfind(nx, ny, filter_radius=5, switchsign=0)

        # search map and identify circular regions of positive and negative charge
        centroids_p = df.func_defectpos(map_p, areathresh=60)
        centroids_m = df.func_defectpos(map_m, areathresh=60)
        phi_p = df.func_defectorient(centroids_p, nx, ny, filter_radius=5, type_str="positive")
        phi_m = df.func_defectorient(centroids_m, nx, ny, filter_radius=5, type_str="negative")
        self.assertEqual(len(phi_p), 1)
        self.assertEqual(len(phi_m), 1)
        self.assertIsNone(assert_allclose(np.mod(phi_p[0], 2*np.pi),5.35589009))
        self.assertIsNone(assert_allclose(np.mod(phi_m[0], 2*np.pi/3),1.89699954))

if __name__ == "__main__":
    unittest.main()

