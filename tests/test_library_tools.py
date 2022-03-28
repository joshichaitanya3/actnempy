import numpy as np
from numpy.testing import assert_allclose
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt 
import unittest
from io import StringIO
from unittest.mock import patch

test_dir = Path(__file__).parent.absolute()
# The modules are two directories up
module_dir = Path(__file__).resolve().parents[1].absolute()
# sys.path.append(module_dir.as_posix())


from actnempy import ActNem
import actnempy.utils as ut
from actnempy.SINDy import library_tools as lt
class TestLibraryTools(unittest.TestCase):

    def test_function(self):
        
        uf = lt.Function("u", maxf=3, maxd=4)
        
        self.assertEqual(uf.__repr__(), "u")
        self.assertEqual(uf.diff_order, 0)
        self.assertEqual(uf.func_order, 1)
        self.assertEqual(uf.max_func_order, 3)
        self.assertEqual(uf.max_diff_order, 4)

    def test_unity(self):
        unity = lt.Function.unity()
        self.assertEqual(unity.diff_order, 0)
        self.assertEqual(unity.func_order, 0)
        self.assertEqual(unity.max_func_order, 0)
        self.assertEqual(unity.max_diff_order, 0)
        self.assertEqual(unity.root, "1")

    def test_multiplyop(self):

        uf = lt.Function("u", maxf=3, maxd=4)
        vf = lt.Function("v", maxf=2, maxd=3)

        wf = uf * vf
        self.assertEqual(wf.__repr__(), "u \u00D7 v")
        self.assertEqual(wf.diff_order, 0)
        self.assertEqual(wf.func_order, 2)
    
    def test_multiplybyunity(self):

        unity = lt.Function.unity()
        vf = lt.Function("v", maxf=2, maxd=3)

        # Checking left-multiply
        wf = unity * vf
        self.assertEqual(wf.__repr__(), "v")
        self.assertEqual(wf.diff_order, 0)
        self.assertEqual(wf.func_order, 1)
        
        # Checking right-multiply
        wf = vf * unity
        self.assertEqual(wf.__repr__(), "v")
        self.assertEqual(wf.diff_order, 0)
        self.assertEqual(wf.func_order, 1)
        
        # Checking unity times unity
        wf = unity * unity
        self.assertEqual(wf.__repr__(), "1")
        self.assertEqual(wf.diff_order, 0)
        self.assertEqual(wf.func_order, 0)

    def test_derivative(self):
        uf = lt.Function("u", maxf=3, maxd=4)

        # Take derivatives of function
        uxf = lt.Derivative(uf, "x")
        self.assertEqual(uxf.__repr__(), "(∂u/∂x)")
        self.assertEqual(uxf.diff_order, 1)
        uxf = lt.Derivative(uf, "x", 2)
        self.assertEqual(uxf.__repr__(), "(∂²u/∂x²)")
        self.assertEqual(uxf.diff_order, 2)
        uxf = lt.Derivative(uf, "x", 3)
        self.assertEqual(uxf.__repr__(), "(∂³u/∂x³)")
        self.assertEqual(uxf.diff_order, 3)
        uxf = lt.Derivative(uf, "x", 4)
        self.assertEqual(uxf.__repr__(), "(∂⁴u/∂x⁴)")
        self.assertEqual(uxf.diff_order, 4)

        # Take derivative of a derivative
        uxf = lt.Derivative(uf, "x")
        ux2f = lt.Derivative(uxf, "x")
        self.assertEqual(ux2f.__repr__(), "(∂²u/∂x²)")
        self.assertEqual(ux2f.diff_order, 2)
    
    def test_multiply_derivative(self):

        uf = lt.Function("u", maxf=3, maxd=4)
        uxf = lt.Derivative(uf, "x")

        # Multiply function with derivative
        wf = uf * uxf 
        self.assertEqual(wf.__repr__(), "u \u00D7 (∂u/∂x)")
        self.assertEqual(wf.func_order, 2)
        self.assertEqual(wf.diff_order, 1)

        # Multiply derivative with derivative
        wf = uxf * uxf 
        self.assertEqual(wf.__repr__(), "(∂u/∂x) \u00D7 (∂u/∂x)")
        self.assertEqual(wf.func_order, 2)
        self.assertEqual(wf.diff_order, 2)
    
    # Testing base expression building
    def test_build_base_expr_1(self):

        uf = lt.Function("u", maxf=1, maxd=3)
        funcs = [uf]
        ivars = ["x"]
        constraints = {
            "func_order": 3,
            "diff_order": 2
        }
        base = lt.build_base_expr(funcs, ivars, constraints)
        self.assertEqual(len(base), 4)
        self.assertEqual(base[0].__repr__(), "u")
        self.assertEqual(base[1].__repr__(), "(∂u/∂x)")
        self.assertEqual(base[2].__repr__(), "(∂²u/∂x²)")
        self.assertEqual(base[3].__repr__(), "1")
    
    def test_build_base_expr_2(self):

        uf = lt.Function("u", maxf=1, maxd=3)
        funcs = [uf]
        ivars = ["x", "y"]
        constraints = {
            "func_order": 3,
            "diff_order": 2
        }
        base = lt.build_base_expr(funcs, ivars, constraints)

        base_exp = ["u", 
                    "(∂u/∂x)", 
                    "(∂u/∂y)", 
                    "(∂²u/∂x²)", 
                    "(∂²u/∂x∂y)", 
                    "(∂²u/∂y²)", 
                    "1"]

        n = len(base_exp)
        self.assertEqual(len(base), n)
        for i in range(n):
            self.assertEqual(base[i].__repr__(), base_exp[i])
    
    def test_build_base_expr_3(self):

        uf = lt.Function("u", maxf=1, maxd=3)
        vf = lt.Function("v", maxf=2, maxd=1)
        funcs = [uf, vf]
        ivars = ["x"]
        constraints = {
            "func_order": 3,
            "diff_order": 2
        }
        base = lt.build_base_expr(funcs, ivars, constraints)

        base_exp = ["u", 
                    "v", 
                    "(∂u/∂x)", 
                    "(∂v/∂x)", 
                    "(∂²u/∂x²)", 
                    "1"]
        
        n = len(base_exp)
        self.assertEqual(len(base), n)
        for i in range(n):
            self.assertEqual(base[i].__repr__(), base_exp[i])

    def test_build_library_expr_with_base(self):

        uf = lt.Function("u", maxf=1, maxd=3)
        vf = lt.Function("v", maxf=2, maxd=1)
        funcs = [uf, vf]
        ivars = ["x"]
        constraints = {
            "func_order": 3,
            "diff_order": 2
        }
        base = lt.build_base_expr(funcs, ivars, constraints)
        lib = lt.build_library_expr_with_base(funcs, ivars, constraints, base)
        # lib = list(lib)

        lib_exp = ["u \u00D7 v \u00D7 v",
                   "u \u00D7 v \u00D7 (∂v/∂x)", 
                   "u \u00D7 v", 
                   "u \u00D7 (∂v/∂x)", 
                   "u", 
                   "v \u00D7 v \u00D7 (∂u/∂x)", 
                   "v \u00D7 v \u00D7 (∂²u/∂x²)", 
                   "v \u00D7 v", 
                   "v \u00D7 (∂u/∂x) \u00D7 (∂v/∂x)", 
                   "v \u00D7 (∂u/∂x)", 
                   "v \u00D7 (∂v /∂x)", 
                   "v \u00D7 (∂²u /∂x²)", 
                   "v", 
                   "(∂v/∂x) \u00D7 (∂u/∂x)", 
                   "(∂u/∂x)", 
                   "(∂v/∂x)", 
                   "(∂²u/∂x²)", 
                   "1"]
        
        n = len(lib_exp)

        self.assertEqual(len(list(lib)), n)
        for term, i in enumerate(lib):
            self.assertEqual(term, lib_exp[i])


if __name__ == "__main__":
    unittest.main()

