import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
from ..utils import get_random_sample, Grid
from tqdm import tqdm
# Check if numpy.random.default_rng is available
no_rng = False
try:
    from numpy.random import default_rng
except ImportError:
    no_rng = True

class TestFunction:
    def __init__(self, p, q, r, window_size, dx, dy, dt):
        
        self.x, self.y, self.t = sym.symbols('x y t')
        self.p = p
        self.q = q
        self.r = r
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.window_size = window_size
    
        xvals = np.linspace(-1,1,window_size[0])
        yvals = np.linspace(-1,1,window_size[1])
        tvals = np.linspace(-1,1,window_size[2])
        self.Xm, self.Ym, self.Tm = np.meshgrid(xvals, yvals, tvals, indexing='ij')
    
        self._define_psi()
    
    def _define_psi(self):
        π = np.pi
        p = self.p
        q = self.q
        r = self.r
        if r is None:
            self.ψ = (self.x**2 -1)**p * (self.y**2 - 1)**q
        else:
            self.ψ = sym.sin(r*π*self.t) * (self.x**2 -1)**p * (self.y**2 - 1)**q
    
    def __str__(self):
        return str(self.ψ)
    
    def grad(self, nx=0, ny=0, nt=0):
        dψ = sym.diff(self.ψ ,self.x, nx, self.y, ny, self.t, nt)

        dpsi = sym.lambdify([self.x, self.y, self.t], dψ, "numpy")

        factor = 1.0 / (self.dx**nx * self.dy**ny * self.dt**nt)
        # return dpsi((self.Xm.T, self.Ym.T, self.Tm.T))
        return factor * dpsi(self.Xm, self.Ym, self.Tm)
    
    def arr(self):

        psi = sym.lambdify([self.x, self.y, self.t], self.ψ, "numpy")

        # return dpsi((self.Xm.T, self.Ym.T, self.Tm.T))
        return psi(self.Xm, self.Ym, self.Tm)
    
    def grad_sym(self, nx=0, ny=0, nt=0):
        return sym.diff(self.ψ ,self.x, nx, self.y, ny, self.t, nt)
    
    
class TestRxx(TestFunction):
    def _define_psi(self):
        π = np.pi
        p = self.p
        q = self.q
        r = self.r
        # self.ψ = sym.sin(r*π*self.t) * (self.x**2 -1)**p * (self.y**2 - 1)**q
        if r is None:
            self.ψ = (self.x**2 -1)**p * (self.y**2 - 1)**q * sym.cos(self.x)
        else:
            self.ψ = sym.sin(r*π*self.t) * (self.x**2 -1)**p * (self.y**2 - 1)**q * sym.cos(self.x)

class TestRxy(TestFunction):
    def _define_psi(self):
        π = np.pi
        p = self.p
        q = self.q
        r = self.r
        # self.ψ = sym.sin(r*π*self.t) * (self.x**2 -1)**p * (self.y**2 - 1)**q
        if r is None:
            self.ψ = (self.x**2 -1)**p * (self.y**2 - 1)**q * sym.sin(self.x)
        else:
            self.ψ = sym.sin(r*π*self.t) * (self.x**2 -1)**p * (self.y**2 - 1)**q * sym.sin(self.x)

if __name__ == "__main__":
    pass
