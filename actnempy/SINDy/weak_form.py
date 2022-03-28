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


def velocity_libraries(u_all,
                       v_all,
                       Qxx_all,
                       Qxy_all,
                       metadata,
                       p=6,
                       r=1):


    sample = metadata['sample']
    num_windows = metadata['num_windows']
    window_size = metadata['window_size']
    (wx, wy, wt) = window_size

    dx = metadata["dx"] * wx / 2 
    dy = metadata["dy"] * wy / 2
    dt = metadata["dt"] * wt / 2

    dx_grid = metadata["dx"]
    dy_grid = metadata["dy"]
    dt_grid = metadata["dt"]
    grid3D = Grid(h=(dx_grid, dy_grid, dt_grid), ndims=3, boundary='regular')

    pts_per_window = wx * wy * wt
    num_points = num_windows * pts_per_window

    # Create arrays to store functions and derivatives
    grad_sq_U_int = np.zeros(num_windows)
    grad_4_U_int = np.zeros(num_windows)
    div_Q_int = np.zeros(num_windows)
    Q_dot_div_Q_int = np.zeros(num_windows)
    U_int = np.zeros(num_windows)
    dt_U_int = np.zeros(num_windows)
    U_dot_grad_U_int = np.zeros(num_windows)
    
    Q_dot_U_int = np.zeros(num_windows)
    TrQ2_U_int = np.zeros(num_windows)

    diff_order = 1

    nb = int(np.ceil(diff_order/2.0))
    inner_view = (slice(nb, -nb), slice(nb, -nb), slice(nb, -nb))
    _, views, = get_random_sample(
        u_all, num_windows, window_size, diff_order, sample)  
    psi = TestFunction(p, p, r, window_size, dx=dx, dy=dy, dt=dt)
    dx_psi = psi.grad(1, 0, 0)
    dy_psi = psi.grad(0, 1, 0)
    
    dx2_psi = psi.grad(2, 0, 0)
    dy2_psi = psi.grad(0, 2, 0)
    dx_dy_psi = psi.grad(1, 1, 0)
    dy_dt_psi = psi.grad(0, 1, 1)
    dx_dt_psi = psi.grad(1, 0, 1)
    

    dy_grad_sq_psi = psi.grad(2, 1, 0) + psi.grad(0, 3, 0)
    dx_grad_sq_psi = psi.grad(3, 0, 0) + psi.grad(1, 2, 0)
    
    dy_grad_4_psi = psi.grad(4, 1, 0) + psi.grad(2, 3, 0) + psi.grad(0, 5, 0)
    dx_grad_4_psi = psi.grad(5, 0, 0) + psi.grad(3, 2, 0) + psi.grad(1, 4, 0)
    

    for key, view in tqdm(views.items()):

        u_view = u_all[view].copy()
        v_view = v_all[view].copy()
        Qxx_view = Qxx_all[view].copy()
        Qxy_view = Qxy_all[view].copy()
        
        u_view = u_all[view]
        v_view = v_all[view]
        Qxx_view = Qxx_all[view]
        Qxy_view = Qxy_all[view]
        
        u = u_view[inner_view] 
        v = v_view[inner_view]
        Qxx = Qxx_view[inner_view]
        Qxy = Qxy_view[inner_view]

        (QXXx_view, QXXy_view, QXXt_view) = grid3D.grad(Qxx_view)
        (QXYx_view, QXYy_view, QXYt_view) = grid3D.grad(Qxy_view)

        dx_Qxx = QXXx_view[inner_view]
        dy_Qxx = QXXy_view[inner_view]
        dt_Qxx = QXXt_view[inner_view]
        dx_Qxy = QXYx_view[inner_view]
        dy_Qxy = QXYy_view[inner_view]
        dt_Qxy = QXYt_view[inner_view]

        dt_U_int[key] = - np.mean(u * dy_dt_psi - v * dx_dt_psi)
        
        U_dot_grad_U_int[key] = np.mean(u * v * (dy2_psi - dx2_psi) + (u**2 - v**2) * dx_dy_psi )

        grad_sq_U_int[key] = np.mean(u * dy_grad_sq_psi - v * dx_grad_sq_psi)
        
        grad_4_U_int[key] = np.mean(u * dy_grad_4_psi - v * dx_grad_4_psi)
        
        div_Q_int[key] = np.mean(Qxy * (dx2_psi - dy2_psi) - 2 * Qxx * dx_dy_psi)
        
        divQx = dx_Qxx + dy_Qxy
        divQy = dx_Qxy - dy_Qxx

        QdivQx = Qxx * divQx + Qxy * divQy
        QdivQy = Qxy * divQx - Qxx * divQy

        Q_dot_div_Q_int[key] = np.mean(QdivQx * dy_psi - QdivQy * dx_psi)

        U_int[key] = np.mean(u * dy_psi - v * dx_psi)
        
        Q_dot_U_int[key] = np.mean( (Qxx*u+Qxy*v) * dy_psi - (Qxy*u-Qxx*v) * dx_psi)
        
        TrQ2_U_int[key] = np.mean( 2 * (Qxx**2 + Qxy**2) * (u * dy_psi - v * dx_psi) )

    lib_NS = np.array(
        [
            ('u.∇u', U_dot_grad_U_int),
            ('∇²u', grad_sq_U_int),
            ('∇⁴u', grad_4_U_int),
            ('∇·Q', div_Q_int),
            ('Q·∇·Q', Q_dot_div_Q_int),
            ('u', U_int),
            ('Q.u', Q_dot_U_int),
            ('(TrQ²)u', TrQ2_U_int)
        ],
        dtype=[('name', 'U50'), ('val', 'float', U_int.shape)]
    )
    
    lib_St = np.array(
        [
            ('∇·Q', div_Q_int),
            ('Q·∇·Q', Q_dot_div_Q_int),
            ('∇⁴u', grad_4_U_int),
            ('u', U_int),
            ('Q.u', Q_dot_U_int),
            ('(TrQ²)u', TrQ2_U_int)

        ],
        dtype=[('name', 'U50'), ('val', 'float', U_int.shape)]
    )

    return (dt_U_int, lib_NS, grad_sq_U_int, lib_St)

def q_tensor(u_all,
             v_all,
             Qxx_all,
             Qxy_all,
             metadata,
             p=3):


    sample = metadata['sample']
    num_windows = metadata['num_windows']
    window_size = metadata['window_size']
    (wx, wy, wt) = window_size

    dx = metadata["dx"] * wx / 2 
    dy = metadata["dy"] * wy / 2
    dt = metadata["dt"] * wt / 2
    
    dx_grid = metadata["dx"]
    dy_grid = metadata["dy"]
    dt_grid = metadata["dt"]
    # This means we need to calculate all the derivatives from scratch.
    grid3D = Grid(h=(dx_grid, dy_grid, dt_grid), ndims=3, boundary='regular')

    pts_per_window = wx * wy * wt
    num_points = num_windows * pts_per_window

    # Create arrays to store functions and derivatives
    dt_Q_int = np.zeros(num_windows)
    U_dot_grad_Q_int = np.zeros(num_windows)
    vort_int = np.zeros(num_windows)
    shear_int = np.zeros(num_windows)
    Q_int = np.zeros(num_windows)
    trQ2_Q_int = np.zeros(num_windows)
    grad_sq_Q_int = np.zeros(num_windows)
    
    diff_order = 1

    nb = int(np.ceil(diff_order/2.0))
    inner_view = (slice(nb, -nb), slice(nb, -nb), slice(nb, -nb))

    _, views, = get_random_sample(
        u_all, num_windows, window_size, diff_order, sample)
    
    Rxxf = TestRxx(p, p, 1, window_size, dx=dx, dy=dy, dt=dt)
    Rxyf = TestRxy(p, p, 1, window_size, dx=dx, dy=dy, dt=dt)
    
    Rxx = Rxxf.arr()
    Rxy = Rxyf.arr()

    dx_Rxx = Rxxf.grad(1, 0, 0)
    dx_Rxy = Rxyf.grad(1, 0, 0)
    
    dy_Rxx = Rxxf.grad(0, 1, 0)
    dy_Rxy = Rxyf.grad(0, 1, 0)
    
    dt_Rxx = Rxxf.grad(0, 0, 1)
    dt_Rxy = Rxyf.grad(0, 0, 1)
    
    dx2_Rxx = Rxxf.grad(2, 0, 0)
    dy2_Rxx = Rxxf.grad(0, 2, 0)
    dx_dy_Rxx = Rxxf.grad(1, 1, 0)
    dy_dt_Rxx = Rxxf.grad(0, 1, 1)
    dx_dt_Rxx = Rxxf.grad(1, 0, 1)

    dx2_Rxy = Rxyf.grad(2, 0, 0)
    dy2_Rxy = Rxyf.grad(0, 2, 0)
    dx_dy_Rxy = Rxyf.grad(1, 1, 0)
    dy_dt_Rxy = Rxyf.grad(0, 1, 1)
    dx_dt_Rxy = Rxyf.grad(1, 0, 1)

    for key, view in tqdm(views.items()):

        u_view = u_all[view]
        v_view = v_all[view]
        Qxx_view = Qxx_all[view]
        Qxy_view = Qxy_all[view]
        
        u = u_view[inner_view] 
        v = v_view[inner_view]
        Qxx = Qxx_view[inner_view]
        Qxy = Qxy_view[inner_view]

        (QXXx_view, QXXy_view, QXXt_view) = grid3D.grad(Qxx_view)
        (QXYx_view, QXYy_view, QXYt_view) = grid3D.grad(Qxy_view)

        dx_Qxx = QXXx_view[inner_view]
        dy_Qxx = QXXy_view[inner_view]
        dt_Qxx = QXXt_view[inner_view]
        dx_Qxy = QXYx_view[inner_view]
        dy_Qxy = QXYy_view[inner_view]
        dt_Qxy = QXYt_view[inner_view]

        nu = Rxx * Qxy - Rxy * Qxx

        dx_nu = (dx_Rxx * Qxy + Rxx * dx_Qxy) - (dx_Rxy * Qxx + Rxy * dx_Qxx)
        dy_nu = (dy_Rxx * Qxy + Rxx * dy_Qxy) - (dy_Rxy * Qxx + Rxy * dy_Qxx)
        
        
        dt_Q_int[key] = -2 * np.mean(Qxx * dt_Rxx + Qxy * dt_Rxy)
        
        U_dot_grad_Q_int[key] = -2 * np.mean( Qxx * (u*dx_Rxx + v*dy_Rxx) + Qxy * (u*dx_Rxy + v*dy_Rxy) )

        vort_int[key] = 2.0 * np.mean( u * dy_nu - v * dx_nu )
        
        shear_int[key] = - np.mean( u * dx_Rxx + v * dx_Rxy + u * dy_Rxy - v * dy_Rxx )

        Q_int[key] = 2 * np.mean( Qxx * Rxx + Qxy * Rxy )

        trQ2_Q_int[key] = 2 * np.mean( 2 * (Qxx**2 + Qxy**2) * (Qxx * Rxx + Qxy * Rxy) )
        
        grad_sq_Q_int[key] = 2 * np.mean( Qxx * (dx2_Rxx + dy2_Rxx) + Qxy * (dx2_Rxy + dy2_Rxy) )

    Dt_Q_int = dt_Q_int + U_dot_grad_Q_int + vort_int

    lib_Q = np.array(
        [
            ('u.∇Q', U_dot_grad_Q_int),
            ('Ω.Q-Q.Ω', vort_int),
            ('E', shear_int),
            ('Q', Q_int),
            ('TrQ² \u00D7 Q', trQ2_Q_int),
            ('∇²Q', grad_sq_Q_int)
        ],
        dtype=[('name', 'U50'), ('val', 'float', U_dot_grad_Q_int.shape)]
    )
    
    lib_DQ = np.array(
        [   
            ('E', shear_int),
            ('Q', Q_int),
            ('TrQ² \u00D7 Q', trQ2_Q_int),
            ('∇²Q', grad_sq_Q_int)
        ],
        dtype=[('name', 'U50'), ('val', 'float', U_dot_grad_Q_int.shape)]
    )
    
    return (dt_Q_int, lib_Q, Dt_Q_int, lib_DQ)

if __name__ == "__main__":
    pass
