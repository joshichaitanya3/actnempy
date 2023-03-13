'''

Active Nematic Identification of Sparse Equations (ANISE)

Main module (Anise)

Code by Chaitanya Joshi (chaitanya@brandeis.edu)

The main object of this module is the Class Anise, which is designed to handle model identification for 2D active nematic data of Q-tensor and velocity. 

'''

from ..utils import func_defectfind, func_defectpos, func_defectorient, func_plotdefects
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np 
import matplotlib.pyplot as plt 
import json
from ..utils import Grid, nematic_plot, add_noise, compute_n, get_random_sample
from .library_tools import Function, build_base_expr, build_constrained_library_array, get_term_val
from .weak_form import TestFunction, TestRxx, TestRxy
from .pde import PDE
from tqdm import tqdm
from pathlib import Path
from scipy.signal import correlate
from ..actnem import ActNem

class Anise(ActNem):
    '''
    
    This class is designed to handle model identification for 2D active nematic data of Q-tensor and velocity. 
    
    Attributes
    ----------

    data_dir : str
        Path to the data directory. This directory must contain the following files:

            `processed_data.npz`: A single .npz file containing 4 arrays: `Qxx_all`, `Qxy_all`, `u_all` and `v_all`, each of dimensions (NX, NY, NT), with X-Y being the spatial dimensions and T being the time dimension. The preprocessing of the experimental / simulation data into this format is done elsewhere.

            `metadata.json` : A json file containing three keys: 'dx', 'dy' and 'dt', specifying the spatial and temoral discretization of the data.
    
    visual_check : bool
        An optional flag to plot the first five frames of the data as a visual check.

    '''
    def __init__(self, data_dir,visual_check=False, run=False):

        self.data_dir = data_dir

        # Setup a directory to store visualizations, if any
        self.vis_dir = f"{data_dir}/visualizations"
        if not os.path.exists(self.vis_dir):
            os.makedirs(self.vis_dir)

        # Import data
        self.processed_data_file = f'{data_dir}/processed_data.npz'
        if os.path.isfile(self.processed_data_file):
            print("Loading all the data...",flush=True)
            with np.load(self.processed_data_file) as data:
                self.u_all = data['u_all']
                self.v_all = data['v_all']
                self.Qxx_all = data['Qxx_all']
                self.Qxy_all = data['Qxy_all']

            # Check that the dimensions match
            vals = [self.u_all.shape, 
                    self.v_all.shape, 
                    self.Qxx_all.shape, 
                    self.Qxy_all.shape]

            if not all(v==self.u_all.shape for v in vals):
                msg = "The shapes of the arrays don't match!"
                raise ValueError(msg)
            else:
                self.size = self.u_all.shape
                print(f'Data loaded. Dimensions: {self.size}')
        else:
            msg = f"Couldn't find the data at {self.processed_data_file}!"
            raise ValueError(msg)
        
        # Import metadata

        with open(f'{data_dir}/sindy_library_specs.json', 'r') as f:
            self.metadata = json.load(f)
        
        with open(f'{data_dir}/metadata.json', 'r') as f:

            self.metadata.update(json.load(f))
            self.dx = self.metadata['dx']
            self.dy = self.metadata['dy']
            self.dt = self.metadata['dt']
        
        self.metadata["data_dir"] = self.data_dir

        (self.NX, self.NY, self.NT) = self.u_all.shape
        self.x = self.dx * np.arange(self.NX)
        self.y = self.dy * np.arange(self.NY)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        self.grid2D = Grid(h=(self.dx,self.dy),boundary="regular",ndims=2)
        
        if visual_check: # Visualize the first 5 frames as a check
            print("Visualizing the first 5 frames...")
            self.visualize(5) 
        
        if run:
            self.sindy_int()

    def generate_libraries_int(self,
                           overdamped=False
                           ):
        '''
        (lib_lhs,lib_Q,lib_Stokes,lib_overdamped) = generate_libraries_int()

        Function to generate SINDy libraries for the Q-tensor and the flow equation. 

        Returns
        -------

        lib_lhs : dtype=[('name','U50'),('val','complex128' (num_windows,))]
            Array containing the terms on the left hand side.
        
        lib_Q : dtype=[('name','U50'),('val','complex128', (num_windows,))] 
            Array containing the terms on the RHS of the Q-tensor equations
        
        lib_Stokes : dtype=[('name','U50'),('val','complex128', (num_windows,))] 
            Array containing the terms on the RHS of the Stokes equation

        lib_overdamped : dtype=[('name','U50'),('val','complex128', (num_windows,))] 
            Array containing the terms on the RHS of the overdamped flow equation

        '''
        sample = self.metadata['sample']
        num_windows = self.metadata['num_windows']
        window_size = self.metadata['window_size']

        
        ## Generate the Function objects and base expressions

        ##### Q tensor ########
        #  
        coords = ['x','y']

        mQ = self.metadata["Q tensor"]

        ufQ = Function('u',**mQ['u'])
        vfQ = Function('v',**mQ['v'])
        QxxfQ = Function('Qxx',**mQ['Qxx'])
        QxyfQ = Function('Qxy',**mQ['Qxy'])

        funcsQ = [ufQ,vfQ,QxxfQ,QxyfQ]
        
        constraints_Q = mQ['constraints']

        base_Q = build_base_expr(funcsQ,coords,constraints_Q)

        # Remove terms containing (∂v/∂y) as they are proportional to 
        # (∂u/∂x) due to incompressibility.

        base_Q = [item for item in base_Q if item.__repr__()!='(∂v/∂y)']
        base_Q = [item for item in base_Q if item.__repr__()!='(∂²v/∂x∂y)']
        base_Q = [item for item in base_Q if item.__repr__()!='(∂²v/∂y²)']

        ####### Fluid #########

        mSt = self.metadata["Fluid"]

        wSt = Function('ω',**mSt["ω"])
        QxxSt = Function('Qxx',**mSt["Qxx"])
        QxySt = Function('Qxy',**mSt["Qxy"])
        uSt = Function('Qxx',**mSt["u"])
        vSt = Function('Qxy',**mSt["v"])

        constraints_Stokes = mSt["constraints"]
        
        funcsStokes = [uSt, vSt, wSt, QxxSt,QxySt]

        base_Stokes = build_base_expr(funcsStokes,
                                        coords,
                                        constraints_Stokes)
        
        ####### Over-damped #########
        if overdamped:
            mOd = self.metadata["Overdamped"]

            QxxOd = Function('Qxx',**mOd["Qxx"])
            QxyOd = Function('Qxy',**mOd["Qxy"])

            constraints_Od = mOd["constraints"]
            
            funcsOd = [QxxOd,QxyOd]

            base_Od = build_base_expr(funcsOd,
                                        coords,
                                        constraints_Od)
            

        arr_type = 'complex128'
        
        pts_per_window = window_size**3
        num_points = num_windows * pts_per_window

        # Initialize arrays to store libraries with 'num_windows' points
        W = np.zeros((num_windows,1))
        U = np.zeros((num_windows,1))
        V = np.zeros((num_windows,1))

        QXX = np.zeros((num_windows,1))
        QXY = np.zeros((num_windows,1))

        Wt = np.zeros((num_windows,1))
        Wx = np.zeros((num_windows,1))
        Wy = np.zeros((num_windows,1))

        Ut = np.zeros((num_windows,1))
        Ux = np.zeros((num_windows,1))
        Uy = np.zeros((num_windows,1))

        Vt = np.zeros((num_windows,1))
        Vx = np.zeros((num_windows,1))

        QXXt = np.zeros((num_windows,1))
        QXXx = np.zeros((num_windows,1))
        QXXy = np.zeros((num_windows,1))

        QXYt = np.zeros((num_windows,1))
        QXYx = np.zeros((num_windows,1))
        QXYy = np.zeros((num_windows,1))

        Wxx = np.zeros((num_windows,1))
        Wxy = np.zeros((num_windows,1))
        Wyy = np.zeros((num_windows,1))

        Uxx = np.zeros((num_windows,1))
        Uxy = np.zeros((num_windows,1))
        Uyy = np.zeros((num_windows,1))

        Vxx = np.zeros((num_windows,1))

        QXXxx = np.zeros((num_windows,1))
        QXXxy = np.zeros((num_windows,1))
        QXXyy = np.zeros((num_windows,1))

        QXYxx = np.zeros((num_windows,1))
        QXYxy = np.zeros((num_windows,1))
        QXYyy = np.zeros((num_windows,1))

        Wxxx = np.zeros((num_windows,1))
        Wxxy = np.zeros((num_windows,1))
        Wxyy = np.zeros((num_windows,1))
        Wyyy = np.zeros((num_windows,1))

        Uxxx = np.zeros((num_windows,1))
        Uxxy = np.zeros((num_windows,1))
        Uxyy = np.zeros((num_windows,1))
        Uyyy = np.zeros((num_windows,1))

        Vxxx = np.zeros((num_windows,1))

        QXXxxx = np.zeros((num_windows,1))
        QXXxxy = np.zeros((num_windows,1))
        QXXxyy = np.zeros((num_windows,1))
        QXXyyy = np.zeros((num_windows,1))

        QXYxxx = np.zeros((num_windows,1))
        QXYxxy = np.zeros((num_windows,1))
        QXYxyy = np.zeros((num_windows,1))
        QXYyyy = np.zeros((num_windows,1))

        Uxxxx = np.zeros((num_windows,1))
        Uxxxy = np.zeros((num_windows,1))
        Uxxyy = np.zeros((num_windows,1))
        Uxyyy = np.zeros((num_windows,1))
        Uyyyy = np.zeros((num_windows,1))

        Vxxxx = np.zeros((num_windows,1))

        QXXxxxx = np.zeros((num_windows,1))
        QXXxxxy = np.zeros((num_windows,1))
        QXXxxyy = np.zeros((num_windows,1))
        QXXxyyy = np.zeros((num_windows,1))
        QXXyyyy = np.zeros((num_windows,1))

        QXYxxxx = np.zeros((num_windows,1))
        QXYxxxy = np.zeros((num_windows,1))
        QXYxxyy = np.zeros((num_windows,1))
        QXYxyyy = np.zeros((num_windows,1))
        QXYyyyy = np.zeros((num_windows,1))
        
        # Initialize the library arrays in which we would later store the
        # regionwise averaged values one by one.

        lib_lhs = np.array(
            [
                ('(∂Qxx/∂t)',QXXt.flatten()),
                ('(∂Qxy/∂t)',QXYt.flatten()),
                ('(∂ω/∂t)',Wt.flatten()),
                ('∇²ω',(Wxx + Wyy).flatten()),
                ('u',U.flatten()),
                ('v',V.flatten())
            ],
            dtype=[('name','U50'),('val',arr_type,U.flatten().shape)]
        )


        db_Q = np.array(
            [
                ('u',U.flatten()),
                ('v',V.flatten()),
                ('Qxx',QXX.flatten()),
                ('Qxy',QXY.flatten()),
                ('(∂u/∂x)',Ux.flatten()),
                ('(∂v/∂x)',Vx.flatten()),
                ('(∂Qxx/∂x)',QXXx.flatten()),
                ('(∂Qxy/∂x)',QXYx.flatten()),
                ('(∂u/∂y)',Uy.flatten()),
                ('(∂Qxx/∂y)',QXXy.flatten()),
                ('(∂Qxy/∂y)',QXYy.flatten()),
                ('(∂²u/∂x²)',Uxx.flatten()),
                ('(∂²v/∂x²)',Vxx.flatten()),
                ('(∂²Qxx/∂x²)',QXXxx.flatten()),
                ('(∂²Qxy/∂x²)',QXYxx.flatten()),
                ('(∂²u/∂x∂y)',Uxy.flatten()),
                ('(∂²Qxx/∂x∂y)',QXXxy.flatten()),
                ('(∂²Qxy/∂x∂y)',QXYxy.flatten()),
                ('(∂²u/∂y²)',Uyy.flatten()),
                ('(∂²Qxx/∂y²)',QXXyy.flatten()),
                ('(∂²Qxy/∂y²)',QXYyy.flatten()),
                ('1',np.ones(U.flatten().shape))
            ],
            dtype=[('name','U50'),('val',arr_type,U.flatten().shape)]
        )

        db_Stokes = np.array(
            [   
                ('u',U.flatten()),
                ('v',V.flatten()),
                ('ω',W.flatten()),
                ('(∂ω/∂x)',Wx.flatten()),
                ('(∂ω/∂y)',Wy.flatten()),
                ('Qxx',QXX.flatten()),
                ('Qxy',QXY.flatten()),
                ('(∂Qxx/∂x)',QXXx.flatten()),
                ('(∂Qxy/∂x)',QXYx.flatten()),
                ('(∂Qxx/∂y)',QXXy.flatten()),
                ('(∂Qxy/∂y)',QXYy.flatten()),
                ('(∂²Qxx/∂x²)',QXXxx.flatten()),
                ('(∂²Qxy/∂x²)',QXYxx.flatten()),
                ('(∂²Qxx/∂x∂y)',QXXxy.flatten()),
                ('(∂²Qxy/∂x∂y)',QXYxy.flatten()),
                ('(∂²Qxx/∂y²)',QXXyy.flatten()),
                ('(∂²Qxy/∂y²)',QXYyy.flatten()),
                ('(∂³Qxx/∂x³)',QXXxxx.flatten()),
                ('(∂³Qxy/∂x³)',QXYxxx.flatten()),
                ('(∂³Qxx/∂x²∂y)',QXXxxy.flatten()),
                ('(∂³Qxy/∂x²∂y)',QXYxxy.flatten()),
                ('(∂³Qxx/∂x∂y²)',QXXxyy.flatten()),
                ('(∂³Qxy/∂x∂y²)',QXYxyy.flatten()),
                ('(∂³Qxx/∂y³)',QXXyyy.flatten()),
                ('(∂³Qxy/∂y³)',QXYyyy.flatten()),
                ('(∂⁴Qxx/∂x⁴)',QXXxxxx.flatten()),
                ('(∂⁴Qxy/∂x⁴)',QXYxxxx.flatten()),
                ('(∂⁴Qxx/∂x³∂y)',QXXxxxy.flatten()),
                ('(∂⁴Qxy/∂x³∂y)',QXYxxxy.flatten()),
                ('(∂⁴Qxx/∂x²∂y²)',QXXxxyy.flatten()),
                ('(∂⁴Qxy/∂x²∂y²)',QXYxxyy.flatten()),
                ('(∂⁴Qxx/∂x∂y³)',QXXxyyy.flatten()),
                ('(∂⁴Qxy/∂x∂y³)',QXYxyyy.flatten()),
                ('(∂⁴Qxx/∂y⁴)',QXXyyyy.flatten()),
                ('(∂⁴Qxy/∂y⁴)',QXYyyyy.flatten()),
                ('1',np.ones(U.flatten().shape))
            ],
            dtype=[('name','U50'),('val',arr_type,U.flatten().shape)]
        )
                
        if overdamped:
            db_overdamped = np.array(
                [
                    ('Qxx',QXX.flatten()),
                    ('Qxy',QXY.flatten()),
                    ('(∂Qxx/∂x)',QXXx.flatten()),
                    ('(∂Qxy/∂x)',QXYx.flatten()),
                    ('(∂Qxx/∂y)',QXXy.flatten()),
                    ('(∂Qxy/∂y)',QXYy.flatten()),
                    ('(∂²Qxx/∂x²)',QXXxx.flatten()),
                    ('(∂²Qxy/∂x²)',QXYxx.flatten()),
                    ('(∂²Qxx/∂x∂y)',QXXxy.flatten()),
                    ('(∂²Qxy/∂x∂y)',QXYxy.flatten()),
                    ('(∂²Qxx/∂y²)',QXXyy.flatten()),
                    ('(∂²Qxy/∂y²)',QXYyy.flatten()),
                    ('(∂³Qxx/∂x³)',QXXxxx.flatten()),
                    ('(∂³Qxy/∂x³)',QXYxxx.flatten()),
                    ('(∂³Qxx/∂x²∂y)',QXXxxy.flatten()),
                    ('(∂³Qxy/∂x²∂y)',QXYxxy.flatten()),
                    ('(∂³Qxx/∂x∂y²)',QXXxyy.flatten()),
                    ('(∂³Qxy/∂x∂y²)',QXYxyy.flatten()),
                    ('(∂³Qxx/∂y³)',QXXyyy.flatten()),
                    ('(∂³Qxy/∂y³)',QXYyyy.flatten()),
                    ('1',np.ones(U.flatten().shape))
                ],
                dtype=[('name','U50'),('val',arr_type,U.flatten().shape)]
            )


        lib_Q = build_constrained_library_array(funcsQ,
                                                base_Q,
                                                db_Q,
                                                coords,
                                                constraints_Q)

        lib_Stokes = build_constrained_library_array(funcsStokes,
                                                        base_Stokes,
                                                        db_Stokes,
                                                        coords,
                                                        constraints_Stokes)
            
        # Add time derivative term to the Stokes library
        dw_dt = get_term_val(lib_lhs,'(∂ω/∂t)')
        # Append ω to the Stokes library
        dw_dt_struct = np.array([('(∂ω/∂t)',dw_dt.flatten())],
                            dtype=lib_Stokes.dtype)
        lib_Stokes = np.append(lib_Stokes,dw_dt_struct)

        if overdamped:
            lib_overdamped = build_constrained_library_array(funcsOd,
                                                        base_Od,
                                                        db_overdamped,
                                                        coords,
                                                        constraints_Od)


        # Now create arrays to store functions and derivatives at each window
        W = np.zeros((pts_per_window,1))
        U = np.zeros((pts_per_window,1))
        V = np.zeros((pts_per_window,1))

        QXX = np.zeros((pts_per_window,1))
        QXY = np.zeros((pts_per_window,1))

        Wt = np.zeros((pts_per_window,1))
        Wx = np.zeros((pts_per_window,1))
        Wy = np.zeros((pts_per_window,1))

        Ut = np.zeros((pts_per_window,1))
        Ux = np.zeros((pts_per_window,1))
        Uy = np.zeros((pts_per_window,1))

        Vt = np.zeros((pts_per_window,1))
        Vx = np.zeros((pts_per_window,1))

        QXXt = np.zeros((pts_per_window,1))
        QXXx = np.zeros((pts_per_window,1))
        QXXy = np.zeros((pts_per_window,1))

        QXYt = np.zeros((pts_per_window,1))
        QXYx = np.zeros((pts_per_window,1))
        QXYy = np.zeros((pts_per_window,1))

        Wxx = np.zeros((pts_per_window,1))
        Wxy = np.zeros((pts_per_window,1))
        Wyy = np.zeros((pts_per_window,1))

        Uxx = np.zeros((pts_per_window,1))
        Uxy = np.zeros((pts_per_window,1))
        Uyy = np.zeros((pts_per_window,1))

        Vxx = np.zeros((pts_per_window,1))

        QXXxx = np.zeros((pts_per_window,1))
        QXXxy = np.zeros((pts_per_window,1))
        QXXyy = np.zeros((pts_per_window,1))

        QXYxx = np.zeros((pts_per_window,1))
        QXYxy = np.zeros((pts_per_window,1))
        QXYyy = np.zeros((pts_per_window,1))

        Wxxx = np.zeros((pts_per_window,1))
        Wxxy = np.zeros((pts_per_window,1))
        Wxyy = np.zeros((pts_per_window,1))
        Wyyy = np.zeros((pts_per_window,1))

        Uxxx = np.zeros((pts_per_window,1))
        Uxxy = np.zeros((pts_per_window,1))
        Uxyy = np.zeros((pts_per_window,1))
        Uyyy = np.zeros((pts_per_window,1))

        Vxxx = np.zeros((pts_per_window,1))

        QXXxxx = np.zeros((pts_per_window,1))
        QXXxxy = np.zeros((pts_per_window,1))
        QXXxyy = np.zeros((pts_per_window,1))
        QXXyyy = np.zeros((pts_per_window,1))

        QXYxxx = np.zeros((pts_per_window,1))
        QXYxxy = np.zeros((pts_per_window,1))
        QXYxyy = np.zeros((pts_per_window,1))
        QXYyyy = np.zeros((pts_per_window,1))

        Uxxxx = np.zeros((pts_per_window,1))
        Uxxxy = np.zeros((pts_per_window,1))
        Uxxyy = np.zeros((pts_per_window,1))
        Uxyyy = np.zeros((pts_per_window,1))
        Uyyyy = np.zeros((pts_per_window,1))

        Vxxxx = np.zeros((pts_per_window,1))

        QXXxxxx = np.zeros((pts_per_window,1))
        QXXxxxy = np.zeros((pts_per_window,1))
        QXXxxyy = np.zeros((pts_per_window,1))
        QXXxyyy = np.zeros((pts_per_window,1))
        QXXyyyy = np.zeros((pts_per_window,1))

        QXYxxxx = np.zeros((pts_per_window,1))
        QXYxxxy = np.zeros((pts_per_window,1))
        QXYxxyy = np.zeros((pts_per_window,1))
        QXYxyyy = np.zeros((pts_per_window,1))
        QXYyyyy = np.zeros((pts_per_window,1))
        
        dx = self.metadata["dx"]
        dy = self.metadata["dy"]
        dt = self.metadata["dt"]

        # grid to compute derivatives in space and time
        grid3D = Grid(h=(dx, dy, dt), ndims=3, boundary='regular')

        # Pick random points in 3D space.
        diff_order = 5
        n_diff = 2 * np.ceil(diff_order/2.0)
        nb = int(np.ceil(diff_order/2.0))
        inner_view = (slice(nb, -nb), slice(nb, -nb), slice(nb, -nb))
        points, views, = get_random_sample(
            self.u_all.shape, num_windows, window_size, diff_order, sample)

        print('Computing derivatives at selected points...')

        for key, view in views.items():
            
            chunk = (..., 0)
            u_view = self.u_all[view].copy()
            v_view = self.v_all[view].copy()
            Qxx_view = self.Qxx_all[view].copy()
            Qxy_view = self.Qxy_all[view].copy()

            (Ux_view, Uy_view, Ut_view) = grid3D.grad(u_view)
            (Vx_view, _, Vt_view) = grid3D.grad(v_view)
            (QXXx_view, QXXy_view, QXXt_view) = grid3D.grad(Qxx_view)
            (QXYx_view, QXYy_view, QXYt_view) = grid3D.grad(Qxy_view)

            (Uxx_view, Uxy_view, _) = grid3D.grad(Ux_view)
            (_, Uyy_view, Uyt_view) = grid3D.grad(Uy_view)
            (Vxx_view, _, Vxt_view) = grid3D.grad(Vx_view)
            (Uxxx_view, Uxxy_view, _) = grid3D.grad(Uxx_view)
            (Uxyy_view, Uyyy_view, _) = grid3D.grad(Uyy_view)
            (Vxxx_view, _, _) = grid3D.grad(Vxx_view)
            (QXXxx_view, QXXxy_view, _) = grid3D.grad(QXXx_view)
            (_, QXXyy_view, _) = grid3D.grad(QXXy_view)
            (QXYxx_view, QXYxy_view, _) = grid3D.grad(QXYx_view)
            (_, QXYyy_view, _) = grid3D.grad(QXYy_view)

            (QXXxxx_view, QXXxxy_view, _) = grid3D.grad(QXXxx_view)
            (QXXxyy_view, QXXyyy_view, _) = grid3D.grad(QXXyy_view)
            (QXYxxx_view, QXYxxy_view, _) = grid3D.grad(QXYxx_view)
            (QXYxyy_view, QXYyyy_view, _) = grid3D.grad(QXYyy_view)

            (Vxxx_view, _, _) = grid3D.grad(Vxx_view)
            W_view = Vx_view - Uy_view

            Wt_view = Vxt_view - Uyt_view

            Wx_view = Vxx_view - Uxy_view
            Wy_view = -Uxx_view - Uyy_view

            Wxx_view = Vxxx_view - Uxxy_view
            Wxy_view = -Uxxx_view - Uxyy_view
            Wyy_view = -Uxxy_view - Uyyy_view


            U[chunk] = u_view[inner_view].flatten()
            V[chunk] = v_view[inner_view].flatten()
            
            QXX[chunk] = Qxx_view[inner_view].flatten()
            QXY[chunk] = Qxy_view[inner_view].flatten()
            
            Ut[chunk] = Ut_view[inner_view].flatten()
            Vt[chunk] = Vt_view[inner_view].flatten()
            
            QXXt[chunk] = QXXt_view[inner_view].flatten()
            QXYt[chunk] = QXYt_view[inner_view].flatten()
            
            Ux[chunk] = Ux_view[inner_view].flatten()
            Uy[chunk] = Uy_view[inner_view].flatten()
            
            Uxx[chunk] = Uxx_view[inner_view].flatten()
            Uxy[chunk] = Uxy_view[inner_view].flatten()
            Uyy[chunk] = Uyy_view[inner_view].flatten()
            
            Vx[chunk] = Vx_view[inner_view].flatten()
            Vxx[chunk] = Vxx_view[inner_view].flatten()
            
            W[chunk] = W_view[inner_view].flatten()
            
            Wt[chunk] = Wt_view[inner_view].flatten()
            
            Wx[chunk] = Wx_view[inner_view].flatten()
            Wy[chunk] = Wy_view[inner_view].flatten()
            
            Wxx[chunk] = Wxx_view[inner_view].flatten()
            Wxy[chunk] = Wxy_view[inner_view].flatten()
            Wyy[chunk] = Wyy_view[inner_view].flatten()
            
            QXXx[chunk] = QXXx_view[inner_view].flatten()
            QXXy[chunk] = QXXy_view[inner_view].flatten()
            
            QXXxx[chunk] = QXXxx_view[inner_view].flatten()
            QXXxy[chunk] = QXXxy_view[inner_view].flatten()
            QXXyy[chunk] = QXXyy_view[inner_view].flatten()
            
            QXXxxx[chunk] = QXXxxx_view[inner_view].flatten()
            QXXxxy[chunk] = QXXxxy_view[inner_view].flatten()
            QXXxyy[chunk] = QXXxyy_view[inner_view].flatten()
            QXXyyy[chunk] = QXXyyy_view[inner_view].flatten()
            
            
            # QXXxxx[p] = Qxx_x_diff[2]
            # QXXxxy[p] = (Qxx_x_diff_yp[1]-Qxx_x_diff_ym[1])/(2*dy)
            # QXXxyy[p] = (Qxx_y_diff_xp[1]-Qxx_y_diff_xm[1])/(2*dx)
            # QXXyyy[p] = Qxx_y_diff[2]
            
            # QXXxxxx[p] = Qxx_x_diff[3]
            # QXXxxxy[p] = (Qxx_x_diff_yp[2]-Qxx_x_diff_ym[2])/(2*dy)
            # QXXxxyy[p] = (Qxx_x_diff_yp[1]+Qxx_x_diff_ym[1]-2*Qxx_x_diff[1])/(dy**2)
            # QXXxyyy[p] = (Qxx_y_diff_xp[2]-Qxx_y_diff_xm[2])/(2*dx)
            # QXXyyyy[p] = Qxx_y_diff[3]
            
            QXYx[chunk] = QXYx_view[inner_view].flatten()
            QXYy[chunk] = QXYy_view[inner_view].flatten()
            
            QXYxx[chunk] = QXYxx_view[inner_view].flatten()
            QXYxy[chunk] = QXYxy_view[inner_view].flatten()
            QXYyy[chunk] = QXYyy_view[inner_view].flatten()
            
            QXYxxx[chunk] = QXYxxx_view[inner_view].flatten()
            QXYxxy[chunk] = QXYxxy_view[inner_view].flatten()
            QXYxyy[chunk] = QXYxyy_view[inner_view].flatten()
            QXYyyy[chunk] = QXYyyy_view[inner_view].flatten()
            
            
            # QXYxxx[p] = Qxy_x_diff[2]
            # QXYxxy[p] = (Qxy_x_diff_yp[1]-Qxy_x_diff_ym[1])/(2*dy)
            # QXYxyy[p] = (Qxy_y_diff_xp[1]-Qxy_y_diff_xm[1])/(2*dx)
            # QXYyyy[p] = Qxy_y_diff[2]
            
            # QXYxxxx[p] = Qxy_x_diff[3]
            # QXYxxxy[p] = (Qxy_x_diff_yp[2]-Qxy_x_diff_ym[2])/(2*dy)
            # QXYxxyy[p] = (Qxy_x_diff_yp[1]+Qxy_x_diff_ym[1]-2*Qxy_x_diff[1])/(dy**2)
            # QXYxyyy[p] = (Qxy_y_diff_xp[2]-Qxy_y_diff_xm[2])/(2*dx)
            # QXYyyyy[p] = Qxy_y_diff[3]

            # Create libraries for each point / window
            lib_lhs_p = np.array(
                [
                    ('(∂Qxx/∂t)',QXXt.flatten()),
                    ('(∂Qxy/∂t)',QXYt.flatten()),
                    ('(∂ω/∂t)',Wt.flatten()),
                    ('∇²ω',(Wxx + Wyy).flatten()),
                    ('u',U.flatten()),
                    ('v',V.flatten())
                ],
                dtype=[('name','U50'),('val',arr_type,U.flatten().shape)]
            )


            db_Q_p = np.array(
                [
                    ('u',U.flatten()),
                    ('v',V.flatten()),
                    ('Qxx',QXX.flatten()),
                    ('Qxy',QXY.flatten()),
                    ('(∂u/∂x)',Ux.flatten()),
                    ('(∂v/∂x)',Vx.flatten()),
                    ('(∂Qxx/∂x)',QXXx.flatten()),
                    ('(∂Qxy/∂x)',QXYx.flatten()),
                    ('(∂u/∂y)',Uy.flatten()),
                    ('(∂Qxx/∂y)',QXXy.flatten()),
                    ('(∂Qxy/∂y)',QXYy.flatten()),
                    ('(∂²u/∂x²)',Uxx.flatten()),
                    ('(∂²v/∂x²)',Vxx.flatten()),
                    ('(∂²Qxx/∂x²)',QXXxx.flatten()),
                    ('(∂²Qxy/∂x²)',QXYxx.flatten()),
                    ('(∂²u/∂x∂y)',Uxy.flatten()),
                    ('(∂²Qxx/∂x∂y)',QXXxy.flatten()),
                    ('(∂²Qxy/∂x∂y)',QXYxy.flatten()),
                    ('(∂²u/∂y²)',Uyy.flatten()),
                    ('(∂²Qxx/∂y²)',QXXyy.flatten()),
                    ('(∂²Qxy/∂y²)',QXYyy.flatten()),
                    ('1',np.ones(U.flatten().shape))
                ],
                dtype=[('name','U50'),('val',arr_type,U.flatten().shape)]
            )

            db_Stokes_p = np.array(
                [
                    ('u',U.flatten()),
                    ('v',V.flatten()),
                    ('ω',W.flatten()),
                    ('(∂ω/∂x)',Wx.flatten()),
                    ('(∂ω/∂y)',Wy.flatten()),
                    ('Qxx',QXX.flatten()),
                    ('Qxy',QXY.flatten()),
                    ('(∂Qxx/∂x)',QXXx.flatten()),
                    ('(∂Qxy/∂x)',QXYx.flatten()),
                    ('(∂Qxx/∂y)',QXXy.flatten()),
                    ('(∂Qxy/∂y)',QXYy.flatten()),
                    ('(∂²Qxx/∂x²)',QXXxx.flatten()),
                    ('(∂²Qxy/∂x²)',QXYxx.flatten()),
                    ('(∂²Qxx/∂x∂y)',QXXxy.flatten()),
                    ('(∂²Qxy/∂x∂y)',QXYxy.flatten()),
                    ('(∂²Qxx/∂y²)',QXXyy.flatten()),
                    ('(∂²Qxy/∂y²)',QXYyy.flatten()),
                    ('(∂³Qxx/∂x³)',QXXxxx.flatten()),
                    ('(∂³Qxy/∂x³)',QXYxxx.flatten()),
                    ('(∂³Qxx/∂x²∂y)',QXXxxy.flatten()),
                    ('(∂³Qxy/∂x²∂y)',QXYxxy.flatten()),
                    ('(∂³Qxx/∂x∂y²)',QXXxyy.flatten()),
                    ('(∂³Qxy/∂x∂y²)',QXYxyy.flatten()),
                    ('(∂³Qxx/∂y³)',QXXyyy.flatten()),
                    ('(∂³Qxy/∂y³)',QXYyyy.flatten()),
                    ('(∂⁴Qxx/∂x⁴)',QXXxxxx.flatten()),
                    ('(∂⁴Qxy/∂x⁴)',QXYxxxx.flatten()),
                    ('(∂⁴Qxx/∂x³∂y)',QXXxxxy.flatten()),
                    ('(∂⁴Qxy/∂x³∂y)',QXYxxxy.flatten()),
                    ('(∂⁴Qxx/∂x²∂y²)',QXXxxyy.flatten()),
                    ('(∂⁴Qxy/∂x²∂y²)',QXYxxyy.flatten()),
                    ('(∂⁴Qxx/∂x∂y³)',QXXxyyy.flatten()),
                    ('(∂⁴Qxy/∂x∂y³)',QXYxyyy.flatten()),
                    ('(∂⁴Qxx/∂y⁴)',QXXyyyy.flatten()),
                    ('(∂⁴Qxy/∂y⁴)',QXYyyyy.flatten()),
                    ('1',np.ones(U.flatten().shape))
                ],
                dtype=[('name','U50'),('val',arr_type,U.flatten().shape)]
            )
                    
            if overdamped:
                db_overdamped_p = np.array(
                    [
                        ('Qxx',QXX.flatten()),
                        ('Qxy',QXY.flatten()),
                        ('(∂Qxx/∂x)',QXXx.flatten()),
                        ('(∂Qxy/∂x)',QXYx.flatten()),
                        ('(∂Qxx/∂y)',QXXy.flatten()),
                        ('(∂Qxy/∂y)',QXYy.flatten()),
                        ('(∂²Qxx/∂x²)',QXXxx.flatten()),
                        ('(∂²Qxy/∂x²)',QXYxx.flatten()),
                        ('(∂²Qxx/∂x∂y)',QXXxy.flatten()),
                        ('(∂²Qxy/∂x∂y)',QXYxy.flatten()),
                        ('(∂²Qxx/∂y²)',QXXyy.flatten()),
                        ('(∂²Qxy/∂y²)',QXYyy.flatten()),
                        ('(∂³Qxx/∂x³)',QXXxxx.flatten()),
                        ('(∂³Qxy/∂x³)',QXYxxx.flatten()),
                        ('(∂³Qxx/∂x²∂y)',QXXxxy.flatten()),
                        ('(∂³Qxy/∂x²∂y)',QXYxxy.flatten()),
                        ('(∂³Qxx/∂x∂y²)',QXXxyy.flatten()),
                        ('(∂³Qxy/∂x∂y²)',QXYxyy.flatten()),
                        ('(∂³Qxx/∂y³)',QXXyyy.flatten()),
                        ('(∂³Qxy/∂y³)',QXYyyy.flatten()),
                        ('1',np.ones(U.flatten().shape))
                    ],
                    dtype=[('name','U50'),('val',arr_type,U.flatten().shape)]
                )


            lib_Q_p = build_constrained_library_array(funcsQ,
                                                        base_Q,
                                                        db_Q_p,
                                                        coords,
                                                        constraints_Q)

            lib_Stokes_p = build_constrained_library_array(funcsStokes,
                                                        base_Stokes,
                                                        db_Stokes_p,
                                                        coords,
                                                        constraints_Stokes)
                
            # Add time derivative term to the Stokes library
            dw_dt_p = get_term_val(lib_lhs_p,'(∂ω/∂t)')
            # Append ω to the Stokes library
            dw_dt_struct_p = np.array([('(∂ω/∂t)',dw_dt_p.flatten())],
                                dtype=lib_Stokes_p.dtype)
            lib_Stokes_p = np.append(lib_Stokes_p,dw_dt_struct_p)

            if overdamped:
                lib_overdamped_p = build_constrained_library_array(funcsOd,
                                                            base_Od,
                                                            db_overdamped_p,
                                                            coords,
                                                            constraints_Od)
            
            for i in range(len(lib_lhs)):
                lib_lhs[i]['val'][key] = np.mean(lib_lhs_p[i]['val'])
            
            for i in range(len(lib_Q)):
                lib_Q[i]['val'][key] = np.mean(lib_Q_p[i]['val'])
            
            for i in range(len(lib_Stokes)):
                lib_Stokes[i]['val'][key] = np.mean(lib_Stokes_p[i]['val'])
            
            if overdamped:
                for i in range(len(lib_overdamped)):
                    lib_overdamped[i]['val'][key] = np.mean(lib_overdamped_p[i]['val'])
        
        if not overdamped:
            lib_overdamped = None
        return (lib_lhs,lib_Q,lib_Stokes,lib_overdamped)
        
    def weak_form_flow_libs(self,
                            num_windows,
                            window_size,
                            sample=1,
                            p=6,
                            r=1):

        '''
        (lib_lhs, lib_NS, lib_St) = weak_form_flow_libs(num_windows,
                                                        window_size,
                                                        sample=1,
                                                        p=6,
                                                        r=1)

        Function to generate SINDy libraries for the flow equation using the weak formulation.

        Parameters
        ----------

        num_windows : int
            Number of integration windows to select randomly
        
        window_size : tuple
            Tuple containing 3 integers (NX, NY, NT), specifying the dimensions of the integration windows in x, y and t. 
        
        sample : int
            Sample number to pick. Default is 1.
        
        p : int
            Exponent of (x**2-1) and (y**2-1) in the test function. Default is 6.
        
        r : int
            

        Returns
        -------

        
        '''

        # sample = self.metadata['sample']
        # num_windows = self.metadata['num_windows']
        # window_size = self.metadata['window_size']
        (wx, wy, wt) = window_size

        dx = self.dx * wx / 2 
        dy = self.dy * wy / 2
        dt = self.dt * wt / 2

        dx_grid = self.dx
        dy_grid = self.dy
        dt_grid = self.dt
        grid3D = Grid(h=(dx_grid, dy_grid, dt_grid), ndims=3, boundary='regular')

        # Create arrays to store functions and derivatives
        grad_sq_U_int = np.zeros(num_windows)
        grad_4_U_int = np.zeros(num_windows)
        div_Q_int = np.zeros(num_windows)
        Q_dot_div_Q_int = np.zeros(num_windows)
        U_int = np.zeros(num_windows)
        dt_U_int = np.zeros(num_windows)
        U_dot_grad_U_int = np.zeros(num_windows)
        U_divU_int = np.zeros(num_windows)
            
        Q_dot_U_int = np.zeros(num_windows)
        TrQ2_U_int = np.zeros(num_windows)

        diff_order = 1

        nb = int(np.ceil(diff_order/2.0))
        inner_view = (slice(nb, -nb), slice(nb, -nb), slice(nb, -nb))
        _, views, = get_random_sample(
            self.size, num_windows, window_size, diff_order, sample)  
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
            
            u_view = self.u_all[view]
            v_view = self.v_all[view]
            Qxx_view = self.Qxx_all[view]
            Qxy_view = self.Qxy_all[view]
            
            u = u_view[inner_view] 
            v = v_view[inner_view]
            Qxx = Qxx_view[inner_view]
            Qxy = Qxy_view[inner_view]

            (QXXx_view, QXXy_view, QXXt_view) = grid3D.grad(Qxx_view)
            (QXYx_view, QXYy_view, QXYt_view) = grid3D.grad(Qxy_view)
            
            (ux_view, uy_view, ut_view) = grid3D.grad(u_view)
            (vx_view, vy_view, vt_view) = grid3D.grad(v_view)

            dx_Qxx = QXXx_view[inner_view]
            dy_Qxx = QXXy_view[inner_view]
            dt_Qxx = QXXt_view[inner_view]
            dx_Qxy = QXYx_view[inner_view]
            dy_Qxy = QXYy_view[inner_view]
            dt_Qxy = QXYt_view[inner_view]

            dx_u = ux_view[inner_view]
            dy_u = uy_view[inner_view]
            dx_v = vx_view[inner_view]
            dy_v = vy_view[inner_view]

            dt_U_int[key] = - np.mean(u * dy_dt_psi - v * dx_dt_psi)
            
            # U_dot_grad_U_int[key] = np.mean(u * v * (dy2_psi - dx2_psi) + (u**2 - v**2) * dx_dy_psi )

            U_dot_grad_U_int[key] = np.mean( (u*dx_u+v*dy_u) * dy_psi - (u*dx_v+v*dy_v) * dx_psi )
            
            div_u = dx_u + dy_v
            U_divU_int[key] = np.mean( (u*div_u) * dy_psi - (v*div_u) * dx_psi )
            
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

        lib_St = np.array(
            [
                ('(∂u/∂t)', dt_U_int),
                ('u.∇u', U_dot_grad_U_int),
                ('u(∇.u)', U_divU_int),
                ('∇·Q', div_Q_int),
                ('Q·∇·Q', Q_dot_div_Q_int),
                # ('∇⁴u', grad_4_U_int),
                ('u', U_int),
                ('Q.u', Q_dot_U_int),
                ('(TrQ²)u', TrQ2_U_int)
            ],
            dtype=[('name', 'U50'), ('val', 'float', U_int.shape)]
        )
        
        
        lib_lhs = np.array(
            [
                ('∇²u', grad_sq_U_int)
            ],
            dtype=[('name', 'U50'), ('val', 'float', grad_sq_U_int.shape)]
        )

        return (lib_lhs, lib_St)

    def q_tensor(self,
                 num_windows,
                 window_size,
                 sample=1,
                 p=3,
                 r=1):


        # sample = self.metadata['sample']
        # num_windows = self.metadata['num_windows']
        # window_size = self.metadata['window_size']
        (wx, wy, wt) = window_size

        dx = self.dx * wx / 2 
        dy = self.dy * wy / 2
        dt = self.dt * wt / 2
        
        dx_grid = self.dx
        dy_grid = self.dy
        dt_grid = self.dt
        # This means we need to calculate all the derivatives from scratch.
        grid3D = Grid(h=(dx_grid, dy_grid, dt_grid), ndims=3, boundary='regular')

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
            self.size, num_windows, window_size, diff_order, sample)
        
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

            u_view = self.u_all[view]
            v_view = self.v_all[view]
            Qxx_view = self.Qxx_all[view]
            Qxy_view = self.Qxy_all[view]

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

        lib_lhs = np.array(
            [
                ('(∂Q/∂t)', dt_Q_int),
                ('(DQ/Dt)', Dt_Q_int)
            ],
            dtype=[('name', 'U50'), ('val', 'float', U_dot_grad_Q_int.shape)]
        )

        return (lib_lhs, lib_Q, lib_DQ)
    
    def sindy_int(self):
        
        print("Generating libraries...")
        (lib_lhs,lib_Q,lib_flow,_) = self.generate_libraries_int()
        print("Computing the PDE for Qxx...")
        self.pde_Qxx = PDE()
        self.pde_Qxx.compute(lib_Q, lib_lhs, '(∂Qxx/∂t)', self.metadata)
        print("Done! Stored under pde_Qxx.\n")
        
        print("Computing the PDE for Qxy...")
        self.pde_Qxy = PDE()
        self.pde_Qxy.compute(lib_Q, lib_lhs, '(∂Qxy/∂t)', self.metadata)
        print("Done! Stored under pde_Qxy.\n")
        
        print("Computing the PDE for the flow...")
        self.pde_St = PDE()
        self.pde_St.compute(lib_flow, lib_lhs, '∇²ω', self.metadata)
        print("Done! Stored under pde_St.\n")

        print("All done!")

    def weak_form(self, 
                  num_windows,
                  window_size,
                  sample=1,
                  p=6,
                  r=1):


        (lib_lhs, lib_St) = self.weak_form_flow_libs(num_windows,
                                                     window_size,
                                                     sample,
                                                     p,
                                                     r)

        self.pde_St_w = PDE()
        self.pde_St_w.compute(lib_St, lib_lhs, '∇²u', self.metadata)

        print("Done! Stored under pde_St_w.\n")

    def weak_form_Q(self, 
                    num_windows,
                    window_size,
                    sample=1,
                    p=3,
                    r=1):


        (lib_lhs, lib_Q, lib_DQ) = self.q_tensor(num_windows,
                                                 window_size,
                                                 sample,
                                                 p,
                                                 r)

        self.pde_Q = PDE()
        self.pde_DQ = PDE()
        self.pde_Q.compute(lib_Q, lib_lhs, '(∂Q/∂t)', self.metadata)
        self.pde_DQ.compute(lib_DQ, lib_lhs, '(DQ/Dt)', self.metadata)

        print("Done! Stored under pde_Q and pde_DQ.\n")


if __name__ == "__main__":
    pass
