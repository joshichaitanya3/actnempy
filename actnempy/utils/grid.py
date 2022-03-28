import numpy as np 
from scipy import ndimage
import itertools
import warnings

class Grid:

    def __init__(self,ndims=2,h=None,boundary='periodic'):
        self.ndims = ndims
        self.h = self.__validate_grid(h,ndims)
        self.boundary = boundary

    def __validate_grid(self,h,ndims):
        # check assumptions on h
        if h is None:
            h = np.ones(ndims)
            return h
        else:
            h = np.asanyarray(h)
        if h.ndim == 0: # h is a scalar
            h = np.ones(ndims) * h
        elif h.shape != (ndims,):
            msg = (
                "Grid spacing must be a scalar or an array "
                "with an element per dimension.\n"
                f"Current dimension is set to {ndims}.\n"
                f"You can change it using Grid.set_ndims({h.shape[0]})"
            )
            raise ValueError(msg)

        if np.any(h <= 0):
            msg = "Grid spacing must be greater than zero"
            raise ValueError(msg)

        return h

    def set_ndims(self,ndims):
        self.ndims = ndims

    def set_h(self,h):
        self.h = self.__validate_grid(h,self.ndims)

    def set_periodic_bc(self):
        self.boundary = "periodic"

    def set_regular_bc(self):
        self.boundary = "regular"

    def __deriv(self,f, ax, h, out, rank=0, boundary=None):
        """
        """
        if boundary is None:
            boundary = self.boundary

        # do the transpose,boundary=boundary
        f_view = np.swapaxes(f, 0, rank + ax)
        out_view = np.swapaxes(out, 0, rank + ax)

        if f_view.shape[0] >= 3:
            inv2h_ax = 0.5/h[ax]
            # Central difference for the inner points
            out_view[1:-1] = inv2h_ax * (f_view[2:] - f_view[:-2])
            if boundary=="regular":
                # Forward difference for the first point
                out_view[0] = 2.0*inv2h_ax * (f_view[1] - f_view[0])
                # Backward difference for the last point
                out_view[-1] = 2.0*inv2h_ax * (f_view[-1] - f_view[-2])
            elif boundary=="periodic":
                out_view[0] = inv2h_ax * (f_view[1] - f_view[-1])
                out_view[-1] = inv2h_ax * (f_view[0] - f_view[-2])
            else:
                msg = (
                    "Invalid boundary condition. "
                    "Please use either `regular` or `periodic`."
                )
                raise ValueError(msg)
        else:
            out_view[:] = 0

    def grad(self, f, ndims=None, h=None, boundary=None):
        """
        grad(f, ndims=None, h=None, boundary=None)

        Computes gradient of tensor field `f` with grid spacing `h`. 
        If `f` is a rank k tensor defined in an `ndims` dimensional
        space, then `f` should have f.ndim = k + ndims. 

        Parameters
        ----------
        f : array_like
            The tensor field of which to compute the gradient
        ndims : int
            The number of dimensions of the system 
            (not including tensor rank)
        h : scalar or array_like or None (default)
            The grid spacing
        boundary: str
            Criterion for the derivatives at the boundary 
            boundary='periodic' indicates periodic boundary conditions
            boudary='regular' indicates the use of forward/backward
            difference at the boundaries
            The default value used, if none is specified, is the 
            default value for the Grid object (which, in turn, is set to 
            'periodic')
            The default boundary condition for one instance of import
            can be changed by using set_periodic_bc() or
            set_regular_bc() on Grid objects.

        Returns
        -------
        gradf : ndarray
            Gradient of tensor field `f` 
            with first index corresponding to the gradient direction

        Raises
        ------
        ValueError
            if `h` has incorrect shape or is not positive
        """
        
        f = np.asanyarray(f)
        if ndims is None:
            ndims = self.ndims
        if h is None:
            h = self.h
        else:
            h = self.__validate_grid(h, ndims)
        if boundary is None:
            boundary = self.boundary
        
        # gradient will add another dimension of length ndims
        gradf = np.empty(tuple([ndims]+list(f.shape)),float)
        fshape = f.shape[:-ndims]
        rank = len(fshape)
        for ax in range(ndims):
            self.__deriv(f, ax, h, gradf[ax], rank=rank,boundary=boundary)

        return gradf

    def div(self, f, ndims=None, h=None, boundary=None):
        """
        div(f, ndims=None, h=None, boundary=None)

        Computes divergence of tensor field `f` with grid spacing `h`,
        If `f` is a rank k tensor defined in an `ndims` dimensional
        space, then `f` should have f.ndim = k + ndims. 

        Parameters
        ----------
        f : array_like
            The tensor field of which to compute the divergence
        ndims : int
            The number of dimensions of the system 
            (not including tensor rank)
        h : scalar or array_like or None (default)
            The grid spacing
        boundary: str
            Criterion for the derivatives at the boundary 
            boundary='periodic' indicates periodic boundary conditions
            boudary='regular' indicates the use of forward/backward
            difference at the boundaries
            The default value used, if none is specified, is the 
            default value for the Grid object (which, in turn, is set to 
            'periodic')
            The default boundary condition for one instance of import
            can be changed by using set_periodic_bc() or
            set_regular_bc() on Grid objects.

        Returns
        -------
        div : ndarray
            Divergence of tensor field `f`. 
            If `f` is indexed f[i][j][k]... , then 
            divf[j][k]... = sum(\partial_i (f[i][j][k]...), 
            i going from 0 to (ndims-1)

        Raises
        ------
        ValueError
            if `h` has incorrect shape or is not positive
        ValueError
            if the rank of `f`, computed as (f.ndim - ndims), is less
            than 1 since the divergence would be ill-defined.
        """
        f = np.asanyarray(f)
        if ndims is None:
            ndims = self.ndims
        if h is None:
            h = self.h
        else:
            h = self.__validate_grid(h, ndims)
        if boundary is None:
            boundary = self.boundary

        rank = f.ndim - ndims

        # divergence will reduce the first dimension
        if rank < 1: # then the object is either a scalar, or is ill-defined
            raise ValueError("Cannot compute divergence of scalar field")

        divf = np.zeros(f.shape[1:])
        temp = np.zeros(f.shape[1:])
        div_dims = min(ndims,f.shape[0])
        if ndims != f.shape[0]:
            warnings.warn("Computing divergence along axis "
                        f"of length {f.shape[0]} in a {ndims}D space.")
        for ax in range(div_dims):
            self.__deriv(f[ax], ax, h, temp, rank=rank-1,boundary=boundary)
            divf += temp

        return divf

    def curl(self, f, ndims=None, h=None, boundary=None):
        """
        curl(f, ndims=None, h=None, boundary=None)

        Computes the curl of tensor field `f` with grid spacing `h`,
        assuming periodic boundary conditions. If `f` is a rank k tensor
        defined in an `ndims` dimensional space, then `f` should have
        f.ndim = k + ndims. 

        Parameters
        ----------
        f : array_like
            The tensor field of which to compute the curl
        ndims : int
            The number of dimensions of the system 
            (not including tensor rank)
        h : scalar or array_like or None (default)
            The grid spacing
        boundary: str
            Criterion for the derivatives at the boundary 
            boundary='periodic' indicates periodic boundary conditions
            boudary='regular' indicates the use of forward/backward
            difference at the boundaries
            The default value used, if none is specified, is the 
            default value for the Grid object (which, in turn, is set to 
            'periodic')
            The default boundary condition for one instance of import
            can be changed by using set_periodic_bc() or
            set_regular_bc() on Grid objects.

        Returns
        -------
        out : ndarray
            Curl of tensor field `f`. If `f` is indexed f[i][j][k]... ,
            the curl[i][j][k]... = sum(levi[i,l,m]\partial_l (f[m][j][k]...), 
            {l,m} going from 0 to (ndims-1)

        Raises
        ------
        ValueError
            if `h` has incorrect shape or is not positive
        ValueError
            if the rank of `f`, computed as (f.ndim - ndims), is less
            than 1 since the curl would be ill-defined.
        ValueError
            if the number of tensor components are less than the system 
            dimension.
        ValueError
            if the number of tensor components is more than 3, in which 
            case the curl is ill-defined.
        """
        
        f = np.asanyarray(f)
        if ndims is None:
            ndims = self.ndims
        if h is None:
            h = self.h
        else:
            h = self.__validate_grid(h, ndims)
        if boundary is None:
            boundary = self.boundary

        rank = f.ndim - ndims
        vdims = f.shape[0]

        if rank < 1:
            msg = "Curl requires at least rank 1 tensor"
            raise ValueError(msg)
        
        if vdims < ndims:
            msg = (
                "Number of tensor components must be "
                "greater or equal to system dimension"
            )
            raise ValueError(msg)
        
        if vdims > 3:
            msg = "Curl is only valid in 2 or 3 dimensions"
            raise ValueError(msg)

        
        if ndims < vdims:
            old_shape = f.shape
            shape = f.shape + (1,) * (vdims - ndims)
            f = f.reshape(shape)
            h_new = np.zeros(vdims)
            h_new[:ndims] = h
            h = h_new

        # build Levi-Civita symbol
        levi = np.zeros([3,3,3])
        levi[0,1,2] = 1
        levi[1,2,0] = 1
        levi[2,0,1] = 1
        levi[0,2,1] = -1
        levi[2,1,0] = -1
        levi[1,0,2] = -1
        
        if vdims == 2:
            out = np.zeros(f.shape[1:])
        else:
            out = np.zeros(f.shape)
        
        temp = np.empty(f.shape[1:])
        for i, j, k in itertools.product(* ([range(3)]*3)):
            sign = levi[i, j, k]
            if sign == 0:
                continue

            if vdims == 2:
                if i != 2: 
                    continue

                self.__deriv(f[k], j, h, temp, rank=rank-1,boundary=boundary)
                out += sign * temp

            else:
                self.__deriv(f[k], j, h, temp, rank=rank-1,boundary=boundary)
                out[i] += sign * temp

        if ndims < vdims:
            out = out.reshape(old_shape)

        return out

    def lap(self, f, ndims=None, h=None, boundary=None):
        """
        lap(f, ndims=None, h=None, boundary=None)

        Computes the laplacian of tensor field `f` with grid spacing
        `h`. If `f` is a rank k tensor defined in an `ndims` dimensional
        space, then `f` should have f.ndim = k + ndims. 

        Parameters
        ----------
        f : array_like
            The tensor field of which to compute the laplacian
        ndims : int
            The number of dimensions of the system 
            (not including tensor rank)
        h : scalar or array_like or None (default)
            The grid spacing
        boundary: str
            Criterion for the derivatives at the boundary 
            boundary='periodic' indicates periodic boundary conditions
            boudary='regular' indicates the use of forward/backward
            difference at the boundaries
            The default value used, if none is specified, is the 
            default value for the Grid object (which, in turn, is set to 
            'periodic')
            The default boundary condition for one instance of import
            can be changed by using set_periodic_bc() or
            set_regular_bc() on Grid objects.

        Returns
        -------
        out : ndarray
            Laplacian of tensor field `f`. If `f` is indexed
            f[i][j][k]... ,
            then
            lapf[i][j][k]... = sum(\partial_l \partial_l (f[i][j][k]...), 
            l going from 0 to (ndims-1)

        Raises
        ------
        ValueError
            if `h` has incorrect shape or is not positive
        """
        
        f = np.asanyarray(f)
        if ndims is None:
            ndims = self.ndims
        if h is None:
            h = self.h
        else:
            h = self.__validate_grid(h, ndims)
        if boundary is None:
            boundary = self.boundary
        
        invh2 = 1.0 / (h * h)
        
        kernel = np.zeros(ndims*[3])
        idx = (...,) + (1,)*(ndims-1)
        for ax in range(ndims):
            kernel[idx] += np.array([1,-2,1]) * invh2[ax]
            # permute
            idx = idx[-1:] + idx[:-1]

        lapf = np.empty(f.shape)
        fshape = f.shape[:-ndims] # Shape of the resulting tensor 

        # Set the mode for convolutions according to the boundary
        # condition.
        
        if boundary=="regular":
            mode='nearest'
        elif boundary=="periodic":
            mode='wrap'
        
        for idx in itertools.product(*[range(s) for s in fshape]):
            ndimage.convolve(f[idx], kernel, mode=mode, output=lapf[idx])
        
        return lapf