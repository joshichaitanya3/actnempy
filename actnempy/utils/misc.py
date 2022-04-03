import numpy as np 
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from numpy import linalg as LA
import matplotlib.pyplot as plt
from .optimal_SVHT_coef import optimal_SVHT_coef
import warnings
from IPython import get_ipython

# Check if cv2 library is available
no_cv2 = False
try:
    import cv2
except ImportError:
    no_cv2 = True

# Check if numpy.random.default_rng is available
no_rng = False
try:
    from numpy.random import default_rng
except ImportError:
    no_rng = True

def denoise(arr):
    """
    denoise(arr)

    Function to denoise spatiotemporal data stored in an array using
    SVD with an optimal threshold algorithm.

    Parameters
    ----------
    arr : 3D ndarray
        array containing the values of a field with shape (nx,ny,nt)
        where t is time.
    
    Returns
    -------
    arr : 3D ndarray
        denoised array
    S : ndarray
        Array containing the sigma matrix of from the SVD
    dim : int
        Optimal threshold for the SVD
    """

    # Compute SVD
    shape = arr.shape
    arr = arr.reshape(np.prod(shape[:-1]),shape[-1])
    n,m = arr.shape
    (U,S,V) = LA.svd(arr, full_matrices=False)
    dim = np.count_nonzero(S > (optimal_SVHT_coef([m/n],0) * np.median(S)))
    
    print(f"Optimal threshold for array = {dim}")

    denoised_arr = (U[:,:dim].dot(np.diag(S[:dim]).dot(V[:dim,:]))).reshape(shape)
    return (denoised_arr, S, dim)

def add_noise(arr, noise_strength=0.01, seed=None):
    """
    add_noise(arr, noise_strength, seed=None)

    Function to add white Gaussian noise to an array. Uses NumPy's 
    random.default_rng() generator if available, and random.randn if not.

    Parameters
    ----------
    arr : ndarray
        Original array
    noise_strength : float
        Strength of the noise relative to the standard deviation 
        of the entries of the array. 
        Default is 0.01
    seed : {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}, optional
        Seed for NumPy's default_rng(). From its description:
        A seed to initialize the `BitGenerator`. If None, then fresh,
        unpredictable entropy will be pulled from the OS. If an ``int`` or
        ``array_like[ints]`` is passed, then it will be passed to
        `SeedSequence` to derive the initial `BitGenerator` state. One may also
        pass in a`SeedSequence` instance
        Additionally, when passed a `BitGenerator`, it will be wrapped by
        `Generator`. If passed a `Generator`, it will be returned unaltered.
        If default_rng is not available, this will be used as a seed 
        for the random.randn
    Returns
    -------
    arr : ndarray
        Array with added noise
    
    Usage
    -----
    >>>velocity = np.ones(5,5)
    >>>velocity
    array([[1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1.]])
    >>>noise_strength = 0.01
    >>>noisy_velocity = add_noise(velocity, noise_strength, seed=12345)
    >>>noisy_velocity
    array([[0.98576175, 1.01263728, 0.99129338, 0.99740827, 0.99924657],
           [0.99259115, 0.98632207, 1.00648893, 1.00361058, 0.98047137],
           [1.0234741 , 1.00968497, 0.99240613, 1.00902198, 0.99533047],
           [0.9993931 , 1.00788844, 0.98743332, 1.00575858, 1.01398979],
           [1.01322298, 0.99700301, 1.00902919, 0.98378417, 0.99841811]])
    """

    if no_rng:

        print('Using random.randn')

        np.random.seed(seed)

        return arr + noise_strength * np.std(arr) * np.random.randn(*arr.shape)
    
    else:
    
        rg = default_rng(seed)

        return arr + noise_strength * np.std(arr) * rg.standard_normal(arr.shape)

def compute_Q(theta,sigma=1,custom_kernel=None):
    """
    compute_Q(theta,sigma=1,custom_kernel=None)
    
    Function to calculate S (scalar order), Qxx and Qxy, given the 
    orientation field theta. 
    The calculation proceeds by computing the molecular tensor 
    m = n \\otimes n - 1/2 I and coarse graining it. 
    The default averaging is done by a gaussian filter with sigma=1.
    A different value of sigma can be provided, or an entirely custom
    kernel can also be provided.
    
    Parameters
    ----------
    theta : ndarray
        2D array containing values of the orientation at a given time.
    sigma : float
        The value of the standard deviation for the gaussian filter.
        Defaults to 1.
    custom_kernel : ndarray
        Array corresponding to a custom kernel to be used for averaging.
        Defaults to None, in which case, it uses the gaussian kernel.

    Returns
    -------
    S : ndarray
        Scalar order parameter
    Qxx : ndarray
        Values of Qxx 
    Qxy : ndarray
        Values of Qxy
    """

    average = lambda f : gaussian_filter(f, sigma=sigma)
    if custom_kernel is not None:
        if no_cv2:
            print("cv2 module not available.")
            print("Cannot generate custom kernel. Using default kernel...")
        else:
            average = lambda f : cv2.filter2D(f,-1,custom_kernel)

    
    nx = np.cos(theta)
    ny = np.sin(theta)
    mQxx = (nx**2 - 0.5)
    mQxy = (nx*ny)
    
    Qxx = average(mQxx)
    Qxy = average(mQxy)
    S = 2*np.sqrt( Qxx**2 + Qxy**2)
    return (S,Qxx,Qxy)

def compute_n(Qxx,Qxy):
    """
    compute_n(Qxx,Qxy)
    
    Function to calculate S (scalar order), nx and ny, given Qxx and
    Qxy. It imposes ny>0, such that theta = np.arctan2(ny,nx) lies
    between 0 to pi.
    
    Parameters
    ----------
    Qxx : ndarray
        2D array containing values of Qxx at a given time.
    Qxy : ndarray
        2D array containing values of Qxy at a given time.
    
    Returns
    -------
    S : ndarray
        Scalar order parameter
    nx : ndarray
        Values of nx such that -1 <= nx <= 1 
    ny : ndarray
        Values of ny such that 0 <= ny <= 1.
        This ensures 0 <= theta <= pi
    """
    S = 2*np.sqrt( Qxx**2 + Qxy**2)
    Qxx = Qxx/S
    Qxy = Qxy/S
    
    # Evaluate nx and ny from normalized Qxx and Qxy
    nx = np.sqrt( Qxx + 0.5 )
    with warnings.catch_warnings(record=True) as wrng:
        ny = np.abs( Qxy / nx ) # This ensures ny>0, such that theta lies between 0 to pi
    if len(wrng)>0:
        ny[nx==0] = 1.0 # RuntimeWarning will appear if nx is exactly zero.
    nx = nx * np.sign( Qxy ) # This gives back nx its correct sign.
    return (S, nx, ny)

def remove_NaNs(field, nematic=False):
    '''
    remove_NaNs(field, nematic=True)

    Function to remove (*in place*) spurious NaNs appearing in the field data. This
    is done by replacing the NaN with the nanmean of its nearest
    neighbors. 
    If the field is an orienatation field of a nematic with angular
    values, the same operation is performed on cos(2theta) and
    sin(2theta) to preserve the nematic symmetry, and then converted
    bavk to the angular values.

    Parameters
    ----------

    field : ndarray
        A 2D array containing the value of a single scalar field (eg.
        theta, vx or vy) at a single point of time (shape NX x NY)
    
    nematic : bool, optional
        Flag to indicate whether the field as an angluar field. Default
        is False.
    
    Returns
    -------
    None (The array is changed in place.)

    '''
    (nx,ny) = field.shape
    ids = np.argwhere(np.isnan(field))
    offsets = [[1,0],[0,1],[-1,0],[0,-1]]
    for pos in ids:
        bdy_ids = (pos+offsets).T
        bdy_ids[0,:] = np.clip(bdy_ids[0,:], 0, nx-1)
        bdy_ids[1,:] = np.clip(bdy_ids[1,:], 0, ny-1)
        bdys = field[tuple(bdy_ids)]
        if nematic:
            # This means that the field is an orientation field of a nematic,
            # meaning, -pi/2 <= field <= pi/2. Hence, the interpolation for
            # this field needs to be handled carefully, taking into account
            # the periodicity and the nematic symmetry.
            c2th, s2th = np.nanmean(np.cos(2*bdys)), np.nanmean(np.sin(2*bdys))
            field[tuple(pos)] = 0.5*np.arctan2(s2th, c2th)
        else:
            field[tuple(pos)] = np.nanmean(bdys)

def count_NaNs(arr):
    """
    count_NaNs(arr)

    Function to count the number of NaNs in an array

    Parameters
    ----------
    arr: ndarray
        Array which potentially contains NaNs
    
    Returns
    -------
    num : int
        Number of NaNs in the array
    """
    return (np.count_nonzero(np.isnan(arr)))

def get_random_sample(shp, num_points, box_size, diff_order, seed=1):
    '''
    get_random_sample(shp, num_points, box_size, diff_order, seed=1)

    Function to sample random boxes out of a 3D array of shape `shp`.

    Parameters
    ----------
    shp : tuple
        Shape of the original array(s) from which to sample
    num_points : int
        Number of samples required
    box_size : 3-tuple or int
        Size of the box to be sampled. This can either be supplied as a 3-tuple, with one integer for each dimension, or an int, in which case a cubical box will be sampled. 
    diff_order : int
        Number of derivatives of the original array the user is expecting to take. This is used to make sure points close to the boundary whose derivatives cannot be taken are not included in the boxes.
    seed : int, optional
        Seed for the random number generator. Defualt is 1.

    Returns
    -------
        points : dict
            Dictionary containing the coordinates of the box centers, with the keys being the indices.
        views : dict
            Dictionary containing the slices that return the box when indexed with.
    
    Usage
    -----
        _, views, = get_random_sample(u_all.shape, 100, 5, 2, 1)
        
        for key, view in views.items():
                
            u_view = u_all[view].copy()
            ...
        
    '''
    if isinstance(box_size, int):
        if box_size < 0:
            raise ValueError(
                "Box size has to be greater than equal to zero!")
        box_size = (box_size,) * len(shp)

    n_diff = 2 * np.ceil(diff_order/2.0)
    
    window = tuple([int(i+n_diff) for i in box_size])
    
    bdy = tuple([int((i+1)/2) for i in window])
    
    box_bdy = tuple([int((i+1)/2) for i in box_size])

    # Now sample random points that lie between bdy and shp-bdy.
    # To speed things up, we will sample random points in each x, y and
    # t directions and combine them. This has a minute risk of sampling
    # the same point twice, which we further reduce by sampling
    # `1.5*num_points` points and then choosing `num_points` number of
    # unique points from it. 

    if no_rng:
        np.random.seed(seed)
        pts_x = np.random.randint(bdy[0],shp[0]-bdy[0],int(1.5*num_points))
        pts_y = np.random.randint(bdy[1],shp[1]-bdy[1],int(1.5*num_points))
        pts_z = np.random.randint(bdy[2],shp[2]-bdy[2],int(1.5*num_points))

    else:
        rg = default_rng(seed)
        pts_x = rg.integers(bdy[0],shp[0]-bdy[0],int(1.5*num_points))
        pts_y = rg.integers(bdy[1],shp[1]-bdy[1],int(1.5*num_points))
        pts_z = rg.integers(bdy[2],shp[2]-bdy[2],int(1.5*num_points))

    picked_points = np.unique( np.array([pts_x,pts_y,pts_z]), axis=1)[:,:num_points]

    # In the ultra-rare case that this gives less than `num_points`
    # number of points, the program should break, asking either for a
    # different seed or for lesser number of points.
    if picked_points.shape[1]<num_points:
        raise ValueError("Couldn't sample enough number of unique points! Try a different seed for `get_random_sample` or try lesser number of points.")
    
    points = {}
    views = {}

    for count in range(num_points):
        points[count] = [picked_points[0][count],
                         picked_points[1][count], picked_points[2][count]]
        views[count] = (slice(points[count][0]-bdy[0]+1, points[count][0]+bdy[0]),
                        slice(points[count][1]-bdy[1]+1,
                              points[count][1]+bdy[1]),
                        slice(points[count][2]-bdy[2] +
                              1, points[count][2]+bdy[2])
                        )

    return points, views

if __name__ == "__main__":
    pass
