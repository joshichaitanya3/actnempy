# ---------------------------------
# Line integration for defect identification in nematics via convolution
# % ------------------------------------------------------------
# Michael M. Norton, Physics @ Rochester Institute of Technology, 2021
# in collaboration w/ Grover Lab (Piyush Grover and Caleb Wagner, Mech. Eng. @ University of Nebraska-Lincoln)
# ---------------------------------

# -----------------
# numerical basics
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# -----------------
# image convolution

from scipy import signal
from scipy import misc

# -----------------
# image processing

from skimage.io import imread, imshow
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops, regionprops_table
from skimage import measure
import math

def func_unitcircle(r):
    # just makes a ring of ones
    # https://stackoverflow.com/questions/39073973/how-to-generate-a-matrix-with-circle-of-ones-in-numpy-scipy
    d = 2*r + 1
    rx, ry = d/2, d/2
    x_grid, y_grid = np.indices((d, d))
    ring_filter = (np.abs(np.hypot(rx - x_grid, ry - y_grid)-r)
                   < 0.5).astype(int)

    x_grid = x_grid-d/2
    y_grid = y_grid-d/2

    r_grid = (x_grid**2+y_grid**2)**0.5

    x_grid[ring_filter == 0] = 0
    y_grid[ring_filter == 0] = 0

    filter_x = -y_grid/r_grid
    filter_y = x_grid/r_grid

    return (x_grid, y_grid, filter_x, filter_y, ring_filter)


def func_defectfind(nx, ny, filter_radius, switchsign):
    """
    func_defectfind(x,y,x0,y0)
    
    Function returns scalar maps the size of the input arrays that identify regions of topological charge
    
    Parameters
    ----------
    nx,ny : components of director field
    filter_radius : radius of line integral region
    switchsign : flips identity of defects +/-  --> -/+ (needed for some data sets)
    
    Returns
    -------
    map : line integral map
    map_p : map for +1/2 defects
    map_m : map for -1/2 defects
    """

    x_grid, y_grid, filter_x, filter_y, ring_filter = func_unitcircle(
        filter_radius)

    Qxx = nx**2-1/2
    Qxy = nx*ny

    Qxx_x, Qxx_y = np.gradient(Qxx)
    Qxy_x, Qxy_y = np.gradient(Qxy)

    denom = (1+4*Qxx+4*Qxy**2+4*Qxx**2)
    dphidx_num = 2*(-2*Qxy*Qxx_x+(1+2*Qxx)*Qxy_x)
    dphidy_num = 2*(-2*Qxy*Qxx_y+(1+2*Qxx)*Qxy_y)

    dphidx = dphidx_num/denom
    dphidy = dphidy_num/denom

    eps_mine = 1E0

    #remove ~0/0 artifacts
    dphidx[(np.abs(denom) < eps_mine) & (np.abs(dphidx_num) < eps_mine)] = 0
    dphidy[(np.abs(denom) < eps_mine) & (np.abs(dphidy_num) < eps_mine)] = 0  #

    map = signal.convolve2d(dphidy, filter_y, boundary='symm', mode='same') + \
        signal.convolve2d(dphidx, filter_x, boundary='symm', mode='same')

    Nrows, Ncolumns = np.shape(map)

    map_m = np.zeros((Nrows, Ncolumns))
    map_p = np.zeros((Nrows, Ncolumns))
    #map_zeros= np.zeros((Nrows, Ncolumns))
    #map_zeros[(map > -0.2) & (map < 0.2)] = 1

    #map_p[map > threshold_otsu(map)] = 1

    #auto-threshold using threshold_otsu acts funny.. just hard-code a threshold

    if switchsign == 1:
        map_m[map > 1] = 1
        map_p[map < -1] = 1
    else:
        map_m[map < -1] = 1
        map_p[map > 1] = 1

    return(map, map_p, map_m)


def func_defectpos(binmap, areathresh):
    """
    func_defectpos(binmap, areathresh)
    
    Function identifies coordinates of defects
    
    Parameters
    ----------
    binmap : logical map of candidate defect regions
    areathresh : area threshold, keep regions greater than threshold 
    
    Returns
    -------
    [[x1, y1],
    [x2, y2], et..]= list of defect coordinates     
    """

    binmap_label = label(binmap)
    regions = regionprops(binmap_label)

    # centroid_list = np.array([])
    centroid_xs = []
    centroid_ys = []
    for props in regions:
        y0, x0 = props.centroid
        area = props.area
        if area > areathresh:
            centroid_xs.append(x0)
            centroid_ys.append(y0)

            # centroid_list = np.append(centroid_list, [x0, y0])

    # N = int(np.size(centroid_list)/2)
    # centroid_list_reshape = np.reshape(centroid_list, [N, 2])

    centroid_list_reshape = np.array([centroid_xs, centroid_ys]).T
    return(centroid_list_reshape)


def func_defectorient(centroids, nx, ny, filter_radius, type_str):
    """
    func_defectorient(centroids, nx, ny, filter_radius,type_str)
    
    Function identifies defect orientation
    
    Parameters
    ----------
    centroids : logical map of candidate defect regions
    nx,ny : director field
    filter_radius : 
    type_str : string that indicates defect type: "positive" or "negative"
    
    Returns
    -------
    [phi1,phi2...phiN] =  list of defect angles [0,2pi]     
    """

    x_grid, y_grid, filter_x, filter_y, ring_filter = func_unitcircle(
        filter_radius)

    pos = np.argwhere(ring_filter > 0)
    x_ring = np.ceil(pos[:, 0] - filter_radius)
    y_ring = np.ceil(pos[:, 1] - filter_radius)

    theta = np.arctan2(y_ring, x_ring)

    theta_sort = np.sort(theta)
    theta_argsort = np.argsort(theta)

    x_ring = x_ring[theta_argsort]
    y_ring = y_ring[theta_argsort]

    centroids_x = centroids[:, 0]
    centroids_y = centroids[:, 1]

    N_defects = np.size(centroids_x)

    phi = np.zeros((N_defects, 1))

    Nrows, Ncolumns = np.shape(nx)

    for ii in range(0, N_defects):

        x0 = centroids_x[ii]
        y0 = centroids_y[ii]

        x = x0 + x_ring
        y = y0 + y_ring

        x[x > (Ncolumns-1)] = Ncolumns-1
        x[x < 0] = 0
        y[y > (Nrows-1)] = Nrows-1
        y[y < 0] = 0

        x = x.astype(int)
        y = y.astype(int)

        # note that to find a coordinate (x,y), need to select (row, column)
        nx_local = nx[y, x]
        ny_local = ny[y, x]

        dotprod = np.abs(nx_local*np.cos(theta_sort) +
                         ny_local*np.sin(theta_sort))

        if np.char.equal(type_str, "positive") == 1:
            phi[ii] = theta_sort[np.argmax(dotprod)]+np.pi
        elif np.char.equal(type_str, "negative") == 1:
            phi[ii] = theta_sort[np.argmax(dotprod)]
        else:
            print("check type_str")

    return(phi)


def func_plotdefects(ax, centroids, phi, color_str, type_str, scale):

    centroids_x = centroids[:, 0]
    centroids_y = centroids[:, 1]

    N_defects = np.size(centroids_x)

    # Flatten phi for convenient indexing below
    phi = phi.flatten()

    for ii in range(0, N_defects):

        x0 = centroids_x[ii]
        y0 = centroids_y[ii]

        x1 = x0 + scale*np.cos(phi[ii])
        y1 = y0 + scale*np.sin(phi[ii])

        ax.scatter(x0, y0, color=color_str)

        if np.char.equal(type_str, "positive") == 1:
            ax.plot((x0, x1), (y0, y1), color_str, linewidth=2.5)
        else:
            for jj in range(0, 3):
                x1 = x0 + scale*np.cos(phi[ii]+(jj+1)*2*np.pi/3)
                y1 = y0 + scale*np.sin(phi[ii]+(jj+1)*2*np.pi/3)
                ax.plot((x0, x1), (y0, y1), color_str, linewidth=2.5)

    return


# 2021.08.17: func_wrap() and func_crop() added to help with periodic boundary conditions

def func_wrap(A, periodic_x, periodic_y, padamount):

    if periodic_x == 1:
        A_pad = A[:, 0:padamount]
        A = np.hstack((A, A_pad))
        if periodic_y == 1:
            A_pad = A[0:padamount, :]
            A = np.vstack((A, A_pad))

    elif periodic_y == 1:
        A_pad = A[0:padamount, :]
        A = np.vstack((A, A_pad))

    return(A)


def func_crop(A, periodic_x, periodic_y, padamount):

    if periodic_x == 1:
        A = A[:, 0:-padamount]
        if periodic_y == 1:
            A = A[0:-padamount, :]

    elif periodic_y == 1:
        A = A[0:-padamount, :]

    return(A)
