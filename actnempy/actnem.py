'''

Active Nematics Analysis Suite

Code by Chaitanya Joshi (chaitanya@brandeis.edu)

The main object of this module is the ActNem class. This class combines key methods for analysis of 2D active nematics data of Q-tensor and velocity data in one place, allowing for easy processing of large datasets. 

Functionalities include: visualizing the director and velocity, computing velocity autocorrelation and orientation autocorrelation functions in time, defect finding (with tracking coming soon). 

'''

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np 
import matplotlib.pyplot as plt 
import json
from .utils import nematic_plot, compute_n, Grid, func_defectfind, func_defectpos, func_defectorient, func_plotdefects
from tqdm import tqdm
from scipy.signal import correlate
import sys
from pathlib import Path

class ActNem:
    '''
    
    The Active Nematics class.
    
    Designed for analysis of 2D active nematics data of Q-tensor and velocity. 
    
    Attributes
    ----------

    data_dir : str
        Path to the data directory. This directory must contain the following files:

            `processed_data.npz`: A single .npz file containing 4 arrays: `Qxx_all`, `Qxy_all`, `u_all` and `v_all`, each of dimensions (NX, NY, NT), with X-Y being the spatial dimensions and T being the time dimension. The preprocessing of the experimental / simulation data into this format is done elsewhere.

            `metadata.json` : A json file containing three keys: 'dx', 'dy' and 'dt', specifying the spatial and temoral discretization of the data.
    
    visual_check : bool
        An optional flag to plot the first five frames of the data as a visual check.

    '''
    def __init__(self, data_dir,visual_check=False, quiet=False):

        self.data_dir = data_dir
        self.quiet = quiet
        # Setup a directory to store visualizations, if any
        self.vis_dir = f"{data_dir}/visualizations"
        if not os.path.exists(self.vis_dir):
            os.makedirs(self.vis_dir)

        # Import data
        self.processed_data_file = f'{data_dir}/processed_data.npz'
        if os.path.isfile(self.processed_data_file):
            self._qprint("Loading all the data...",flush=True)
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
                self._qprint(f'Data loaded. Dimensions: {self.size}')
        else:
            msg = f"Couldn't find the data at {self.processed_data_file}!"
            raise ValueError(msg)
        
        # Import metadata

        with open(f'{data_dir}/metadata.json', 'r') as f:
            self.metadata = json.load(f)

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

    def _qprint(self, x, **kwargs):
        if not self.quiet:
            return print(x, **kwargs)
    
    def reset_data(self):
        '''
        Function to reload the data from the file, in case the data
        arrays are modified.

        '''

        print("Reloading the data...",flush=True)

        with np.load(self.processed_data_file) as data:
                self.u_all = data['u_all']
                self.v_all = data['v_all']
                self.Qxx_all = data['Qxx_all']
                self.Qxy_all = data['Qxy_all']

        print("Data has been reset.")
    
    def visualize(self, nframes, save=True):
        '''
        visualize(nframes, save=False)

        Function to visualize the director/order and flow-field/vorticity in the data. 

        Parameters
        ----------

        nframes : int
            Number of frames. The first `nframes` frames will be visualized.
        save : bool
            Flag to indicate whether to save the visualizations. True by default. 
        '''
        plt.ion()
        fig = plt.figure(figsize=(10,4))
        for t in range(nframes):
            u = self.u_all[:,:,t]
            v = self.v_all[:,:,t]
            Qxx = self.Qxx_all[:,:,t]
            Qxy = self.Qxy_all[:,:,t]
            (S,nx,ny) = compute_n(Qxx,Qxy)
            velocity = np.array([u,v])
            vorticity = self.grid2D.curl(velocity)
            plt.clf()
            plt.subplot(1,2,1)
            plt.pcolor(self.x, self.y, S.T)
            plt.colorbar()
            nematic_plot(self.x, self.y, nx.T,ny.T, density=2.0, color='gray')
            plt.title('Director and Order')
            plt.subplot(1,2,2)
            plt.pcolor(self.x[1:-1], self.y[1:-1], vorticity[1:-1,1:-1].T, cmap='bwr')
            plt.colorbar()
            plt.streamplot(self.x[1:-1], self.y[1:-1], u[1:-1,1:-1].T, v[1:-1,1:-1].T, density=2.0, color='gray')
            plt.title('Flow Field and Vorticity (in 1/seconds)')
            plt.suptitle(f"t = {self.dt*t:.1f} seconds")
            fig.canvas.draw()
            if save:
                plt.savefig(f'{self.vis_dir}/frame_{t}.png')
        plt.ioff()
        plt.show()

    def compute_divergence(self, frame=0, plot=False, show=False):
        '''
        compute_divergence(self, frame=0, plot=False)

        Function to compute the divergence of the velocity field at a
        given frame

        Parameters
        ----------

        frame : int, optional
            Time frame at which to compute the divergence. Default is 0
        
        plot : bool, optional
            Flag to indicate whether to plot the spatial map of the
            divergence. Default is false
        show: bool, optional
            Flag to indicate whether to show the output of the plot.
            Default is False
        
        Returns
        -------

        divu : ndarray
            2D array containing the spatial map of the divergence at the
            given time frame.

        '''

        
        u = self.u_all[:, :, frame]
        v = self.v_all[:, :, frame]
        velocity = np.array([u, v])
        divu = self.grid2D.div(velocity)
        if plot:
            _ = plt.figure(figsize=(5, 5))
            Qxx = self.Qxx_all[:, :, frame]
            Qxy = self.Qxy_all[:, :, frame]
            (_, nx, ny) = compute_n(Qxx, Qxy)
            plt.pcolor(self.x, self.y, divu.T, cmap='bwr')
            plt.colorbar()
            nematic_plot(self.x, self.y, nx.T, ny.T, density=2.0, color='gray')
            plt.title('Director and Divergence')
            plt.savefig(f'{self.vis_dir}/divergence_{frame}.png')
            if show:
                plt.show()
            else:
                plt.close(fig)
        return divu

    def check_imcompressibility(self, plot=False):
        '''
        check_imcompressibility(self, plot=True)

        Function to check incompressibility in the flow field by
        computing the mean and s.e.m. of the divergence of the velocity
        across all time. 

        Parameters
        ----------

        plot : bool, optional
            Flag to indicate whether to plot the spatial map of the
            divergence. Default is false
        
        Returns:

        means : float
            Mean of div(u) at each point in time
        
        '''

        means = np.zeros(self.NT)
        for i in range(self.NT):
            divu = self.compute_divergence(frame=i, plot=plot)
            means[i] = divu.mean()
        
        return means

    def _autocorr_vector(self, vx, vy):
        '''
        
        Function to compute the autocorrelation in time of a vector.
        \\expval{ \\vec{v}(t+t') \cdot \\vec{v}(t') }_{t'}

        '''

        vxa = correlate(vx, vx, mode='full')
        vya = correlate(vy, vy, mode='full')
        va = vxa+vya
        mid_pt = len(va)//2
        return va[mid_pt:] / va[mid_pt]

    def velocity_autocorr(self):
        '''
        vcorr = velocity_autocorr()

        Computes the spatially averaged velocity time correlation function from the data. It then prints the correlation time by computing when the correlation function drops to 1/e.

        Returns
        -------

        vcorr : ndarray
            Array of dimensions (NT,) containing the velocity time correlation function.
        
        tc : float
            The autocorrelation time in physical time units (dt * autocorelation time in frames)
        
        '''
        corrs = np.zeros([self.NX*self.NY,self.NT])
        for i in range(self.NX):
            for j in range(self.NY):
                idx = j + (i-1)*self.NY
                vxp = self.u_all[i,j,:]
                vyp = self.v_all[i,j,:]
                corrs[idx, :] = self._autocorr_vector(vxp, vyp)
        vcorr = np.nanmean(corrs, axis=0)
        tcorr = np.argmin(vcorr>1/np.exp(1)) # In frames
        tc = self.dt * tcorr
        return vcorr, tc
    
    def orientation_autocorr(self):
        '''
        ocorr = orientation_autocorr()

        Computes the spatially averaged orientation time correlation function from the data. It then prints the correlation time by computing when the correlation function drops to 1/e.

        Returns
        -------

        ocorr : ndarray
            Array of dimensions (NT,) containing the orientation time correlation function.

        tc : float
            The autocorrelation time in physical time units (dt * autocorelation time in frames)
        
        '''
        corrs = np.zeros([self.NX*self.NY,self.NT])
        # (_, nx_all, ny_all) = compute_n(self.Qxx_all, self.Qxy_all)
        for i in range(self.NX):
            for j in range(self.NY):
                idx = j + (i-1)*self.NY
                Qxxp = self.Qxx_all[i,j,:]
                Qxyp = self.Qxy_all[i,j,:]
                corrs[idx, :] = self._autocorr_vector(Qxxp, Qxyp)

        ocorr = np.nanmean(corrs, axis=0)
        tcorr = np.argmin(ocorr>1/np.exp(1)) # In frames
        tc = self.dt * tcorr
        return ocorr, tc

    def find_defects(self, filter_radius=5, size_thresh=60, switchsign=0, frame=0, plot=False, show=False):
        '''
        find_defects(filter_radius=5, size_thresh=60, switchsign=0,
        frame=0, plot=False)
        
        Function to find defects in a given frame.

        Parameters
        ----------

        frame : int, optional
            Frame at which to find defects. 
            Default is 0
        filter_radius : float, optional
            Radius of line integral region. 
            Default is 5
        size_thresh : float, optional
            area threshold, keep regions greater than threshold. 
            Default is 60
        switchsign : flips identity of defects +/-  --> -/+ (needed for
        some data sets)
        frame: int, optional
            Frame number. Default is 0
        plot: bool, optional
            Flag to indicate whether to plot the frame with defects.
            Default is False
        show: bool, optional
            Flag to indicate whether to show the output of the plot.
            Default is False
        Returns
        -------
        centroids_p : ndarray
            Array containing the x-y coordinates of the +1/2 defects
        centroids_m : ndarray
            Array containing the x-y coordinates of the -1/2 defects
        phi_p : ndarray
            Array containing the orientations of the +1/2 defects
        phi_m : ndarray
            Array containing the orientations of the -1/2 defects

        '''
        (_, nx, ny) = compute_n(self.Qxx_all[:,:,frame], self.Qxy_all[:,:,frame])
        
        nx = nx.T
        ny = ny.T
        # create charge density map
        _, map_p, map_m = func_defectfind(nx, ny, filter_radius, switchsign)
        

        # search map and identify circular regions of positive and negative charge
        centroids_p = func_defectpos(map_p, size_thresh)
        centroids_m = func_defectpos(map_m, size_thresh)

        # get the oriengation of defects
        phi_p = func_defectorient(centroids_p, nx, ny, filter_radius, "positive")
        phi_m = func_defectorient(centroids_m, nx, ny, filter_radius, "negative")
        
        if plot:
            # plot defects on top of order parameter and director
            fig, ax = plt.subplots(figsize=(4.3, 4.3))

            nematic_plot(self.x, self.y, nx, ny, density=3.0)
            #ax.imshow(fluorescence, cmap=plt.cm.gray)
            ax.set_aspect('equal', adjustable='box')
            color_p = 'magenta'
            color_m = 'cyan'
            defect_scale = 5

            cp = centroids_p*self.dx
            cm = centroids_m*self.dy
            func_plotdefects(ax, cp, phi_p, color_p, "positive", defect_scale)
            func_plotdefects(ax, cm, phi_m, color_m, "negative", defect_scale)
            plt.xlim([0, self.x[-1]])
            plt.ylim([0, self.y[-1]])
            plt.xlabel(r"x (in $\mu$m)")
            plt.ylabel(r"y (in $\mu$m)")
            plt.title(f"t = {frame*self.dt} seconds")
            plt.tight_layout()
            plt.savefig(f'{self.vis_dir}/defects_{frame}.png')
            if show:
                plt.show()
            else:
                plt.close(fig)
        return centroids_p, centroids_m, phi_p, phi_m
    
    def num_defects_all(self, filter_radius=5, size_thresh=60, switchsign=0, plot=False):
        '''
        num_defects_all(filter_radius=5, size_thresh=60, switchsign=0,
        plot=False)
        
        Compute the number of +1/2 and -1/2 defects in each frame

        Parameters
        ----------

        filter_radius : float, optional
            Radius of line integral region. 
            Default is 5
        size_thresh : float, optional
            area threshold, keep regions greater than threshold. 
            Default is 60
        switchsign : flips identity of defects +/-  --> -/+ (needed for
        some data sets) 
        plot: bool, optional
            Flag to indicate whether to plot the frame with defects
        
        Returns
        -------
        num_p : ndarray
            1D array containing the number of +1/2 defects at each frame
        num_m : ndarray
            1D array containing the number of -1/2 defects at each frame

        '''
        num_p = np.zeros(self.NT)
        num_m = np.zeros(self.NT)

        for frame in tqdm(range(self.NT)):
            (cp, cm, _, _) = self.find_defects(filter_radius=filter_radius,
                                               size_thresh=size_thresh, 
                                               switchsign=switchsign, 
                                               frame=frame, 
                                               plot=plot)
            num_p[frame] = len(cp)
            num_m[frame] = len(cm)
        
        return num_p, num_m

if __name__ == "__main__":
    pass
