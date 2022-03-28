'''

Active Nematic Identification of Sparse Equations (ANISE)

PDE Module

Code by Chaitanya Joshi (chaitanya@brandeis.edu)

This module provides the Class PDE, which is designed to conveniently analyze 
the result of the sparse identification framework. In addition, it provides 
some modifications of the functions available in the original PDE-FIND framework.


'''

import numpy as np
import matplotlib.pyplot as plt 
from .PDE_FIND import TrainSTRidge
from .library_tools import get_term_val
import warnings 
from ..utils.grid import Grid
from pathlib import Path
from sklearn.metrics import r2_score

### Matplotlib settings for PRX / PRL figures
parent = Path(__file__).parent.parent

style_file = (parent / "prx.mplstyle").resolve()
plt.style.use(str(style_file))


###

def HRidge(X0, y, lam, normalize = 2):
    """
    (w_all, r2) = HRidge(X0, y, lam, normalize = 2)
    
    Heirarchical Ridge Regression: algorithm for finding a heirarchy of 
    successively sparser approximations to X0^{-1}y. 

    This assumes y is only one column
    
    Instead of cutting off values smaller than an arbitrary tolerance, 
    the value with the smallest coefficient (after normalization) is cut-off 
    one at a time.

    It will return the whole heirarchy of models, along with their R2 scores.
    
    Parameters
    ----------
    
    X0 : ndarray
        Array of shape (n,d) containing the n observations of the d terms on 
        the right hand side.
    
    y : ndarray
        Array of shape (n,1) containing the n observations of the left hand side
    
    normalize : int
        Optional argument for the ord of the normalization. Default is 2.
    
    Returns
    -------

    w_all : ndarray
        Array of shape (d,d) containing the d coefficients at all the d levels 
        of sparsity. Currently, w_all[0,:] contains the densest model and 
        w_all[-1,:] contains the sparsest.
    

    r2 : ndarray
        Array of shape (d,) containing the corresponding r-squared values at 
        each level of sparsity.
    
    """

    n,d = X0.shape
    # Allocating memory for all the heirarchical solutions
    w_all = np.zeros([d, d], dtype=np.complex64)
    r2 = np.zeros(d)
    X = np.zeros((n, d), dtype=np.complex64)
    # First normalize data
    if normalize != 0:
        Mreg = np.zeros((d,1))
        for i in range(0,d):
            Mreg[i] = 1.0/(np.linalg.norm(X0[:,i],normalize))
            X[:,i] = Mreg[i]*X0[:,i]
    else: X = X0
    
    # Get the standard ridge esitmate
    if lam != 0: w = np.linalg.lstsq(X.T.dot(X) + lam*np.eye(d), X.T.dot(y), rcond=None )[0]
    else: w = np.linalg.lstsq(X,y, rcond=None)[0]

    # biginds = np.where( abs(w) > np.min(abs(w)))[0]
    # w_all[0] = w.flatten().copy()
    # print(np.linalg.lstsq(X,y, rcond=None)[0].flatten().shape)
    w_all[0] = np.linalg.lstsq(X,y, rcond=None)[0].flatten() # The solution with all the terms included is just least squares
    
    # Threshold and continue
    lhs = np.squeeze(np.real(y))

    if normalize != 0:
            wr = np.multiply(Mreg, w)
            w_all[0] = np.multiply(Mreg, w_all[0][:, np.newaxis]).flatten()
    else:
        wr = w.copy()
    rhs = np.squeeze(np.real(X0.dot(wr)))

    r2[0] = r2_score(lhs, rhs)
    
    for j in range(1,d):

        # Figure out the position of the term with the smallest coefficient
        smallid = np.where(abs(w) <= np.min(abs(w[np.nonzero(w)])))[0]
        biginds = [i for i in range(d) if i not in smallid]

        w[smallid] = 0
        w_all[j][biginds] = np.linalg.lstsq(X[:, biginds], y, rcond=None)[0].flatten()

        if normalize != 0:
            w_all[j] = np.multiply(Mreg, w_all[j][:, np.newaxis]).flatten()

        rhs = np.squeeze(np.real(X0.dot(w_all[j])))

        r2[j] = r2_score(lhs, rhs)

        if lam != 0: w[biginds] = np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) + lam*np.eye(len(biginds)),X[:, biginds].T.dot(y), rcond=None)[0]
        else: w[biginds] = np.linalg.lstsq(X[:, biginds],y, rcond=None)[0]
    
    return (w_all, r2)

def kfold_cv(X, y, k=10):
    '''
    (w_all_train, r2train, r2test, variance) = kfold_cv(X, y, k=10)

    Perform k-fold cross validation (CV) for the PDE fit found using the
    HRidge algorithm for y=X.a. If k is not specified, a 10-fold CV is
    performed by default. If enough data-points aren't available for
    10-fold CV (# of data-points < 10*nparameters), then the maximum k
    possible is picked.

    Parameters
    ----------

    X : ndarray
        Array of shape (n,d) containing the n observations of the d terms on 
        the right hand side.
    
    y : ndarray
        Array of shape (n,1) containing the n observations of the left hand side
    
    k : int
        Number of folds in the cross-validation. Default is 10, however
        a smaller value is automatically chosen if there isn't enough data.
    
    Returns
    -------
    
    w_all_train : ndarray
        Array of shape (k,d,d) containing the d coefficients at all the d levels 
        of sparsity, for all the k training sets. Currently, w_all[i,0,:] 
        contains the densest model and w_all[i,-1,:] contains the sparsest.

    r2train : ndarray
        Array of shape (k,d) containing the corresponding r-squared values at 
        each level of sparsity for all the k training sets.

    r2test : ndarray
        Array of shape (k,d) containing the r-squared values at each level of 
        sparsity for all the k test sets. This r-squared is obtained by seeing 
        how well the parameters obtained from the training sets fit to the test
        sets.
    
    variance : ndarrray
        Array of shape (d,1) containing the variance of the model across the 
        k-folds. This variance is measured only in binary. If the optimal model 
        contains the same m terms at the m-th level of sparsity across all the 
        k-folds, then variance is False (0), else it is True (1).
        
    '''

    n = y.shape[0] # Total number of data-points
    nparameters = X.shape[1]

    rng = np.random.default_rng()
    ids = np.arange(n)
    rng.shuffle(ids)
    X = X[ids,:]
    y = y[ids,:]

    if n<k*nparameters:
        k = np.floor(n/nparameters).astype(int)
        print("k is too large for the amount of data present...")
        print(f"Switching to k={k}")
    m = np.round(n/k).astype(int) # Rough number of data-points in each subsample

    ids = np.arange(n)
    ids = ids[::m]
    ids = np.append(ids,n)

    r2train = np.zeros([k, nparameters])
    r2test = np.zeros([k, nparameters])
    w_all_train = np.zeros([k,nparameters,nparameters], dtype=np.complex_)
    for i in range(k):
        subsample = slice(ids[i],ids[i+1])
        Xtrain = np.delete(X, subsample, axis=0)
        ytrain = np.delete(y, subsample, axis=0)

        Xtest = X[subsample,:].copy()
        ytest = y[subsample].copy()

        (w_all_train[i], r2train[i]) = HRidge(Xtrain,ytrain,10**-5) 
        
        lhs_test = np.squeeze(np.real(ytest))

        for j in range(nparameters):

            rhs_test = np.squeeze(np.real(Xtest.dot(w_all_train[i,j])))

            r2test[i,j] = r2_score(lhs_test, rhs_test)
    
    n_terms = np.arange(1,nparameters+1) 
    variance = (np.count_nonzero(np.prod(w_all_train,axis=0),axis=1) - np.flipud(n_terms))!=0 
    return (w_all_train, r2train, r2test, variance)


def print_pde(w, rhs_description, ut='u_t'):
    '''
    print_pde(w, rhs_description, ut = 'u_t')

    Function to print the PDE model from a coefficient vector `w` and
    the corresponding list of term names `rhs_description`.

    Parameters
    ----------

    w : ndarray Vector of coefficients, shape (n,1)

    rhs_description: list of str
        List of length n containing the names of the terms on the right
        hand side.

    ut : str, optional
        String corresponding to the name of the left hand side term.
        Default is 'u_t'

    Returns
    -------

    None

    '''
    pde = ut + ' =   '
    first = True
    for i in range(len(w)):
        if w[i] != 0:
            if not first:
                pde = pde + len(ut)*' ' + '+ '
            pde = pde + "(%g) " % w[i].real + rhs_description[i] + "\n   "
            first = False
    print(pde)

class PDE:
    '''
    
    This class is designed to conveniently analyze the result of the 
    sparse identification framework.
    
    
    Attributes
    ----------
    
    rhs : dtype=[('name','U50'),('val','complex128' (num_windows,))]
        Array containing the terms on the right hand side.

    lhs : dtype=[('name','U50'),('val','complex128' (num_windows,))]
        Array containing the terms on the left hand side.

    ut : str
        'name' of the term to pick from the left hand side, e.g. '(∂ω/∂t)'.
    
    
        [Feature Request] (eliminate the need for ut, and pass lhs with only
                           one term)
    
    metadata : dictionary
        Dictionary containing the values for the data directory "data_dir" and 
        the value of the number of folds "k" for cross-validation.
        
    ridge_lam : float, optional
        Value of lambda to be used in the Ridge regression at each level of 
        sparsity. Default is set to 10**-5. ridge_lam = 0 implies least squares.
        
    
    '''
    def __init__(self, filename=None):
        
        self.grid = Grid(h=1, ndims=1)

        self.ut = "u_t" # Default value

        if filename is not None:
            
            self.load(filename)

            self._calculate_quantities()

    def compute(self, rhs, lhs, ut, metadata, ridge_lam=10**-5):

        num_windows = metadata['num_windows']
        window_size = metadata['window_size']
        
        try:
            k = metadata["k"]
        except KeyError:
            k = 10
        self.metadata = metadata
        self.rhs = rhs
        self.lhs = lhs
        self.ut = ut
        self.lam = ridge_lam
        self.y = get_term_val(lhs, ut)
        try:
            self.ya = np.mean(self.y.reshape(num_windows,window_size**3),axis=-1)[:,np.newaxis]
        except ValueError:
            self.ya = self.y[:,np.newaxis]
        self.desc = list(rhs['name'])
        self.X = rhs['val']

        self.nt = self.X.shape[0]
        try:
            self.Xa = np.mean(self.X.reshape(self.nt,num_windows,window_size**3),axis=-1).T
        except ValueError:
            self.Xa = self.X.T
        
        # Using HRidge to get the heirarchy of models
        (self.w_all, self.r2) = HRidge(self.Xa,self.ya,self.lam) 
        
        # Independently, doing the same with k-fold cross-validation
        (self.w_all_train, self.r2train, self.r2test, self.variance) = kfold_cv(self.Xa, self.ya, k=k)
        
        self.variance = np.flipud(self.variance)
        self._calculate_quantities()

    def _calculate_quantities(self):

        self.w_all_k = self.w_all_train

        self.w_all_k[self.w_all_k == 0] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.w_all_avg = np.nanmean(self.w_all_k, axis=0)
            self.w_all_std = np.nanstd(self.w_all_k, axis=0)
        self.w_all_k[np.isnan(self.w_all_k)] = 0
        self.w_all_avg[np.isnan(self.w_all_avg)] = 0
        self.w_all_std[np.isnan(self.w_all_std)] = 0

        self.r2train_avg = np.mean(self.r2train, axis=0)
        self.r2test_avg = np.mean(self.r2test, axis=0)
        self.r2train_min = np.min(self.r2train, axis=0)
        self.r2test_min = np.min(self.r2train, axis=0)
        self.r2train_max = np.max(self.r2train, axis=0)
        self.r2test_max = np.max(self.r2train, axis=0)

        # Number of terms goes from 1 to n
        self.n_terms = np.arange(1, len(self.r2)+1)
        # Fraction of Variance Unexplained (FVU). The indices of r2 start with
        # the densest model and down to the sparsest, so we use flipud
        self.fvu = np.flipud(1-self.r2)
        self.fvu_mean = np.flipud(1-self.r2train_avg)
        self.fvu_min = np.flipud(1-self.r2train_max)
        self.fvu_max = np.flipud(1-self.r2train_min)
        self.fvu_err = np.array([self.fvu_min, self.fvu_max])

        self.fvut_mean = np.flipud(1-self.r2test_avg)
        self.fvut_min = np.flipud(1-self.r2test_max)
        self.fvut_max = np.flipud(1-self.r2test_min)
        self.fvut_err = np.array([self.fvut_min, self.fvut_max])

        # self.variance = np.flipud(self.variance)

        # The shoulder of the fvu vs n is determined by the maxima of lap(log(fvu))

        lapfvu = self.grid.lap(np.log(self.fvu))

        # +1 is to account for 0 indexing
        self.nopt = np.argmax(lapfvu[:-1]) + 1

        print("Optimal model: ")
        self.display_model(self.nopt)


    def save(self, path):    
        '''
        
        save(path)
        
        Save all the relevant results to a .npz file. 
        
        Includes: w_all, w_all_train, r2, r2train, r2test, variance and desc
        
        Parameters
        ----------
        path : str
            Full path to the .npz file in which to save the result

        Returns
        -------
        None.

        '''
        np.savez(path,
                 w_all = self.w_all,
                 w_all_train = self.w_all_train,
                 r2 = self.r2,
                 r2train = self.r2train,
                 r2test = self.r2test,
                 variance = self.variance,
                 desc = self.desc,
                 )

    def load(self, filename):
    
        try:
            pde = np.load(filename)
            self.w_all = pde["w_all"]
            self.w_all_train = pde["w_all_train"]
            self.r2 = pde["r2"]
            self.r2train = pde["r2train"]
            self.r2test = pde["r2test"]
            self.variance = pde["variance"]
            self.desc = pde["desc"]

        except KeyError:
            print(f"Some keys not found. The PDE isn't saved properly!")
    
    def stridge(self, lam=10**-5, d_tol=10**-5):
        '''
        stridge(lam, d_tol)
        
        Use the method of Sequential Thresholded Ridge Regression (STRidge)
        from (Rudy2017) to get the sparse model. The optimal model is  
        printed, along with the r-squared score.

        Parameters
        ----------
        lam : float, optional
            Penalty for the 2-norm of the coefficients for Ridge Regression.
            Default is 10**-5
        d_tol : float, optional
            Hyperparameter to specify the sparsity level
            Default is 10**-5

        Returns
        -------
        w : ndarray
            Vector of coefficients in the optimal model
        r2 : float
            r-squared score for the optimal model

        '''
        (w, r2) = TrainSTRidge(self.Xa, self.ya, lam, d_tol)
        print(f"R-squared: {r2:g}")
        print_pde(w, self.desc)
        return (w, r2)
    
    def plot_fvu(self,nterms=None,var='var',filename=None):
        '''
        plot_fvu()
          
        Generate a plot of FVU (Fraction of Variance Unexplained) vs number of 
        non-zero terms found using the HRidge method.
        
        Parameters
        ----------
        nterms : int, optional
            Number of terms to truncate the x-axis for the plot. 
            The default is None, in which case the x-axis will not be truncated
        var : str, optional
            Name to include in the filename of the saved plot. 
            The default is 'var'.
        filename : str, optional
            Full path to the filename of the saved plot. 
            The default is None, in which case it will be saved as 
            f'fvu_{var}_nterms_{nterms}' in the PDE's data_dir

        Returns
        -------
        None.

        '''
        if nterms is None:
            nterms = self.nt
        if var is None:
            var = 'var'
        # Get the default figsize meant for PRX half-column figures
        [fig_width, fig_height] = plt.rcParams["figure.figsize"]
        # Generate figure for full column
        fig = plt.figure(figsize=(fig_width, 0.6*fig_height))
        plt.semilogy(self.n_terms,self.fvu,'o-')
        plt.xlabel('Number of non-zero terms')
        plt.ylabel(r'$1 - R^2$')
        plt.xlim([0,nterms+0.5])
        ax = plt.gca()
        ax.tick_params(direction="in")
        ax.tick_params(which="minor",direction="in")
        plt.tight_layout()
        if filename is None:
            filename = f'{self.metadata["data_dir"]}/fvu_{var}_nterms_{nterms}'
        fig.savefig(f"{filename}.png",dpi=300)
        fig.savefig(f"{filename}.svg",dpi=300)
        fig.savefig(f"{filename}.pdf",dpi=300)
        
        plt.show()
        
    def plot_fvu_kfold(self,nterms=None,var='var',filename=None):
        '''
        plot_fvu_kfold()
          
        Generate a plot of FVU (Fraction of Variance Unexplained) vs number of 
        non-zero terms found using k-fold cross-validation on the HRidge method.
        
        Training FVU's are plotted in blue, while test FVU's are plotted in 
        orange. Models with no variance across the k-folds are indicated with 
        circles whereas the models with variance are indicated with triangles.
        
        Parameters
        ----------
        nterms : int, optional
            Number of terms to truncate the x-axis for the plot. 
            The default is None, in which case the x-axis will not be truncated
        var : str, optional
            Name to include in the filename of the saved plot. 
            The default is 'var'.
        filename : str, optional
            Full path to the filename of the saved plot. 
            The default is None, in which case it will be saved as 
            f'fvu_{var}_nterms_{nterms}_cv' in the PDE's data_dir

        Returns
        -------
        None.

        '''
        if nterms is None:
            nterms = self.nt
        if var is None:
            var = 'var'
        [fig_width, fig_height] = plt.rcParams["figure.figsize"]
        # Generate figure for full column
        fig = plt.figure(figsize=(2*fig_width, fig_height))
        zv = self.variance==0 # ids with zero variance
        p1 = plt.errorbar(self.n_terms[zv],self.fvu_mean[zv],yerr=self.fvu_err[:,zv],fmt='o',color='tab:orange')
        p2 = plt.errorbar(self.n_terms[~zv],self.fvu_mean[~zv],yerr=self.fvu_err[:,~zv],fmt='^',color='tab:orange')
        p3 = plt.errorbar(self.n_terms[zv],self.fvut_mean[zv],yerr=self.fvut_err[:,zv],fmt='o',color='tab:blue')
        p4 = plt.errorbar(self.n_terms[~zv],self.fvut_mean[~zv],yerr=self.fvut_err[:,~zv],fmt='^',color='tab:blue')
        yl,yh = plt.gca().get_ylim()
        plt.yscale('log')
        plt.xlabel('Number of non-zero terms')
        plt.ylabel(r'$1 - R^2$')

        plt.legend([(p1,p2),(p3,p4)],['Train data','Test data'],scatterpoints=2)
        plt.xlim([0,nterms+0.5])
        plt.tight_layout()
        if filename is None:
            filename = f'{self.metadata["data_dir"]}/fvu_{var}_nterms_{nterms}_cv'
        fig.savefig(f"{filename}.png",dpi=300)
        fig.savefig(f"{filename}.pdf",dpi=300)
        plt.show()
    
    def plot_fvu_kfold_paper(self,nterms=None,var='var',filename=None):
        '''
        plot_fvu_kfold_paper()

        Generate a plot of average FVU (Fraction of Variance
        Unexplained) vs number of non-zero terms found using k-fold
        cross-validation on the HRidge method.

        Models with no variance across the k-folds are indicated with
        circles whereas the models with variance are indicated with
        triangles.

        Parameters
        ----------
        nterms : int, optional Number of terms to truncate the x-axis
            for the plot. The default is None, in which case the x-axis
            will not be truncated var : str, optional Name to include in
            the filename of the saved plot. The default is 'var'.
            filename : str, optional Full path to the filename of the
            saved plot. The default is None, in which case it will be
            saved as f'fvu_{var}_nterms_{nterms}_cv' in the PDE's
            data_dir

        Returns
        -------
        None.

        '''
        if nterms is None:
            nterms = self.nt
        if var is None:
            var = 'var'
        [fig_width, fig_height] = plt.rcParams["figure.figsize"]
        # Generate figure for full column
        fig = plt.figure(figsize=(fig_width, 0.5*fig_height))
        # fig = plt.figure()
        zv = self.variance==0 # ids with zero variance

        p1 = plt.errorbar(self.n_terms[zv], self.fvu_mean[zv],
                          yerr=self.fvu_err[:, zv], fmt='o', color='tab:blue')
        p2 = plt.errorbar(self.n_terms[~zv], self.fvu_mean[~zv],
                          yerr=self.fvu_err[:, ~zv], fmt='^', color='tab:blue')

        # p3 = plt.errorbar(self.n_terms[zv],self.fvut_mean[zv],yerr=self.fvut_err[:,zv],fmt='o',color='tab:blue')
        # p4 = plt.errorbar(self.n_terms[~zv],self.fvut_mean[~zv],yerr=self.fvut_err[:,~zv],fmt='^',color='tab:blue')
        yl,yh = plt.gca().get_ylim()
        plt.yscale('log')
        plt.xlabel('Number of non-zero terms')
        plt.ylabel(r'$1 - R^2$')

        # plt.legend([p3,p4],['Indicates same model under cross-validation','Indicates different model under cross-validation'],scatterpoints=2)
        plt.xlim([0,nterms+0.5])
        plt.tight_layout()
        if filename is None:
            filename = f'{self.metadata["data_dir"]}/fvu_paper_{var}_nterms_{nterms}_cv'
        fig.savefig(f"{filename}.png",dpi=300)
        fig.savefig(f"{filename}.pdf",dpi=300)
        plt.show()
    
    def display_model(self,n=None):
        '''
        display_model()
        
        Display the model at the optimal sparsity. An optional argument can be 
        provided to display the model at a specific sparsity.
        
        Parameters
        ----------
        n : int, optional
            Displays the model with n non-zero terms. The default is None, in
            which case, the optimal model is displayed. The optimal model is 
            the level at which lap(log(fvu)) is maximum.
 
        Returns
        -------
        None.

        '''
        if n is None:
            n = self.nopt
            
        idx = -n
        beta = self.w_all[idx][:, np.newaxis]
        print(f"R2 score: {self.r2[idx]}")
        print(f"Contains {np.count_nonzero(beta)}/{len(beta)} terms...")
        print_pde(beta, self.desc, self.ut)
    
    def display_model_avg(self,n=None):
        '''
        display_model_avg()
        
        Display the model at the optimal sparsity, averaged across the k-folds.
        An optional argument can be provided to display the model at a
        specific sparsity.
        
        Parameters
        ----------
        n : int, optional
            Displays the model with n non-zero terms. The default is None, in
            which case, the optimal model is displayed. The optimal model is 
            the level at which lap(log(fvu)) is maximum.
 
        Returns
        -------
        None.

        '''
        if n is None:
            n = self.nopt
        self.idx = -n
        self.beta = self.w_all_avg[self.idx][:, np.newaxis]
        print(f"R2 score: {self.r2[self.idx]}")
        print(f"Contains {np.count_nonzero(self.beta)}/{len(self.beta)} terms...")
        print_pde(self.beta, self.desc, self.ut)


if __name__ == "__main__":
    pass
    
