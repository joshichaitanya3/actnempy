import numpy as np 
import scipy.integrate as integrate

def optimal_SVHT_coef(beta, sigma_known):

    """
    optimal_SVHT_coef(beta, sigma_known)

    Coefficient determining optimal location of Hard Threshold for Matrix
    Denoising by Singular Values Hard Thresholding when noise level is known or
    unknown.  

    See D. L. Donoho and M. Gavish, "The Optimal Hard Threshold for Singular
    Values is 4/sqrt(3)", http://arxiv.org/abs/1305.5870

    Parameters
    ---------- 

    beta : float 
        aspect ratio m/n of the matrix to be denoised, 0<beta<=1. 
        beta may be a vector 
    sigma_known: bool
        1 if noise level known, 0 if unknown

    Returns
    -------

    coef : int
        optimal location of hard threshold, up the median data singular
        value (sigma unknown) or up to sigma*sqrt(n) (sigma known) 
        a vector of the same dimension as beta, where coef(i) is the 
        coefficient correcponding to beta(i)

    Examples
    --------

    Usage in known noise level:

    Given an m-by-n matrix Y known to be low rank and observed in white noise 
    with mean zero and known variance sigma^2, form a denoised matrix Xhat by:

    >>>U,s,V = np.linalg.svd(Y, full_matrices = False)
    >>>(n,m) = Y.shape
    >>>s[ s < (optimal_SVHT_coef(m/n,1) * np.sqrt(n) * sigma) ] = 0
    >>>Xhat = U * diag(y) * V'


    Usage in unknown noise level:

    Given an m-by-n matrix Y known to be low rank and observed in white
    noise with mean zero and unknown variance, form a denoised matrix 
    Xhat by:

    >>>U,s,V = np.linalg.svd(Y, full_matrices = False)
    >>>(n,m) = Y.shape
    >>>s[ s < (optimal_SVHT_coef(m/n,0) * np.median(s)) ] = 0 
    >>>Xhat = ( U.dot( np.diag(s).dot(V) ) )
    -----------------------------------------------------------------------------
    Authors: Matan Gavish and David Donoho <lastname>@stanford.edu, 2013

    This program is free software: you can redistribute it and/or modify it under
    the terms of the GNU General Public License as published by the Free Software
    Foundation, either version 3 of the License, or (at your option) any later
    version.

    This program is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY without even the implied warranty of MERCHANTABILITY or FITNESS
    FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
    details.

    You should have received a copy of the GNU General Public License along with
    this program.  If not, see <http://www.gnu.org/licenses/>.
    -----------------------------------------------------------------------------
    """
    if sigma_known:
        return optimal_SVHT_coef_sigma_known(beta)
    else:
        return optimal_SVHT_coef_sigma_unknown(beta)
    

def optimal_SVHT_coef_sigma_known(beta):

    beta = np.array(beta)

    assert((beta > 0.0).all())
    assert((beta <= 1.0).all())
    
    assert( np.prod(beta.shape) == max(beta.shape) ) #  beta must be a vector
    
    beta = beta.flatten()

    w = (8 * beta) / (beta + 1 + np.sqrt(beta**2 + 14 * beta +1)) 
    return np.sqrt(2 * (beta + 1) + w)

def optimal_SVHT_coef_sigma_unknown(beta):

    # warning('off','MATLAB:quadl:MinStepSize')
    beta = np.array(beta)

    assert((beta > 0.0).all())
    assert((beta <= 1.0).all())
    
    if beta.shape != (): # If beta is not a float
        assert( np.prod(beta.shape) == max(beta.shape) ) #  beta must be a vector
    
    beta = beta.flatten()
    
    coef = optimal_SVHT_coef_sigma_known(beta)
    
    mp_median = [median_marcenko_pastur(beta_i) for beta_i in beta]
    
    return coef / np.sqrt(mp_median)


def median_marcenko_pastur(x,beta):

    if (beta <= 0) or (beta > 1) :
        msg = "Invalid value of beta. Beta should lie between 0 and 1"
        raise ValueError(msg)
    
    lower_bound = ( 1 - np.sqrt(beta) )**2
    higher_bound = ( 1 + np.sqrt(beta) )**2
    
    if (x < lower_bound) or (x > higher_bound) :
        msg = "Invalid value of x. " + \
              "x should lie between (1-sqrt(beta))**2 and  (1+sqrt(beta))**2"
        raise ValueError(msg)

    dens = lambda t : np.sqrt( (higher_bound-t) * (t-lower_bound) ) / (2 * np.pi * beta * t)

    (I, _) = integrate.quad(dens, lower_bound, x)
    print(f'x = {x:.3f}, beta = {beta:.3f}, I = {I:.3f}')

    return I


def median_marcenko_pastur(beta):

    mar_pas = lambda x: 1 - inc_mar_pas(x,beta,0)
    lower_bound = ( 1 - np.sqrt(beta) )**2
    higher_bound = ( 1 + np.sqrt(beta) )**2

    change = 1
    while change and (higher_bound - lower_bound > .001):
      change = 0
      x = np.linspace(lower_bound, higher_bound, 5)

      y = np.array([mar_pas(xi) for xi in x])
    #   for i=1:length(x),
    #       y(i) = MarPas(x(i))
    #   end
      if (y < 0.5).any():
         lower_bound = max(x[y < 0.5])
         change = 1
      if (y > 0.5).any():
         higher_bound = min(x[y > 0.5])
         change = 1
    
    return (higher_bound + lower_bound) / 2

def inc_mar_pas(x0,beta,gamma):

    if (beta > 1) :
        msg = "Invalid value of beta. Beta should be greater than 1"
        raise ValueError(msg)
    
    top_spec = (1 + np.sqrt(beta))**2
    bot_spec = (1 - np.sqrt(beta))**2

    mar_pas = lambda x: if_else( Q=(top_spec-x) * (x-bot_spec) >0,
                                 point=np.sqrt( (top_spec-x) * (x-bot_spec)) / (beta * x) / (2 * np.pi),
                                 counter_point=0)
    if gamma != 0:
       fun = lambda x : ( x**gamma * mar_pas(x) )
    else:
       fun = lambda x : mar_pas(x)

    (I, _) = integrate.quad(fun, x0, top_spec)
    
    return I

def if_else(Q,point,counter_point):

    y = point
    counter_point = np.array(counter_point)
    if (~Q).any():
        if counter_point.size == 1:
            counter_point = np.ones(Q.shape) * counter_point

        if Q.size == 1:
            y = counter_point
        else:
            y[~Q] = counter_point[~Q]

    return y

    