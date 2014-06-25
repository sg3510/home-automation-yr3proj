"""
Hyperpriors for log likelihood calculation
------------------------------------------
This module contains a set of commonly used priors for GP models.
Note some priors are available in a log transformed space and non-transformed space
"""

import scipy as SP
import numpy as NP
import pdb
import scipy.special as SPs
import itertools


def lnFobgp(x,params):
    """Fobenius norm prior on latent space:
    x: factors [N x Q]
    params: scaling parameter 
    params[0]: prior cost
    Note: this prior only works if the paramter is constraint to be strictly positive
    """
    lng = NP.zeros_like(x)
    dlng = NP.zeros_like(x)
    # f = -
    #TODO: normalize the x vectors to unit variance,ie.
    #x2 = x/SP.dot(x[:,0],x[:,0])**2 etc.
    #this also needs to enter the gradients ...

    if 0: 
        lng[0,0] = 1.0 * ((NP.dot(x.T, x))**2).sum()
        chain = 4.0*NP.dot(x.T, x)
        dlng = 1.0* NP.dot(chain, x.T).T

    if 1:
        for i,j in itertools.combinations(xrange(x.shape[1]),2):
            norm_x_i = NP.sqrt(NP.dot(x[:, i:i+1].T, x[:, i:i+1]))
            norm_x_j = NP.sqrt(NP.dot(x[:, j:j+1].T, x[:, j:j+1]))
            vect_dot = NP.dot(x[:, i:i+1].T, x[:, j:j+1])[0,0]
            inner = NP.sqrt(vect_dot**2)
            lng[0,0] += (inner / (norm_x_i * norm_x_j))
            
            dlng[:, i:i+1] += (1.0/inner * vect_dot * x[:,j:j+1])/(norm_x_i*norm_x_j) - (inner/(norm_x_i*norm_x_j)**2 * x[:, i:i+1]*norm_x_j)/norm_x_i
            dlng[:, j:j+1] += (1.0/inner * vect_dot * x[:,i:i+1])/(norm_x_i*norm_x_j) - (inner/(norm_x_i*norm_x_j)**2 * x[:, j:j+1]*norm_x_i)/norm_x_j              
                
                # dlng[:, i:i+1] += -1.0/(inner**2)*2.0*NP.dot(x[:, i:i+1].T, x[:, j:j+1])[0,0]*x[:, j:j+1]/(norm_x_i * norm_x_j) - (inner/(norm_x_i * norm_x_j)**2 * (x[:, i:i+1]*norm_x_j)/norm_x_i)
                # dlng[:, j:j+1] +=  -1.0/(inner**2)*2.0*NP.dot(x[:, i:i+1].T, x[:, j:j+1])[0,0]*x[:, i:i+1]/(norm_x_i * norm_x_j) - (inner/(norm_x_i * norm_x_j)**2 * (x[:, j:j+1]*norm_x_i)/norm_x_j)

    if 0:
        for i in range(x.shape[1]):
            for j in range(x.shape[1]):
                lng[0,0] += NP.dot(x[:, i:i+1].T, x[:, j:j+1])[0,0]

        dlng = NP.zeros_like(x)
        for i in range(x.shape[1]):
            for j in range(x.shape[1]):
                dlng[:, i:i+1] += x[:, j:j+1]
                dlng[:, j:j+1] += x[:, i:i+1]

    # lng = (NP.abs(x)**2).sum()
    # lng = x**2
    # dlng = 2*x
    lng = lng * params[0]
    dlng = dlng * params[0]   
    # lng = SP.zeros_like(x)
    # dlng = SP.zeros_like(x)
    return [lng,dlng]



def lnL1(x,params):
    """L1 type prior defined on the non-log weights
    params[0]: prior cost
    Note: this prior only works if the paramter is constraint to be strictly positive
    """
    l = SP.double(params[0])
    x_ = 1./x

    lng = -l * x_
    dlng = + l*x_**2
    return [lng,dlng]



def lnGamma(x,params):
    """
    Returns the ``log gamma (x,k,t)`` distribution and its derivation with::
    
        lngamma     = (k-1)*log(x) - x/t -gammaln(k) - k*log(t)
        dlngamma    = (k-1)/x - 1/t
    
    
    **Parameters:**
    
    x : [double]
        the interval in which the distribution shall be computed.
    
    params : [k, t]
        the distribution parameters k and t.
    
    """
    #explicitly convert to double to avoid int trouble :-)
    k=SP.double(params[0])
    t=SP.double(params[1])

    lng     = (k-1)*SP.log(x) - x/t -SPs.gammaln(k) - k*SP.log(t)
    dlng    = (k-1)/x - 1/t
    return [lng,dlng]


def lnGammaExp(x,params):
    """
    
    Returns the ``log gamma (exp(x),k,t)`` distribution and its derivation with::
    
        lngamma     = (k-1)*log(x) - x/t -gammaln(k) - k*log(t)
        dlngamma    = (k-1)/x - 1/t
   
    
    **Parameters:**
    
    x : [double]
        the interval in which the distribution shall be computed.
    
    params : [k, t]
        the distribution parameters k and t.
    
    """
    #explicitly convert to double to avoid int trouble :-)
    ex = SP.exp(x)
    rv = lnGamma(ex,params)
    rv[1]*= ex
    return rv


def lnGauss(x,params):
    """
    Returns the ``log normal distribution`` and its derivation in interval x,
    given mean mu and variance sigma::

        [N(params), d/dx N(params)] = N(mu,sigma|x).

    **Note**: Give mu and sigma as mean and variance, the result will be logarithmic!

    **Parameters:**

    x : [double]
        the interval in which the distribution shall be computed.

    params : [k, t]
        the distribution parameters k and t.
        
    """
    mu = SP.double(params[0])
    sigma = SP.double(params[1])
    halfLog2Pi = 0.91893853320467267 # =.5*(log(2*pi))
    N = SP.log(SP.exp((-((x-mu)**2)/(2*(sigma**2))))/sigma)- halfLog2Pi
    dN = -(x-mu)/(sigma**2)
    return [N,dN]

def lnuniformpdf(x,params):
    """
    Implementation of ``lnzeropdf`` for development purpose only. This
    pdf returns always ``[0,0]``.  
    """
    return [0,0]

def _plotPrior(X,prior):
    import pylab as PL
    Y = SP.array(prior[0](X,prior[1]))
    PL.hold(True)
    PL.plot(X,SP.exp(Y[0,:]))
    #plot(X,(Y[0,:]))
    #plot(X,Y[1,:],'r-')
    PL.show()
    
if __name__ == "__main__":
    prior = [lnGammaExp,[4,2]]
    X = SP.arange(0.01,10,0.1)
    _plotPrior(X,prior)
