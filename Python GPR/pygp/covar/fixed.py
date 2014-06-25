"""fixed covariance functions
Classes for fixed covarinace functions
======================================
FixedCF

"""

import scipy as SP
from pygp.covar.covar_base import BayesianStatisticsCF, CovarianceFunction
import sys

class FixedCF(CovarianceFunction):
    """
    Fixed covariance function with learnt scaling parameter
    """

    def __init__(self,K,n_dimensions=0,*args,**kw_args):
        """K: the fixed covariance structure"""
        super(FixedCF, self).__init__(n_dimensions,*args,**kw_args)
        self._K = K 
        self.n_hyperparameters = 1

    def K(self,theta,x1,x2=None):
        #get input data:
        x1, x2 = self._filter_input_dimensions(x1, x2)
        #scale parameter
        A  = SP.exp(2*theta[0])
        #if dimensions match return K
        if (x1.shape[0]==self._K.shape[0]) and (x2.shape[0]==self._K.shape[1]):
            # debug:
            #import pdb;pdb.set_trace()
            return A*self._K
        else:
            #sys.stdout.write("WARNING: Objects are not aligned in fixedCF\r")
            return SP.zeros([x1.shape[0],x2.shape[0]])

    def Kdiag(self,theta,x1):
        """self covariance"""
        if (x1.shape[0]==self._K.shape[0]):
            return self._K.diagonal()
        else:
            return SP.zeros([x1.shape[0]])

    def Kgrad_theta(self,theta,x1,i):
        RV = self.K(theta,x1)
        #derivative w.r.t. to amplitude
        RV*=2
        return RV

    def Kgrad_x(self,theta,x1,x2,d):
        RV = SP.zeros([x1.shape[0],x2.shape[0]])
        return RV
    
    def Kgrad_xdiag(self,theta,x1,d):
        """derivative w.r.t diagonal of self covariance matrix"""
        RV = SP.zeros([x1.shape[0]])
        return RV

    def get_hyperparameter_names(self):
        names = []
        names.append('FixedCF Amplitude')
        return names

class FixedCFPsiStat(FixedCF, BayesianStatisticsCF):

    def update_stats(self, theta, mean, variance, inducing_variables):
        super(FixedCFPsiStat, self).update_stats(theta, mean, variance, inducing_variables)
        self.reparametrize(theta)
        
        self.psi0 = self.N * self.sigma
        self.psi1 = self.K(self.theta, self.mean, self.inducing_variables)
        self.psi2 = self.N * self.K(theta, self.inducing_variables)**2
        return BayesianStatisticsCF.update_stats(self, theta, mean, variance, inducing_variables)


    def reparametrize(self, theta=None, **kwargs):
        if theta is not None:
            self.sigma = SP.exp(2*theta[0])


    def psi(self):
        return self.psi0, self.psi1, self.psi2

    def psigrad_theta(self):
        return 2*self.psi0, 0., 4*self.psi2[:,:,None,None]


    def psigrad_mean(self):
        return 0., 0., 0.


    def psigrad_variance(self):
        return 0., 0., 0.
        

    def psigrad_inducing_variables(self):
        return 0., 0., 0.
    
class SqexpFixed(CovarianceFunction):
    """
    Fixed covariance function with lengthscale parameter for exponentiation
    (e.g. matrix of pairwise distances)
    """

    def __init__(self,K,**kw_args):
        """K: the fixed pairwise distance matrix"""
        super(CovarianceFunction, self).__init__(**kw_args)
        self._K = K 
        self.n_hyperparameters = 2

    def K(self,theta,x1,x2=None):
        #get input data:
        if x2 is None:
            x2 = x1
        #scale parameter
        A  = SP.exp(2*theta[0])
        #lengthscale
        L  = SP.exp(theta[1])
        #rescaled distances (squared)
        Dr  = (self._K/L)**2
        K   = SP.exp(-0.5*Dr)
        #if dimensions match return K
        if (x1.shape[0]==self._K.shape[0]) & (x2.shape[0]==self._K.shape[1]):
            return A*K
        else:
            return SP.zeros([x1.shape[0],x2.shape[0]])

    def Kgrad_theta(self,theta,x1,i):
        #calc covariance
        RV = self.K(theta,x1)
        if i==0:
            #derivative w.r.t. to amplitude
            RV*=2
        elif i==1:
            #lengthscale
            L  = SP.exp(theta[1])
            #rescaled distances (squared)
            Dr  = (self._K/L)**2
            RV *= Dr         
        return RV


    def get_hyperparameter_names(self):
        names = []
        names.append('SqexpFixed Amplitude')
        names.append('SqexpFixed Lengthscale')
        return names
