"""
Noise covariance function
-------------------------

NoiseCFISO
NoiseCFReplicates
"""

from pygp.covar.covar_base import BayesianStatisticsCF, CovarianceFunction
import numpy
import scipy as SP
import sys
sys.path.append("../")


# import python / numpy:




class NoiseCFISO(CovarianceFunction):
    """
    Covariance function for Gaussian observation noise for
    all datapoints as a whole.
    """
    
    def __init__(self,n_dimensions=-1, *args,**kw_args):
        CovarianceFunction.__init__(self,n_dimensions, *args,**kw_args)
        self.n_hyperparameters = 1

    def get_hyperparameter_names(self):
        #"""return the names of hyperparameters to make identificatio neasier"""
        names = []
        names.append('Noise Amplitude')
        return names

    def K(self,theta,x1,x2=None):
        """
        Get Covariance matrix K with given hyperparameters theta and inputs *args* = X[, X']. Note that this covariance function will only get noise as hyperparameter!

        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction` 
        """
        
        #noise is only presenet if have a single argument
        x1,x2 = self._filter_input_dimensions(x1, x2)
        #noise = (x1 == x2)*SP.exp(2*theta[0])    
        if x1.shape==x2.shape and (x1 == x2).all():
            return numpy.eye(x1.shape[0]) * SP.exp(2*theta[0])
        else:
            return 0

    def Kgrad_theta(self,theta,x1,i):
        """
        The derivative of the covariance matrix with
        respect to i-th hyperparameter.

        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction`
        """
        #1. calculate kernel
        #no noise
        K = self.K(theta,x1)
        assert i==0, 'unknown hyperparameter'
        return 2*K

    def Kgrad_x(self,theta,x1,x2,d):
        RV = SP.zeros([x1.shape[0],x2.shape[0]])
        return RV

    def Kgrad_xdiag(self,theta,x1,d):
        RV = SP.zeros([x1.shape[0]])
        return RV
    
    

class NoiseCFISOPsiStat(NoiseCFISO, BayesianStatisticsCF):
    
    def psi(self):
        psi_0 = psi_1 = psi_2 = 0
        return [psi_0, psi_1, psi_2]

    def psigrad_theta(self):
        return [0]*3
#        psi_grad_0 = numpy.zeros((self.theta.shape[0],1))
#        psi_grad_1 = numpy.zeros((self.N, self.M, self.Q))
#        psi_grad_2 = numpy.zeros((self.M, self.M, self.Q))
#        return [psi_grad_0, psi_grad_1, psi_grad_2]


    def psigrad_mean(self):
        psi_grad = 0
        return [psi_grad]*3
    
    
    def psigrad_variance(self):
        psi_grad = 0
        return [psi_grad]*3
    

    def psigrad_inducing_variables(self):
        psi_grad = 0
        return [psi_grad]*3
        

#    def psi_0(self, theta, mean, variance, inducing_variables):
#        return 0
#
#
#    def psi_0grad_theta(self, theta, mean, variance, inducing_variables):
#        return numpy.zeros_like(theta)
#
#
#    def psi_0grad_mean(self, theta, mean, variance, inducing_variables):
#        return numpy.zeros((mean.size))
#
#
#    def psi_0grad_variance(self, theta, mean, variance, inducing_variables):
#        return numpy.zeros((variance.size))
#
#
#    def psi_0grad_inducing_variables(self, theta, mean, variance, inducing_variables):
#        return numpy.zeros((inducing_variables.size))
#
#
#    def psi_1(self, theta, mean, variance, inducing_variables):
#        return numpy.zeros((mean.shape[0], inducing_variables.shape[0]))
#
#
#    def psi_1grad_theta(self, theta, mean, variance, inducing_variables):
#        return numpy.zeros((mean.shape[0], inducing_variables.shape[0], theta.shape[0]))
#
#
#    def psi_1grad_mean(self, theta, mean, variance, inducing_variables):
#        return numpy.zeros((mean.shape[0], inducing_variables.shape[0], mean.size))
#
#
#    def psi_1grad_variance(self, theta, mean, variance, inducing_variables):
#        return numpy.zeros((mean.shape[0], inducing_variables.shape[0], variance.size))
#
#
#    def psi_1grad_inducing_variables(self, theta, mean, variance, inducing_variables):
#        return numpy.zeros((mean.shape[0], inducing_variables.shape[0], inducing_variables.size))
#
#
#    def psi_2(self, theta, mean, variance, inducing_variables):
#        return numpy.zeros((inducing_variables.shape[0], inducing_variables.shape[0]))
#
#
#    def psi_2grad_theta(self, theta, mean, variance, inducing_variables):
#        return numpy.zeros((inducing_variables.shape[0], inducing_variables.shape[0], theta.shape[0]))
#
#
#    def psi_2grad_mean(self, theta, mean, variance, inducing_variables):
#        return numpy.zeros((inducing_variables.shape[0], inducing_variables.shape[0], mean.size))
#
#
#    def psi_2grad_variance(self, theta, mean, variance, inducing_variables):
#        return numpy.zeros((inducing_variables.shape[0], inducing_variables.shape[0], variance.size))
#
#
#    def psi_2grad_inducing_variables(self, theta, mean, variance, inducing_variables):
#        return numpy.zeros((inducing_variables.shape[0], inducing_variables.shape[0], inducing_variables.size))
pass
    

class NoiseCFReplicates(CovarianceFunction):
    """Covariance function for replicate-wise Gaussian observation noise"""

    def __init__(self, replicate_indices,*args,**kw_args):
        CovarianceFunction.__init__(self,*args,**kw_args)
        self.replicate_indices = replicate_indices
        self.n_hyperparameters = len(SP.unique(replicate_indices))

    def get_hyperparameter_names(self):
        #"""return the names of hyperparameters to make identificatio neasier"""
        names = ["Sigma %i" % (i) for i in range(self.n_hyperparameters)]
        return names

    def K(self,theta,x1,x2=None):
        """
        Get Covariance matrix K with given hyperparameters theta and inputs *args* = X[, X']. Note that this covariance function will only get noise as hyperparameters!

        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction` 
        """
        assert len(theta)==self.n_hyperparameters,'Too many hyperparameters'
        #noise is only present if have a single argument
        if(x2 is None):
            noise = SP.eye(x1.shape[0])
            for i_,n in enumerate(theta):
                noise[self.replicate_indices==i_] *= SP.exp(2*n)
        else:
            noise = 0 
        return noise

    def Kgrad_theta(self,theta,x1,i):
        '''
        The derivative of the covariance matrix with
        respect to i-th hyperparameter.

        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction`
        '''
        #1. calculate kernel
        #no noise
        assert i<self.n_hyperparameters, 'unknown hyperparameters'
        K = SP.eye(x1.shape[0])
        K[self.replicate_indices==i] *= SP.exp(
            2*theta[i])
        K[self.replicate_indices!=i] *= 0
        return 2*K  

    def Kgrad_x(self,theta,x1,x2,d):
        RV = SP.zeros([x1.shape[0],x2.shape[0]])
        return RV

    def Kgrad_xdiag(self,theta,x1,d):
        RV = SP.zeros([x1.shape[0]])
        return RV
