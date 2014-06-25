'''
Created on 29 May 2012

@author: maxz
'''
from pygp.covar.covar_base import CovarianceFunction, BayesianStatisticsCF
import numpy

class BiasCF(CovarianceFunction):
    '''
    Bias CF with   r$\alpha \cdot \boldsymbol{1}$
    '''
    def __init__(self,n_dimensions=-1,**kwargs):
        super(BiasCF, self).__init__(n_dimensions,**kwargs)
        self.n_hyperparameters = 1

    def K(self, theta, x1, x2=None):
        x1, x2 = self._filter_input_dimensions(x1, x2)
        sigma = numpy.exp(2*theta[0])
        return sigma * numpy.ones((x1.shape[0], x2.shape[0]))


    def Kdiag(self, theta, x1):
        x1 = self._filter_x(x1)
        sigma = numpy.exp(2*theta[0])
        return sigma * numpy.ones(x1.shape[0])


    def Kgrad_theta(self, theta, x1, i):
        assert i==0, "only one bias parameter"
        return 2 * self.K(theta, x1)


    def Kgrad_x(self, theta, x1, x2, d):
        x1, x2 = self._filter_input_dimensions(x1, x2)
        return numpy.zeros((x1.shape[0], x2.shape[0]))


    def Kgrad_xdiag(self, theta, x1, d):
        x1 = self._filter_x(x1)
        return numpy.zeros((x1.shape[0]))


    def get_hyperparameter_names(self):
        return ['BiasCF Amplitude']
    
class BiasCFPsiStat(BiasCF, BayesianStatisticsCF):
    """
    Psi Statistics for BiasCF
    """
    def __init__(self, n_dimensions=-1, **kwargs):
        super(BiasCFPsiStat, self).__init__(n_dimensions, **kwargs)

    def update_stats(self, theta, mean, variance, inducing_variables):
        super(BiasCFPsiStat, self).update_stats(theta, mean, variance, inducing_variables)
        self.reparametrize()
        
        self.psi0 = self.N * self.sigma
        self.psi1 = self.K(self.theta, self.mean, self.inducing_variables)
        self.psi2 = self.N * self.K(theta, self.inducing_variables)**2
        return BayesianStatisticsCF.update_stats(self, theta, mean, variance, inducing_variables)


    def reparametrize(self):
        self.sigma = numpy.exp(2*self.theta[0])


    def psi(self):
        return self.psi0, self.psi1, self.psi2

    def psigrad_theta(self):
        return 2*self.psi0, 2*self.sigma, 4*self.psi2[:,:,None,None]


    def psigrad_mean(self):
        return 0., 0., 0.


    def psigrad_variance(self):
        return 0., 0., 0.
        

    def psigrad_inducing_variables(self):
        return 0., 0., 0.