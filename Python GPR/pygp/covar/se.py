"""
Squared Exponential Covariance functions
========================================

This class provides some ready-to-use implemented squared exponential covariance functions (SEs).
These SEs do not model noise, so combine them by a :py:class:`pygp.covar.combinators.SumCF`
or :py:class:`pygp.covar.combinators.ProductCF` with the :py:class:`pygp.covar.noise.NoiseISOCF`, if you want noise to be modelled by this GP.
"""

import numpy
from pygp.covar.covar_base import BayesianStatisticsCF, CovarianceFunction
from pygp.covar import dist

class SqexpCFARD(CovarianceFunction):
    """
    Standart Squared Exponential Covariance function.

    **Parameters:**
    
    - dimension : int
        The dimension of this SE. For instance a 2D SE has
        hyperparameters like::
        
          covar_hyper = [Amplitude,1stD Length-Scale, 2ndD Length-Scale]

    - dimension_indices : [int]
        Optional: The indices of the get_n_dimensions() in the input.
        For instance the get_n_dimensions() of inputs are in 2nd and
        4th dimension dimension_indices would have to be [1,3].

    """   
    def __init__(self, *args, **kwargs):
        super(SqexpCFARD, self).__init__(*args, **kwargs)
        self.n_hyperparameters = self.get_n_dimensions() + 1
        pass

    def get_hyperparameter_names(self):
        """
        return the names of hyperparameters to
        make identification easier
        """
        names = []
        names.append('SECF Amplitude')
        for dim in self.get_dimension_indices():
            names.append(('{0!s:>%is}D Length-Scale' % len(str(self.get_dimension_indices().max()))).format(dim))
        return names
   
    def get_number_of_parameters(self):
        """
        Return the number of hyperparameters this CF holds.
        """
        return self.get_n_dimensions() + 1;

    def K(self, theta, x1, x2=None):
        """
        Get Covariance matrix K with given hyperparameters
        and inputs X=x1 and X\`*`=x2.

        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction`
        """
        #1. get inputs
        x1, x2 = self._filter_input_dimensions(x1, x2)
        #2. exponentiate parameters
        V0 = numpy.exp(2 * theta[0])
        L = numpy.exp(theta[1:1 + self.get_n_dimensions()])
        sqd = dist.sq_dist(x1 / L, x2 / L)
        #sqd = dist.sq_dist( numpy.exp( numpy.log(numpy.abs(x1)) - L) , numpy.exp( numpy.log(numpy.abs(x2)) - L) )
        
        #3. calculate the whole covariance matrix:
        rv = V0 * numpy.exp(-0.5 * sqd)
        return rv

    def Kdiag(self, theta, x1):
        """
        Get diagonal of the (squared) covariance matrix.

        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction`
        """
        #diagonal is independent of data
        x1 = self._filter_x(x1)
        V0 = numpy.exp(2 * theta[0])
        return V0 * numpy.exp(0) * numpy.ones([x1.shape[0]])
    
    def Kgrad_theta(self, theta, x1, i):
        """
        The derivatives of the covariance matrix for
        each hyperparameter, respectively.

        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction`
        """
        x1 = self._filter_x(x1)
        # 2. exponentiate params:
        V0 = numpy.exp(2 * theta[0])
        L = numpy.exp(theta[1:1 + self.get_n_dimensions()])
        # calculate squared distance manually as we need to dissect this below
        x1_ = x1 / L
        
        d = dist.dist(x1_)
        sqd = (d * d)
        sqdd = sqd.sum(-1)

        #3. calcualte withotu derivatives, need this anyway:
        rv0 = V0 * numpy.exp(-0.5 * sqdd)
        if i == 0:
            return 2 * rv0
        else:
            grad = rv0 * sqd[:, :, i - 1]
#            import ipdb;ipdb.set_trace()
            return grad

    
    def Kgrad_x(self, theta, x1, x2, d):
        """
        The partial derivative of the covariance matrix with
        respect to x, given hyperparameters `theta`.

        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction`
        """
        # if we are not meant return zeros:
        if(d not in self.get_dimension_indices()):
            return numpy.zeros([x1.shape[0], x2.shape[0]])
        rv = self.K(theta, x1, x2)
#        #1. get inputs and dimension
        x1, x2 = self._filter_input_dimensions(x1, x2)
        d -= self.get_dimension_indices().min()
#        #2. exponentialte parameters
#        V0 = numpy.exp(2*theta[0])
#        L  = numpy.exp(theta[1:1+self.get_n_dimensions()])[d]
        L2 = numpy.exp(2 * theta[1:1 + self.get_n_dimensions()])
#        # get squared distance in right dimension:
#        sqd = dist.sq_dist(x1[:,d]/L,x2[:,d]/L)
#        #3. calculate the whole covariance matrix:
#        rv = V0*numpy.exp(-0.5*sqd)
        #4. get non-squared distance in right dimesnion:
        nsdist = dist.dist(x1, x2)[:, :, d] / L2[d]
        
        return rv * nsdist
    
    def Kgrad_xdiag(self, theta, x1, d):
        """"""
        #digaonal derivative is zero because d/dx1 (x1-x2)^2 = 0
        #because (x1^d-x1^d) = 0
        RV = numpy.zeros([x1.shape[0]])
        return RV

    def get_ard_dimension_indices(self):
        return numpy.array(self.get_dimension_indices() + 1, dtype='int')

class SqexpCFARDInv(CovarianceFunction):
    """
    Standart Squared Exponential Covariance function.

    **Parameters:**
    
    - dimension : int
        The dimension of this SE. For instance a 2D SE has
        hyperparameters like::
        
          covar_hyper = [Amplitude,1stD Length-Scale, 2ndD Length-Scale]

    - dimension_indices : [int]
        Optional: The indices of the get_n_dimensions() in the input.
        For instance the get_n_dimensions() of inputs are in 2nd and
        4th dimension dimension_indices would have to be [1,3].

    """   
    def __init__(self, n_dimensions= -1, dimension_indices=None):
        super(SqexpCFARDInv, self).__init__(n_dimensions, dimension_indices)
        self.n_hyperparameters = self.get_n_dimensions() + 1
        pass

    def get_hyperparameter_names(self):
        """
        return the names of hyperparameters to
        make identification easier
        """
        names = []
        names.append('SECF Amplitude')
        for dim in self.get_dimension_indices():
            names.append(('SECF ARD {0:%s}' % len(str(self.get_dimension_indices().max()))).format(dim))
        return names
   
    def get_number_of_parameters(self):
        """
        Return the number of hyperparameters this CF holds.
        """
        return self.get_n_dimensions() + 1;

    def K(self, theta, x1, x2=None):
        """
        Get Covariance matrix K with given hyperparameters
        and inputs X=x1 and X\`*`=x2.

        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction`
        """
        #1. get inputs
        x1, x2 = self._filter_input_dimensions(x1, x2)
        #2. exponentialte parameters
        V0 = numpy.exp(-2 * theta[0])
        L = numpy.exp(theta[1:1 + self.get_n_dimensions()])
        Linv = L #1./L
        
        sqd = dist.sq_dist(x1 * Linv, x2 * Linv)
        #sqd = dist.sq_dist( numpy.exp( numpy.log(numpy.abs(x1)) - L) , numpy.exp( numpy.log(numpy.abs(x2)) - L) )
        
        #3. calculate the whole covariance matrix:
        rv = V0 * numpy.exp(-0.5 * sqd)
        return rv

    def Kdiag(self, theta, x1):
        """
        Get diagonal of the (squared) covariance matrix.

        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction`
        """
        #diagonal is independent of data
        x1 = self._filter_x(x1)
        V0 = numpy.exp(-2 * theta[0])
        return V0 * numpy.exp(0) * numpy.ones([x1.shape[0]])
    
    def Kgrad_theta(self, theta, x1, i):
        """
        The derivatives of the covariance matrix for
        each hyperparameter, respectively.

        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction`
        """
        x1 = self._filter_x(x1)
        # 2. exponentiate params:
        V0 = numpy.exp(-2 * theta[0])
        L = numpy.exp(theta[1:1 + self.get_n_dimensions()])
        # calculate squared distance manually as we need to dissect this below
        Linv = L #1./L
        x1_ = x1 * Linv
        d = dist.dist(x1_, x1_)
        sqd = (d * d)
        sqdd = sqd.sum(axis=2)
        #3. calcualte withotu derivatives, need this anyway:
        rv0 = V0 * numpy.exp(-0.5 * sqdd)
        if i == 0:
            return -2 * rv0
        else:
            return rv0 * -sqd[:, :, i - 1]

    
    def Kgrad_x(self, theta, x1, x2, d):
        """
        The partial derivative of the covariance matrix with
        respect to x, given hyperparameters `theta`.

        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction`
        """
        # if we are not meant return zeros:
        if(d not in self.get_dimension_indices() and self.get_n_dimensions() > 0):
            return numpy.zeros([x1.shape[0], x2.shape[0]])
        rv = self.K(theta, x1, x2)
#        #1. get inputs and dimension
        x1, x2 = self._filter_input_dimensions(x1, x2)
        d -= self.get_dimension_indices().min()
#        #2. exponentialte parameters
#        V0 = numpy.exp(2*theta[0])
#        L  = numpy.exp(theta[1:1+self.get_n_dimensions()])[d]
        L2 = numpy.exp(2 * theta[1:1 + self.get_n_dimensions()])
#        # get squared distance in right dimension:
#        sqd = dist.sq_dist(x1[:,d]/L,x2[:,d]/L)
#        #3. calculate the whole covariance matrix:
#        rv = V0*numpy.exp(-0.5*sqd)
        #4. get non-squared distance in right dimesnion:
        nsdist = dist.dist(x1, x2)[:, :, d] * L2[d]
        
        return rv * nsdist
    
    def Kgrad_xdiag(self, theta, x1, d):
        """"""
        #digaonal derivative is zero because d/dx1 (x1-x2)^2 = 0
        #because (x1^d-x1^d) = 0
        RV = numpy.zeros([x1.shape[0]])
        return RV
    
    def get_reparametrized_theta(self, theta):
        theta = theta.copy()
        theta[0] = -theta[0]
        rep = numpy.exp(2 * theta)
        return rep


    def get_de_reparametrized_theta(self, theta):
        rep = .5 * numpy.log(theta)
        rep[0] = -rep[0]
        return rep


    def get_ard_dimension_indices(self):
        return numpy.array(self.dimension_indices + 1, dtype='int')

    

class SqexpCFARDPsiStat(SqexpCFARDInv, BayesianStatisticsCF):
    def update_stats(self, theta, mean, variance, inducing_variables):
        super(SqexpCFARDPsiStat, self).update_stats(theta, mean, variance, inducing_variables)
        self.reparametrize()
        self.sigma = self.theta[0]
        self.alpha = self.theta[1:1 + self.get_n_dimensions()]
        # place for cached computations:
        self.dist_mu_Z = (self.mean[:, None] - self.inducing_variables)
        self.dist_mu_Z_sq = self.dist_mu_Z ** 2
        alpha_variance = (self.alpha * self.variance)
        self.norm_full = alpha_variance + 1
        self.norm_half = alpha_variance + .5
        self.dist_Z_Z = self.inducing_variables[:, None] - self.inducing_variables
        self.dist_Z_Z_sq = self.dist_Z_Z ** 2
        self.Z_hat = (self.inducing_variables[:, None] + self.inducing_variables) / 2.
        self.dist_mu_Z_hat = self.mean[:, None, None] - self.Z_hat
        self.dist_mu_Z_hat_sq = self.dist_mu_Z_hat ** 2
        self.psi1 = self._psi_1()
        self.psi2_n_expanded = self.sigma ** 2 * numpy.exp(self._psi_2_exponent().sum(-1))
        self.psi2 = self.psi2_n_expanded.sum(0)
##        compute psi2 slow
#        psi2_n_expanded = numpy.zeros((self.N,self.M,self.M,self.Q))
#        psi2 = numpy.zeros((self.M,self.M))
#        for n in xrange(self.N):    
#            psi2_inner = numpy.zeros((self.M,self.M,self.Q))
#            for m in xrange(self.M):
#                for mprime in xrange(self.M):
#                    for q in xrange(self.Q):
#                        mu = self.mean[n,q]
#                        S = self.variance[n,q]
#                        alpha = self.alpha[q]
#                        z = self.inducing_variables[m,q]
#                        zprime = self.inducing_variables[mprime,q]
#                        z_hat = .5*(z+zprime)
#
#                        p1 = .5 * ((z - zprime)**2)
#                        p2 = ((mu - z_hat)**2) / (alpha*S + .5)
#                        p1plus2 = alpha * (p1 + p2)
#                        p3 = numpy.log(2*alpha*S + 1)
#                        #psi2_n_expanded[n,m,mprime,q] = -.5 * (p1plus2 + p3)
#                        psi2_inner[m,mprime,q] += -.5 * (p1plus2 + p3)
#                        pass
#                    pass
#                pass
#            psi2_n_expanded[n] = psi2_inner
#            psi2 += self.sigma**2 * numpy.exp(psi2_inner.sum(-1))
#            
#        
        
    def reparametrize(self):
        # constrain positive:
        self.theta = self.get_reparametrized_theta(self.theta)
        self.variance = numpy.exp(2 * self.variance)
    
    def psi(self):
        try:
            assert self.theta.shape[0] == self.get_number_of_parameters()
        except NameError:
            raise BayesianStatisticsCF.CacheMissingError("Cache missing, perhaps a missing self.update_stats!")
        except AssertionError:
            raise CovarianceFunction.HyperparameterError("Wrong number of parameters!")
        psi_0 = self.N * self.sigma # n_hyperparams x 1
        psi_1 = self.psi1 # N x M
        psi_2 = self.psi2 # M x M
        
        return [psi_0, psi_1, psi_2]
    
    def psigrad_theta(self):
        psi_0 = numpy.zeros(self.get_number_of_parameters())
        psi_0[0] = self.N * -2 * self.sigma
        
        #psi_1 = numpy.zeros((self.N,self.M,self.get_number_of_parameters()))
        psi_1 = numpy.tile(self.psi1[:, :, None], self.get_number_of_parameters())
        psi_1[:, :, 0] *= -2#*self.psi1#self._exp_psi_1(self.theta[1:1+self.get_get_n_dimensions()()], self.mean, self.variance, self.inducing_variables)
        psi_1_inner = (self.dist_mu_Z_sq / (self.norm_full ** 2)[:, None])
        psi_1_inner += (self.variance / self.norm_full)[:, None]
        psi_1[:, :, 1:] *= -self.alpha * psi_1_inner # -alpha bc of exp(2*theta)
        # psi_1: N x M x Q x 1
        # calculate psi2grad_theta slow:
#        for q in xrange(self.Q):
#            distances = (numpy.atleast_2d(self.mean[:,q]).T - numpy.atleast_2d(self.inducing_variables[:,q]))**2
#            alpha_q = self.alpha[q]
#            S_q = numpy.atleast_2d(self.variance[:,q]).T
#            normalizing_factor = (alpha_q * S_q) + 1
#            dexp_psi_1 = -.5 * ( (distances / normalizing_factor**2) + (S_q / normalizing_factor))  
#            psi1[:,:,q+1] = -alpha * self.sigma * self._psi_1() * numpy.exp(dexp_psi_1)
#
#        psi_2 = numpy.zeros((self.M, self.M, self.get_number_of_parameters()))
#        psi_2[:,:,0]  = self.psi2
#        psi_2[:,:,0] *= 4
#        test = 0
#        for n in xrange(self.N):
#            psi_2_inner = numpy.zeros((self.M, self.M, self.Q))
#            for m in xrange(self.M):
#                for mprime in xrange(self.M):
#                    for q in xrange(self.Q):
#                        mu = self.mean[n,q]
#                        S = self.variance[n,q]
#                        alpha = self.alpha[q]
#                        z = self.inducing_variables[m,q]
#                        zprime = self.inducing_variables[mprime,q]
#                        z_hat = .5*(z+zprime)
#
#                        p1  = .5*(z-zprime)**2
#                        p2  = .5*(mu - z_hat)**2 / (alpha*S + .5)**2
#                        p3  = S / (alpha*S + .5)
#                        p = (p1 + p2 + p3)
#                        
#                        inner = -.5 * p
#                        psi_2_inner[m,mprime,q] = 2*alpha*inner
#                        pass
#                    pass
#                pass
#            real = self.sigma**2 * numpy.exp(self.psi2_n_expanded[n].sum(-1))[:,:,None]
#            test += real
#            psi_2[:,:,1:] += psi_2_inner * real 
#        
        psi_2 = numpy.zeros((self.M, self.M, self.get_number_of_parameters()))
        psi_2[:, :, 0] = self.psi2 * -4
        psi_2_inner = (.5 * self.dist_mu_Z_hat_sq / (self.norm_half ** 2)[:, None, None])
        psi_2_inner += (.5 * self.dist_Z_Z_sq)[None]
        psi_2_inner += (self.variance / self.norm_half)[:, None, None]
        psi_2_inner *= -.5 * 2 * self.alpha
        psi_2[:, :, 1:] = (psi_2_inner * self.psi2_n_expanded[:, :, :, None]).sum(0)

        return psi_0[:, None], psi_1[:, :, :, None], psi_2[:, :, :, None]
        pass
        
    def psigrad_inducing_variables(self):
        psi_0 = 0#numpy.zeros((self.M * self.Q))
        
        psi_1_inner = self.alpha * self.dist_mu_Z / self.norm_full[:, None]
        psi_1_outer = self.psi1[:, :, None] * psi_1_inner
        psi_1 = numpy.eye(self.M, self.M)[None, :, :, None] * psi_1_outer[:, :, None]
        psi_1 = psi_1#.reshape((self.N, self.M, self.M * self.Q))
        
        psi_2_inner = -.5 * self.alpha * ((self.dist_Z_Z[None]) - (self.dist_mu_Z_hat / self.norm_half[:, None, None]))
        psi_2_inner = numpy.eye(self.M)[None, :, None, :, None] * psi_2_inner[:, :, :, None]        
        psi_2_inner *= self.psi2_n_expanded[:, :, :, None, None]
        
        psi_2_inner = psi_2_inner + psi_2_inner.swapaxes(1, 2)
        
        psi_2 = psi_2_inner.sum(0)#.reshape((self.M,self.M,self.M * self.Q))
        psi_2 = psi_2.swapaxes(0, 1)

#        psi_2_inner = - .5 * self.alpha * ( ( self.dist_Z_Z[None] ) - ( self.dist_mu_Z_hat / self.norm_half[:,None,None] ) )
#        psi_2_inner = numpy.eye(self.M)[None,:,None,:,None] * psi_2_inner[:,:,:,None]        
#        psi_2_inner *= self.psi2_n_expanded[:,:,:,None]
#        psi_2_inner = psi_2_inner + psi_2_inner.swapaxes(1,2)
#        psi_2 = psi_2_inner
#        psi_2 = psi_2_inner.sum(0)#.reshape((self.M,self.M,self.M * self.Q))
#        import ipdb;ipdb.set_trace()
#        psi_2 = psi_2.swapaxes(0,1)
        
        return psi_0, psi_1, psi_2
    
    def psigrad_mean(self):
        psi_0 = 0#numpy.zeros((self.N * self.Q))
        
        psi_1_inner = -self.alpha * self.dist_mu_Z / self.norm_full[:, None]
        psi_1 = self.psi1[:, :, None] * psi_1_inner
        psi_1 = psi_1[:, :, None, :]
        #psi_1_ = numpy.eye(self.N)[:,None,:,None] * psi_1[:,:,None]
        #psi_1.resize((self.N, self.M, self.N * self.Q))
        
        psi_2_inner = -self.alpha * self.dist_mu_Z_hat / self.norm_half[:, None, None]
        psi_2 = (psi_2_inner * self.psi2_n_expanded[:, :, :, None])
        psi_2 = psi_2.swapaxes(0, 2)
        #psi_2 = psi_2.reshape((self.M,self.M,self.N * self.Q))
        #psi_2 = numpy.zeros((self.N, self.M, self.M, self.Q))
        return psi_0, psi_1, psi_2

    def psigrad_variance(self):
        psi_0 = 0#numpy.zeros((self.N * self.Q))
        
        psi_1_inner = -(self.alpha ** 2 * self.dist_mu_Z_sq) / self.norm_full[:, None]
        psi_1_inner += self.alpha
        #psi_1 = self.psi1[:,:,None] * -.5 * (psi_1_inner / self.norm_full[:,None])
        # reparametrization constrain psoitive
        psi_1 = self.psi1[:, :, None] * -(psi_1_inner / self.norm_full[:, None]) * self.variance[:, None]
        psi_1 = psi_1[:, :, None]
#        psi_1 = numpy.eye(self.N)[:,None,:,None] * psi_1[:,:,None]
        #psi_1.resize((self.N, self.M, self.N * self.Q))
        
        psi_2_inner = (self.alpha * self.dist_mu_Z_hat_sq / (self.norm_half)[:, None, None]) - 1
        psi_2 = self.psi2_n_expanded[:, :, :, None] * .5 * self.alpha * (psi_2_inner / self.norm_half[:, None, None])

        psi_2 = psi_2.swapaxes(0, 2) * 2 * self.variance[None, None]
        #psi_2 = psi_2.reshape((self.M,self.M,self.N * self.Q))
        
        return [psi_0, psi_1, psi_2]
        
    def _psi_1(self):
        #        psi1 = numpy.zeros((self.N, self.M))
#        for n in xrange(self.N):
#            for m in xrange(self.M):
#                psi_expo = 0
#                for q in xrange(self.Q):
#                    dist = self.alpha[q] * (self.mean[n,q] - self.inducing_variables[m,q])**2
#                    norm = (self.alpha[q] * self.variance[n,q]) + 1
#                    psi_expo += -.5 * ( (dist / norm) + numpy.log(norm) )
#                psi1[n,m] = psi_expo
#        psi1 = numpy.exp(psi1)        
        summand = -.5 * ((self.alpha * self.dist_mu_Z_sq / self.norm_full[:, None]) + numpy.log(self.norm_full[:, None]))
        return self.sigma * numpy.exp(summand.sum(-1))
#    
    def _psi_2_exponent(self):
#        old = self.psi_2(self.theta, self.mean, self.variance, self.inducing_variables)
        summand = -.5 * (self.alpha * ((.5 * self.dist_Z_Z_sq)[None] + (self.dist_mu_Z_hat_sq / self.norm_half[:, None, None])) + numpy.log(2 * self.norm_half[:, None, None]))
#        new = 
#        import ipdb;ipdb.set_trace()
#        assert numpy.allclose(old,new)
        return summand
#    def psi_0(self, theta, mean, variance, inducing_points):
#        return mean.shape[0] * theta[0]
#
#    def psi_0grad_theta(self, theta, mean, variance, inducing_points, i):
#        """
#        Gradients with respect to each hyperparameter, respectively. 
#        """
#        # no gradients except of A
#        if i == 0:
#            return mean.shape[0]
#        return 0
#    
#    def psi_1(self, theta, mean, variance, inducing_points):
#        alpha = theta[1:1+self.get_get_n_dimensions()()]
#        exp_psi_1 = self._exp_psi_1(alpha, mean, variance, inducing_points)
#        return theta[0] * exp_psi_1
#        
#    def _inner_sum_psi_1(self, alpha, mean, variance, inducing_points):
#        distances = alpha * (mean.T - inducing_points)**2
#        normalizing_factor = (alpha * variance) + 1
#        summand = -.5 * ((distances / normalizing_factor.T) + numpy.log(normalizing_factor.T))
#        #import pdb;pdb.set_trace()
#        return summand
#    
#    def psi_1grad_theta(self, theta, mean, variance, inducing_points, i):
#        if i==0:
#            return self._exp_psi_1(theta[1:1+self.get_get_n_dimensions()()], mean, variance, inducing_points)
#        q = i-1
#        distances = (numpy.atleast_2d(mean[:,q]).T - numpy.atleast_2d(inducing_points[:,q]))**2
#        alpha_q = theta[i]
#        S_q = numpy.atleast_2d(variance[:,q]).T
#        normalizing_factor = (alpha_q * S_q) + 1
#        dexp_psi_1 = -.5 * ( (distances / normalizing_factor**2) + (S_q / normalizing_factor))  
#        return self.psi_1(theta, mean, variance, inducing_points) * dexp_psi_1
#
#    def _exp_psi_1(self, alpha, mean, variance, inducing_points):
#        return numpy.exp(numpy.add.reduce(\
#                                [self._inner_sum_psi_1(alpha[q], 
#                                                       numpy.atleast_2d(mean[:,q]), 
#                                                       numpy.atleast_2d(variance[:,q]), 
#                                                       numpy.atleast_2d(inducing_points[:,q]))\
#                                 for q in xrange(self.get_get_n_dimensions()())], 
#                                axis=0))
#
#    
#    def psi_2(self, theta, mean, variance, inducing_points):
#        alpha = theta[1:1+self.get_get_n_dimensions()()]
#        summ = numpy.add.reduce([
#                              numpy.exp(numpy.add.reduce( [
#                                                        self._inner_sum_wrapper_psi_2(q, n, alpha, mean, variance, inducing_points) 
#                                                        for q in xrange(self.get_get_n_dimensions()())
#                                                        ], 
#                                                      axis = 0))
#                              for n in xrange(mean.shape[0])
#                              ], axis = 0)
##        summ = numpy.zeros((inducing_points.shape[0], inducing_points.shape[0]))
##        for n in xrange(mean.shape[0]):
##            inner_inner = numpy.zeros((inducing_points.shape[0], inducing_points.shape[0]))
##            for q in xrange(self.get_n_dimensions()):
##                inner_inner += self._inner_sum_wrapper_psi_2(q, n, alpha, mean, variance, inducing_points)
##            summ += numpy.exp(inner_inner)
#            
#        return mean.shape[0] * theta[0]**2 * summ
#
#    def _inner_sum_psi_2(self, alpha, mean, variance, inducing_points):
#        inducing_points_distances = ((inducing_points.T - inducing_points)**2 ) / 2.
#        normalizing_factor = (alpha * variance) + .5
#        
#        inducing_points_cross_means = (inducing_points.T + inducing_points) / 2.
#        inducing_points_cross_distances = ((mean - inducing_points_cross_means)**2) / normalizing_factor
#        
#        summand = -.5 * ( alpha * (inducing_points_distances + inducing_points_cross_distances) + numpy.log(2 * normalizing_factor.T))
#        if 0 or numpy.iscomplex(summand).any():
#            import pdb;pdb.set_trace()
#        return summand
#    
#    def _inner_sum_wrapper_psi_2(self, q, n, alpha, mean, variance, inducing_points):
#        return self._inner_sum_psi_2(alpha[q],
#                                     numpy.atleast_2d(mean[n,q]), 
#                                     numpy.atleast_2d(variance[n,q]), 
#                                     numpy.atleast_2d(inducing_points[:,q]))
#        

        
