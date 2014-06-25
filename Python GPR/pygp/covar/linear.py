"""
Classes for linear covariance function
======================================
Linear covariance functions

LinearCFISO
LinearCFARD

"""

#TODO: fix ARD covariance 

import numpy
from pygp.covar.covar_base import BayesianStatisticsCF, CovarianceFunction

class LinearCFISO(CovarianceFunction):
    """
    isotropic linear covariance function with a single hyperparameter
    """

    def __init__(self, *args, **kw_args):
        super(LinearCFISO, self).__init__(*args, **kw_args)
        self.n_hyperparameters = 1

    def K(self, theta, x1, x2=None):
        #get input data:
        x1, x2 = self._filter_input_dimensions(x1, x2)
        # 2. exponentiate params:
        A = numpy.exp(2 * theta[0])
        RV = A * numpy.dot(x1, x2.T)
        return RV

    def Kdiag(self, theta, x1):
        x1 = self._filter_x(x1)
        RV = numpy.dot(x1, x1.T).sum(axis=1)
        RV *= 2
        return RV


    def Kgrad_theta(self, theta, x1, i):
        assert i == 0, 'LinearCF: Kgrad_theta: only one hyperparameter for linear covariance'
        RV = self.K(theta, x1)
        #derivative w.r.t. to amplitude
        RV *= 2
        return RV


    def Kgrad_x(self, theta, x1, x2, d):
        x1, x2 = self._filter_input_dimensions(x1, x2)
        RV = numpy.zeros([x1.shape[0], x2.shape[0]])
        if d not in self.get_dimension_indices():
            return RV
        d -= self.get_dimension_indices().min()
        A = numpy.exp(2 * theta[0])
        RV[:, :] = A * x2[:, d]
        return RV

    
    def Kgrad_xdiag(self, theta, x1, d):
        """derivative w.r.t diagonal of self covariance matrix"""
        x1 = self._filter_x(x1)
        RV = numpy.zeros([x1.shape[0]])
        if d not in self.get_dimension_indices():
            return RV
        d -= self.get_dimension_indices().min()
        A = numpy.exp(2 * theta[0])
        RV[:] = 2 * A * x1[:, d]
        return RV

    def get_hyperparameter_names(self):
        names = []
        names.append('LinearCFISO Amplitude')
        return names
    
class LinearCF(CovarianceFunction):

    def __init__(self, n_dimensions=1, dimension_indices=None):
        super(LinearCF, self).__init__(n_dimensions, dimension_indices)
        self.n_hyperparameters = self.get_n_dimensions()

    def get_reparametrized_theta(self, theta):
        return numpy.exp(2*theta)

    def get_ard_dimension_indices(self):
        return numpy.array(self.dimension_indices,dtype='int')

    def get_de_reparametrized_theta(self, theta):
        return .5*numpy.log(theta)

#        if (len(self.get_dimension_indices()) > 0):
#            self.n_dimensions = len(self.get_dimension_indices())
#            self.n_hyperparameters = self.get_n_dimensions()
#        else:
#            self.n_dimensions = 0
#            self.n_hyperparameters = 0
        
        
    def get_hyperparameter_names(self):
        return ['Linear ARD %i'%i for i in self.get_dimension_indices()]

    def K(self, logtheta, x1, x2=None):
        x1, x2 = self._filter_input_dimensions(x1, x2)
                # 2. exponentiate params:
        # L  = SP.exp(2*logtheta[0:self.get_n_dimensions()])
        # RV = SP.zeros([x1.shape[0],x2.shape[0]])
        # for i in xrange(self.get_n_dimensions()):
        #     iid = self.get_dimension_indices()[i]
        #     RV+=L[i]*SP.dot(x1[:,iid:iid+1],x2[:,iid:iid+1].T)
        if self.get_n_dimensions() > 0:
            M = numpy.diag(numpy.exp(2 * logtheta[:self.get_n_dimensions()]))
            RV = numpy.dot(numpy.dot(x1, M), x2.T)
        else:
            RV = numpy.zeros([x1.shape[0], x2.shape[0]])
            
        return RV

    def Kgrad_theta(self, logtheta, x1, i):
        iid = self.get_dimension_indices()[i]
        Li = numpy.exp(2 * logtheta[i])
        RV = 2 * Li * numpy.dot(x1[:, iid:iid + 1], x1[:, iid:iid + 1].T)
        return RV
    

    def Kgrad_x(self, logtheta, x1, x2, d):
        RV = numpy.zeros([x1.shape[0], x2.shape[0]])
        if d not in self.get_dimension_indices():
            return RV
        #get corresponding amplitude:
        i = numpy.nonzero(self.get_dimension_indices() == d)[0][0]
        A = numpy.exp(2 * logtheta[i])
        RV[:, :] = A * x2[:, d]
        return RV

    
    def Kgrad_xdiag(self, logtheta, x1, d):
        """derivative w.r.t diagonal of self covariance matrix"""
        RV = numpy.zeros([x1.shape[0]])
        if d not in self.get_dimension_indices():
            return RV
        #get corresponding amplitude:
        i = numpy.nonzero(self.get_dimension_indices() == d)[0][0]
        A = numpy.exp(2 * logtheta[i])
        RV = numpy.zeros([x1.shape[0]])
        RV[:] = 2 * A * x1[:, d]
        return RV

class LinearCFARD(CovarianceFunction):
    """identical to LinearCF, however alternative paramerterisation of the ard parameters"""

    def __init__(self, n_dimensions=-1, dimension_indices=None):
        super(LinearCFARD, self).__init__(n_dimensions=n_dimensions, dimension_indices=dimension_indices)        
        self.n_hyperparameters = self.get_n_dimensions()
        
    def get_hyperparameter_names(self):
        names = []
        names.append('Amplitude')
        return names

    def K(self, theta, x1, x2=None):
        if x2 is None:
            x2 = x1
        # 2. exponentiate params:
        #L  = SP.exp(-2*theta[0:self.get_n_dimensions()])
        # L  = 1./theta[0:self.get_n_dimensions()]

        # RV = SP.zeros([x1.shape[0],x2.shape[0]])
        # for i in xrange(self.get_n_dimensions()):
        #     iid = self.get_dimension_indices()[i]
        #     RV+=L[i]*SP.dot(x1[:,iid:iid+1],x2[:,iid:iid+1].T)
        RV = numpy.dot(numpy.dot(x1[:, self.get_dimension_indices()], self._A(theta)), x2[:, self.get_dimension_indices()].T)
        return RV

    def Kgrad_theta(self, theta, x1, i):
        iid = self.get_dimension_indices()[i]
        #Li = SP.exp(-2*theta[i])
        Li = 1. / theta[i]
        RV = -1 * Li ** 2 * numpy.dot(x1[:, iid:iid + 1], x1[:, iid:iid + 1].T)
        return RV
    

    def Kgrad_x(self, theta, x1, x2, d):
        RV = numpy.zeros([x1.shape[0], x2.shape[0]])
        if d not in self.get_dimension_indices():
            return RV
        #get corresponding amplitude:
        i = numpy.nonzero(self.get_dimension_indices() == d)[0][0]
        #A = SP.exp(-2*theta[i])
        RV[:, :] = self._A(theta, i) * x2[:, d]
        return RV

    
    def Kgrad_xdiag(self, theta, x1, d):
        """derivative w.r.t diagonal of self covariance matrix"""
        RV = numpy.zeros([x1.shape[0]])
        if d not in self.get_dimension_indices():
            return RV
        #get corresponding amplitude:
        i = numpy.nonzero(self.get_dimension_indices() == d)[0][0]
        #A = SP.exp(-2*theta[i])
        RV = numpy.zeros([x1.shape[0]])
        RV[:] = 2 * self._A(theta, i) * x1[:, d]
        return RV
    
    def _A(self, theta, i=None):
        if i is None:
            return numpy.diagflat(1. / theta[0:self.get_n_dimensions()])
        return 1. / theta[i]
        # return numpy.diag(theta[0:self.get_n_dimensions()])

class LinearCFPsiStat(LinearCF, BayesianStatisticsCF):
    
    #__slots__ = super(LinearCFPsiStat, self).__slots__.extend(["thetaindv", "thetamean", "thetaindv"
    #                                                          "K_inner", "thetaindvK", "psi1"])

    def __init__(self, *args, **kwargs):
        super(LinearCFPsiStat, self).__init__(*args, **kwargs)

    def update_stats(self, theta, mean, variance, inducing_variables):
        super(LinearCFPsiStat, self).update_stats(theta, mean, variance, inducing_variables)
        
        self.reparametrize() # rescale ARD params, note that the dimension are chosen by super!
        
        self.thetamean = self.theta * self.mean
        self.thetaindv = self.inducing_variables * self.theta
        self.K_inner = numpy.dot(self.mean.T, self.mean) + numpy.diag(numpy.sum(self.variance, 0))
        self.thetaindvK = numpy.dot(self.thetaindv, self.K_inner)
        self.psi1 = numpy.dot(self.mean, self.thetaindv.T)

    def reparametrize(self):
        # soft constrain positive
        self.theta = numpy.exp(2 * self.theta) 
        self.variance = numpy.exp(2 * self.variance)
        
    def psi(self):
        try:
            assert self.theta.shape[0] == self.get_number_of_parameters()
        except AttributeError:
            raise type(self).CacheMissingError("Cache Missing: Perhaps a missing self.update_stats!")
        except AssertionError:
            raise type(self).HyperparameterError("Wrong number of parameters!")
        psi_0 = numpy.sum(self.theta * (self.mean ** 2 + self.variance))
        psi_1 = numpy.dot(self.mean, self.thetaindv.T)
        # psi_2:
        psi_2 = numpy.dot(self.thetaindv, numpy.dot(self.K_inner, self.thetaindv.T))
        return psi_0, psi_1, psi_2


    def psigrad_theta(self):
        """
        returns the gradients w.r.t. _...
        """
        psi_0grad_theta = 2 * numpy.sum(self.theta * (self.mean ** 2 + self.variance), 0)
        psi_1grad_theta = 2 * self.thetamean[:, None, :] * self.inducing_variables
#        psi_1grad_theta = 2 * self.thetamean[:, None, :] * self.inducing_variables
        prod = self.thetaindvK[:, None] * self.thetaindv
        psi_2grad_theta = 2.* (prod.swapaxes(0, 1) + prod)        
        return psi_0grad_theta[:,None], psi_1grad_theta[:,:,:,None], psi_2grad_theta[:,:,:,None]
               
    def psigrad_mean(self):
        """
        returns the gradients w.r.t. _...
        """
        psi_0grad_mean = 2 * self.thetamean                           # < needs flatten
#        psi_1grad_mean = (numpy.eye(self.N)[:, None, :, None] * self.thetaindv[None, :, None])#.reshape(self.N, self.M, self.N * self.Q)
        psi_1grad_mean = self.thetaindv[None,:,None]
#        psi_1grad_mean = (numpy.ones(self.N)[None, :, None] * self.thetaindv[:, None])#.reshape(self.N, self.M, self.N * self.Q)
#        psi_1grad_mean.resize(1, self.M, self.N*self.Q)
#        psi_1grad_mean = (self.thetaindv[None] * numpy.ones((self.N,1,1))).reshape(self.M,self.N*self.Q)[None]
        prod = (self.psi1.T[:, None, :, None] * self.thetaindv[None, :, None])#.reshape(self.M, self.M, self.N * self.Q)
        psi_2grad_mean = prod + prod.swapaxes(0, 1)
        return psi_0grad_mean, psi_1grad_mean, psi_2grad_mean

    def psigrad_variance(self):
        """
        returns the gradients w.r.t. _...
        """
        #psi_0grad_variance = self.theta * (numpy.ones_like(self.variance)) # < without reparametrization
        psi_0grad_variance = 2 * self.theta * self.variance # < needs flatten
        #psi_1grad_variance = numpy.zeros((self.N, self.M, self.N * self.Q))
        psi_1grad_variance = 0#numpy.zeros((1,1,self.N,self.Q))
        #psi_2grad_variance = ((self.thetaindv[:, None] * self.thetaindv)[:, :, None, :] * numpy.ones((1, 1, self.N, 1)))#.reshape(self.M, self.M, self.N * self.Q)
        # without reparametrization:
        #psi_2grad_variance = (self.thetaindv[:, None] * self.thetaindv)[:, :, None, :]#.reshape(self.M, self.M, self.N * self.Q)
        psi_2grad_variance = 2 * self.variance * ((self.thetaindv[:, None] * self.thetaindv)[:, :, None, :])
        return psi_0grad_variance, psi_1grad_variance, psi_2grad_variance
                
    def psigrad_inducing_variables(self):
        psi_1grad_inducing_variables = (numpy.eye(self.M)[None, :, :, None] * self.thetamean[:, None, None, :])#.reshape(self.N, self.M, self.M * self.Q)
        #psi_1grad_inducing_variables = (self.thetamean[:,:,None] * numpy.ones(self.M)).reshape(self.N,self.M*self.Q)[:,None,:]
        prod = (numpy.eye(self.M)[:, None, :, None] * (self.thetaindvK * self.theta)[None, :, None])#.reshape(self.M, self.M, self.M * self.Q)
        psi_2grad_inducing_variables = prod.swapaxes(0, 1) + prod
        return [0,#numpy.zeros(self.M * self.Q), 
                psi_1grad_inducing_variables, 
                psi_2grad_inducing_variables]

    
#    
#    def psi_0(self, theta, mean, variance, inducing_variables):
#        # old = numpy.sum([numpy.trace(numpy.dot(self._A(theta), numpy.dot(mean[n:n+1,:].T, mean[n:n+1,:]) + numpy.diagflat(variance[n,:]))) for n in xrange(mean.shape[0])])
#        # As you would read it:
#        #        new_n = 0
#        #        A = numpy.diag(theta)
#        #        for i in range(mean.shape[0]):
#        #            mu = mean[i:i+1].T
#        #            M = numpy.dot(mu, mu.T)
#        #            S = numpy.diag(variance[i])
#        #            new = numpy.dot(A, M + S)
#        #            new_n += numpy.trace(new)
#        #        return new_n
#        # as you would implement:
#        new_new = numpy.sum(numpy.exp(2 * theta) * (mean ** 2 + variance))
#        return new_new
#
#    def psi_0grad_theta(self, theta, mean, variance, inducing_variables):
#        # diag_sum = 0
#        # As you would read it:
#        #        for i in range(mean.shape[0]):
#        #            mu = mean[i:i+1].T
#        #            M = numpy.dot(mu, mu.T)
#        #            S = numpy.diag(variance[i])
#        #            new = M + S
#        #            diag_sum += numpy.diag(new)
#        #            
#        #        full_sum = numpy.zeros_like(theta)
#        #        
#        #        for i in range(mean.shape[0]):
#        #            mu = mean[i:i+1].T
#        #            M = numpy.dot(mu, mu.T)
#        #            S = numpy.diag(variance[i])
#        #            new = M + S
#        #            for q in range(mean.shape[1]):
#        #                full_sum[q] += new[q,q]
#        # as you could implement:
#        new = 2 * numpy.sum(numpy.exp(2 * theta) * (mean ** 2 + variance), 0)
#        return new
#
#    def psi_0grad_mean(self, theta, mean, variance, inducing_variables):
#        new = 2 * numpy.exp(2 * theta) * mean
#        return new.flatten()
#
#    def psi_0grad_variance(self, theta, mean, variance, inducing_variables):
#        new = numpy.exp(2 * theta) * (numpy.ones_like(variance))
#        return new.flatten()
#
#    def psi_1(self, theta, mean, variance, inducing_variables):
#        A = numpy.diagflat(numpy.exp(2 * theta))
#        new = numpy.dot(mean, numpy.dot(A, inducing_variables.T))
#        #assert numpy.allclose(new, numpy.array([[numpy.dot(numpy.atleast_2d(mean[n,:]), numpy.dot(A, numpy.atleast_2d(inducing_variables[m,:]).T)) for m in xrange(inducing_variables.shape[0])] for n in xrange(mean.shape[0])])[:,:,0,0])
#        return new 
#            
#    def psi_1grad_theta(self, theta, mean, variance, inducing_variables):
#        new = mean[:, None, :] * inducing_variables * 2 * numpy.exp(2 * theta)
##        old = numpy.zeros((mean.shape[0],inducing_variables.shape[0],len(theta)))
##        for i in xrange(self.get_n_dimensions()):
##            iid = self.get_dimension_indices()[i]
##            Li = numpy.exp(2*theta[i])
##            old[:,:,i] = 2*Li*numpy.dot(mean[:,iid:iid+1],inducing_variables[:,iid:iid+1].T)
##        assert numpy.allclose(new, old)
#        return new
#    
#    def psi_1grad_mean(self, theta, mean, variance, inducing_variables):
#        grads = (numpy.ones_like(mean)[:, None] * inducing_variables * numpy.exp(2 * theta))
#        filta = numpy.eye(mean.shape[0])
#        grad = (filta[:, :, None, None] * grads).swapaxes(1, 2)
#        new = grad.reshape(mean.shape[0], inducing_variables.shape[0], -1)
#        return new
#
#    def psi_1grad_variance(self, theta, mean, variance, inducing_variables):
#        return numpy.zeros((mean.shape[0], inducing_variables.shape[0], variance.size))
#    
#    def psi_1grad_inducing_variables(self, theta, mean, variance, inducing_variables):
##        A = numpy.diagflat(numpy.exp(2*theta))
##        old = numpy.dot(mean, A)#        import pdb;pdb.set_trace()
##        #import pdb;pdb.set_trace()
#        #M = inducing_variables.shape[0]
#        #N = mean.shape[0] 
#        #Q = mean.shape[1]
#        #A = numpy.ones_like(inducing_variables) * numpy.exp(2*theta)
#        #new = numpy.dot(mean, A.T)[:,:,None] * numpy.ones(mean.shape[1])
#        #new = mean[:,None] * A
#        prod = numpy.dot(mean, numpy.diagflat(numpy.exp(2 * theta)))
#        new = prod[:, None, None, :] * numpy.eye(inducing_variables.shape[0])[None, :, :, None]
##        ret = numpy.zeros((N,M,M,Q))
##        for m in xrange(M):
##            for q in xrange(Q):
##                ret[:,m,:,q] = numpy.dot(prod, self._J(M,Q,m,q).T)
#        return new.reshape(mean.shape[0], inducing_variables.shape[0], -1)
#    
#    def psi_2(self, theta, mean, variance, inducing_variables):
##        ZA = numpy.dot(inducing_variables, numpy.diagflat(numpy.exp(2*theta)))
##        inner = numpy.dot(mean.T, mean) + numpy.diag(variance.sum(0))
##        new = numpy.dot(ZA, numpy.dot(inner, ZA.T))
##        A = numpy.diagflat(numpy.exp(2*theta))
##        old = numpy.zeros((inducing_variables.shape[0], inducing_variables.shape[0]))
##        for n in xrange(mean.shape[0]):
##            for m in xrange(inducing_variables.shape[0]):
##                for mprime in xrange(inducing_variables.shape[0]):
##                    mu = mean[n:n+1,:].T
##                    M = numpy.dot(mu,mu.T)
##                    S = numpy.diagflat(variance[n:n+1,:])
##                    MS = M+S
##                    AMSA = numpy.dot(A,numpy.dot(MS,A))
##                    zAMSA = numpy.dot(inducing_variables[m:m+1,:],AMSA)
##                    zAMSAz = numpy.dot(zAMSA, inducing_variables[mprime:mprime+1,:].T)
##                    old[m,mprime] += zAMSAz
##        
#        outer = inducing_variables * numpy.exp(2 * theta)
#        inner = numpy.dot(mean.T, mean) + numpy.diag(numpy.sum(variance, 0))
#        new = numpy.dot(outer, numpy.dot(inner, outer.T))
##        assert numpy.allclose(old, new)
#        
#        return new
#            
#    def psi_2grad_theta(self, theta, mean, variance, inducing_variables):
##        A = numpy.diagflat(numpy.exp(2*theta))
##        old = numpy.zeros((inducing_variables.shape[0], inducing_variables.shape[0], len(theta)))
##        for n in xrange(mean.shape[0]):
##            for m in xrange(inducing_variables.shape[0]):
##                for mprime in xrange(inducing_variables.shape[0]):
##                    mu = mean[n:n+1,:].T
##                    M = numpy.dot(mu,mu.T)
##                    S = numpy.diagflat(variance[n:n+1,:])
##                    MS = M+S
##                    Zm = inducing_variables[m:m+1,:]
##                    Zmprime = inducing_variables[mprime:mprime+1,:]
###                    AMSA = numpy.dot(A,numpy.dot(MS,A))
###                    zAMSA = numpy.dot(inducing_variables[m:m+1,:],AMSA)
###                    zAMSAz = numpy.dot(zAMSA, inducing_variables[mprime:mprime+1,:].T)
###                    old[m,mprime] += zAMSAz
##                    left = numpy.dot(Zm,numpy.dot(MS,numpy.dot(A,Zmprime.T)))
##                    right = numpy.dot(Zm,numpy.dot(A, numpy.dot(MS,Zmprime.T))).T
##                    old[m,mprime,:] = left+right
#        outer = inducing_variables * numpy.exp(2 * theta)
#        inner = numpy.dot(mean.T, mean) + numpy.diagflat(numpy.sum(variance, 0))
#        outin = numpy.dot(outer, inner)
#        grad = outin[:, None] * outer
#        test = grad.swapaxes(0, 1) + grad
#        #import pdb;pdb.set_trace()
#        return 2.*test
#    
#    def psi_2grad_mean(self, theta, mean, variance, inducing_variables):
##        A = numpy.diagflat(numpy.exp(2*theta))
##        ZA = numpy.dot(inducing_variables, A)
##        AZ = numpy.dot(A, inducing_variables.T)
##        ZAAZ = numpy.dot(ZA,AZ)
##        AZZA = numpy.dot(AZ,ZA)
##        UUT = numpy.dot(mean, mean.T)
##        grad = numpy.dot(mean, AZZA)
##        M = inducing_variables.shape[0]
##        Q = inducing_variables.shape[1]
##        N = mean.shape[0]
##        grad = numpy.zeros((N,M,Q))
##        A = numpy.exp(2*theta)
##        for q in xrange(Q):
##            test = numpy.dot(psi1, numpy.dot(numpy.ones((M,1)), inducing_variables[:,q:q+1].T))
##            grad[:,:,q] = A[q] * test
###        
##        new = numpy.dot(psi1, inducing_variables * A)
#        psi1 = self.psi_1(theta, mean, variance, inducing_variables)
#        prod = numpy.dot(psi1[:, :, None], (inducing_variables * numpy.exp(2 * theta))[:, None])
#        grad = prod + prod.swapaxes(1, 2)
#        return numpy.rollaxis(grad, 0, 3).reshape(inducing_variables.shape[0], inducing_variables.shape[0], -1)
#    
#    def psi_2grad_variance(self, theta, mean, variance, inducing_variables):
#        A = numpy.diagflat(numpy.exp(2 * theta))
#        Z = inducing_variables
#        ZAA = numpy.dot(numpy.dot(Z, A), A)
#        grad = (Z[:, None] * ZAA)[:, :, None, :] * numpy.ones((1, 1, variance.shape[0], 1))
#        return grad.reshape(inducing_variables.shape[0], inducing_variables.shape[0], -1)
#        
#    def psi_2grad_inducing_variables(self, theta, mean, variance, inducing_variables):
#        M = inducing_variables.shape[0]
##        outer = inducing_variables * numpy.exp(2*theta)
##        outer_ = numpy.ones_like(inducing_variables) * numpy.exp(2*theta)
##        inner = numpy.dot(mean.T,mean) + numpy.diagflat(numpy.sum(variance,0))
##        outin = numpy.dot(outer, inner)
##        new = numpy.dot(outin, outer_.T)[:,:,None] * numpy.eye(M,M)
##        grad = numpy.rollaxis(new,0) + numpy.rollaxis(new,1)
#        outer = inducing_variables * numpy.exp(2 * theta)
#        outer_ = numpy.ones_like(inducing_variables) * numpy.exp(2 * theta)
#        inner = numpy.dot(mean.T, mean) + numpy.diagflat(numpy.sum(variance, 0))
#        outin = numpy.dot(outer, inner)
##        new = numpy.dot(outin, numpy.exp(2*theta))
#        #new = outin[:,None] * outer_
#        K = (outin * outer_)
#        new = numpy.eye(M, M)[:, :, None, None] * K
#        grad = new.swapaxes(1, 2) + new.swapaxes(0, 2)
##        grad = numpy.rollaxis(grad,3).T.swapaxes
##        grad = numpy.rollaxis(new,0) + numpy.rollaxis(new,1)
##        A = numpy.diagflat(numpy.exp(2*theta))
##        K = lambda n: numpy.dot(mean[n:n+1,:], mean[n:n+1,:].T) + numpy.diagflat(variance[n:n+1,:])
##        M = inducing_variables.shape[0]
##        old = numpy.zeros((M,M,M))
##        for m in xrange(M):
##            for mprime in xrange(M):
##                old[m, mprime] = numpy.add.reduce([numpy.dot(A,numpy.dot(K(n),numpy.dot(A,inducing_variables[mprime:mprime+1,:].T))) for n in xrange(mean.shape[0])]).sum()
#        return grad.reshape(inducing_variables.shape[0], inducing_variables.shape[0], -1)
#    
#    def _J(self, m, n, i, j):
#        z = numpy.zeros((m, n))
#        z[i, j] = 1.
#        return z
