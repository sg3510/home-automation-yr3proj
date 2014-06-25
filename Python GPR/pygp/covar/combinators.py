"""
Covariance Function Combinators
-------------------------------

Each combinator is a covariance function (CF) itself. It combines one or several covariance function(s) into another. For instance, :py:class:`pygp.covar.combinators.SumCF` combines all given CFs into one sum; use this class to add noise.

"""

import scipy as sp
import numpy
import itertools
import operator

from pygp.covar.dist import dist
from pygp.covar.covar_base import BayesianStatisticsCF, CovarianceFunction
import __future__

class SumCF(CovarianceFunction):
    """
    Sum Covariance function. This function adds
    up the given CFs and returns the resulting sum.

    *covars* : [:py:class:`pygp.covar.CovarianceFunction`]
    
        Covariance functions to sum up.
    """

#    __slots__ = ["n_params_list","covars","covars_theta_I"]

    def __init__(self, covars, names=[], *args, **kw_args):
        #1. check that all covars are covariance functions
        #2. get number of params
        super(SumCF, self).__init__(*args, **kw_args)
        self.n_params_list = []
        self.covars = []
        self.covars_theta_I = []
        self.covars_covar_I = []
        
        self.covars = covars
        if names and len(names) == len(self.covars):
            self.names = names
        elif names:
            self.names = []
            print "names provided, but shapes not matching (names:{}) (covars:{})".format(len(names), len(covars))
        else:
            self.names = []
        
        i = 0
        
        for nc in xrange(len(covars)):
            covar = covars[nc]
            assert isinstance(covar, CovarianceFunction), 'SumCF: SumCF is constructed from a list of covaraince functions'
            Nparam = covar.get_number_of_parameters()
            self.n_params_list.append(Nparam)
            self.covars_theta_I.append(sp.arange(i, i + covar.get_number_of_parameters()))
            self.covars_covar_I.extend(sp.repeat(nc, Nparam))
            i += covar.get_number_of_parameters()
        
        self.n_params_list = sp.array(self.n_params_list)
        self.n_hyperparameters = self.n_params_list.sum()

    def get_dimension_indices(self):
        """
        returns list of dimension indices for covariance functions
        """
        return numpy.array(reduce(numpy.append,
                                  map(lambda x: x.get_dimension_indices(),
                                      self.covars)),
                           dtype='int')

    def get_n_dimensions(self):
        return reduce(operator.add, map(lambda x:x.get_n_dimensions(), self.covars))

    def get_hyperparameter_names(self):
        """return the names of hyperparameters to make identification easier"""
        return reduce(numpy.append, map(lambda x: x.get_hyperparameter_names(), self.covars))

    def get_theta_by_names(self, theta, names):
        theta_n = []        
        for nc in xrange(len(self.covars)):
            _theta = theta[self.covars_theta_I[nc]]
            if self.names[nc] in names:
                theta_n.append(_theta)
        return reduce(numpy.append, theta_n)

    def get_reparametrized_theta_by_names(self, theta, names):
        return self.get_theta_by_names(self.get_reparametrized_theta(theta), names)
        
    def K(self, theta, x1, x2=None, names=[]):
        """
        Get Covariance matrix K with given hyperparameters
        theta and inputs x1 and x2. The result
        will be the sum covariance of all covariance
        functions combined in this sum covariance.

        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction` 
        """
        #1. check theta has correct length
        assert theta.shape[0] == self.get_number_of_parameters(), 'K: theta has wrong shape'
        #2. create sum of covarainces..
        for nc in xrange(len(self.covars)):
            covar = self.covars[nc]
            _theta = theta[self.covars_theta_I[nc]]
            if not names or not self.names or self.names[nc] in names:
                K_ = covar.K(_theta, x1, x2)
                try:
                    K += K_ #@UndefinedVariable
                except:
                    K = K_
        return K

    def Kgrad_theta(self, theta, x1, i):
        '''
        The partial derivative of the covariance matrix with
        respect to i-th hyperparameter.

        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction`
        '''
        #1. check theta has correct length
        assert theta.shape[0] == self.n_hyperparameters, 'K: theta has wrong shape'
        nc = self.covars_covar_I[i]
        covar = self.covars[nc]
        d = self.covars_theta_I[nc].min()
        j = i - d
        return covar.Kgrad_theta(theta[self.covars_theta_I[nc]], x1, j)
        

    #derivative with respect to inputs
    def Kgrad_x(self, theta, x1, x2, d):
        assert theta.shape[0] == self.n_hyperparameters, 'K: theta has wrong shape'
        RV = sp.zeros([x1.shape[0], x1.shape[0]])
        for nc in xrange(len(self.covars)):
            covar = self.covars[nc]
            _theta = theta[self.covars_theta_I[nc]]
            RV += covar.Kgrad_x(_theta, x1, x2, d)
        return RV
    
    #derivative with respect to inputs
    def Kgrad_xdiag(self, theta, x1, d):
        assert theta.shape[0] == self.n_hyperparameters, 'K: theta has wrong shape'
        RV = sp.zeros([x1.shape[0]])
        for nc in xrange(len(self.covars)):
            covar = self.covars[nc]
            _theta = theta[self.covars_theta_I[nc]]
            RV += covar.Kgrad_xdiag(_theta, x1, d)
        return RV

    def get_reparametrized_theta(self, theta):
        return numpy.concatenate([covar.get_reparametrized_theta(theta[nc]) for covar, nc in zip(self.covars, self.covars_theta_I)])

    def get_de_reparametrized_theta(self, theta):
        return numpy.concatenate([covar.get_de_reparametrized_theta(theta[nc]) for covar, nc in zip(self.covars, self.covars_theta_I)])
    
    def get_ard_dimension_indices(self):
        return numpy.int_(reduce(numpy.append, [covar.get_ard_dimension_indices() for covar in self.covars]))


class SumCFPsiStat(SumCF, BayesianStatisticsCF):

    def __init__(self, *args, **kwargs):
        super(SumCFPsiStat, self).__init__(*args, **kwargs)

    def update_stats(self, theta, mean, variance, inducing_variables):
        super(SumCFPsiStat, self).update_stats(theta, mean, variance, inducing_variables)
        self.psi_list = []
        for covar, nc in zip(self.covars, self.covars_theta_I):
            covar.update_stats(theta[nc], mean, variance, inducing_variables)
            
            self.psi_list.append(covar.psi())
    
    def psi(self):
        psis = reduce(lambda x, y: [numpy.add(a, b) for a, b in zip(x, y)], self.psi_list)
        # cross terms occuring for psi2
        crossterms = 0
        for a, b in itertools.combinations([p[1] for p in self.psi_list], 2):
            prod = numpy.multiply(a, b).sum(0)
            crossterms += prod[None] + prod[:, None]
        psis[2] += crossterms
        return psis
    
#        psi0 = numpy.zeros(1)
#        psi1 = numpy.zeros((self.N, self.M))
#        psi2 = numpy.zeros((self.M, self.M))
#        def append_at(covar, index):
#            grads = covar.psi()
#            psi0[index] = grads[0]
#            psi1[:, :, index] = grads[1]
#            psi2[:, :, index] = grads[2]
#        [append_at(covar, index) for covar, index in zip(self.covars, self.covars_theta_I)]
#        
#        # psi2 cross terms
#        for grad_index, psi1_index in itertools.permutations(range(len(self.psi_list)), 2):
#            theta_index = self.covars_theta_I[grad_index]
#            psi2_cross = (psi1[:, :, theta_index] * numpy.expand_dims(numpy.expand_dims(self.psi_list[psi1_index][1], -1), -1)).sum(0)
#            psi2[:, :, theta_index] += (psi2_cross[None] + psi2_cross[:, None])
        
        
    def psigrad_theta(self):
        #logging.debug("combinators.SumCF.psigrad_inducing_varaibles: cross terms for psi_2 not implemented yet!")
        psi0 = numpy.zeros((self.get_number_of_parameters(), 1))
        psi1 = numpy.zeros((self.N, self.M, self.get_number_of_parameters(), 1))
        psi2 = numpy.zeros((self.M, self.M, self.get_number_of_parameters(), 1))
        def append_at(covar, index):
            grads = covar.psigrad_theta()
            psi0[index] = grads[0]
            psi1[:, :, index] = grads[1]
            psi2[:, :, index] = grads[2]
        [append_at(covar, index) for covar, index in zip(self.covars, self.covars_theta_I)]
        
        # psi2 cross terms
        for grad_index, psi1_index in itertools.permutations(range(len(self.psi_list)), 2):
            theta_index = self.covars_theta_I[grad_index]
            psi2_cross = (psi1[:, :, theta_index] * numpy.expand_dims(numpy.expand_dims(self.psi_list[psi1_index][1], -1), -1)).sum(0)
            psi2[:, :, theta_index] += (psi2_cross[None] + psi2_cross[:, None])
            
        return psi0, psi1, psi2
            
    def psigrad_mean(self):
        psi_grads = [covar.psigrad_mean() for covar in self.covars]
        psi0 = numpy.zeros((self.N, self.get_n_dimensions()))
        psi1 = numpy.zeros((self.N, self.M, 1, self.get_n_dimensions()))
        psi2 = numpy.zeros((self.M, self.M, self.N, self.get_n_dimensions()))
        for covar, psigrad in zip(self.covars, psi_grads):
            dims = covar.get_dimension_indices()
            psi0[:, dims] = psi0[:, dims] + psigrad[0]
            psi1[:, :, :, dims] = psi1[:, :, :, dims] + psigrad[1]
            psi2[:, :, :, dims] = psi2[:, :, :, dims] + psigrad[2]
        
        psis = [psi0, psi1, psi2]#reduce(lambda x,y: [numpy.add(a,b) for a,b in zip(x,y)], psi_grads)

        # psi2 cross terms
        for grad_index, psi1_index in itertools.permutations(range(len(self.psi_list)), 2):
            psi1 = numpy.expand_dims(numpy.expand_dims(self.psi_list[psi1_index][1], -1), -1)
            #psi2_cross = (numpy.sum(psi1,0) * numpy.sum(psi_grads[grad_index][1],0)) 
            dims = self.covars[grad_index].get_dimension_indices() # of gradient
            if dims.size > 0:
                psi2_cross = (psi1 * psi_grads[grad_index][1])
                if len(psi2_cross.shape) == 4 and psi2_cross.shape[2] == 1:
                    # NOTE: Here we got rid of one N dimension, so we have to take account for that -.-
                    psi2_cross = psi2_cross.swapaxes(0, 2)
                psi2_cross = psi2_cross.sum(0)
    #            if grad_index == 0 or psi1_index == 0:
    #                import ipdb;ipdb.set_trace()
                psis[2][:, :, :, dims] = psis[2][:, :, :, dims] + (numpy.expand_dims(psi2_cross, 0) + numpy.expand_dims(psi2_cross, 1)) 
        
        return psis 
    
    def psigrad_variance(self):
        psi_grads = [covar.psigrad_variance() for covar in self.covars]
        psi0 = numpy.zeros((self.N, self.get_n_dimensions()))
        psi1 = numpy.zeros((self.N, self.M, 1, self.get_n_dimensions()))
        psi2 = numpy.zeros((self.M, self.M, self.N, self.get_n_dimensions()))
        for covar, psigrad in zip(self.covars, psi_grads):
            dims = covar.get_dimension_indices()
            psi0[:, dims] = psi0[:, dims] + psigrad[0]
            psi1[:, :, :, dims] = psi1[:, :, :, dims] + psigrad[1]
            psi2[:, :, :, dims] = psi2[:, :, :, dims] + psigrad[2]
        
        psis = [psi0, psi1, psi2]#reduce(lambda x,y: [numpy.add(a,b) for a,b in zip(x,y)], psi_grads)

        # psi2 cross terms
        for grad_index, psi1_index in itertools.permutations(range(len(self.psi_list)), 2):
            psi1 = numpy.expand_dims(numpy.expand_dims(self.psi_list[psi1_index][1], -1), -1)
            #psi2_cross = (numpy.sum(psi1,0) * numpy.sum(psi_grads[grad_index][1],0)) 
            dims = self.covars[grad_index].get_dimension_indices() # of gradient
            if dims.size > 0:
                psi2_cross = (psi1 * psi_grads[grad_index][1])
                if len(psi2_cross.shape) == 4 and psi2_cross.shape[2] == 1:
                    # NOTE: Here we got rid of one N dimension, so we have to take account for that -.-
                    psi2_cross = psi2_cross.swapaxes(0, 2)
                psi2_cross = psi2_cross.sum(0)
    #            if grad_index == 0 or psi1_index == 0:
    #                import ipdb;ipdb.set_trace()
                psis[2][:, :, :, dims] = psis[2][:, :, :, dims] + (numpy.expand_dims(psi2_cross, 0) + numpy.expand_dims(psi2_cross, 1)) 
        
        return psis 
#        psis = reduce(lambda x,y: [numpy.add(a,b) for a,b in zip(x,y)], psi_grads)
#        
#        # psi2 cross terms
#        for grad_index, psi1_index in itertools.permutations(range(len(self.psi_list)),2):
#            psi1 = numpy.expand_dims(numpy.expand_dims(self.psi_list[psi1_index][1],-1),-1)
#            psi2_cross = (psi1 * psi_grads[grad_index][1])
#            if len(psi2_cross.shape) == 4 and psi2_cross.shape[2] == 1:
#                # NOTE: Here we got rid of one N dimension, so we have to take account for that -.-
#                psi2_cross = psi2_cross.swapaxes(0,2)
#            psi2_cross = psi2_cross.sum(0)
#            psis[2][:,:,:,self.covars[grad_index].get_dimension_indices()] = psis[2][:,:,:,self.covars[grad_index].get_dimension_indices()] + (numpy.expand_dims(psi2_cross,0) + numpy.expand_dims(psi2_cross,1)) 
#        
#        return psis

    def psigrad_inducing_variables(self):
        psi_grads = [covar.psigrad_inducing_variables() for covar in self.covars]
        
        psi0 = numpy.zeros((self.M, self.get_n_dimensions()))
        psi1 = numpy.zeros((self.N, self.M, self.M, self.get_n_dimensions()))
        psi2 = numpy.zeros((self.M, self.M, self.M, self.get_n_dimensions()))
        for covar, psigrad in zip(self.covars, psi_grads):
            dims = covar.get_dimension_indices()
            psi0[:, dims] = psi0[:, dims] + psigrad[0]
            psi1[:, :, :, dims] = psi1[:, :, :, dims] + psigrad[1]
            psi2[:, :, :, dims] = psi2[:, :, :, dims] + psigrad[2]
        
        psis = [psi0, psi1, psi2]#reduce(lambda x,y: [numpy.add(a,b) for a,b in zip(x,y)], psi_grads)

        # psi2 cross terms
        for grad_index, psi1_index in itertools.permutations(range(len(self.psi_list)), 2):
            psi1 = numpy.expand_dims(numpy.expand_dims(self.psi_list[psi1_index][1], -1), -1)
            dims = self.covars[grad_index].get_dimension_indices() # of gradient
            #psi2_cross = (numpy.sum(psi1,0) * numpy.sum(psi_grads[grad_index][1],0)) 
            if dims.size > 0: # We only need to add something, when the covariance is dependent on dimensions
                psi2_cross = (psi1 * psi_grads[grad_index][1])
                if len(psi2_cross.shape) == 4 and psi2_cross.shape[2] == 1:
                    # NOTE: Here we got rid of one N dimension, so we have to take account for that -.-
                    psi2_cross = psi2_cross.swapaxes(0, 2)
                psi2_cross = psi2_cross.sum(0)
    #            if grad_index == 0 or psi1_index == 0:
    #                import ipdb;ipdb.set_trace()
                psis[2][:, :, :, dims] = psis[2][:, :, :, dims] + (numpy.expand_dims(psi2_cross, 0) + numpy.expand_dims(psi2_cross, 1)) 
        
        return psis 

        
#        psis = reduce(lambda x,y: [numpy.add(a,b) for a,b in zip(x,y)], psi_grads)
#        
#        # psi2 cross terms
#        for grad_index, psi1_index in itertools.permutations(range(len(self.psi_list)),2):
#            psi1 = numpy.expand_dims(numpy.expand_dims(self.psi_list[psi1_index][1],-1),-1)
#            psi2_cross = (psi1 * psi_grads[grad_index][1]).sum(0)
#            psis[2] = psis[2] + (numpy.expand_dims(psi2_cross,0) + numpy.expand_dims(psi2_cross,1))
#        
#        return psis
         

#    def psigrad(self, theta, mean, variance, inducing_variables):
#        def sum_grads(x, y):
#            theta1 = x.pop()
#            theta2 = y.pop()
##            theta_grad = x.pop().concatenate(y.pop())
#            other_grads = [numpy.add(a,b) for a,b in zip(x,y)]
#            grads = [theta_grad].extend(other_grads)
#            stop
#            return theta_grad, other_grads
#        psi_list = [covar.psigrad(theta[nc], mean, variance, inducing_variables) for covar, nc in zip(self.covars, self.covars_theta_I)]
#        return reduce(lambda x,y: [sum_grads(x,y)], psi_list)
#    def psi_0(self, theta, mean, variance, inducing_variables):
#        assert theta.shape[0] == self.n_hyperparameters, 'K: theta has wrong shape'
#        psi = 0
#        for i, nc in enumerate(self.covars_theta_I):
#            psi += self.covars[i].psi_0(theta[nc], mean, variance, inducing_variables)
#        return psi
#
#
#    def psi_0grad_theta(self, theta, mean, variance, inducing_variables):
#        grad = numpy.zeros_like(theta)
#        for i, nc in enumerate(self.covars_theta_I):
#            grad[nc] = self.covars[i].psi_0grad_theta(theta[nc], mean, variance, inducing_variables)
#        return grad
#
#
#    def psi_0grad_mean(self, theta, mean, variance, inducing_variables):
##        grad_ = numpy.zeros((mean.size))
#        grad = numpy.add.reduce([covar.psi_0grad_mean(the, mean, variance, inducing_variables) for covar, the in zip(self.covars, self.covars_theta_I)])
##        for i, nc in enumerate(self.covars_theta_I):
##            grad_ += self.covars[i].psi_0grad_mean(theta[nc], mean, variance, inducing_variables)
##        import pdb;pdb.set_trace()
#        return grad
#
#
#    def psi_0grad_variance(self, theta, mean, variance, inducing_variables):
#        grad = numpy.zeros((variance.size))
#        for i, nc in enumerate(self.covars_theta_I):
#            grad += self.covars[i].psi_0grad_variance(theta[nc], mean, variance, inducing_variables)
#        return grad
#
#
#    def psi_1(self, theta, mean, variance, inducing_variables):
#        psi = numpy.zeros((mean.shape[0], inducing_variables.shape[0]))
#        for i, nc in enumerate(self.covars_theta_I):
#            psi += self.covars[i].psi_1(theta[nc], mean, variance, inducing_variables)
#        return psi
#
#
#    def psi_1grad_theta(self, theta, mean, variance, inducing_variables):
#        grad = numpy.zeros((mean.shape[0], inducing_variables.shape[0], theta.shape[0]))
#        for i, nc in enumerate(self.covars_theta_I):
#            grad[:,:,nc] = self.covars[i].psi_1grad_theta(theta[nc], mean, variance, inducing_variables)
#        return grad
#
#
#    def psi_1grad_mean(self, theta, mean, variance, inducing_variables):
#        grad = numpy.zeros((mean.shape[0], inducing_variables.shape[0], mean.size))
#        for i, nc in enumerate(self.covars_theta_I):
#            grad += self.covars[i].psi_1grad_mean(theta[nc], mean, variance, inducing_variables)
#        return grad
#                           
#
#    def psi_1grad_variance(self, theta, mean, variance, inducing_variables):
#        grad = numpy.zeros((mean.shape[0], inducing_variables.shape[0], variance.size))
#        for i, nc in enumerate(self.covars_theta_I):
#            grad += self.covars[i].psi_1grad_variance(theta[nc], mean, variance, inducing_variables)
#        return grad
#
#
#    def psi_1grad_inducing_variables(self, theta, mean, variance, inducing_variables):
#        grad = numpy.zeros((mean.shape[0], inducing_variables.shape[0], inducing_variables.size))
#        for i, nc in enumerate(self.covars_theta_I):
#            grad += self.covars[i].psi_1grad_inducing_variables(theta[nc], mean, variance, inducing_variables)
#        return grad
#
#
#    def psi_2(self, theta, mean, variance, inducing_variables):
#        psi = numpy.zeros((inducing_variables.shape[0], inducing_variables.shape[0]))
#        for i, nc in enumerate(self.covars_theta_I):
#            psi += self.covars[i].psi_2(theta[nc], mean, variance, inducing_variables)
#        return psi
#
#
#    def psi_2grad_theta(self, theta, mean, variance, inducing_variables):
#        grad = numpy.zeros((inducing_variables.shape[0], inducing_variables.shape[0], theta.shape[0]))
#        for i, nc in enumerate(self.covars_theta_I):
#            grad[:,:,nc] = self.covars[i].psi_2grad_theta(theta[nc], mean, variance, inducing_variables)
#        return grad
#
#
#    def psi_2grad_mean(self, theta, mean, variance, inducing_variables):
#        grad = numpy.zeros((inducing_variables.shape[0], inducing_variables.shape[0], mean.size))
#        for i, nc in enumerate(self.covars_theta_I):
#            grad += self.covars[i].psi_2grad_mean(theta[nc], mean, variance, inducing_variables)
#        return grad
#
#
#    def psi_2grad_variance(self, theta, mean, variance, inducing_variables):
#        grad = numpy.zeros((inducing_variables.shape[0], inducing_variables.shape[0], variance.size))
#        for i, nc in enumerate(self.covars_theta_I):
#            grad += self.covars[i].psi_2grad_variance(theta[nc], mean, variance, inducing_variables)
#        return grad
#
#    def psi_2grad_inducing_variables(self, theta, mean, variance, inducing_variables):
#        grad = numpy.zeros((inducing_variables.shape[0], inducing_variables.shape[0], inducing_variables.size))
#        for i, nc in enumerate(self.covars_theta_I):
#            grad += self.covars[i].psi_2grad_inducing_variables(theta[nc], mean, variance, inducing_variables)
#        return grad
    pass

class ProductCF(CovarianceFunction):
    """
    Product Covariance function. This function multiplies
    the given CFs and returns the resulting product.
    
    **Parameters:**
    
    covars : [CFs of type :py:class:`pygp.covar.CovarianceFunction`]
    
        Covariance functions to be multiplied.
        
    """
    #    __slots__=["n_params_list","covars","covars_theta_I"]
    
    def __init__(self, covars, *args, **kw_args):
        super(ProductCF, self).__init__()
        self.n_params_list = []
        self.covars = []
        self.covars_theta_I = []
        self.covars_covar_I = []

        self.covars = covars
        i = 0
        
        for nc in xrange(len(covars)):
            covar = covars[nc]
            #assert isinstance(covar, CovarianceFunction), 'ProductCF: ProductCF is constructed from a list of covaraince functions'
            Nparam = covar.get_number_of_parameters()
            self.n_params_list.append(Nparam)
            self.covars_theta_I.append(sp.arange(i, i + covar.get_number_of_parameters()))
            self.covars_covar_I.extend(sp.repeat(nc, Nparam))
#            for ip in xrange(Nparam):
#                self.covars_covar_I.append(nc)
            
            i += Nparam
            
        self.n_params_list = sp.array(self.n_params_list)
        self.n_hyperparameters = self.n_params_list.sum()

    def get_hyperparameter_names(self):
        """return the names of hyperparameters to make identificatio neasier"""
        names = []
        for covar in self.covars:
            names.extend(covar.get_hyperparameter_names())
        return names


    def K(self, theta, x1, x2=None):
        """
        Get Covariance matrix K with given hyperparameters
        theta and inputs x1 and x2. The result
        will be the product covariance of all covariance
        functions combined in this product covariance.

        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction` 
        """
        #1. check theta has correct length
        assert theta.shape[0] == self.n_hyperparameters, 'ProductCF: K: theta has wrong shape'
        #2. create sum of covarainces..
        if x2 is None:
            K = sp.ones([x1.shape[0], x1.shape[0]])
        else:
            K = sp.ones([x1.shape[0], x2.shape[0]])
        for nc in xrange(len(self.covars)):
            covar = self.covars[nc]
            _theta = theta[self.covars_theta_I[nc]]
            K *= covar.K(_theta, x1, x2)
        return K


    def Kgrad_theta(self, theta, x, i):
        '''The derivatives of the covariance matrix for
        the i-th hyperparameter.
        
        **Parameters:**
        See :py:class:`pygp.covar.CovarianceFunction`
        '''
        #1. check theta has correct length
        assert theta.shape[0] == self.n_hyperparameters, 'ProductCF: K: theta has wrong shape'
        # nc = self.covars_covar_I[i]
        # covar = self.covars[nc]
        # d  = self.covars_theta_I[nc].min()
        # j  = i-d
        # return covar.Kd(theta[self.covars_theta_I[nc]],x1,j)
        # rv = sp.ones([self.n_hyperparameters,x1.shape[0],x2.shape[0]])
        # for nc in xrange(len(self.covars)):
        #     covar = self.covars[nc]
        #     #get kernel and derivative
        #     K_ = covar.K(theta[self.covars_theta_I[nc]],*args)
        #     Kd_= covar.Kd(theta[self.covars_theta_I[nc]],*args)
        #     #for the parmeters of this covariance multiply derivative
        #     rv[self.covars_theta_I[nc]] *= Kd_
        #     #for all remaining ones kernel
        #     rv[~self.covars_theta_I[nc]] *= K_
        nc = self.covars_covar_I[i]
        covar = self.covars[nc]
        d = i - self.covars_theta_I[nc].min()
        Kd = covar.Kgrad_theta(theta[self.covars_theta_I[nc]], x, d)
        for ind in xrange(len(self.covars)):
            if(ind != nc):
                _theta = theta[self.covars_theta_I[ind]]
                Kd *= self.covars[ind].K(_theta, x)
        return Kd

    #derivative with respect to inputs
    def Kgrad_x(self, theta, x1, x2, d):
        assert theta.shape[0] == self.n_hyperparameters, 'Product CF: K: theta has wrong shape'
        x1, x2 = self._filter_input_dimensions(x1, x2)
        RV_sum = sp.zeros([x1.shape[0], x2.shape[0]])
        for nc in xrange(len(self.covars)):
            RV_prod = sp.ones([x1.shape[0], x2.shape[0]])
            for j in xrange(len(self.covars)):
                _theta = theta[self.covars_theta_I[j]]
                covar = self.covars[j]
                if(j == nc):
                    RV_prod *= covar.Kgrad_x(_theta, x1, x2, d)
                else:
                    RV_prod *= covar.K(_theta, x1, x2)
            RV_sum += RV_prod
        return RV_sum
#            covar = self.covars[nc]
#            if(d in covar.dimension_indices):
#                dims = covar.dimension_indices.copy()
#                #covar.set_dimension_indices([d])
#                _theta = theta[self.covars_theta_I[nc]]
#                K = covar.K(_theta,x1,x2)
#                RV_sum += covar.Kgrad_x(_theta, x1, x2, d)/K
#                #import pdb;pdb.set_trace()
#                RV_prod *= K
#                #covar.set_dimension_indices(dims)
#        return RV_sum*RV_prod
    
    def Kgrad_xdiag(self, theta, x1, d):
        assert theta.shape[0] == self.n_hyperparameters, 'K: theta has wrong shape'
#        RV_sum = sp.zeros([x1.shape[0], x1.shape[0]])
#        RV_prod = sp.ones([x1.shape[0], x1.shape[0]])
#        for nc in xrange(len(self.covars)):
#            covar = self.covars[nc]
#            _theta = theta[self.covars_theta_I[nc]]
#            if(d in covar.dimension_indices):
#                dims = covar.dimension_indices.copy()
#                #covar.set_dimension_indices([d])
#                _theta = theta[self.covars_theta_I[nc]]
#                K = covar.Kdiag(_theta,x1)
#                RV_sum += covar.Kgrad_xdiag(_theta, x1, d)/K
#                RV_prod *= K
#                #covar.set_dimension_indices(dims)
#        return RV_sum * RV_prod

#        pdb.set_trace()
        RV_sum = sp.zeros([x1.shape[0]])
        for nc in xrange(len(self.covars)):
            RV_prod = sp.ones([x1.shape[0]])
            for j in xrange(len(self.covars)):
                _theta = theta[self.covars_theta_I[j]]
                covar = self.covars[j]
                if(j == nc):
                    RV_prod *= covar.Kgrad_xdiag(_theta, x1, d)
                else:
                    RV_prod *= covar.Kdiag(_theta, x1)
            RV_sum += RV_prod
        return RV_sum

#    def get_Iexp(self, theta):
#        Iexp = []
#        for nc in xrange(len(self.covars)):
#            covar = self.covars[nc]
#            _theta = theta[self.covars_theta_I[nc]]
#            Iexp = sp.concatenate((Iexp, covar.get_Iexp(_theta)))
#        return sp.array(Iexp, dtype='bool')

class ProductCFPsiStat(ProductCF, BayesianStatisticsCF):

    def update_stats(self, theta, mean, variance, inducing_variables):
        super(ProductCFPsiStat, self).update_stats(theta, mean, variance, inducing_variables)
        self.psi_list = []
        for covar, nc in zip(self.covars, self.covars_theta_I):
            covar.update_stats(theta[nc], mean, variance, inducing_variables)
            self.psi_list.append(covar.psi())
        
    def psi(self):
        psis = reduce(lambda x, y: [numpy.multiply(a, b) for a, b in zip(x, y)], self.psi_list)
        return psis

    def psigrad_theta(self):
        return BayesianStatisticsCF.psigrad_theta(self)


    def psigrad_mean(self):
        return BayesianStatisticsCF.psigrad_mean(self)


    def psigrad_variance(self):
        return BayesianStatisticsCF.psigrad_variance(self)


    def psigrad_inducing_variables(self):
        return BayesianStatisticsCF.psigrad_inducing_variables(self)

    
    
    pass

class ShiftCF(CovarianceFunction):
    """
    Time Shift Covariance function. This covariance function depicts
    the time shifts induced by the data and covariance function given
    and passes the shifted inputs to the covariance function given.
    To calculate the shifts of the inputs make shure the covariance
    function passed implements the derivative after the input
    Kd_dx(theta, x).
    
    covar : CF of type :py:class:`pygp.covar.CovarianceFunction`
    
        Covariance function to be used to depict the time shifts.
    
    replicate_indices : [int]

        The indices of the respective replicates, corresponding to
        the inputs. For instance: An input with three replicates:

        ===================== ========= ========= =========
        /                     rep1      rep2      rep3
        ===================== ========= ========= =========
        input = [             -1,0,1,2, -1,0,1,2, -1,0,1,2]
        replicate_indices = [ 0,0,0,0,  1,1,1,1,  2,2,2,2]
        ===================== ========= ========= =========

            
        Thus, the replicate indices represent
        which inputs correspond to which replicate.
        
    """
#    __slots__=["n_params_list","covars","covars_theta_I"]

    def __init__(self, covar, replicate_indices, *args, **kw_args):
        super(ShiftCF, self).__init__(*args, **kw_args)
        #1. check that covar is covariance function
        assert isinstance(covar, CovarianceFunction), 'ShiftCF: ShiftCF is constructed from a CovarianceFunction, which provides the partial derivative for the covariance matrix K with respect to input X'
        #2. get number of params
        self.replicate_indices = replicate_indices
        self.n_replicates = len(sp.unique(replicate_indices))
        self.n_hyperparameters = covar.get_number_of_parameters() + self.n_replicates
        self.covar = covar

    def get_reparametrized_theta(self, theta):
        covar_n_hyper = self.covar.get_number_of_parameters()
        return sp.concatenate((self.covar.get_reparametrized_theta(theta[:covar_n_hyper]), theta[covar_n_hyper:]))

    def get_de_reparametrized_theta(self, theta):
        covar_n_hyper = self.covar.get_number_of_parameters()
        return sp.concatenate((self.covar.get_de_reparametrized_theta(theta[:covar_n_hyper]), theta[covar_n_hyper:]))

    def get_timeshifts(self, theta):
        covar_n_hyper = self.covar.get_number_of_parameters()
        return theta[covar_n_hyper:]

    def get_hyperparameter_names(self):
        """return the names of hyperparameters to make identificatio neasier"""
        return sp.concatenate((self.covar.get_hyperparameter_names(), ["Time-Shift rep%i" % (i) for i in sp.unique(self.replicate_indices)]))

    def K(self, theta, x1, x2=None):
        """
        Get Covariance matrix K with given hyperparameters
        theta and inputs x1 and x2. The result
        will be the covariance of the covariance
        function given, calculated on the shifted inputs x1,x2.
        The shift is determined by the last n_replicate parameters of
        theta, where n_replicate is the number of replicates this
        CF conducts.

        **Parameters:**

        theta : [double]
            the hyperparameters of this CF. Its structure is as follows:
            [theta of covar, time-shift-parameters]
        
        Others see :py:class:`pygp.covar.CovarianceFunction` 
        """
        #1. check theta has correct length
        assert theta.shape[0] == self.n_hyperparameters, 'ShiftCF: K: theta has wrong shape'
        #2. shift inputs of covarainces..
        # get time shift parameter
        covar_n_hyper = self.covar.get_number_of_parameters()
        # shift inputs
        T = theta[covar_n_hyper:covar_n_hyper + self.n_replicates]
        shift_x1 = self._shift_x(x1.copy(), T)
        K = self.covar.K(theta[:covar_n_hyper], shift_x1, x2)
        return K


    def Kgrad_theta(self, theta, x, i):
        """
        Get Covariance matrix K with given hyperparameters
        theta and inputs x1 and x2. The result
        will be the covariance of the covariance
        function given, calculated on the shifted inputs x1,x2.
        The shift is determined by the last n_replicate parameters of
        theta, where n_replicate is the number of replicates this
        CF conducts.

        **Parameters:**

        theta : [double]
            the hyperparameters of this CF. Its structure is as follows::
            [theta of covar, time-shift-parameters]

        i : int
            the partial derivative of the i-th
            hyperparameter shal be returned. 
            
        """
        #1. check theta has correct length
        assert theta.shape[0] == self.n_hyperparameters, 'ShiftCF: K: theta has wrong shape'
        covar_n_hyper = self.covar.get_number_of_parameters()
        T = theta[covar_n_hyper:covar_n_hyper + self.n_replicates]
        shift_x = self._shift_x(x.copy(), T)        
        if i >= covar_n_hyper:
            Kdx = self.covar.Kgrad_x(theta[:covar_n_hyper], shift_x, shift_x, 0)
            c = sp.array(self.replicate_indices == (i - covar_n_hyper),
                         dtype='int').reshape(-1, 1)
            cdist = dist(c, c)
            cdist = cdist.transpose(2, 0, 1)
            try:
                return Kdx * cdist
            except ValueError:
                return Kdx

        else:
            return self.covar.Kgrad_theta(theta[:covar_n_hyper], shift_x, i)

#    def get_Iexp(self, theta):
#        """
#        Return indices of which hyperparameters are to be exponentiated
#        for optimization. Here we do not want
#        
#        **Parameters:**
#        See :py:class:`pygp.covar.CovarianceFunction`
#        """
#        covar_n_hyper = self.covar.get_number_of_parameters()
#        Iexp = sp.concatenate((self.covar.get_Iexp(theta[:covar_n_hyper]),
#                               sp.zeros(self.n_replicates)))
#        Iexp = sp.array(Iexp,dtype='bool')
#        return Iexp
        
    def _shift_x(self, x, T):
        # subtract T, respectively
        if(x.shape[0] == self.replicate_indices.shape[0]):
            for i,ri in enumerate(sp.unique(self.replicate_indices)):
                x[self.replicate_indices == ri] -= T[i]
        return x
