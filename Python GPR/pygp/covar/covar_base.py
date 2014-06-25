"""
Covariance Functions
====================

We implemented several different covariance functions (CFs) you can work with. To use GP-regression with these covariance functions it is highly recommended to model the noise of the data in one extra covariance function (:py:class:`pygp.covar.noise.NoiseISOCF`) and add this noise CF to the CF you are calculating by putting them all together in one :py:class:`pygp.covar.combinators.SumCF`.

For example to use the squared exponential CF with noise::

    from pygp.covar import se, noise, combinators
    
    #Feature dimension of the covariance: 
    dimensions = 1
    
    SECF = se.SEARDCF(dim)
    noise = noise.NoiseISOCF()
    covariance = combinators.SumCF((SECF,noise))

"""

# import scipy:
import scipy
import logging

class CovarianceFunction(object):
    class HyperparameterError(Exception):
        pass
    
    """
    *Abstract super class for all implementations of covariance functions:*
    
    **Important:** *All Covariance Functions have
    to inherit from this class in order to work
    properly with this GP framework.*

    **Parameters:**

    n_dimensions : int

        standard: n_dimension = 1. The number of
        dimensions (i.e. features) this CF holds.

    dimension_indices : [int]

        The indices of dimensions (features) this CF takes into account.

    """
    __slots__ = ["n_hyperparameters",
                "n_dimensions",
                "dimension_indices"]

    def __init__(self, n_dimensions= -1, dimension_indices=None):
        """
        initialise covariance function:
        n_dimensions: int [1]
            number of dimensions of the input to work on
            set zero for independent of input dimensions
        """
        
        self.n_hyperparameters = scipy.nan
        #set relevant dimensions for coveriance function
        #either explicit index
        if dimension_indices is not None:
            self.dimension_indices = scipy.array(dimension_indices, dtype='int32')
        #or via the number of used dimensions
        elif n_dimensions > 0:
            self.dimension_indices = scipy.arange(n_dimensions)
        #or just what comes in
        else:
            self.dimension_indices = scipy.array([])
        self.n_dimensions = len(self.dimension_indices)
        pass

    def K(self, theta, x1, x2=None):
        """
        Get Covariance matrix K with given hyperparameters
        theta and inputs x1 and optional x2.
        If only x1 is given the covariance
        matrix is computed with x1 against x1.

        **Parameters:**

        theta : [double]

            The hyperparameters for which the covariance
            matrix shall be computed. *theta* are the
            hyperparameters for the respective covariance function.
            For instance :py:class:`pygp.covar.se.SEARDCF`
            holds hyperparameters as follows::

                `[Amplitude, Length-Scale(s)]`.

        x1 : [double]
        
            The training input X, for which the
            pointwise covariance shall be calculated.

        x2 : [double]
        
            The interpolation input X\`*`, for which the
            pointwise covariance shall be calculated.

        """
        logging.critical("implement K")
        print("%s: Function K not yet implemented" % (self.__class__))
        return None
        pass

    def Kdiag(self, theta, x1):
        """
        Get diagonal of the (squared) covariance matrix.

        *Default*: Return the diagonal of the fully
        calculated Covariance Matrix. This may be overwritten
        more efficiently.
        """
        logging.debug("No Kdiag specified; Return default naive computation.")
        #print("%s: Function Kdiag not yet implemented"%(self.__class__))
        return self.K(theta, x1).diagonal()
        
    def Kgrad_theta(self, theta, x1, i):
        """
        Get partial derivative of covariance matrix K
        with respect to the i-th given
        hyperparameter `theta[i]`.

        **Parameters:**

        theta : [double]

            The hyperparameters for covariance.

        x1 : [double]
        
            The training input X.

        i : int

            The index of the hyperparameter, which's
            partial derivative shall be returned.

        """
        logging.critical("implement Kd")
        print "please implement Kd"
        pass

    def Kgrad_x(self, theta, x1, x2, d):
        """
        Partial derivatives of K[X1,X2] with respect to x1(:)^d
        RV: matrix of size [x1,x2] containin all values of
        d/dx1^{i,d} K(X1,X2)
        """
        logging.critical("implement Kgrad_x")
        #print("%s: Function Kgrad_x not yet implemented"%(self.__class__))
        return None

    def Kgrad_xdiag(self, theta, x1, d):
        """
        Diagonal of partial derivatives of K[X1,X1] w.r.t. x1(:)^d
        RV: vector of size [x1] cotaining all partial derivatives
        d/dx1^{i,d} diag(K(X1,X2))
        """
        logging.critical("implement the partial derivative w.r.t x")
        #print("%s: Function Kgrad_xdiag not yet implemented"%(self.__class__))
        return None

    def get_reparametrized_theta(self, theta):
        """
        returns the reparametrized version of theta
        """
        return scipy.exp(2*theta)
    
    def get_de_reparametrized_theta(self, theta):
        """
        Return the version of theta, as it was before reparametrizing it. 
        Basically:
            get_de_reparametrized_theta(get_reparametrized_theta(theta)) == theta
        """
        return .5* scipy.log(theta)

    def get_hyperparameter_names(self):
        """
        Return names of hyperparameters to make
        identification easier
        """
        print "%s: WARNING: hyperparamter name not specified yet!" % (self.__class__)
        return []

    def get_number_of_parameters(self):
        """
        Return number of hyperparameters, specified by user.
        """
        return self.n_hyperparameters
    
    def get_default_hyperparameters(self, x=None, y=None):
        """
        Return default hyperpameters.
        
        *Default:*: No hyperparameters; Returns an empty array.
        """
        return {'covar':scipy.array([])}

    def get_n_dimensions(self):
        """
        Returns the number of dimensions, specified by user.
        Zero if independent of dimensions.
        """
        if self.n_dimensions > 0:
            return self.n_dimensions
        else:
            return 0
    
    def set_dimension_indices(self, active_dimension_indices):
        """
        Get the active_dimensions for this covariance function, i.e.
        the indices of the feature dimensions of the training inputs, which shall
        be used for the covariance.
        """
        self.n_dimensions = active_dimension_indices[-1] - active_dimension_indices[0]
        self.dimension_indices = active_dimension_indices
        pass

    def get_dimension_indices(self):
        """
        Get the indices of the input dimensions to work on, i.e.
        the indices of the feature dimensions of the training inputs, which shall
        be used for the covariance.
        Empty if independent of dimensions.
        """
        if self.get_n_dimensions() > 0:
            return self.dimension_indices
        else:
            return scipy.array([], dtype="bool")
    
    def get_ard_dimension_indices(self):
        """
        Return ard dimension indices. Empty if no ard scales.
        """
        return []

    def _filter_x(self, x):
        """
        Filter out the dimensions, which not correspond to a feature.
        (Only self.dimension_inices are needed)
        """
        if self.get_n_dimensions() > 0:
            return x[:, self.get_dimension_indices()]
        else:
            return x

    def _filter_input_dimensions(self, x1, x2):
        """
        Filter out all dimensions, which not correspond to a feature.
        Filter is self.dimension_indices.
        Returns : filtered x1, filtered x2
        """
        if x2 == None:
            return self._filter_x(x1), self._filter_x(x1)
        return self._filter_x(x1), self._filter_x(x2)


class BayesianStatisticsCF():
    class CacheMissingError(Exception):
        pass
    
    grad_theta_id = 0
    grad_means_id = 1
    grad_vars_id = 2
    grad_indv_id = 3
    
    #__slots__ = ["theta", "N", "M", "Q", "mean", "variance", "inducing_variables"]
    
    def __init__(self, *args, **kwargs):
        super(BayesianStatisticsCF, self).__init__(*args, **kwargs)
    
    def update_stats(self, theta, mean, variance, inducing_variables):
        """
        call this function to renew cache with given parameters

        always call super in this function!!! 
        """
        self.N = mean.shape[0]
        self.M = inducing_variables.shape[0]
        self.Q = self.get_n_dimensions()
        
        self.theta = theta
        self.mean = self._filter_x(mean)
        self.variance = self._filter_x(variance)
        self.inducing_variables = self._filter_x(inducing_variables)
        #logging.info("no update required?")
    
    def reparametrize(self):
        """
        overwrite this for different scaling purposes of hyperparameters
        """
        pass
    
    def psi(self):
        """
        returns psi statistics as list [psi_0, psi_1, psi_2]
        """
        # connected to be able to cache stuff!
        logging.critical("need psi statistics")

    def psigrad_theta(self):
        """
        returns gradients of psi statistics in following dict:
        [ psi_{0..2}grad_theta,
          psi_{0..2}grad_mean,
          psi_{0..2}grad_variance,
          psi_{0..2}grad_inducing_variables ]
        """
        logging.critical("need gradients of psi statistics")
    
    def psigrad_mean(self):
        """
        returns gradients of psi statistics in following dict:
        [ psi_{0..2}grad_theta,
          psi_{0..2}grad_mean,
          psi_{0..2}grad_variance,
          psi_{0..2}grad_inducing_variables ]
        """
        logging.critical("need gradients of psi statistics")
    
    def psigrad_variance(self):
        """
        returns gradients of psi statistics in following dict:
        [ psi_{0..2}grad_theta,
          psi_{0..2}grad_mean,
          psi_{0..2}grad_variance,
          psi_{0..2}grad_inducing_variables ]
        """
        logging.critical("need gradients of psi statistics")
    
    def psigrad_inducing_variables(self):
        """
        returns gradients of psi statistics in following dict:
        [ psi_{0..2}grad_theta,
          psi_{0..2}grad_mean,
          psi_{0..2}grad_variance,
          psi_{0..2}grad_inducing_variables ]
        """
        logging.critical("need gradients of psi statistics")
        
#    def psi_0(self, theta, mean, variance, inducing_variables):
#        logging.critical("implement psi statistics!")
#
#    def psi_0grad_theta(self, theta, mean, variance, inducing_variables):
#        logging.critical("implement psi statistics!")
#
#    def psi_0grad_mean(self, theta, mean, variance, inducing_variables):
#        logging.critical("implement psi statistics!")
#
#    def psi_0grad_variance(self, theta, mean, variance, inducing_variables):
#        logging.critical("implement psi statistics!")
#
#    def psi_0grad_inducing_variables(self, theta, mean, variance, inducing_variables):
#        return numpy.zeros(inducing_variables.size)
#
#    def psi_1(self, theta, mean, variance, inducing_variables):
#        logging.critical("implement psi statistics!")
#            
#    def psi_1grad_theta(self, theta, mean, variance, inducing_variables):
#        logging.critical("implement psi statistics!")
#    
#    def psi_1grad_mean(self, theta, mean, variance, inducing_variables):
#        logging.critical("implement psi statistics!")
#
#    def psi_1grad_variance(self, theta, mean, variance, inducing_variables):
#        logging.critical("implement psi statistics!")
#    
#    def psi_1grad_inducing_variables(self, theta, mean, variance, inducing_variables):
#        logging.critical("implement psi statistics!")
#    
#    def psi_2(self, theta, mean, variance, inducing_variables):
#        logging.critical("implement psi statistics!")
#            
#    def psi_2grad_theta(self, theta, mean, variance, inducing_variables):
#        logging.critical("implement psi statistics!")
#    
#    def psi_2grad_mean(self, theta, mean, variance, inducing_variables):
#        logging.critical("implement psi statistics!")
#    
#    def psi_2grad_variance(self, theta, mean, variance, inducing_variables):
#        logging.critical("implement psi statistics!")
#                
#    def psi_2grad_inducing_variables(self, theta, mean, variance, inducing_variables):
#        logging.critical("implement psi statistics!")
