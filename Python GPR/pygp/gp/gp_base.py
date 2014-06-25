"""
Gaussian Process Package
========================

Holds all Gaussian Process classes, which hold all informations for a Gaussian Process to work porperly.

.. class **GP**: basic class for GP regression:
   * claculation of log marginal likelihood
   * prediction
   * data rescaling
   * transformation into log space


"""

import copy
import numpy.linalg as linalg
import scipy.lib.lapack.flapack
from pygp.linalg.linalg_matrix import jitChol, solve_chol
import logging
import numpy

class GP(object):
    """
    Gaussian Process regression class. Holds all information
    for the GP regression to take place.

    **Parameters:**

    covar_func : :py:class:`pygp.covar`
        The covariance function, which calculates the covariance
        of the outputs

    x : [double]
        training inputs (might be high dimensional,
        depending on which covariance function is chosen)
        Note: x must be of dimension `(-1,d)`

    y : [double]
        training targets of dimension `(-1,1)`

    Detailed descriptions of the fields of this class:

    ================================ ============ ===========================================
    Data                             Type/Default Explanation
    ================================ ============ ===========================================
    x                                array([])    inputs
    t                                array([])    targets
    n                                0            size of training data
    mean                             0            mean of the data

    **Settings:**

    **Covariance:**
    covar                            None         Covariance function

    **caching of covariance-stuff:**
    alpha                            None         cached alpha
    L                                None         chol(K)
    Nlogtheta                        0            total number of hyperparameters
                                                  for set kernel etc.
                                                  which if av. will be used
                                                  for predictions
    ================================ ============ ===========================================
    """
    # Smean : boolean
    # Subtract mean of Data
    __slots__ = ["_x", "_y", "n", "d", "covar", "likelihood", \
                "_covar_cache", '_active_set_indices', '_active_set_indices_changed',
                "_xcache", "_ycache", '_finite_filter']

    def __init__(self, covar_func=None, likelihood=None, x=None, y=None):
        '''GP(covar_func,likleihood,Smean=True,_x=None,_y=None)
        covar_func: Covariance
        likelihood: likelihood model
        _x/_y:        training input/targets
        '''
        self._invalidate_cache()
        if not (x is None) and not y is None:
            self.setData(x=x, y=y)
        # Store the constructor parameters
        self.covar = covar_func
        self.likelihood = likelihood
        pass

    def getData(self):
        """ Returns the data [x,y], currently set for this GP"""
        return [self._x, self._y]

    def setData(self, x, y):
        """
        setData(_x,t) with **Parameters:**

        x : inputs: [N x D]

        y : targets/outputs [N x d]
        #note d dimensional data structure only make sense for GPLVM
        """
        self._invalidate_cache()
        if x.ndim <= 1:
            x = x[:, None]
        if y.ndim <= 1:
            y = y[:, None]

        self._xcache = x
        self._ycache = y
        
        self._finite_filter = self._get_finite_filter(x, y)
        
        self._x = self._get_filtered(x)
        self._y = self._get_filtered(y)

        # squeeeze targets; this should only be a vector
        # self._y = _y.squeeze()
        # assert shapes
        assert self._x.shape[0] == self._y.shape[0], 'input/target shape missmatch'
        self.n = self._x.shape[0]
        # for GPLVM models:
        self.d = self._y.shape[1]
        pass

    def set_active_set_indices(self, active_set_indices):
        if active_set_indices is None:
            active_set_indices = slice(None)
        self._active_set_indices_changed = True
        self._active_set_indices = active_set_indices
        if self._xcache is not None and self._ycache is not None:
            self._finite_filter = self._get_finite_filter(self._xcache, self._ycache)
        
        
    def LML(self, hyperparams, priors=None):
        """
        Calculate the log Marginal likelihood
        for the given logtheta.

        **Parameters:**

        hyperparams : {'covar':CF_hyperparameters, ... }
            The hyperparameters for the log marginal likelihood.

        priors : [:py:class:`pygp.priors`]
            the prior beliefs for the hyperparameter values

        Ifilter : [bool]
            Denotes which hyperparameters shall be optimized.
            Thus ::

                Ifilter = [0,1,0]

            has the meaning that only the second
            hyperparameter shall be optimized.

        kw_args :
            All other arguments, explicitly annotated
            when necessary.

        """
        LML = self._LML_covar(hyperparams)

        # account for prior
        if priors is not None:
            plml = self._LML_prior(hyperparams, priors=priors)
            LML -= numpy.array([p[0, :].sum() for p in plml.values()]).sum()
        return LML


    def LMLgrad(self, hyperparams, priors=None, **kw_args):
        """
        Returns the gradient of the log Marginal likelihood for the given hyperparameters hyperparams.

        **Parameters:**

        hyperparams : {'covar':CF_hyperparameters, ...}
            The hyperparameters which shall be optimized and derived

        priors : [:py:class:`pygp.priors`]
            Priors which shall be optimized and derived

        """
        # Ideriv :
        #      indicator which derivativse to calculate (default: all)


        RV = self._LMLgrad_covar(hyperparams)
        if self.likelihood is not None:
            RV.update(self._LMLgrad_lik(hyperparams))
        # prior
        if priors is not None:
            # pdb.set_trace()
            plml = self._LML_prior(hyperparams, priors=priors, **kw_args)
            for key in RV.keys():
                RV[key] -= plml[key][1, :]
        return RV

    def get_covariances(self, hyperparams):
        """
        Return the Cholesky decompositions L and alpha::

            K
            L     = chol(K)
            alpha = solve(L,t)
            return [covar_struct] = get_covariances(hyperparam)

        **Parameters:**

        hyperparams: dict
            The hyperparameters for cholesky decomposition

        _x, _y: [double]
            input x and output _y for cholesky decomposition.
            If one/both is/are set, there will be no chaching allowed

        """
        if self._is_cached(hyperparams) and not self._active_set_indices_changed:
            pass
        else:
            Knoise = 0
            # 1. use likelihood object to perform the inference
            if self.likelihood is not None:
                Knoise = self.likelihood.K(hyperparams['lik'], self._x)
            K = self.covar.K(hyperparams['covar'], self._x)
            K += Knoise
            L = jitChol(K)[0].T  # lower triangular
            alpha = solve_chol(L, self._y)
            # DPOTRI computes the inverse of a real symmetric positive definite
            # matrix A using the Cholesky factorization
            Linv = scipy.lib.lapack.flapack.dpotri(L)[0]
            # Copy the matrix and kill the diagonal (we don't want to have 2*var)
            Kinv = Linv.copy()
            numpy.fill_diagonal(Linv, 0)
            # build the full inverse covariance matrix. This is correct: verified
            # by doing numpy.allclose(Kinv, linalg.inv(K))
            Kinv += Linv.T

            self._covar_cache = {'K': K, 'L':L, 'alpha':alpha, 'Kinv': Kinv}
            # store hyperparameters for cachine
            self._covar_cache['hyperparams'] = copy.deepcopy(hyperparams)
            self._active_set_indices_changed = False
        return self._covar_cache


    def predict(self, hyperparams, xstar, output=0, var=True):
        '''
        Predict mean and variance for given **Parameters:**

        hyperparams : {}
            hyperparameters in logSpace

        xstar    : [double]
            prediction inputs

        var      : boolean
            return predicted variance

        interval_indices : [ int || bool ]
            Either numpy array-like of boolean indicators,
            or numpy array-like of integer indices, denoting
            which x indices to predict from data.

        output   : output dimension for prediction (0)
        '''
        KV = self.get_covariances(hyperparams)

        # cross covariance:
        Kstar = self.covar.K(hyperparams['covar'], self._x, xstar)
        mu = numpy.dot(Kstar.transpose(), KV['alpha'][:, output])
        if(var):
            Kss_diag = self.covar.Kdiag(hyperparams['covar'], xstar).copy()
            if self.likelihood is not None:
                Kss_diag += self.likelihood.Kdiag(hyperparams['lik'], xstar)
            v = linalg.solve(KV['L'], Kstar)
            S2 = Kss_diag - sum(v * v, 0).transpose()
            S2 = abs(S2)
            return [mu, S2]
        else:
            return mu


    ########PRIVATE FUNCTIONS########


    def _LML_covar(self, hyperparams):
        """

	log marginal likelihood contributions from covariance hyperparameters

	"""
        try:
            KV = self.get_covariances(hyperparams)
        except linalg.LinAlgError:
            logging.error("exception caught (%s)" % (str(hyperparams)))
            return 1E6

        # Change: no supports multi dimensional stuff for GPLVM
        lml_quad = 0.5 * (KV['alpha'] * self._y).sum()
        lml_det = self._get_target_dimension() * (numpy.log(KV['L'].diagonal()).sum())
        lml_const = 0.5 * self._get_target_dimension() * self._get_input_dimension() * numpy.log(2 * numpy.pi)
        return (lml_quad + lml_det + lml_const)


    def _LMLgrad_covar(self, hyperparams):
        # currently only support derivatives of covar params
        logtheta = hyperparams['covar']
        try:
            KV = self.get_covariances(hyperparams)
        except linalg.LinAlgError:
            logging.error("exception caught (%s)" % (str(hyperparams)))
            return {'covar':numpy.zeros(len(logtheta))}
        # _ = self._get_input_dimension()
        # _ = KV['L']

        alpha = KV['alpha']
        W = self._get_target_dimension() * KV['Kinv'] - numpy.dot(alpha, alpha.transpose())
        self._covar_cache['W'] = W


        LMLgrad = numpy.zeros(len(logtheta))
        for i in xrange(len(logtheta)):
            Kd = self.covar.Kgrad_theta(hyperparams['covar'], self._x, i)
            LMLgrad[i] = 0.5 * (W * Kd).sum()
        RV = {'covar': LMLgrad}
        return RV


    def _LMLgrad_lik(self, hyperparams):
        """derivative of the likelihood parameters"""
        logtheta = hyperparams['lik']

        # note: we assume hard codede that this is called AFTER LMLgrad_covar has been called
        KV = self._covar_cache
        W = KV['W']

        LMLgrad = numpy.zeros(len(logtheta))
        for i in xrange(len(logtheta)):
            Kd = self.likelihood.Kgrad_theta(logtheta, self._x, i)
            LMLgrad[i] = 0.5 * (W * Kd).sum()
        RV = {'lik': LMLgrad}
        return RV


    def _invalidate_cache(self):
        """reset cache structure"""
        self._active_set_indices = slice(None)
        self._active_set_indices_changed = False
        self._covar_cache = {}
        self._xcache = None
        self._ycache = None
        pass

    def _LML_prior(self, hyperparams, priors={}):
        """calculate the prior contribution to the log marginal likelihood"""
        if priors is None:
            priors = {}
        RV = {}
        for key, value in hyperparams.iteritems():
            LL = list(value.shape)
            LL.insert(0, 2)
            pvalues = numpy.zeros(LL)
            if key == 'x':
                # speical treatment for x which has a single joint prior over all
                if key in priors:
                    plist = priors[key]
                    theta = hyperparams[key]
                    [lng, dlng] = plist[0](theta, plist[1])
                    pvalues[0] = lng
                    pvalues[1] = dlng
            else:

                # NOTE: removed the chain rule for exponentiated hyperparams
                if key in priors:
                    plist = priors[key]
                    theta = copy.deepcopy(hyperparams[key])
                    for i in xrange(len(theta)):
                        pvalues[:, i] = plist[i][0](theta[i], plist[i][1])
            RV[key] = pvalues
        return RV

    def _is_cached(self, hyperparams, keys=None):
        """check whether model parameters are cached"""
        if self._covar_cache is None:
            return False
        elif not 'hyperparams' in self._covar_cache.keys():
            return False
        else:
            if keys is None:
                keys = hyperparams.keys()
            # compare
            for key in keys:
                if not numpy.alltrue(self._covar_cache['hyperparams'][key] == hyperparams[key]):
                    return False
            # otherwise they are cached:
            return True

    def _get_filtered(self, x):
        return x[self._active_set_indices][self._finite_filter, :]

    def _get_filtered_zeros(self, x, newx=None):
        if newx is None:
            newx = numpy.zeros_like(self._xcache)
        newx[self._active_set_indices][self._finite_filter, :] = x
        return newx

    def _get_finite_filter(self, x, y):
        return (~(~numpy.isfinite(x[self._active_set_indices])).any(1) * ~(~numpy.isfinite(y[self._active_set_indices])).any(1)).flatten()

    #def _x(self, *args, **kw_args):
    #    return self._get_filtered(self._x)

    #def _y(self, *args, **kw_args):
    #    """get _y values"""
    #    return self._get_filtered(self._y)

#    def _get_active_set(self, x):
#        # if(self._active_set_indices is None):
#        #    return _x
#        # else:
#        return x[self._active_set_indices]

    def _get_target_dimension(self):
#        if(self._active_set_indices is None):
#            return self.d
#        else:
        return self._y.shape[1]

    def _get_input_dimension(self):
#        if(self._active_set_indices is None):
#            return self.n
#        else:
        return self._x.shape[1]

    def plot(self, hyperparams):
        """
        plot current state of the model
        """
        raise NotImplementedError("Plotting not yet implemented")
