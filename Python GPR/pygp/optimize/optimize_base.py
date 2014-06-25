"""
Package for Gaussian Process Optimization
=========================================

This package provides optimization functionality
for hyperparameters of covariance functions
:py:class:`pygp.covar` given. 

"""


# import scipy:
import scipy as SP
import scipy.optimize as OPT
import logging as LG
import pdb
import numpy
import scipy.linalg
from scipy.linalg import LinAlgError
from scipy.optimize._minimize import minimize
import sys

# LG.basicConfig(level=LG.INFO)

def param_dict_to_list(di, skeys=None):
    """convert from param dictionary to list"""
    # sort keys
    # RV = SP.concatenate([di[key].flatten() for key in skeys])
    # RV = reduce(numpy.append,[di[key] for key in skeys])
    RV = numpy.concatenate([di[key].ravel() for key in skeys])
    
    return RV
    pass

def param_list_to_dict(li, param_struct, skeys):
    """convert from param dictionary to list
    param_struct: structure of parameter array
    """
    RV = []
    i0 = 0
    for key in skeys:
        val = param_struct[key]
        shape = SP.array(val) 
        np = shape.prod()
        i1 = i0 + np
        params = li[i0:i1].reshape(shape)
        RV.append((key, params))
        i0 = i1
    return dict(RV)

def checkgrad(f, fprime, x, verbose=True, hyper_names=None, tolerance=1E-1, *args, **kw_args):
    """
    Analytical gradient calculation using a 3-point method
    
    """
    
    import numpy as np
    
    # using machine precision to choose h
    
    eps = np.finfo(float).eps
    step = np.sqrt(eps)  # *max(1E-6, x.min())
    
    # shake things up a bit by taking random steps for each x dimension
    h = step * np.sign(np.random.uniform(-1, 1, x.shape))
    
    f_ph = f(x + h, *args, **kw_args)
    f_mh = f(x - h, *args, **kw_args)
    numerical_gradient = (f_ph - f_mh) / (2 * h)
    analytical_gradient_all = fprime(x, *args, **kw_args)

    # ratio = (f_ph - f_mh)/(2*np.dot(h.T, analytical_gradient))

    # if True:
    h = np.zeros_like(x)
    dim = 1
    if f_ph.shape:
        dim = len(f_mh)
    res = np.zeros((len(x), 4, dim))

    if verbose:
        if hyper_names is None:
            hyper_names = map(str, range(len(x)))
        m = max_length(hyper_names)
        m = "{0!s:%is}" % m
        format_string = m + "  NUM:{1:13.5G}  ANA:{2:13.5G}  RAT:{3: 4.3F}  DIF:{4: 4.3F}"
        # format_string = format_string%(m,'s','!s','f','f', 'f')
        
    for i in range(len(x)):
        h[i] = step
        f_ph = f(x + h, *args, **kw_args)
        f_mh = f(x - h, *args, **kw_args)
    
        numerical_gradient = (f_ph - f_mh) / (2 * step)
        
        # analytical_gradient = fprime(x, *args, **kw_args)[i] ## <---- Exhaustive
        analytical_gradient = analytical_gradient_all[i]
        
        difference = numerical_gradient - analytical_gradient
        
        if np.abs(step * analytical_gradient) > 0:
            ratio = (f_ph - f_mh) / (2 * step * analytical_gradient)
        elif np.abs(difference) < tolerance:
            ratio = 1.
        else:
            ratio = 0
        h[i] = 0
        if verbose:
            try:
                if(np.abs(ratio - 1) < tolerance):
                    try:
                        mark = " " + '\033[92m' + u"\u2713" + '\033[0m'  # green checkmark
                    except:
                        mark = ":)"
                else:
                    try:
                        mark = " " + '\033[91m' + u"\u2717" + '\033[0m'  # red crossmark
                    except:
                        mark = "X("
                print format_string.format(hyper_names[i],
                                       numerical_gradient,
                                       analytical_gradient,
                                       ratio,
                                       difference), mark
            except Exception:
                import ipdb;ipdb.set_trace()
        res[i, 0, :] = numerical_gradient
        res[i, 1, :] = analytical_gradient
        res[i, 2, :] = ratio
        res[i, 3, :] = difference
        
    return res

def max_length(x_names):
    m = 0
    for name in x_names:
        m = max(m, len(str(name)))
    return m

def opt_hyper(gpr, hyperparams,
              Ifilter=None, maxiter=1000,
              gradcheck=False,
              bounds=None, callback=None,
              # optimizer=OPT.fmin_tnc, 
              gradient_tolerance=1e-8,
              messages=True, *args, **kw_args):
    """
    Optimize hyperparemters of :py:class:`pygp.gp.basic_gp.GP` ``gpr`` starting from given hyperparameters ``hyperparams``.

    **Parameters:**

    gpr : :py:class:`pygp.gp.basic_gp`
        GP regression class
    hyperparams : {'covar':logtheta, ...}
        Dictionary filled with starting hyperparameters
        for optimization. logtheta are the CF hyperparameters.
    Ifilter : [boolean]
        Index vector, indicating which hyperparameters shall
        be optimized. For instance::

            logtheta = [1,2,3]
            Ifilter = [0,1,0]

        means that only the second entry (which equals 2 in
        this example) of logtheta will be optimized
        and the others remain untouched.

    bounds : [[min,max]]
        Array with min and max value that can be attained for any hyperparameter

    maxiter: int
        maximum number of function evaluations
    gradcheck: boolean 
        check gradients comparing the analytical gradients to their approximations
    
    ** argument passed onto LML**

    priors : [:py:class:`pygp.priors`]
        non-default prior, otherwise assume
        first index amplitude, last noise, rest:lengthscales
    """
    # optimizer: :py:class:`scipy.optimize`
    # which scipy optimizer to use? (standard lbfgsb)

    global __last_lml__
    __last_lml__ = 0
#    i=0
#    mill = {0:'-',1:'\\',2:"|",4:'-',3:'/'}
#    def callback(x):
#        i = (i+1)%5
#        import sys; 
#        sys.stdout.flush()
#        sys.stdout.write("optimizing...{}\r".format(mill[i]))

    def f(x):
        x_ = X0
        x_[Ifilter_x] = x
        global __last_lml__
        __last_lml__ = gpr.LML(param_list_to_dict(x_, param_struct, skeys), *args, **kw_args)
        # LG.debug("L("+str(x_)+")=="+str(rv))
        if SP.isnan(__last_lml__):
            return 1E6
        return __last_lml__
    
    def df(x):
        x_ = X0
        x_[Ifilter_x] = x
        rv = gpr.LMLgrad(param_list_to_dict(x_, param_struct, skeys), *args, **kw_args)
        rv = param_dict_to_list(rv, skeys)
        # LG.debug("dL("+str(x_)+")=="+str(rv))
        if not SP.isfinite(rv).all():  # SP.isnan(rv).any():
            In = ~SP.isfinite(rv)#SP.isnan(rv)
            rv[In] = 1E6
        return rv[Ifilter_x]

    # 0. store parameter structure
    skeys = SP.sort(hyperparams.keys())
    param_struct = dict([(name, hyperparams[name].shape) for name in skeys])

    # 1. convert the dictionaries to parameter lists
    X0 = param_dict_to_list(hyperparams, skeys)
    if Ifilter is not None:
        Ifilter_x = SP.array(param_dict_to_list(Ifilter, skeys), dtype='bool')
    else:
        Ifilter_x = slice(None)#SP.ones(len(X0), dtype='bool')

    # 2. bounds
    if bounds is not None:
        # go through all hyperparams and build bound array (flattened)
        _b = []
        for key in skeys:
            if key in bounds.keys():
                _b.extend(bounds[key])
            else:
                _b.extend([(-SP.inf, +SP.inf)] * hyperparams[key].size)
        bounds = SP.array(_b)
        bounds = bounds[Ifilter_x]
        pass
       
        
    # 2. set starting point of optimization, truncate the non-used dimensions
    x = X0.copy()[Ifilter_x]
        
    LG.debug("startparameters for opt:" + str(x))
    
    if gradcheck:
        checkgrad(f, df, x, hyper_names=None)
        LG.info("check_grad (pre) (Enter to continue):" + str(OPT.check_grad(f, df, x)))
        raw_input()
# #        
    
    LG.debug("start optimization")
    # general optimizer interface
    # note: x is a subset of X, indexing the parameters that are optimized over
    # Ifilter_x pickes the subest of X, yielding x
    try:
        #=======================================================================
        # TNC:
        # opt_RV, opt_lml = optimizer(f, x, fprime=df, maxfun=int(maxiter), pgtol=gradient_tolerance, messages=messages, bounds=bounds)[:2]
        #=======================================================================
        
        #=======================================================================
        # L BFGS:
        # opt_RV, opt_lml = OPT.fmin_l_bfgs_b(f, x, fprime=df, maxfun=int(maxiter), approx_grad=False, iprint=0, bounds=bounds, factr=10.0 , pgtol=gradient_tolerance)[:2]
        #=======================================================================
        
        #=======================================================================
        # L_BFGS_B minimize
        iprint = -1
        if messages:
            iprint = 1
        res = minimize(f, x, method="L-BFGS-B", jac=df,
                       bounds=bounds, callback=callback,
                       tol=gradient_tolerance,
                       options=dict(maxiter=maxiter,
                                    iprint=iprint))
        opt_RV, opt_lml = res.x, res.fun
        #=======================================================================
        
        opt_x = opt_RV
        
    except LinAlgError as error:
        print error
        opt_x = X0
        opt_lml = __last_lml__
    # opt_RV = OPT.fmin_bfgs(f, x, fprime=df, maxiter=int(maxiter), disp=1)
    # opt_x = opt_RV
    
    # relate back to X
    Xopt = X0.copy()
    Xopt[Ifilter_x] = opt_x
    # convert into dictionary
    opt_hyperparams = param_list_to_dict(Xopt, param_struct, skeys)
    # get the log marginal likelihood at the optimum:
    # opt_lml = gpr.LML(opt_hyperparams,**kw_args)

    if gradcheck:
        checkgrad(f, df, Xopt, hyper_names=None)
        LG.info("check_grad (post) (Enter to continue):" + str(OPT.check_grad(f, df, opt_x)))
        pdb.set_trace()
#        # raw_input()

    # LG.debug("old parameters:")
    # LG.debug(str(hyperparams))
    # LG.debug("optimized parameters:")
    # LG.debug(str(opt_hyperparams))
    # LG.debug("grad:"+str(df(opt_x)))
    
    return [opt_hyperparams, opt_lml]
