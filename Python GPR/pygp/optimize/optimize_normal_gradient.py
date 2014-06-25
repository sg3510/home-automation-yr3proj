'''
Created on 23 May 2012

@author: maxz
'''
from pygp.optimize.optimize_base import opt_hyper, param_list_to_dict,\
    param_dict_to_list, checkgrad
import scipy.optimize as OPT
import logging as LG
from pygp.covar import se, noise, combinators
from pygp.demo.demo_bayesian_gplvm import random_inputs
from pygp.gp.bayesian_gplvm import vars_id, mean_id
from pygp.gp import bayesian_gplvm
import numpy
import pylab
from scipy.optimize.linesearch import line_search, line_search_wolfe2,\
    line_search_BFGS, line_search_armijo, line_search_wolfe1

def opt_hyper_gaussian_natural_gradient(gpr, hyperparams, mean_key, variance_key, 
                                        maxiter=500,inner_iter=10, 
                                        Ifilter=None,gradcheck=False,
                                        bounds = None,callback=None, 
                                        optimizer=OPT.fmin_tnc,gradient_tolerance=-1,
                                        messages=False, *args,**kw_args):
    
    def f(x, *args):
        x_ = X0
        x_[Ifilter_x] = x
        rv =  gpr.LML(param_list_to_dict(x_,param_struct,skeys),*args,**kw_args)
        #LG.debug("L("+str(x_)+")=="+str(rv))
        if numpy.isnan(rv):
            return 1E6
        return rv
    
    def df(x, *args):
        x_ = X0
        x_[Ifilter_x] = x
        rv =  gpr.LMLgrad(param_list_to_dict(x_,param_struct,skeys),*args,**kw_args)
        rv = param_dict_to_list(rv,skeys)
        #LG.debug("dL("+str(x_)+")=="+str(rv))
        if not numpy.isfinite(rv).all(): #numpy.isnan(rv).any():
            In = numpy.isnan(rv)
            rv[In] = 1E6
        return rv[Ifilter_x]

    #0. store parameter structure
    skeys = numpy.sort(hyperparams.keys())
    param_struct = dict([(name,hyperparams[name].shape) for name in skeys])

    
    #1. convert the dictionaries to parameter lists
    X0 = param_dict_to_list(hyperparams,skeys)
    if Ifilter is not None:
        Ifilter_x = numpy.array(param_dict_to_list(Ifilter,skeys),dtype='bool')
    else:
        Ifilter_x = numpy.ones(len(X0),dtype='bool')

    #2. bounds
    if bounds is not None:
        #go through all hyperparams and build bound array (flattened)
        _b = []
        for key in skeys:
            if key in bounds.keys():
                _b.extend(bounds[key])
            else:
                _b.extend([(-numpy.inf,+numpy.inf)]*hyperparams[key].size)
        bounds = numpy.array(_b)
        bounds = bounds[Ifilter_x]
        pass
       
        
    #2. set stating point of optimization, truncate the non-used dimensions
    x  = X0.copy()[Ifilter_x]
        
    LG.debug("startparameters for opt:"+str(x))
    
    if gradcheck:
        checkgrad(f, df, x)
        LG.info("check_grad (pre) (Enter to continue):" + str(OPT.check_grad(f,df,x)))
        raw_input()
##        
    LG.debug("start optimization")


    
    if gradient_tolerance < 0:
        gradient_tolerance = numpy.sqrt(numpy.finfo(float).eps)
    
    hyper_for_opt = hyperparams.copy()
    normal_keys = [mean_key, variance_key]
    hyperparam_keys = [v for v in hyperparams.keys() if not v in normal_keys]
    last_lml = gpr.LML(hyperparams)
    curr_lml = last_lml + 2*gradient_tolerance
    
    direction_dict = dict([(k,numpy.zeros_like(v)) for k,v in hyperparams.iteritems()])
    Q = hyper_for_opt[variance_key].shape[1]
    
    while maxiter > 0 and numpy.abs(last_lml-curr_lml) > gradient_tolerance:
        print "Iteration Loops left %s" % maxiter
        last_lml = gpr.LML(hyper_for_opt)
        # optimize non gaussian parameters
        #general optimizer interface
        #note: x is a subset of X, indexing the parameters that are optimized over
        # Ifilter_x pickes the subest of X, yielding x
        hyper_for_opt = optimizer(f, x, fprime=df, args=[hyperparam_keys], maxfun=int(inner_iter),
                           pgtol=gradient_tolerance, messages=messages, bounds=bounds)
        #    optimizer = OPT.fmin_l_bfgs_b
        #    opt_RV=optimizer(f, x, fprime=df, maxfun=int(maxiter),iprint =1, bounds=bounds, factr=10.0, pgtol=1e-10)

        Xopt = X0.copy()
        Xopt[Ifilter_x] = hyper_for_opt[0]
        #convert into dictionary
        hyper_for_opt = param_list_to_dict(hyper_for_opt[0],param_struct,skeys)
        curr_lml = gpr.LML(hyper_for_opt)
    
        # optimize natural gradient parameters:
        print("  NIT   NF   F                       GTG")
        grad_mean_last = numpy.ones((1,Q)); grad_variance_last=numpy.ones((0,Q)); direction_last = 0
        for i in xrange(inner_iter):
            #mean = hyper_for_opt[mean_key]
            variance = hyper_for_opt[variance_key]

            grad_gaussian = gpr.LMLgrad(hyper_for_opt, hyperparam_keys=normal_keys)
            grad_mean = grad_gaussian[mean_key]
            grad_variance = grad_gaussian[variance_key]
            
            grad = numpy.append(grad_mean, grad_variance, 0)
            grad_ = numpy.append(grad_mean / variance, grad_variance / 2., 0)
            grad_last = numpy.append(grad_mean_last, grad_variance_last, 0)
            beta = ((grad_ * (grad - grad_last)) / (grad * grad_).sum(0)).sum(0)

            direction = -grad_ + beta * direction_last

            direction_dict[mean_key] = direction[:grad_mean.shape[0],:]
            direction_dict[variance_key] = direction[grad_mean.shape[0]:,:]
            
            alpha = line_search_armijo(f, param_dict_to_list(hyper_for_opt, skeys), 
                                param_dict_to_list(direction_dict, skeys), 
                                0,
                                [normal_keys])
            
            hyper_for_opt[mean_key] = alpha[0] * direction[:grad_mean.shape[0],:]
            hyper_for_opt[variance_key] = alpha[0] * direction[grad_mean.shape[0]:,:]
            
            grad_mean_last = grad_mean
            grad_variance_last = grad_variance
            direction_last = direction
            
        maxiter -= 1
    
    
    
    
    #relate back to X
    Xopt = X0.copy()
    Xopt[Ifilter_x] = hyper_for_opt
    #convert into dictionary
    opt_hyperparams = param_list_to_dict(Xopt,param_struct,skeys)
    #get the log marginal likelihood at the optimum:
    opt_lml = gpr.LML(opt_hyperparams,**kw_args)

    
    return hyper_for_opt, curr_lml






def transform_by_fishers_information(grad_mean, grad_variance, variance):
    grad_mean *= variance
    grad_variance *= 2
    return grad_mean, grad_variance    

if __name__ == '__main__':
    N = 4
    M = 3
    Qlearn = 2
    D = 12
    # Input time steps
    
#        cov = linear.LinearCFPsiStat(Qlearn)
    cov = se.SqexpCFARDPsiStat(Qlearn)
    noi = noise.NoiseCFISOPsiStat()
    covar = combinators.SumCFPsiStat((cov,noi))
    hyperparams, signals, Y = random_inputs(N, M, Qlearn, D, covar)
    #hyperparams[variance_key] = numpy.random.randn(N,Qlearn) 
    g = bayesian_gplvm.BayesianGPLVM(y=Y, 
                                     covar_func=covar, 
                                     gplvm_dimensions=None, 
                                     n_inducing_variables=M)
    
    opt_hyperparams, lml = opt_hyper_gaussian_natural_gradient(g, hyperparams, 
                                                               mean_id, 
                                                               vars_id, 
                                                               messages=True)
    pylab.ion()
    g.plot(opt_hyperparams)