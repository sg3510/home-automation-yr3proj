"""
Python package for Gaussian process regression in python
========================================================

demo_gpr.py explains how to perform basic regression tasks.
demo_gpr_robust.py shows how to apply EP for robust Gaussian process regression.

gpr.py Basic gp regression package
gpr_ep.py GP regression with EP likelihood models

covar: covariance functions
"""

import pkgutil

__all__ = []
for loader, module_name, is_pkg in  pkgutil.walk_packages(__path__):
    __all__.append(module_name)
    loader.find_module(module_name).load_module(module_name)
    
del loader, module_name, is_pkg

#from pygp.optimize.optimize_base import param_list_to_dict
#import scipy
#__import__('pkg_resources').declare_namespace(__name__)
#
#
#def GPModel():
#    
#    def __init__(self, gp, hyperparams):
#        self.gp = gp
#        self.beginning_hyperparams = hyperparams
#        self.randomize()
#        
#    def randomize(self):
#        self.n_function_calls = 0
#        self.hyperparams = self.beginning_hyperparams
#        
#    def update_hyperparams(**kwargs):
#        pass
#    
#    def optimize(self, maxiter=1000, callback=None):
#        def f(x):
#            self.hyperparams.update(param_list_to_dict(x))
#            if callback is not None:
#                callback(self.hyperparparams)
#            return 
#        
#        
#        skeys = scipy.sort(hyperparams.keys())
#        param_struct = dict([(name,hyperparams[name].shape) for name in skeys])
#
#
#    def negative_log_marginal_likelihood(self):
#        return self.gp.LML(self.hyperparams)
#    
#    def negative_log_marginal_likelihood_gradient(self):
#        return self.gp.LML_grad(self.hyperparams)