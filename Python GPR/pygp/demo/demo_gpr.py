"""
Application Example of GP regression
====================================

This Example shows the Squared Exponential CF
(:py:class:`covar.se.SEARDCF`) combined with noise
:py:class:`covar.noise.noiseCF` by summing them up
(using :py:class:`covar.combinators.sumCF`).
"""

import logging as LG
import numpy.random as random

from pygp.covar import se, noise, combinators
import pygp.plot.gpr_plot as gpr_plot
import pygp.priors.lnpriors as lnpriors

import pylab as PL
import scipy as SP
from pygp.likelihood.likelihood_base import GaussLikISO
from pygp.gp.gp_base import GP
from pygp.optimize.optimize_base import opt_hyper
import sys


def create_toy_data():
    #0. generate Toy-Data; just samples from a superposition of a sin + linear trend
    n = 14
    
    xmin = 1
    xmax = 2.5*SP.pi
    x = SP.linspace(xmin,xmax,n)
    
    C = 2       #offset
    sigma = 0.01
    
    b = 0
    
    y  = b*x + C + 1*SP.sin(x)
#    dy = b   +     1*SP.cos(x)
    y += sigma*random.randn(y.shape[0])
    
    y-= y.mean()
    
    x = x[:,SP.newaxis]
    
    x[SP.random.randint(n)] = SP.nan
    y[SP.random.randint(n)] = SP.nan

    return [x,y]


def run_demo():
    LG.basicConfig(level=LG.INFO)
    random.seed(1)

    #1. create toy data
    [x,y] = create_toy_data()
    n_dimensions = 1
    
    #2. location of unispaced predictions
    X = SP.linspace(0,10,200)[:,SP.newaxis]
        

    if 0:
        #old interface where the covaraince funciton and likelihood are one thing:
        #hyperparamters
        covar_parms = SP.log([1,1,1])
        hyperparams = {'covar':covar_parms}       
        #construct covariance function
        SECF = se.SqexpCFARD(n_dimensions=n_dimensions)
        noiseCF = noise.NoiseCFISO()
        covar = combinators.SumCF((SECF,noiseCF))
        covar_priors = []
        #scale
        covar_priors.append([lnpriors.lnGammaExp,[1,2]])
        covar_priors.extend([[lnpriors.lnGammaExp,[1,1]] for i in xrange(n_dimensions)])
        #noise
        covar_priors.append([lnpriors.lnGammaExp,[1,1]])
        priors = {'covar':covar_priors}
        likelihood = None

    if 1:
        #new interface with likelihood parametres being decoupled from the covaraince function
        likelihood = GaussLikISO()
        covar_parms = SP.log([1,1])
        hyperparams = {'covar':covar_parms,'lik':SP.log([1])}       
        #construct covariance function
        SECF = se.SqexpCFARD(n_dimensions=n_dimensions)
        covar = SECF
        covar_priors = []
        #scale
        covar_priors.append([lnpriors.lnGammaExp,[1,2]])
        covar_priors.extend([[lnpriors.lnGammaExp,[1,1]] for i in xrange(n_dimensions)])
        lik_priors = []
        #noise
        lik_priors.append([lnpriors.lnGammaExp,[1,1]])
        priors = {'covar':covar_priors,'lik':lik_priors}

        

    
    gp = GP(covar,likelihood=likelihood,x=x,y=y)
    
    gp.set_active_set_indices(slice(0,-2))
    
    opt_model_params = opt_hyper(gp,hyperparams,priors=priors,gradcheck=True,messages=False)[0]
    
    #predict
    [M,S] = gp.predict(opt_model_params,X)

    #create plots
    gpr_plot.plot_sausage(X,M,SP.sqrt(S))
    gpr_plot.plot_training_data(x,y)
    PL.show()
    


if __name__ == '__main__':
    run_demo()
