"""
Demo Application for Gaussian process latent variable models
====================================

"""
import logging as LG
import numpy as NP

from pygp.gp import gplvm
from pygp.covar import linear, se
from pygp.util.pca import PCA
from pygp.likelihood.likelihood_base import GaussLikISO
from pygp.optimize.optimize_base import opt_hyper

import copy

import scipy as SP
import pylab
from pygp.priors import lnpriors

# global params
N = 20
K = 3
D = 500

def run_demo():
    LG.basicConfig(level=LG.INFO)

    # 1. simulate data from a linear PCA model

    SP.random.seed(1)
    
    t = SP.linspace(0, 2 * SP.pi, N)[:, None]
    t = SP.tile(t, K)
    t += SP.linspace(SP.pi / 4., 3. / 4. * SP.pi, K)
    S = .5 * t + 1.2 + SP.sin(t)
    # S = SP.random.randn(N,K) # True Latent Input    
    W = SP.random.randn(K, D)  # 

    Y = SP.dot(S, W)  # Linear Output

    Y += 0.2 * SP.random.randn(N, D)

    # use "standard PCA"
    # [Spca,Wpca] = gplvm.PCA(Y,K)
    pc = PCA(Y)
    Spca = pc.project(Y, K)
    
    # reconstruction
    # Y_ = SP.dot(Spca,Wpca.T)

    if 1:
        # use linear kernel
        covariance = linear.LinearCF(n_dimensions=K)
        hyperparams = {'covar': SP.log([2] * (K))}
    if 0:
        # use ARD kernel
        covariance = se.SqexpCFARD(n_dimensions=K)
        hyperparams = {'covar': SP.log([1] * (K + 1))}

    # initialization of X at arandom
    X0 = SP.random.randn(N, K)
    # X0 = Spca
    hyperparams['x'] = copy.deepcopy(X0)

    priors = {'x': [lnpriors.lnFobgp, [10000]]}

    # standard Gaussian noise
    likelihood = GaussLikISO()
    hyperparams['lik'] = SP.log(SP.sqrt([0.5]))
    g = gplvm.GPLVM(covar_func=covariance, likelihood=likelihood, x=copy.deepcopy(X0), y=Y, gplvm_dimensions=SP.arange(X0.shape[1]))
    
    # specify optimization bounds:

    bounds = {}
    bounds['lik'] = SP.array([[-10., 10.]])
    hyperparams['x'] = X0

    print "running standard gplvm"
    Ifilter = {}
    Ifilter['covar'] = SP.array([False])
    Ifilter['lik'] = SP.ones(hyperparams['lik'].shape, dtype='bool')
    Ifilter['x'] = SP.ones(hyperparams['x'].shape, dtype='bool')

    Ifilter = None
    
    [opt_hyperparams, opt_lml2] = opt_hyper(g, copy.deepcopy(hyperparams),
                                            messages=True, gradcheck=False,
                                            priors=priors, bounds=bounds,
                                            Ifilter=Ifilter)

    Sgplvm = opt_hyperparams['x']
    
    K_learned = covariance.K(opt_hyperparams['covar'], Sgplvm)
    
    vmax = None  # max(S.max(), Sgplvm.max(), Spca.max())
    vmin = None  # min(S.min(), Sgplvm.min(), Spca.min())
    cmap = pylab.get_cmap('gray')
    
    pylab.ion()
    ax1 = pylab.subplot(131)
    pylab.title("Truth")
    pylab.imshow(SP.dot(S, S.T), cmap=cmap, interpolation='bilinear', vmin=vmin, vmax=vmax, aspect='auto')
    pylab.xlabel('Sample')
    pylab.ylabel('Sample')
    #pylab.yticks(range(N))
    #pylab.xticks(range(N))
    ax2 = pylab.subplot(132, sharex=ax1, sharey=ax1)
    pylab.title("GPLVM")
    pylab.imshow(K_learned, cmap=cmap, interpolation='bilinear', vmin=vmin, vmax=vmax, aspect='auto')
#    pylab.ylabel('Latent dimension')
    pylab.xlabel('Sample')
    pylab.setp( ax2.get_yticklabels(), visible=False)
    #pylab.yticks([])
    #pylab.xticks(range(N))
    ax3 = pylab.subplot(133, sharex=ax1, sharey=ax1)
    pylab.title("PCA")
    pylab.imshow(SP.dot(Spca, Spca.T), cmap=cmap, interpolation='bilinear', vmin=vmin, vmax=vmax, aspect='auto')
#    pylab.ylabel('Latent dimension')
    pylab.xlabel('Sample')
    pylab.setp( ax3.get_yticklabels(), visible=False)
    #pylab.xticks(range(N))
    #pylab.yticks([])
    # pylab.colorbar()
    pylab.subplots_adjust(left=.1, bottom=.1, right=.9, top=.9,
                          wspace=.05, hspace=None)

    # print opt_hyperparams['x']
#    import ipdb;ipdb.set_trace()
    return g, hyperparams, opt_hyperparams, S, Spca, Sgplvm

if __name__ == '__main__':
    pylab.close('all')
    g, hyperparams, opt_hyperparams, S, Spca, Sgplvm = run_demo()
    pearson_pca = SP.zeros((K, K))
    pearson_gplvm = SP.zeros((K, K))
    import scipy.stats
    
    for k in xrange(K):
        for kprime in xrange(K):
            pearson_pca[k, kprime] = scipy.stats.pearsonr(S[:, k], Spca[:, kprime])[0]
            pearson_gplvm[k, kprime] = scipy.stats.pearsonr(S[:, k], Sgplvm[:, kprime])[0]
            
    print "PCA correlations:\t", SP.absolute(pearson_pca).max(0)
    print "GPLVM correlations:\t", SP.absolute(pearson_gplvm).max(0)
    
    X = opt_hyperparams['x']

#    M = SP.array([[scipy.stats.pearsonr(X[:, i], X[:, j])[0] for i in range(X.shape[1])] for j in range(X.shape[1])])

#    PL.ion()
#    PL.matshow(PL.absolute(M))
#    PL.colorbar()
#    PL.show()
#     calc prior:
#    x = X
#    LL = 0
#    for i in range(x.shape[1]):
#        for j in range(x.shape[1]):
#            norm_x_i = NP.sqrt(NP.dot(x[:, i:i + 1].T, x[:, i:i + 1]))
#            norm_x_j = NP.sqrt(NP.dot(x[:, j:j + 1].T, x[:, j:j + 1]))
#            inner = (NP.dot(x[:, i:i + 1].T, x[:, j:j + 1])[0, 0])
#            LL += inner / (norm_x_i * norm_x_j)
#    print SP.absolute(M).mean()
#    print LL

