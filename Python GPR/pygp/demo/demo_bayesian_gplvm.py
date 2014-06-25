'''
Created on 5 Apr 2012

@author: maxz
'''
from pygp.covar import linear, combinators, noise, bias, se
from pygp.gp.bayesian_gplvm import mean_id, vars_id, ivar_id
from pygp.gp import bayesian_gplvm
import numpy
from pygp.optimize.optimize_base import opt_hyper
import copy
import pylab

def random_inputs(N,M,Qlearn,D,covar,noise=.5,Qreal=None):
    if Qreal is None:
        Qreal = max(1,Qlearn/4)
    # Signals are variations of sin and cos
    x = numpy.linspace(0,2*numpy.pi,N)
    signals = [lambda x: numpy.sin(x)]#, lambda x: numpy.cos(x), ]
    signals = numpy.array(map(lambda f: f(x), signals)).T
#    signals = numpy.random.randn(N,Qreal)

    inputs = numpy.zeros((N, Qlearn))
    inputs[:,:signals.shape[1]] = signals
    inputs += .3*numpy.random.randn(*inputs.shape)
    
    W = 1+.3*numpy.random.randn(Qlearn,D)
    # map to output space:
    Y = numpy.dot(inputs, W)
    # Add noise
    Y += noise*numpy.random.randn(N, D)
    
    mean = numpy.random.randn(*inputs.shape) + inputs
    variance = .5 * numpy.ones(mean.shape)
    inducing_variables = numpy.random.rand(M, Qlearn) * 2*numpy.pi #numpy.array([numpy.linspace(0,2*numpy.pi, M)]*Qlearn).reshape(-1,Qlearn)        
    
    hyperparams = {'covar':numpy.log(numpy.sqrt(numpy.repeat(.5, covar.get_number_of_parameters()))),
                   'beta':numpy.array([.5]),
                   mean_id:mean, # N x Qlearn
                   vars_id:numpy.log(numpy.sqrt(variance)), # N x Qlearn
                   ivar_id:inducing_variables # M x Qlearn
                   }
    
    return hyperparams, inputs, Y

if __name__ == '__main__':
    pylab.ion()
    pylab.close('all')

    smaller = []
    n_runs = 1
    for i in xrange(n_runs):
        # create data (simulation)
        N = 45
        Qreal = 2
        M = Qreal+1
        Qlearn = 6
        D = 20
        # Input time steps
        
        lin = linear.LinearCFPsiStat(Qlearn)
        noi = noise.NoiseCFISOPsiStat()
        bia = bias.BiasCFPsiStat()
        covar = combinators.SumCFPsiStat((lin,bia,noi))
        hyperparams, signals, Y = random_inputs(N, M, Qlearn, D, covar, Qreal = Qreal)
        
        g = bayesian_gplvm.BayesianGPLVM(y=Y, covar_func=covar, gplvm_dimensions=None, n_inducing_variables=M)
        
    #    psi_0 = covar.psi_0(hyperparams['covar'], mean, variance, inducing_variables)
    #    psi_1 = covar.psi_1(hyperparams['covar'], mean, variance, inducing_variables)
    #    psi_2 = covar.psi_2(hyperparams['covar'], mean, variance, inducing_variables)
    #    

        #go through all hyperparams and build bound array (flattened)
     

        # learn latent variables
        skeys = numpy.sort(hyperparams.keys())
        param_struct = dict([(name,hyperparams[name].shape) for name in skeys])
        
        bounds = {}#{'beta': [(numpy.log(1./1E-3))]}
        
        hyper_for_optimizing = copy.deepcopy(hyperparams)
        #hyper_for_optimizing[vars_id] = numpy.log(hyper_for_optimizing[vars_id])
        opt_hyperparams, opt_lml = opt_hyper(g, hyper_for_optimizing, bounds=bounds, maxiter=10000, messages=True)
        #opt_hyperparams[vars_id] = numpy.exp(opt_hyperparams[vars_id])
#        opt_covar = numpy.exp(2*opt_hyperparams['covar'])
#        param_diffs = numpy.abs(numpy.subtract.outer(opt_covar,opt_covar)).round(5)
#        while (param_diffs < .1).all():
##            pylab.close('all')
##            pylab.bar(range(Qlearn),numpy.exp(2*opt_hyperparams['covar']))
##            pylab.figure()
##            opt_covar = numpy.exp(2*opt_hyperparams['covar'])
##            s = numpy.arange(0,Qlearn)[opt_hyperparams['covar']==opt_hyperparams['covar'].max()][0]      
##            for s in xrange(Qlearn):
##                pylab.plot(signals[:,s],'--o')
##                pylab.errorbar(range(N), opt_hyperparams[mean_id][:,s], yerr=numpy.sqrt(opt_hyperparams[vars_id])[:,s], ls='-', alpha=1./numpy.sqrt(opt_hyperparams[vars_id])[:,s].max())
#            yn = 'y' #raw_input("want to restart? [y]/n: ")
#            if len(yn) < 1:
#                yn = 'y'
#            if 'y' in yn:
#                print "restarting..." 
#                hyperparams, signals_new, Y_new = random_inputs(N, M, Qlearn, D)
#                g = bayesian_gplvm.BayesianGPLVM(y=Y, covar_func=covar, gplvm_dimensions=None, n_inducing_variables=M)
#                hyper_for_optimizing = copy.deepcopy(hyperparams)
#                #hyper_for_optimizing[vars_id] = numpy.log(hyper_for_optimizing[vars_id])
#                opt_hyperparams, opt_lml = opt_hyper(g, hyper_for_optimizing, bounds=bounds, maxiter=1000, messages=True)
#                #opt_hyperparams[vars_id] = numpy.exp(opt_hyperparams[vars_id])
#                opt_covar = numpy.exp(2*opt_hyperparams['covar'])
#                param_diffs = numpy.abs(numpy.subtract.outer(opt_covar,opt_covar))
#            else:
#                break

        g.plot(opt_hyperparams, marker='x', color='b')
        
        Ystar_mean, Ystar_variance = g.predict(opt_hyperparams, opt_hyperparams[mean_id])
        
        pylab.figure()
        pylab.subplot(121)
        pylab.imshow(numpy.dot(signals, numpy.ones((Qlearn, D))), aspect='auto', cmap='jet', interpolation='nearest')
        pylab.title("Noise Free Training Outputs")
        pylab.subplot(122)
        pylab.imshow(Ystar_mean, aspect='auto', cmap='jet', interpolation='nearest')
        pylab.title("Predicted Training Outputs")
        

        
#        pylab.close('all')
#        pylab.bar(range(Qlearn),numpy.exp(2*opt_hyperparams['covar']))
#        pylab.figure()
#        opt_covar = numpy.exp(2*opt_hyperparams['covar'])
#        plots=[]
#        for s in xrange(Qlearn):
#            plots.append(pylab.plot(signals[:,s],'--o',alpha=.5))
#        alpha = 1#(1-numpy.sqrt(opt_hyperparams[vars_id])[:,s].max())**2
#        s = numpy.arange(0,Qlearn)[opt_hyperparams['covar']==opt_hyperparams['covar'].max()][0]
#        pylab.errorbar(range(N), opt_hyperparams[mean_id][:,s], yerr=numpy.sqrt(opt_hyperparams[vars_id])[:,s], ls='-', alpha=alpha, color='g')#plots[s][0].get_color())
#
#        pylab.show()
        #opt_x = optimize.fmin(f, x, maxiter=1000, callback=lambda x: sys.stdout.write("current dist: %s\n"%(x.reshape(N,Qlearn)-signals))).reshape(N,Qlearn)   
        
#        pylab.figure()
#        
#        opt_covar = numpy.exp(2*opt_hyperparams['covar'])
#        s = numpy.arange(0,Qlearn)[opt_hyperparams['covar']==opt_hyperparams['covar'].max()][0]      
#        for s in xrange(Qlearn):
#            pylab.plot(signals[:,s],ls='o')
#            pylab.errorbar(range(N), opt_covar[s] * opt_hyperparams[mean_id][:,s], yerr=numpy.sqrt(2*opt_hyperparams[vars_id])[:,s], ls='-')
#        
        # predict data
        
                
#    print "True g.LML < random g.LML: %.1f%% out of %i runs" % (100.*(numpy.array(smaller).sum()/(1.*n_runs)), n_runs)

#        
