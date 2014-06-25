'''
Created on 10 Sep 2012

@author: maxz
'''
import numpy
import pylab


class PCA(object):
    """
    PCA module with automatic primal/dual usage.
    """
    def __init__(self, X):
        self.mu = X.mean(0)
        self.sigma = X.std(0)
        
        X = self.center(X)
        
        #self.X = input
        if X.shape[0] >= X.shape[1]:
            #print "N >= D: using primal"
            self.eigvals, self.eigvectors = self._primal_eig(X)
        else:
            #print "N < D: using dual"
            self.eigvals, self.eigvectors = self._dual_eig(X)
        self.sort = numpy.argsort(self.eigvals)[::-1]
        self.eigvals = self.eigvals[self.sort]
        self.eigvectors = self.eigvectors[:,self.sort]
        self.fracs = self.eigvals/self.eigvals.sum()
        self.Q = self.eigvals.shape[0]

    def center(self,X):
        X = X-self.mu
        X = X/self.sigma
        return X
    
    def _primal_eig(self, X):
        return numpy.linalg.eigh(numpy.cov(X.T)) #X.T.dot(X)
    
    def _dual_eig(self, X):
        dual_eigvals, dual_eigvects = numpy.linalg.eigh(numpy.cov(X)) #X.dot(X.T))
        relevant_dimensions = numpy.argsort(numpy.abs(dual_eigvals))[-X.shape[1]:]
        eigvals = dual_eigvals[relevant_dimensions]
        eigvects = dual_eigvects[:,relevant_dimensions]
        eigvects = (1./numpy.sqrt(X.shape[0]*numpy.abs(eigvals)))*X.T.dot(eigvects)
        eigvects /= numpy.sqrt(numpy.diag(eigvects.T.dot(eigvects)))
        return eigvals, eigvects
        
    def project(self, X, Q=None):
        """
        Project X into PCA space, defined by the Q highest eigenvalues.
        
        Y = X dot V
        """        
        if Q is None:
            Q = self.Q
        if Q > X.shape[1]:
            raise IndexError("requested dimension larger then input dimension")
        X = self.center(X)
        return X.dot(self.eigvectors[:,:Q])

    def plot_fracs(self, Q=None, ax=pylab.gca()):
        if Q is None:
            Q = self.Q
        ticks = numpy.arange(Q)
        ax.bar(ticks-.4,self.fracs[:Q])
        ax.set_xticks(ticks, map(lambda x: r"${}$".format(x), ticks+1))
        ax.set_ylabel("Eigenvalue fraction")
        ax.set_xlabel("PC")
        ax.set_ylim(0,ax.get_ylim()[1])
        ax.set_xlim(ticks.min()-.5, ticks.max()+.5)
        try:
            pylab.tight_layout()
        except:
            pass
        
    def plot_2d(self, X, labels=None, s=20, c='b', marker='+', ax=pylab.gca(), ulabels=None, **kwargs):
        if labels is None:
            labels = numpy.zeros(X.shape[0])
        if ulabels is None:
            ulabels = numpy.unique(labels)
        nlabels = len(ulabels)
        if len(c) != nlabels:
            from matplotlib.cm import jet # @UnresolvedImport
            c = [jet(float(i)/nlabels) for i in range(nlabels)]
        X = self.project(X,2)
        kwargs.update(dict(s=s))
        plots = list()
        for i,l in enumerate(ulabels):
            kwargs.update(dict(color=c[i], marker=marker[i%len(marker)]))
            plots.append(ax.scatter(*X[labels==l,:].T, label=str(l), **kwargs))
        ax.set_xlabel(r"PC$_1$")
        ax.set_ylabel(r"PC$_2$")
        try:
            pylab.tight_layout()
        except:
            pass
        return plots
