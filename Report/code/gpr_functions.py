#!/usr/bin/python

from numpy import *
from math import *


#COVARIANCE FUNCTION
#Calculates the covariance, i.e. a measure of the similarity between two points 
#using the error squared or Gaussian kernel
def cov_func(x1, x2, M, sig_pow):
        diff = x1 - x2
        temp = dot(diff.transpose(), M)
        k = sig_pow*exp(-0.5*dot(temp, diff))
	#print k
        return k




#CALCULATE COVARIANCE MATRIX K
#Uses the covariance function to calculate the covariance matrix of the multivariate gaussian
#distribution of the training data points.
def calc_K(X, theta):

	#Number of Input features
	m = len(X[0])
	#Number of data points
	n = len(X)

	#Initialise matrix M
	M = zeros((m,m))
	for i in range(m):
		M[i][i] = 1/(theta[i+2]*theta[i+2])
	
	#Initialise then calculate 
	K = zeros((n,n))
	for i in range(n):
        	for j in range(i, n):
        		K[i][j] = cov_func(X[i].transpose(), X[j].transpose(), M, theta[0])
			K[j][i] = K[i][j]
	return K

	
#CALCULATE 'C'
#'C' is just the notation for the covariance matrix K plus a noise term that represents
#jitter in the training data
def calc_C(K, noise_pow):
	a = len(K)
	temp = noise_pow*eye(a)
	C = K + temp
	return C


#PARTIAL DIFFERENTIATE C WITH RESPECT TO HYPERPARAMETERS
#Funtion is necessary to carry out gradient descent. Returns a vector containing the 
#evaluated partial differential of NLML with respect to each of the hyperparameters
def grad_C(theta, X, K, C):
	
	#a is the number of hyperparameters
	a = len(theta)
	#n is the number of data points in the training set, note that K and C are nxn
	n = len(C)
	#m is the number of data features
	m = len(X[0])
	
	
	
	#Initialise gradient of C
	partial_diff_C = zeros((a,n,n))
	
	#Find partial diff with respect to signal power
	partial_diff_C[0] = 2*theta[0]*K
	
	#Find partial diff with respect to the noise power	
	partial_diff_C[1] = 2*theta[1]*eye(n)

	#Find partial diff with respect to the different scale lengths
	for k in range(2,a):
		for i in range(n):
                	for j in range(i, n):
				diff = (X[i][k-2] - X[j][k-2])**2
        			l = theta[k]
				temp = (diff/(l**3))	
        	        	partial_diff_C[k][i][j] = temp*K[i][j]
                        	partial_diff_C[k][j][i] = partial_diff_C[k][i][j]#Note that C and K are symmetric, therefore only need to calculate half the elements 
	
	return partial_diff_C 



#CALCULATE THE GRADIENT OF THE NEGATIVE LOG MARGINAL LIKLIHOOD (NLML) WITH RESPECT TO ONE OF THE FEATURES
#Calculates the partial differential of the NLML with respect to a single hyper parameter
def calc_pdiff_nlml(C, y, pdiff_C):
	
	inv_C = linalg.inv(C)		 
	diff_L = 0.5*(trace(dot(inv_C, pdiff_C)) - dot(y.transpose(), dot(inv_C, dot(pdiff_C, dot(inv_C,y)))))
	return diff_L
	
	
				
#CALCULATE THE NEGATIVE LOG MARGINAL LIKLIHOOD (NLML)
#Calculates the Negative Log Marginal Likelihood
def calc_nlml(C, y):
	n = len(C)
	L = 0.5*(log(linalg.det(C)) + dot(y.transpose(),dot(linalg.inv(C), y)) + n*log(2*pi))
	return L
	

#GRADIENT DESCENT
#Carry out gradient descent to find a minima to optimize the hyper parameters with respect to 
#minimising the NLML. 
def gradient_descent(alpha, init_theta, X, y, error_level, max_iters):
	m = len(init_theta)
	L_hist = ones(max_iters+2)
	pdiff_L = zeros(m)
	theta = init_theta
	j = 0

	K = calc_K(X, theta)
        C = calc_C(K,theta[1])
        L_hist[j] = calc_nlml(C,y)

	successful = True
	while(L_hist[j] > error_level):
		j = j+1
		K = calc_K(X, theta)
		C = calc_C(K,theta[1])
		L_hist[j] = calc_nlml(C,y) 
		pdiffCs = grad_C(theta, X, K, C)	
		for k in range(m):
			p_diffL = calc_pdiff_nlml(C, y, pdiffCs[k])
			theta[k] = theta[k] - alpha*p_diffL 	
			

		if(j>max_iters):
			print("Failure to converge below desired error level after number of iterations =")
			print j
			print("Consider increasing value of alpha to decrease convergence time")
			successful = False
			break

		if(L_hist[j] > L_hist[j-1]):
			print("Failure to converge below required error level, nlml not decreasing with each iteration, local minima reached, consider decreasing alpha")
			print("Number of iterations =")
			print j
			successful = False
			break

	if(successful == True):
		print("Convergence below desired error level successful after number of iterations =")
		print j 
	
	return (theta, successful, L_hist[j], L_hist, j)
		


#CALCULATE K_*
#Calculate the covariance of the input data point with each the points in the training set
def calc_K_star(x, X, theta):
	
	#Number of Input features
        m = len(X[0])
        #Number of data points
        n = len(X)
        #Initialise matrix M
        M = zeros((m,m))
        for i in range(m):
                M[i][i] = 1/(theta[i+2]*theta[i+2])	
	K_star = zeros(n)
	
	for j in range(n):
		K_star[j] = cov_func(x.transpose(), X[j].transpose(), M, theta[0]) 
	return K_star

#CALCULATE EXPECTED THERMOSTAT SETTING
def calc_E_TS(inv_C, K_star, y):
	expected_TS = dot(K_star, dot(inv_C, y))
	return expected_TS

#NORMALIZE INPUT DATA BEFORE CARRYING OUT GRADIENT DESCENT
#Normalisation is necessary becuase otherwise differences in the order of magnitude between
#different features would result in a warped (and much slower) path of descent 
def normalize_data(X):	
	X = X.transpose()
	numfeatures = len(X)
	numdata = len(X[0])
	norm_X = zeros((numfeatures, numdata))	

	for j in range(numfeatures):
		u = mean(X[j])	
		u = u*ones(numdata)
		stan_dev = std(X[j])
		stan_dev
		norm_X[j] = (1/stan_dev)*(X[j] - u)	
	return norm_X.transpose()
	
#NORMALIZE TRAINING AND INPUT DATA ACCORDING TO TRAINING DATA NORMALISATION
#Normalizes the training data and also the input data. Note the training data is normalized 
#first and then the mean and std are taken and used to normalize the input. This ensures the same 
#transformation occurs on the input data as on the training data and therfore that the relationship 
#with the thermostat setting is the same.
def normalize_input(X, x):
	X = X.transpose()
        numfeatures = len(X)
        numdata = len(X[0])
        norm_X = zeros((numfeatures, numdata))
	norm_x = zeros(numfeatures)	

        for j in range(numfeatures):
                u = mean(X[j])
                u_vec = u*ones(numdata)
                stan_dev = std(X[j])
                stan_dev
                norm_X[j] = (1/stan_dev)*(X[j] - u_vec)
		norm_x[j] = (1/stan_dev)*(x[j] - u) 
        return norm_X.transpose(), norm_x
	

#Function used to find the index and value of an element in an
#array that is closest to the argument 'value'. Example application
#is finding closest time to the present time in a database.	
def find_nearest(array,value):
	
	min_index = 0
	min_diff = array[min_index] - value
	diff = zeros(len(array))
	min_value = array[min_index]
	
	for j in range(len(array)):
		diff[j] = abs(array[j] - value)
		if(diff[j] <= min_diff):
			min_index = j
			min_diff = diff[min_index]
			min_value = array[j]
	
	return min_index, min_value, min_diff

#Calculates a 'similarity' metric for two vectors. An example application is in finding entries in the training data base whose
#external conditions best match a new training data point that needs to be swapped in.
def calc_similarity(a, b):
	if(len(a) != len(b)):
		print("Error: cannot calculate similarity between vectors as do not have the same number of elements")
	sum = 0
	
	for j in range(len(a)):
		sum = sum + abs(a[j] - b[j])
	
	return sum
		
	









