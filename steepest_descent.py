from math import log
import numpy as np
from numpy import linalg as nm  

'''
@authors: Robert Lewis, Joseph Palazzo
CSCI-416

'''

class Quadratic(object):
	def f(self, x):
		return 0.5 * (x[0]**2 + 10*x[1] **2)
		
	def g(self, x):
		return np.array ([x[0], 10*x[1]]).T

#The negative log-likelihood for logistic regression
class Logistic_Regression(object):

	def __init__(self, X_train, y_train, fit_intercept = True, lr=0.01, num_iter=100000, verbose = False):
		#This constructor stores the training data and also prepends a column
		#of ones. Each row of X_traincorresponds to a single training case.
		self.num_cases = X_train.shape[0]
		e = np.ones((self.num_cases, 1))
		self.X_train = np.append(e, X_train, axis =1)
		self.y_train = y_train
		self.lr = lr
		self.num_iter = num_iter
		self.verbose = verbose
		
	def sigmoid(self, z):
		return 1 / (1 + np.exp(-z))
		
	def f(self, w):
		#Return the negative log-likelihood. Remember that we wish to 
		#maximize the log-log-likelihood, so we minimize its negative.
		z = self.X_train.dot(w)
		p = 1 / (1 + np.exp(-z))
		log_likelihood = 0
		
		for i in range(0, self.num_cases):
			if (y_train[i] ==1):
				log_likelihood -= log(p[i])
			else:
				log_likelihood -= log(1 - p[i])
				
		return log_likelihood

	def g(self, w):
	
		w = np.zeros(self.X_train.shape[1])
		z = self.X_train.dot(w)
		p = 1 / (1 + np.exp(-z))
		
		gradient = np.dot(self.X_train.T, p)

		for i in range(0, gradient.size):
			if y_train[i] == 1:
				break;
			else:
				gradient[i] -= gradient[i] * .01
		return gradient
		
def condition(armijo_factor, alpha, g_c, abs_grad_tol, rel_grad_tol, f_c, x_t, 
	x_c, abs_stepsize_tol, rel_stepsize_tol):

	cond_0 = obj.f(x_t) < obj.f(x_c) - armijo_factor * alpha * nm.norm(g_c, 2)
	# small gradient in an absolute sense
	cond_1 = (nm.norm(g_c) < abs_grad_tol)

	# Gradient small relative to the magnitude of f.
	cond_2 = (nm.norm(g_c) < rel_grad_tol * max(abs(f_c), 1))

	# small step in an absolute sense
	cond_3 = (nm.norm(x_t - x_c) < abs_stepsize_tol)

	# samll step relative to magnitude of x_c
	cond_4 = (nm.norm(x_t - x_c) <(rel_stepsize_tol * max(nm.norm(x_c), 1)))

	return (cond_0 and cond_1 and cond_2 and cond_3 and cond_4)
		
#Steepest descent for minimization using a linesearch.
def gd0(obj, x0, use_sgd=False, max_it=100,
	abs_grad_tol=1.0e-04, rel_grad_tol=1.0e-04,
	abs_stepsize_tol=1.0e-06, rel_stepsize_tol=1.0e-06,
	armijo_factor=1.0e-04):
	#x0 is the starting point for the minimization, as a numpy column vector.
	x_c = x0
	f_c = obj.f(x_c)
	g_c = obj.g(x_c)


	for it in range(0, max_it):
		#Try the Cauchy step.
		alpha = 1
		x_t = x_c - alpha*g_c
		f_t = obj.f(x_t)
		
		#Perform the linesearch if needed.
		while(condition(armijo_factor, alpha, g_c, abs_grad_tol, rel_grad_tol, f_c, x_t, x_c, 
		abs_stepsize_tol, rel_stepsize_tol)):
			
			alpha /= 2
			x_t = x_c - alpha*g_c
			f_t = obj.f(x_t)
				
		#Accept the new iterate.
		x_c = x_t
		f_c = f_t
		g_c = obj.g(x_c)

	return x_c


if(__name__ == '__main__'):
	np.random.seed(42)

	obj = Quadratic()
	x0 = np.array([2, 3]).T

	from sklearn import datasets


	# Load the diabetes dataset
	diabetes = datasets.load_diabetes()


	# Use only one feature
	X = diabetes.data[:, np.newaxis, 2]

	# Split the data into training/testing sets
	X_train = X[:-20]
	X_test = X[-20:]

	# Split the targets into training/testing sets
	y_train = diabetes.target[:-20]
	y_test = diabetes.target[-20:]

	
	#Check the gradient at x0.
	f0 = obj.f(x0)
	h = 1.0e-05
	dx = 2 * (np.random.random_sample(x0.shape) - 0.5)
	f1 = obj.f(x0 + h*dx)
	fd = (f1 - f0)/h
	g_c = obj.g(x0)
	an = np.dot(g_c, dx)

	print('The following should be close.')
	print('finite difference: ', fd)
	print('analytical: ', an)

	#Minimize the objective.
	x = gd0(obj, x0, max_it=100)
	print('solution: ', x)

	#Logistic_Regression
	#X_train = np.array([(+1, +1), (+1, -1), (-1, +1), (-1, -1)])
	#y_train = np.array([1, 1, 0, 0])
	obj = Logistic_Regression(X_train, y_train)

	w0 = np.array([1,1]).T
	#print ('f(w0):', np.exp(-obj.f(w0)))

	#obj = Logistic_Regression(X_train, y_train)
	w = gd0(obj, w0, max_it=2000)
	print('solution: ', w)
	#print('f(x):', np.exp(-obj.f(w)))
