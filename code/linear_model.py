import numpy as np
from numpy.linalg import solve
import findMin
import sys
from scipy.optimize import approx_fprime


# Original Least Squares
class LeastSquares:
    # Class constructor
    def __init__(self):
        pass

    def fit(self,X,y):
        # Solve least squares problem

        a = np.dot(X.T, X)
        b = np.dot(X.T, y)
        self.w = solve(a, b)
        # print(self.w)

    def predict(self, Xhat):

        w = self.w
        yhat = np.dot(Xhat, w)
        # print(yhat.shape)
        return yhat

# Least Squares where each sample point X has a weight associated with it.
class WeightedLeastSquares:

    def __init__(self):
        pass

    def fit(self,X,y,z):


        # X.T*W*X*Xhat = X.TWy
        a = np.dot(X.T, z)
        a = np.dot(a, X)

        b = np.dot(X.T, z)
        b = np.dot(b, y)
        self.w = solve(a, b)


    def predict(self,Xhat):

        w = self.w

        yhat = np.dot(Xhat, w)
        # print(yhat.shape)
        return yhat

class LinearModelGradient:

    def __init__(self):
        pass

    def fit(self,X,y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros((d, 1))

        # check the gradient
        estimated_gradient = approx_fprime(self.w, lambda w: self.funObj(w,X,y)[0], epsilon=1e-6)
        implemented_gradient = self.funObj(self.w,X,y)[1]
        if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
            print('User and numerical derivatives differ: %s vs. %s' % (estimated_gradient, implemented_gradient));
        else:
            print('User and numerical derivatives agree.')

        self.w, f = findMin.findMin(self.funObj, self.w, 100, X, y)

    def predict(self,Xtest):

        w = self.w
        yhat = Xtest*w
        return yhat

    def funObj(self,w,X,y):


        # Calculate the function value
        # f = (1/2)* np.sum((X.dot(w)-y)**2)
        f = (1/2)* np.sum(np.log(np.exp(np.dot(X,w)-y)+np.exp(y-np.dot(X,w))))
        # print(X.shape)
        # print(y.shape)


        # Calculate the gradient value
        # g = X.T.dot(X.dot(w) - y)
        # print(np.exp(np.dot(X,w)-y))
        # g = (1/2)* np.sum((w*np.exp(np.dot(X,w)-y)-w*np.exp(y-np.dot(X,w)))/(np.exp(np.dot(X,w)-y)+np.exp(y-np.dot(X,w))))

        g = 0
        for i in range(X.shape[0]):
            g += (1/2)* (X[i]*(np.exp(w*X[i]-y[i])-np.exp(y[i]-w*X[i])))/(np.exp(w*X[i]-y[i])+np.exp(y[i]-w*X[i]))
        # print(g)
        return f,g
