import numpy as np
def logistic_func(theta, x):                  # function value
    return 1./(np.exp(-x.dot(theta)) + 1.)
logistic_func_all = logistic_func
lf = logistic_func
def cross_entropy_loss(theta, X, y):        # MSE value of current regression
    return np.sum(y*np.log(lf(theta, X)) + (1 - y)*np.log(1. - lf(theta, X)))
def grad_cross_entropy_loss(theta, X, y):
    return -X.dot(y - lf(theta, X))
print(cross_entropy_loss(np.ones(2), np.ones((2, 2)), np.ones(2)))