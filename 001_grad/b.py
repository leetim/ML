import numpy as np

def linear_func(theta, x):
    return np.dot(theta, x)

def linear_func_all(theta, X):
    return np.dot(theta, np.transpose(X))

def mean_squared_error(theta, X, y):
    return np.average((y - linear_func_all(theta, X))**2)

def grad_mean_squared_error(theta, X, y):
    return -2.0/len(y)*np.dot(y - linear_func_all(theta, X), X)

# X = np.array([[1,2],[3,4],[4,5]])

# theta = np.array([5, 6])

# y = np.array([1, 2, 1])

# print(linear_func_all(theta, X)) # —> array([17, 39, 50])

# print(mean_squared_error(theta, X, y)) # —> 1342.0

# print(grad_mean_squared_error(theta, X, y)) # —> array([215.33333333, 283.33333333])