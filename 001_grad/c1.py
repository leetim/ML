import numpy as np

def linear_func(theta, x):
    return np.dot(theta, x)

def linear_func_all(theta, X):
    return np.dot(theta, np.transpose(X))

def mean_squared_error(theta, X, y):
    return np.average((y - linear_func_all(theta, X))**2)

def grad_mean_squared_error(theta, X, y):
    return -2.0/len(y)*np.dot(y - linear_func_all(theta, X), X)

def fit_linear_regression(X, y):
    m, n = np.shape(X)
    theta = (np.random.rand(n)- 0.5)*40
    alpha0 = 1e-5
    sq_sum = np.zeros(n)
    eps = 0.0001
    while True:
        g = grad_mean_squared_error(theta, X, y)
        if g.dot(g) < eps:
            break
        alpha = alpha0/(np.sqrt(sq_sum + eps) + 1)
        mv = g*alpha
        theta -= mv
        # print("Theta:")
        # print(theta)
        # print("Grad:")
#        print(g.dot(g))

        sq_sum += g**2
    return theta

#X1 = 20*np.random.randn(40000, 5)
# print(X1)
#f = lambda X: 70 + np.sum(np.array([1, 4, 5, 2, 3])*X)
#Y = np.array([f(x) for x in X1]) + np.random.randn(40000)*4
# print(Y)
#print(fit_linear_regression(X1, Y))
# print(np.any(np.array([1, 3]) < 2))

# print(np.array([[1, 2, 3], [1, 2, 3]]) - np.array([1, 1, 1])