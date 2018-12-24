import numpy as np

def fit_linear_regression(X, y):
    A = np.dot(np.transpose(X), X)
    B = np.dot(y, X)
    return np.linalg.solve(A, B)

# X1 = 20*np.random.randn(40000, 5)
# f = lambda X: 70 + np.sum(np.array([1, 4, 5, 2, 3]*X))
# Y = np.array([f(x) for x in X1]) + np.random.randn(40000)*4
# # print(Y)
# print(fit_linear_regression(X1, Y))

# print(np.array([[1, 2, 3], [1, 2, 3]]) - np.array([1, 1, 1]))
