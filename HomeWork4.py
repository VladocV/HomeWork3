import numpy as np

#1, a:
A = np.array([[1,54],[1,6],[1,7]])
print('1, a:\n',A.T, sep='')

#1, b:
A = np.array([[1, 7, 8],[4, 2, 9],[5, 6, 3]])
print('1, b:\n',A.T, sep='')

#2, a:
#число столбцов первой матрицы равно числу строк второй - False

#2, b:
A = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
B = np.array([[1, 1, 0],[0, 1, 1],[1, 0, 1]])
print('2, b:\n',np.dot(A, B))

#3:
def weights_func(X, Y, lr=0.1, epochs=10):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    w0 = 0
    for i in range(epochs):
        Y_pred = np.dot(X, weights) + w0
        print(X.T)
        weights += 2 * lr * (1 / n_samples) * X.T @ (Y - Y_pred)
        w0 += 2 * lr * (1 / n_samples) * np.sum(Y - Y_pred)
    return weights, w0

X = [[1,145],[2,163],[3,240],[3,350],[4,421],[4,397],[5,620]]
Y = [[80],[170],[100],[220],[200],[270],[500]]
X = np.array(X)
Y = np.array(Y)
X = (X - X.mean(axis=0)) / X.std(axis=0)
X_train = X[:6, :]
X_test = X[6:, :]
Y_train = Y[:6]
Y_test = Y[6:]

weight, w0 = weights_func(X_train, Y_train, epochs=100)
