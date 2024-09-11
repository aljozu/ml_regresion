import numpy as np
import matplotlib.pyplot as plt
import time

class PolynomialRegression:
    def __init__(self, degree, learning_rate, iterations, alpha=0.1):
        self.degree = degree
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.alpha = alpha  # Regularization strength

    def transform(self, X):
        X_transform = np.ones((len(X), self.degree + 1))
        for j in range(1, self.degree + 1):
            X_transform[:, j] = X ** j
        return X_transform

    def normalize(self, X):
        X_mean = np.mean(X[:, 1:], axis=0)
        X_std = np.std(X[:, 1:], axis=0)
        X[:, 1:] = (X[:, 1:] - X_mean) / X_std
        return X

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.m = self.X.shape[0]
        
        X_transform = self.transform(self.X)
        X_normalize = self.normalize(X_transform)

        XtX = np.dot(X_normalize.T, X_normalize)  # X^T * X
        Xty = np.dot(X_normalize.T, self.y)        # X^T * y
        
        self.W = np.linalg.solve(XtX, Xty)  # Solve for weights

    def fit_with_l1(self, X, y):
        self.X = X
        self.y = y
        self.m = self.X.shape[0]

        X_transform = self.transform(self.X)
        X_normalize = self.normalize(X_transform)

        # Initialize weights
        self.W = np.zeros(X_normalize.shape[1])

        # Coordinate Descent
        for _ in range(self.iterations):
            for j in range(len(self.W)):
                if j == 0:  # Bias term, no regularization
                    self.W[j] = np.sum(self.y * X_normalize[:, j]) / np.sum(X_normalize[:, j] ** 2)
                else:
                    rho_j = np.dot(X_normalize[:, j], self.y - np.dot(X_normalize, self.W) + self.W[j] * X_normalize[:, j])
                    self.W[j] = np.sign(rho_j) * max(0, abs(rho_j) - self.alpha) / np.sum(X_normalize[:, j] ** 2)

    def fit_with_l2(self, X, y):
        self.X = X
        self.y = y
        self.m = self.X.shape[0]

        X_transform = self.transform(self.X)
        X_normalize = self.normalize(X_transform)

        XtX = np.dot(X_normalize.T, X_normalize)  # X^T * X
        Xty = np.dot(X_normalize.T, self.y)        # X^T * y

        # Add L2 regularization term
        reg_term = self.alpha * np.eye(XtX.shape[0])
        reg_term[0, 0] = 0  # No regularization for the bias term

        self.W = np.linalg.solve(XtX + reg_term, Xty)  # Solve for weights with L2 regularization

    def predict(self, X):
        X_transform = self.transform(X)
        X_normalize = self.normalize(X_transform)
        return np.dot(X_normalize, self.W)

file_path = "datos.txt"
with open(file_path, 'r') as file:
    data = file.read()
Y = np.array([float(x.strip()) for x in data.split(',')])
X = np.arange(len(Y))

# model training
#model = PolynomialRegression(degree=80, learning_rate=0.01, iterations=1000, alpha=0.1)

deg = 10
model_no_reg = PolynomialRegression(degree=deg, learning_rate=0.01, iterations=1000, alpha=0.1)
model_ridge = PolynomialRegression(degree=deg, learning_rate=0.01, iterations=1000, alpha=0.1)
model_lasso = PolynomialRegression(degree=deg, learning_rate=0.01, iterations=1000, alpha=0.1)

model_no_reg.fit(X, Y)  # Entrenar el modelo
Y_no_reg = model_no_reg.predict(X)  # Hacer predicciones



model_ridge.fit_with_l1(X, Y)  # Entrenar con L1 regularización
Y_ridge = model_ridge.predict(X)  # Hacer predicciones con Ridge


model_lasso.fit_with_l2(X, Y)  # Entrenar con L2 regularización
Y_lasso = model_lasso.predict(X)  # Hacer predicciones con Lasso



# Visualization
plt.scatter(X, Y, color='blue', label='Actual')
#plt.plot(X, Y_no_reg, color='purple', label='no-reg')
#plt.plot(X, Y_ridge, color='green', label='l1')
plt.plot(X, Y_lasso, color='red', label='l2')
plt.title('Polynomial Regression - Degree ' + str(deg))
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.savefig("l2-10.svg", format='svg')
plt.show()
