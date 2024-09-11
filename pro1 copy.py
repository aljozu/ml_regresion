import numpy as np
import matplotlib.pyplot as plt

class PolynomialRegression:
    def __init__(self, degree, learning_rate, iterations):
        self.degree = degree
        self.learning_rate = learning_rate
        self.iterations = iterations

    def transform(self, X):
        # Initialize the design matrix with ones
        X_transform = np.ones((len(X), self.degree + 1))
        
        # Fill in polynomial terms
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
model = PolynomialRegression(degree=100, learning_rate=0.01, iterations=500)
model.fit(X, Y)

# Prediction on training set
Y_pred = model.predict(X)

# Visualization
plt.scatter(X, Y, color='blue', label='Actual')
plt.plot(X, Y_pred, color='orange', label='Predicted')
plt.title('Polynomial Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()


import timeit

# Entrenamiento del modelo
def entrenar_modelo():
    model_no_reg.fit(X, Y)  # Entrenar el modelo

# Medir el tiempo de entrenamiento
tiempo_entrenamiento = timeit.timeit(entrenar_modelo, number=100)
print(f"Tiempo promedio de entrenamiento: {tiempo_entrenamiento / 100} segundos")

# Predicción del modelo
def predecir_modelo():
    model_no_reg.predict(X)  # Hacer predicciones

# Medir el tiempo de predicción
tiempo_prediccion = timeit.timeit(predecir_modelo, number=100)
print(f"Tiempo promedio de predicción: {tiempo_prediccion / 100} segundos")


# Entrenamiento y predicción con Ridge (L1 regularización)
def entrenar_modelo_ridge():
    model_ridge.fit_with_l1(X, Y)  # Entrenar con L1 regularización

def predecir_modelo_ridge():
    model_ridge.predict(X)  # Hacer predicciones con Ridge

# Medir el tiempo de entrenamiento y predicción para Ridge
tiempo_entrenamiento_ridge = timeit.timeit(entrenar_modelo_ridge, number=100)
tiempo_prediccion_ridge = timeit.timeit(predecir_modelo_ridge, number=100)

print(f"Tiempo promedio de entrenamiento (Ridge L1): {tiempo_entrenamiento_ridge / 100} segundos")
print(f"Tiempo promedio de predicción (Ridge): {tiempo_prediccion_ridge / 100} segundos")


# Entrenamiento y predicción con Lasso (L2 regularización)
def entrenar_modelo_lasso():
    model_lasso.fit_with_l2(X, Y)  # Entrenar con L2 regularización

def predecir_modelo_lasso():
    model_lasso.predict(X)  # Hacer predicciones con Lasso

# Medir el tiempo de entrenamiento y predicción para Lasso
tiempo_entrenamiento_lasso = timeit.timeit(entrenar_modelo_lasso, number=100)
tiempo_prediccion_lasso = timeit.timeit(predecir_modelo_lasso, number=100)

print(f"Tiempo promedio de entrenamiento (Lasso L2): {tiempo_entrenamiento_lasso / 100} segundos")
print(f"Tiempo promedio de predicción (Lasso): {tiempo_prediccion_lasso / 100} segundos")
