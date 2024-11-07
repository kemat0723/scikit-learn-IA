import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Cargar el conjunto de datos California Housing
California = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(California.data, California.target, test_size=0.2, random_state=42)

# Crear el modelo de regresion lineal
model = LinearRegression()

# Entrenar el modelo
model.fit(X_train, y_train)

# Predecir los precios de las viviendas para los datos de prueba
y_pred = model.predict(X_test)

# Calcular el error cuadratico medio (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Error cuadratico medio (MSE):", mse)
