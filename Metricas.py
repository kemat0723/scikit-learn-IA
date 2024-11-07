from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifierLinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#Cargar el conjunto de datos Iris
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, 
                                                    iris.target, 
                                                    test_size=0.2, 
                                                    random_state=42)

#crear el clasificador de vecinos más cercanos
clf = KNeighborsClassifier(n_neighbors=3)

#Entrenar el clasificador
clf.fit(X_train, y_train)

predecir las etiquetas para los datos de prueba
y_pred = clf.predict(X_test)

#calcular la precision del clasificador
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del clasificador:", accuracy)

#prediciones del  modelo
y_pred = clf.predict(X_test)

#calcular metricas de evaluacion del modelo
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pre, average='weighted')
print("Precisión:", accuracy)
print("Precisión promedio ponderada:", precision)
print("Precisión promedio ponderada:", recall)
print("Precisión promedio ponderada:", f1)