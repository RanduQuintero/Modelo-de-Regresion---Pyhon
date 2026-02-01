from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score 
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Randu\Desktop\Modelos de machine learning\Housing.csv")

#Limpieza rapida de los datos
#Eliminar duplicados exactos
df = df.drop_duplicates()
#Eliminar filas con valores nulos del df
df = df.dropna()
#Estandarizar los nombres de las columnas
df.columns = df.columns.str.lower() #Minusculas
df.columns = df.columns.str.replace(" ", "_") #Remplazar espacios en blanco
df.columns = df.columns.str.replace("-", "_") #Remplazar guion por guion bajo

"""Hacer un cambio de nombre de las columnas
df = df.rename(columns={
    "bedrooms": "cuartos",
    "bathrooms": "banos", #Evitar la Ñ
    "stories": "pisos",
    "parking": "cochera",
    "price": "precio"
})"""
#One-Hot_Encoding
df = pd.get_dummies(
    df,
    columns=[
        "mainroad",
        "guestroom",
        "basement",
        "hotwaterheating",
        "airconditioning",
        "prefarea",
        "furnishingstatus"
    ],
    drop_first=True
)

Datos_entrada = df.drop(columns="price")
Datos_etiqueta = df["price"]

#Division de los datos en entrenamiento y explotacion
X_train, X_test, y_train, y_test = \
    train_test_split(Datos_entrada, Datos_etiqueta , test_size=0.2)

#Escalamos los datos 
scaler = StandardScaler()
Datos_train_escalados = scaler.fit_transform(X_train)
Datos_test_escalados = scaler.transform(X_test)

#Aplicacion del modelo
modelo = LinearRegression()
modelo.fit(Datos_train_escalados, y_train)

predicciones = modelo.predict(Datos_test_escalados)
print(predicciones)
print(mean_squared_error(y_test, predicciones))
print("R2:", r2_score(y_test, predicciones))

plt.scatter(y_test, predicciones)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)
plt.xlabel("Precio real")
plt.ylabel("Precio predicho")
plt.title("Real vs Predicción")
plt.show()
