import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

#Dataframe improvisado
df = pd.DataFrame({
    "edad": [23, 25, 31, 45, 52, 48, 33, 29],
    "ingreso": [12000, 15000, 22000, 48000, 52000, 50000, 26000, 21000],
    "gasto_mensual": [2000, 2500, 3200, 6000, 6800, 6400, 3500, 3000]
})
clusters = {
    0: "Clientes jóvenes - bajo gasto",
    1: "Clientes medio - gasto moderado",
    2: "Clientes premium - alto gasto"
}
#Uso de una pipe para que todos sigan el mismo proceso 
# y lo que pasa internamnte
'''
StandardScaler.fit(df)
StandardScaler.transform(df)
KMeans.fit(datos_escalados)
'''
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(n_clusters=3))
])

pipeline.fit(df)
#Almacenar el mmodelo entrenado
joblib.dump(pipeline, "modelo_clustering.pkl")
#Cargar el modelo entrenado para utilizarlo
modelo = joblib.load("modelo_clustering.pkl")

#Cliente a asiganar 
nuevo_cliente = pd.DataFrame([{
    "edad": 50,
    "ingreso": 55000,
    "gasto_mensual": 28000
}])

cluster = modelo.predict(nuevo_cliente)
print("Cluster asignado:", cluster[0])
print(clusters[cluster[0]])
