import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from sklearn.metrics import silhouette_score

#Dataframe improvisado
df = pd.read_csv(r"C:\Users\Randu\Desktop\Modelos de machine learning\datos_clientes.csv")

clusters = {
    0: "Clientes jóvenes - bajo gasto",
    1: "Clientes medio - gasto moderado",
    2: "Clientes premium - alto gasto"
}
#Uso de una pipe para que todos sigan el mismo proceso 
# y lo que pasa internamente
'''
StandardScaler.fit(df)
StandardScaler.transform(df)
KMeans.fit(datos_escalados)
'''
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(n_clusters=3))
])
#Entrenamos el modelo
pipeline.fit(df)
#Almacenar el mmodelo entrenado
joblib.dump(pipeline, "modelo_clustering.pkl")
#Cargar el modelo entrenado para utilizarlo
modelo = joblib.load("modelo_clustering.pkl")

#Cliente a asiganar 
nuevo_cliente = pd.DataFrame([{
    "edad": 50,
    "ingreso": 20000,
    "gasto_mensual": 15000
}])

cluster = modelo.predict(nuevo_cliente)
print("Cluster asignado:", cluster[0])
print(clusters[cluster[0]])
