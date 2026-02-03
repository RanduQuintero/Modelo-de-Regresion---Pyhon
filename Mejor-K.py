import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.DataFrame({
    "edad": [23, 25, 31, 45, 52, 48, 33, 29],
    "ingreso": [12000, 15000, 22000, 48000, 52000, 50000, 26000, 21000],
    "gasto_mensual": [2000, 2500, 3200, 6000, 6800, 6400, 3500, 3000]
})

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

inercia = []

K = range(1, 9)
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X_scaled)
    print(k,kmeans.inertia_)
    inercia.append(kmeans.inertia_)

plt.plot(K, inercia, marker='o')
plt.xlabel("Número de clusters (k)")
plt.ylabel("Inercia")
plt.title("Método del Codo")
plt.show()