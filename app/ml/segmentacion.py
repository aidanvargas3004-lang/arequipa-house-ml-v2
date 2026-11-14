"""Segmentación exploratoria independiente del modelo de tasación.

K-Means agrupa por características, sin utilizar precio como entrada. Los
precios se resumen después: no son predicciones ni evidencia de demanda.
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .constants import BOOLEANAS, CATEGORICAS, NUMERICAS
from .pipeline import preparar


def ajustar_segmentos(df: pd.DataFrame) -> dict:
    """Prueba de 2 a 6 grupos y conserva la mayor silueta reproducible.

    Diez inicializaciones k-means++ reducen la dependencia de un punto de
    partida. La silueta (-1 a 1) mide separación; no es precisión predictiva.
    Se evalúan como máximo 1000 filas para limitar el coste cuadrático.
    """
    if len(df) < 30:
        raise ValueError("Se requieren al menos 30 registros sintéticos para segmentar.")
    x = preparar(df)
    if not np.isfinite(x[NUMERICAS].to_numpy()).all():
        raise ValueError("Los registros contienen valores numéricos no válidos.")
    # Terreno y pisos no aportan información comparable en departamentos.
    departamentos = x["tipo"] == "Departamento"
    x.loc[departamentos, ["area_terreno", "pisos"]] = 0
    cantidades = [c for c in NUMERICAS if c not in BOOLEANAS]
    pre = ColumnTransformer(
        [("cantidades", StandardScaler(), cantidades),
         ("categorias", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS),
         ("extras", "passthrough", BOOLEANAS)],
        # No estandarizamos extras raros: tendrían un peso excesivo.
        transformer_weights={"cantidades": 1.0, "categorias": 0.7, "extras": 0.35},
    )
    matriz = pre.fit_transform(x)
    distintos = len(np.unique(matriz, axis=0))
    if distintos < 3:
        raise ValueError("No hay suficientes perfiles diferentes para comparar segmentos.")
    evaluaciones = []
    modelos = {}
    for k in range(2, min(6, len(df) // 10, distintos - 1) + 1):
        modelo = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        etiquetas = modelo.fit_predict(matriz)
        muestra = np.random.RandomState(42).choice(len(df), min(len(df), 1000), replace=False)
        # Una muestra podría contener un único grupo en datasets muy desiguales.
        if not 2 <= len(np.unique(etiquetas[muestra])) < len(muestra):
            continue
        valor = float(silhouette_score(matriz[muestra], etiquetas[muestra]))
        evaluaciones.append({"k": k, "silueta": valor})
        modelos[k] = modelo
    if not evaluaciones:
        raise ValueError("No fue posible obtener grupos distintos evaluables.")
    mejor = max(evaluaciones, key=lambda e: e["silueta"])
    modelo = modelos[mejor["k"]]
    grupos = []
    for etiqueta in range(mejor["k"]):
        filas = df.iloc[np.flatnonzero(modelo.labels_ == etiqueta)]
        grupos.append({
            "id": etiqueta, "n": len(filas), "porcentaje": len(filas) / len(df) * 100,
            "area": float(filas.area_construida.median()),
            "habitaciones": float(filas.habitaciones.median()),
            "banos": float(filas.banos.median()),
            "antiguedad": float(filas.antiguedad.median()),
            "precio": float(filas.precio.median()),
            "precio_m2": float((filas.precio / filas.area_construida).median()),
            "tipo": str(filas.tipo.mode().iloc[0]),
            "distrito": str(filas.distrito.mode().iloc[0]),
        })
    # Numeración por superficie mediana para facilitar la lectura del panel.
    grupos.sort(key=lambda g: (g["area"], g["id"]))
    return {"modelo": Pipeline([("pre", pre), ("kmeans", modelo)]),
            "grupos": grupos, "evaluaciones": evaluaciones, "k": mejor["k"],
            "silueta": mejor["silueta"], "n": len(df),
            "fecha": datetime.now(timezone.utc).isoformat()}


def asignar_segmentos(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    """Asigna publicaciones a centroides existentes; no modifica los grupos."""
    if df.empty:
        return np.array([], dtype=int)
    x = preparar(df)
    x.loc[x["tipo"] == "Departamento", ["area_terreno", "pisos"]] = 0
    return bundle["modelo"].predict(x)
