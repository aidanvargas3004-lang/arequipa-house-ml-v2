"""Compara varios algoritmos de regresión con la misma partición que usa el sistema.

Sirve de respaldo para el informe: muestra por qué se eligieron Random Forest y
Gradient Boosting frente a otras técnicas (regresión lineal, árbol de decisión,
KNN y SVM). Usa el dataset sintético de demostración, así que los valores
validan la metodología y no la precisión sobre el mercado real.

Uso:  PYTHONPATH=. python scripts/comparar_algoritmos.py [--json salida.json]
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from app.ml.constants import CATEGORICAS, NUMERICAS, TARGET
from app.ml.datagen import generar_dataset
from app.ml.pipeline import preparar

SEMILLA = 42


def construir(estimador, escalar: bool):
    pasos = [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS)]
    if escalar:
        pasos.append(("num", StandardScaler(), NUMERICAS))
        pre = ColumnTransformer(pasos)
    else:
        pre = ColumnTransformer(pasos, remainder="passthrough")
    return TransformedTargetRegressor(
        regressor=Pipeline([("pre", pre), ("est", estimador)]), func=np.log1p, inverse_func=np.expm1
    )


MODELOS = {
    "Regresión lineal": (LinearRegression(), True),
    "Árbol de decisión": (DecisionTreeRegressor(max_depth=12, min_samples_leaf=3, random_state=SEMILLA), False),
    "KNN (k = 7)": (KNeighborsRegressor(n_neighbors=7, weights="distance"), True),
    "SVM (SVR, kernel RBF)": (SVR(kernel="rbf", C=3.0, epsilon=0.05), True),
    "Random Forest": (
        RandomForestRegressor(n_estimators=160, min_samples_leaf=2, max_features=0.6, n_jobs=1, random_state=SEMILLA),
        False,
    ),
    "Gradient Boosting": (
        GradientBoostingRegressor(n_estimators=350, learning_rate=0.07, max_depth=4, subsample=0.9, random_state=SEMILLA),
        False,
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="Guardar los resultados en un archivo JSON")
    args = ap.parse_args()

    df = generar_dataset(4000, SEMILLA)
    x = preparar(df)
    y = df[TARGET].astype(float).to_numpy()
    idx = np.arange(len(df))
    i_tr, i_tmp = train_test_split(idx, test_size=0.30, random_state=SEMILLA)
    i_val, i_te = train_test_split(i_tmp, test_size=0.50, random_state=SEMILLA)

    filas = []
    for nombre, (est, escalar) in MODELOS.items():
        modelo = construir(est, escalar)
        t0 = time.perf_counter()
        modelo.fit(x.iloc[i_tr], y[i_tr])
        seg = time.perf_counter() - t0
        pred = modelo.predict(x.iloc[i_te])
        fila = {
            "modelo": nombre,
            "mae": round(float(mean_absolute_error(y[i_te], pred)), 0),
            "rmse": round(float(math.sqrt(mean_squared_error(y[i_te], pred))), 0),
            "r2": round(float(r2_score(y[i_te], pred)), 4),
            "mdape": round(float(np.median(np.abs(pred - y[i_te]) / y[i_te]) * 100), 2),
            "segundos": round(seg, 1),
        }
        filas.append(fila)
        print(f"{nombre:<24} MAE {fila['mae']:>9,.0f}  RMSE {fila['rmse']:>9,.0f}  R² {fila['r2']:.4f}  "
              f"error mediano {fila['mdape']:>5.2f} %  ({fila['segundos']} s)")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"particion": {"entrenamiento": len(i_tr), "validacion": len(i_val), "prueba": len(i_te)}, "resultados": filas},
                      f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
