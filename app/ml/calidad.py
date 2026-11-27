"""Control de calidad del dataset: detección de registros atípicos.

Un anuncio real puede traer un precio mal escrito, un área equivocada o una
vivienda muy distinta a las de su zona. Esos pocos casos distorsionan el
entrenamiento y hacen que las estimaciones varíen mucho. Aquí se detectan con
una regla robusta, fácil de explicar y de auditar:

1. Se calcula el precio por m² construido de cada registro.
2. Se compara, en escala logarítmica, con la mediana de su grupo (mismo tipo y
   distrito; si el grupo es pequeño, mismo tipo; si también, todos).
3. Se usa la desviación absoluta mediana (MAD) como medida de dispersión, que no
   se deja arrastrar por los propios valores extremos.
4. Es atípico el registro cuyo puntaje robusto supera el umbral (por defecto 3,5,
   el valor recomendado por Iglewicz y Hoaglin, 1993) o cuyo precio por m² es
   imposible en la práctica.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

UMBRAL_Z = 3.5          # puntaje robusto a partir del cual un registro es atípico
MIN_GRUPO = 12          # registros mínimos para usar un grupo como referencia
M2_MIN_PLAUSIBLE = 300  # S/ por m² por debajo del cual el dato no es creíble
M2_MAX_PLAUSIBLE = 30000
MAX_FRACCION_EXCLUIDA = 0.10  # nunca se descarta más del 10 % para entrenar
MAD_MINIMO = 0.08       # evita dividir entre casi cero cuando el grupo es muy homogéneo


def _referencia(df: pd.DataFrame, columna: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Mediana y MAD (en logaritmo) del precio por m² de cada registro según su grupo."""
    mediana = pd.Series(np.nan, index=df.index)
    mad = pd.Series(np.nan, index=df.index)
    niveles = (["tipo", "distrito"], ["tipo"], [])
    for cols in niveles:
        grupos = df.groupby(cols) if cols else [(None, df)]
        for _, idx in ((k, g.index) for k, g in grupos):
            faltan = mediana.loc[idx].isna()
            if not faltan.any() or len(idx) < MIN_GRUPO and cols:
                continue
            valores = columna.loc[idx]
            med = float(valores.median())
            dispersion = float((valores - med).abs().median())
            sel = idx[faltan.to_numpy()]
            mediana.loc[sel] = med
            mad.loc[sel] = dispersion
    return mediana, mad


def detectar_atipicos(df: pd.DataFrame, umbral: float = UMBRAL_Z) -> pd.DataFrame:
    """Devuelve una fila por registro atípico, ordenada del más extremo al menos.

    Columnas: ``indice`` (índice del DataFrame original), ``precio_m2``,
    ``referencia_m2`` (mediana de su zona), ``puntaje`` y ``motivo``.
    """
    columnas = ["indice", "precio_m2", "referencia_m2", "puntaje", "motivo"]
    if df.empty:
        return pd.DataFrame(columns=columnas)
    d = df[["tipo", "distrito", "area_construida", "precio"]].copy()
    d["precio"] = pd.to_numeric(d["precio"], errors="coerce")
    d["area_construida"] = pd.to_numeric(d["area_construida"], errors="coerce")
    d = d.dropna()
    d = d[(d["precio"] > 0) & (d["area_construida"] > 0)]
    m2 = d["precio"] / d["area_construida"]
    log_m2 = np.log(m2)
    mediana, mad = _referencia(d, log_m2)
    sigma = np.maximum(1.4826 * mad, MAD_MINIMO)
    puntaje = (log_m2 - mediana) / sigma
    referencia = np.exp(mediana)

    motivo = pd.Series("", index=d.index)
    imposible = (m2 < M2_MIN_PLAUSIBLE) | (m2 > M2_MAX_PLAUSIBLE)
    motivo[imposible & (m2 < M2_MIN_PLAUSIBLE)] = "Precio por m² imposible (demasiado bajo)"
    motivo[imposible & (m2 > M2_MAX_PLAUSIBLE)] = "Precio por m² imposible (demasiado alto)"
    atipico_z = (puntaje.abs() > umbral) & ~imposible
    motivo[atipico_z & (puntaje > 0)] = "Precio por m² muy alto para su zona"
    motivo[atipico_z & (puntaje < 0)] = "Precio por m² muy bajo para su zona"

    sel = motivo != ""
    res = pd.DataFrame({
        "indice": d.index[sel], "precio_m2": m2[sel].round(0).to_numpy(),
        "referencia_m2": referencia[sel].round(0).to_numpy(), "puntaje": puntaje[sel].round(2).to_numpy(),
        "motivo": motivo[sel].to_numpy(),
    })
    res["_abs"] = res["puntaje"].abs()
    return res.sort_values("_abs", ascending=False).drop(columns="_abs").reset_index(drop=True)


def separar_atipicos(df: pd.DataFrame, *, minimo_filas: int = 60, umbral: float = UMBRAL_Z) -> tuple[pd.DataFrame, dict]:
    """Devuelve ``(df_limpio, informe)`` sin los registros atípicos.

    Salvaguardas: nunca se descarta más del 10 % del dataset (si hay más
    sospechosos, solo los más extremos) ni se deja menos de ``minimo_filas``.
    """
    det = detectar_atipicos(df, umbral)
    tope = int(len(df) * MAX_FRACCION_EXCLUIDA)
    excluir = det.head(tope)
    if len(df) - len(excluir) < minimo_filas:
        excluir = det.iloc[0:0]
    limpio = df.drop(index=excluir["indice"].tolist())
    informe = {
        "total_inicial": int(len(df)), "detectados": int(len(det)), "excluidos": int(len(excluir)),
        "umbral": umbral, "tope_aplicado": bool(len(det) > len(excluir) and len(excluir) > 0),
    }
    return limpio, informe
