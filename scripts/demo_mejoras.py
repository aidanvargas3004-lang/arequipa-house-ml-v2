"""Demuestra con números qué aportan el control de calidad y el rango calibrado.

Se usa el dataset sintético. A los datos de entrenamiento se les inyectan errores de
captura (precios mal escritos: ×6 o ÷6) para imitar el ruido de los anuncios reales,
y el modelo se evalúa siempre sobre datos limpios que nunca vio.

Uso:  PYTHONPATH=. python scripts/demo_mejoras.py [--errores 0.05] [--json salida.json]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from app.ml import pipeline
from app.ml.datagen import generar_dataset


def metricas(real, pred):
    ape = np.abs(pred - real) / real
    return {"mae": float(np.mean(np.abs(pred - real))), "mdape": float(np.median(ape) * 100),
            "p90_error": float(np.quantile(ape, 0.9) * 100), "mape": float(ape.mean() * 100)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--errores", type=float, default=0.05, help="fracción de errores de captura en el entrenamiento")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--json")
    args = ap.parse_args()

    df = generar_dataset(args.n, 5)
    df["precio"] = df["precio"].astype(float)
    rng = np.random.default_rng(5)
    orden = rng.permutation(len(df))
    corte = int(len(df) * 0.8)
    train, prueba = df.iloc[orden[:corte]].copy(), df.iloc[orden[corte:]].copy()   # la prueba queda limpia
    k = int(len(train) * args.errores)
    malos = rng.choice(len(train), k, replace=False)
    factor = np.where(np.arange(k) < k // 2, 6.0, 1 / 6.0)
    train.iloc[malos, train.columns.get_loc("precio")] *= factor
    print(f"Entrenamiento: {len(train)} registros, {k} con error de captura ({args.errores:.0%}). Prueba limpia: {len(prueba)}.\n")

    real = prueba["precio"].to_numpy()
    resultados = {}
    for etiqueta, excluir in (("Sin depuración", False), ("Con depuración", True)):
        r = pipeline.entrenar(train, excluir_atipicos=excluir)
        b = r["bundle"]
        pred = pipeline.predecir(b, prueba.to_dict("records"))
        m = metricas(real, pred)
        # rango antiguo: porcentaje fijo según el error mediano de validación
        margen = max(0.05, b["error_relativo"])
        cob_antiguo = float(np.mean((real >= pred * (1 - margen)) & (real <= pred * (1 + margen))) * 100)
        q = np.array([pipeline.limites_intervalo(b["intervalo"], t) for t in prueba["tipo"]])
        cob_nuevo = float(np.mean((real >= pred * np.exp(q[:, 0])) & (real <= pred * np.exp(q[:, 1]))) * 100)
        resultados[etiqueta] = {**m, "excluidos": r["metricas"]["calidad"]["excluidos"], "cobertura_rango_fijo": cob_antiguo,
                                "cobertura_rango_calibrado": cob_nuevo, "ancho_fijo": margen * 100,
                                "ancho_calibrado": float(np.median((np.exp(q[:, 1]) - np.exp(q[:, 0])) / 2) * 100)}
    print(f"{'':<16}{'Excluidos':>10}{'MAE (S/)':>12}{'Error mediano':>15}{'Error p90':>11}")
    for e, v in resultados.items():
        print(f"{e:<16}{v['excluidos']:>10}{v['mae']:>12,.0f}{v['mdape']:>14.2f}%{v['p90_error']:>10.1f}%")
    print("\nRango probable (meta: contener el precio real en el 80 % de los casos):")
    v = resultados["Con depuración"]
    print(f"  Rango fijo (antes):        cobertura {v['cobertura_rango_fijo']:.1f} %  (±{v['ancho_fijo']:.1f} %)")
    print(f"  Rango calibrado (ahora):   cobertura {v['cobertura_rango_calibrado']:.1f} %  (±{v['ancho_calibrado']:.1f} % mediano)")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(resultados, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
