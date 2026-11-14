"""Búsqueda KNN compartida por tasaciones y detalles del catálogo.

No predice precios: ordena publicaciones por distancia entre características.
El índice se construye con el catálogo vigente, sin entrenar ni modificar el
modelo de tasación. Las escalas son reglas iniciales de producto, ajustables
cuando se disponga de evaluaciones de relevancia de los compradores.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .constants import ACABADOS, BOOLEANAS, DISTRITO_NOMBRES, ESTADOS, TIPOS

# Una diferencia de 50 m² equivale a una unidad de distancia en área construida.
# Usamos escalas fijas para que un catálogo pequeño o un inmueble extremo no
# cambien la importancia de todas las demás características.
ESCALAS = {
    "area_construida": 50, "area_terreno": 100, "habitaciones": 2,
    "banos": 2, "pisos": 2, "antiguedad": 20, "cochera": 2,
}


def _vector(datos: dict) -> list[float]:
    """Convierte categorías, cantidades y extras en coordenadas comparables."""
    vector = []
    # One-hot evita inventar un orden entre distritos o tipos de inmueble.
    # Cada categoría distinta aporta peso² a la distancia cuadrática total.
    for campo, catalogo, peso in (
        ("tipo", TIPOS, 2.0), ("distrito", DISTRITO_NOMBRES, 2.0),
    ):
        vector.extend(peso / np.sqrt(2) * (datos[campo] == valor) for valor in catalogo)
    for campo, escala in ESCALAS.items():
        # Terreno y pisos no diferencian departamentos en nuestra tasación.
        valor = 0 if datos["tipo"] == "Departamento" and campo in ("area_terreno", "pisos") else datos[campo]
        vector.append(float(valor) / escala)
    vector.extend((ESTADOS.index(datos["estado_conservacion"]) / 2,
                   ACABADOS.index(datos["acabados"]) / 2))
    # Los extras aportan menos individualmente que tipo, distrito o superficie.
    vector.extend(0.5 * bool(datos[campo]) for campo in BOOLEANAS)
    return vector


def motivos_similitud(consulta: dict, candidato: dict) -> list[str]:
    """Explica coincidencias reales; la distancia no es un porcentaje de acierto."""
    motivos = []
    if consulta["tipo"] == candidato["tipo"]:
        motivos.append("Mismo tipo de vivienda")
    if consulta["distrito"] == candidato["distrito"]:
        motivos.append("Mismo distrito")
    if abs(float(consulta["area_construida"]) - float(candidato["area_construida"])) <= 0.15 * float(consulta["area_construida"]):
        motivos.append("Área construida similar (±15 %)")
    if consulta["habitaciones"] == candidato["habitaciones"]:
        motivos.append("Igual número de habitaciones")
    return motivos or ["Alternativa más cercana entre las publicaciones disponibles"]


def vecinos_cercanos(consulta: dict, candidatos: list[dict], *, limite: int = 3) -> list[int]:
    """Devuelve índices ordenados por distancia; admite cero o pocos candidatos.

    El precio se omite deliberadamente: buscamos similitud física y de distrito,
    no afirmamos que las recomendaciones se ajusten al presupuesto del usuario.
    Los empates conservan el orden recibido para evitar tarjetas inestables.
    """
    if not candidatos or limite <= 0:
        return []
    matriz = np.asarray([_vector(c) for c in candidatos], dtype=float)
    indice = NearestNeighbors(metric="euclidean", algorithm="brute", n_jobs=1).fit(matriz)
    # Consultamos todos para resolver empates de forma estable antes del corte.
    distancias, posiciones = indice.kneighbors([_vector(consulta)], n_neighbors=len(candidatos))
    orden = sorted(zip(distancias[0], posiciones[0]), key=lambda par: (par[0], par[1]))
    return [int(i) for _, i in orden[:limite]]
