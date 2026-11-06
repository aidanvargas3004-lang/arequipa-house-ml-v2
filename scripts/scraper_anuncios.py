"""Extrae anuncios de venta de viviendas de Arequipa desde un portal inmobiliario y los guarda en CSV.

IMPORTANTE (léelo antes de usarlo)
----------------------------------
* Úsalo solo con fines académicos y con pocas páginas. Revisa los términos de uso del portal.
* El script consulta el archivo robots.txt del portal y SE DETIENE si no permite leer esa dirección.
* Espera unos segundos entre peticiones y se detiene si el portal responde 403 o 429
  (no intenta saltarse ninguna protección).
* No guarda nombres ni teléfonos; solo datos de la vivienda y el enlace del anuncio.
* Los precios de los anuncios son precios de OFERTA, no de venta real.

Instalación (una sola vez, con el entorno virtual activo):
    pip install requests beautifulsoup4 lxml

Uso (desde la carpeta raíz del proyecto):
    # 1) Leer el catálogo en línea. Copia la URL del navegador (casas o departamentos en venta en Arequipa).
    python scripts/scraper_anuncios.py --url "URL_DEL_CATALOGO" --paginas 2

    # 2) Si la URL tiene un número de página, usa {pagina} en su lugar:
    python scripts/scraper_anuncios.py --url "https://portal.pe/venta-arequipa-pagina-{pagina}" --paginas 3

    # 3) Sin conexión: guarda la página con Ctrl+S en el navegador y léela desde el archivo.
    python scripts/scraper_anuncios.py --html pagina_guardada.html

El resultado es ventas_reales_arequipa.csv. Luego se carga en el sistema desde
Administración → Dataset → Importar CSV (el sistema valida cada fila).
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from datetime import date
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Permite importar el lector de texto de anuncios del propio proyecto (app/ml/anuncio.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.ml.anuncio import analizar_anuncio  # noqa: E402

# Nos identificamos con honestidad (sin datos personales) para que el portal sepa quién consulta
USER_AGENT = "ArequipaHouseML-academico/1.0 (proyecto universitario; lectura de pocas paginas)"

# Selectores CSS que se prueban, en orden, para encontrar cada "tarjeta" de anuncio.
# Cada portal usa nombres distintos: si ninguno funciona, mira la página con F12 (Inspeccionar),
# busca la etiqueta que envuelve un anuncio y pásala con --selector-tarjeta.
SELECTORES_TARJETA = [
    "[data-qa*='posting']", "[data-posting-type]", "article", "div[class*='posting']",
    "div[class*='card']", "li[class*='result']", "li[class*='item']",
]

# Columnas del CSV final: las que entiende el importador del sistema + datos de trazabilidad
COLUMNAS = [
    "tipo", "distrito", "area_construida", "area_terreno", "habitaciones", "banos", "pisos", "cochera",
    "antiguedad", "estado_conservacion", "acabados", "jardin", "piscina", "ascensor", "seguridad",
    "cerca_colegios", "cerca_hospitales", "cerca_comercial", "transporte_publico",
    "precio", "fuente_url", "fecha_captura",
]
# Sin estos datos un anuncio no sirve para entrenar el modelo
OBLIGATORIAS = ["tipo", "distrito", "area_construida", "habitaciones", "precio"]


def robots_permite(url: str) -> bool:
    """Consulta robots.txt del portal y dice si se puede leer esa URL."""
    partes = urlparse(url)
    robots_url = f"{partes.scheme}://{partes.netloc}/robots.txt"
    rp = RobotFileParser()
    try:
        # Descargamos robots.txt con nuestra identificación y un límite de tiempo
        r = requests.get(robots_url, headers={"User-Agent": USER_AGENT}, timeout=20)
        if r.status_code in (401, 403):      # acceso prohibido: se asume que NO está permitido
            return False
        rp.parse(r.text.splitlines() if r.status_code == 200 else [])  # sin robots.txt = sin restricciones
    except requests.RequestException as e:
        print(f"No se pudo leer robots.txt ({e}). Por precaución no se continúa.")
        return False
    return rp.can_fetch(USER_AGENT, url)


def descargar(url: str) -> str | None:
    """Descarga una página. Devuelve su HTML, o None si el portal pide detenerse."""
    r = requests.get(url, headers={"User-Agent": USER_AGENT, "Accept-Language": "es-PE,es;q=0.9"}, timeout=25)
    if r.status_code in (403, 429):          # el portal nos bloquea: respetamos y paramos
        print(f"El portal respondió {r.status_code} (bloqueo o exceso de peticiones). Se detiene aquí.")
        return None
    r.raise_for_status()                     # cualquier otro error (404, 500…) lanza una excepción
    return r.text


def buscar_tarjetas(soup: BeautifulSoup, selector: str | None):
    """Devuelve las etiquetas HTML que representan cada anuncio del catálogo."""
    candidatos = [selector] if selector else SELECTORES_TARJETA
    for css in candidatos:
        tarjetas = soup.select(css)
        # Una tarjeta válida debe mostrar un precio (S/ o $) y un área en m²
        validas = [t for t in tarjetas if _parece_anuncio(t.get_text(" ", strip=True))]
        if len(validas) >= 3 or (selector and validas):
            return validas
    return []


def _parece_anuncio(texto: str) -> bool:
    t = texto.lower().replace("m²", "m2")
    return ("m2" in t) and any(s in t for s in ("s/", "usd", "us$", "$", "soles"))


def extraer(tarjeta, base_url: str) -> dict:
    """Convierte una tarjeta HTML en un diccionario con los datos de la vivienda."""
    texto = tarjeta.get_text(" ", strip=True)        # todo el texto visible de la tarjeta
    lectura = analizar_anuncio(texto)                # busca precio, distrito, m², habitaciones, baños…
    fila = {c: "" for c in COLUMNAS}
    fila.update(lectura["valores"])                  # datos detectados en el texto
    fila["precio"] = lectura["precio"]               # precio ya convertido a soles
    enlace = tarjeta.find("a", href=True) or (tarjeta if tarjeta.name == "a" and tarjeta.get("href") else None)
    fila["fuente_url"] = urljoin(base_url, enlace["href"]) if enlace else ""   # enlace al anuncio
    fila["fecha_captura"] = date.today().isoformat()
    return fila


def siguiente_pagina(soup: BeautifulSoup, url_actual: str) -> str | None:
    """Busca el enlace «Siguiente» del catálogo (si la URL no usa {pagina})."""
    a = soup.select_one("a[rel='next']") or next(
        (x for x in soup.find_all("a", href=True) if x.get_text(strip=True).lower() in ("siguiente", "next", "›", "»")), None)
    return urljoin(url_actual, a["href"]) if a else None


def limpiar(filas: list[dict], antiguedad_supuesta: int | None) -> tuple[pd.DataFrame, int]:
    """Estructura con Pandas, limpia y devuelve (tabla limpia, cantidad descartada)."""
    df = pd.DataFrame(filas, columns=COLUMNAS)
    total = len(df)
    for c in ("precio", "area_construida"):
        df[c] = pd.to_numeric(df[c], errors="coerce")                       # texto → número (o vacío)
    df = df.dropna(subset=OBLIGATORIAS)                                      # sin datos clave, se descarta
    df = df[df["tipo"].astype(str).str.len() > 0]
    df = df[(df["precio"] >= 20000) & (df["precio"] <= 20_000_000)]          # precios absurdos fuera
    df = df[(df["area_construida"] >= 15) & (df["area_construida"] <= 3000)]  # áreas absurdas fuera
    df = df.drop_duplicates(subset=["tipo", "distrito", "area_construida", "habitaciones", "precio"])
    df["precio"] = df["precio"].round().astype(int)                          # precio en soles, sin decimales
    df["area_construida"] = df["area_construida"].map(lambda x: f"{x:g}")    # 85.0 → 85
    if antiguedad_supuesta is not None:                                      # solo si tú lo pides, de forma explícita
        df["antiguedad"] = df["antiguedad"].replace("", pd.NA).fillna(antiguedad_supuesta)
    return df.reset_index(drop=True), total - len(df)


def main() -> int:
    ap = argparse.ArgumentParser(description="Extrae anuncios de viviendas de Arequipa a un CSV.")
    ap.add_argument("--url", help="URL del catálogo (usa {pagina} si la URL lleva el número de página).")
    ap.add_argument("--html", help="Archivo HTML guardado con Ctrl+S (modo sin conexión).")
    ap.add_argument("--paginas", type=int, default=2, help="Cuántas páginas leer (máx. 10; por defecto 2).")
    ap.add_argument("--pausa", type=float, default=4.0, help="Segundos de espera entre páginas (mín. 2).")
    ap.add_argument("--selector-tarjeta", help="Selector CSS de cada anuncio, si la detección automática falla.")
    ap.add_argument("--antiguedad-supuesta", type=int, help="Años a usar cuando el anuncio no dice la antigüedad.")
    ap.add_argument("--salida", default="ventas_reales_arequipa.csv", help="Archivo CSV de salida.")
    args = ap.parse_args()
    if not args.url and not args.html:
        ap.error("Indica --url o --html.")

    filas: list[dict] = []
    if args.html:                                            # modo sin conexión: leer un archivo guardado
        soup = BeautifulSoup(Path(args.html).read_text(encoding="utf-8", errors="ignore"), "lxml")
        filas += [extraer(t, "") for t in buscar_tarjetas(soup, args.selector_tarjeta)]
    else:                                                    # modo en línea
        paginas = max(1, min(args.paginas, 10))              # tope de 10 páginas para no abusar del portal
        pausa = max(2.0, args.pausa)
        url = args.url.format(pagina=1) if "{pagina}" in args.url else args.url
        if not robots_permite(url):
            print("El portal NO permite leer esta dirección de forma automática (robots.txt). No se continúa.")
            print("Alternativa: guarda la página con Ctrl+S y usa --html, o usa la captura asistida del sistema.")
            return 2
        for n in range(1, paginas + 1):
            print(f"Leyendo página {n}: {url}")
            html = descargar(url)
            if html is None:
                break
            soup = BeautifulSoup(html, "lxml")               # convierte el HTML en un árbol de etiquetas
            tarjetas = buscar_tarjetas(soup, args.selector_tarjeta)
            print(f"  anuncios encontrados: {len(tarjetas)}")
            if not tarjetas:
                print("  No se reconocieron anuncios. Revisa la página con F12 y usa --selector-tarjeta.")
                break
            filas += [extraer(t, url) for t in tarjetas]
            if n == paginas:
                break
            url = args.url.format(pagina=n + 1) if "{pagina}" in args.url else siguiente_pagina(soup, url)
            if not url:
                break
            time.sleep(pausa + random.uniform(0, 1.5))       # pausa educada entre páginas

    if not filas:
        print("No se extrajo ningún anuncio.")
        return 1
    df, descartados = limpiar(filas, args.antiguedad_supuesta)
    df.to_csv(args.salida, index=False, encoding="utf-8-sig")  # utf-8-sig: Excel abre bien las tildes
    print(f"\nListo: {len(df)} anuncios guardados en {args.salida} ({descartados} descartados por datos incompletos).")
    if df["antiguedad"].replace("", pd.NA).isna().any():
        print("AVISO: hay anuncios sin antigüedad; el importador del sistema los rechazará. "
              "Complétalos en Excel o usa --antiguedad-supuesta N (suposición declarada).")
    print("Siguiente paso: Administración → Dataset → Importar CSV.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
