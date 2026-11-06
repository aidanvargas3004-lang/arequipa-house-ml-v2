"""Lectura asistida de anuncios inmobiliarios pegados como texto.

El administrador copia el texto de un anuncio (de un portal, de WhatsApp, de un
folleto…) y esta función propone valores para el formulario del dataset. No visita
ningún sitio web: solo interpreta el texto recibido, y la persona revisa y confirma
cada dato antes de guardarlo.
"""
from __future__ import annotations

import re

from .constants import ACABADOS, BOOLEANAS, ESTADOS
from .schema import _DISTRITOS, _TIPOS, norm

TIPO_CAMBIO_REFERENCIAL = 3.75  # S/ por US$; el administrador puede cambiarlo

_NUM = r"(?<![\d.,])(\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?|\d{1,3}(?: \d{3})+(?!\d)(?!\s*(?:m2|m²))|\d+(?:[.,]\d+)?)"

# palabra clave → campo booleano (se busca sobre el texto sin tildes)
_EXTRAS = {
    "jardin": r"\bjardin(?:es)?\b",
    "piscina": r"\bpiscina\b|\bterma\b",
    "ascensor": r"\bascensor(?:es)?\b",
    "seguridad": r"\bseguridad\b|\bvigilancia\b|\bserenazgo\b|\bcondominio cerrado\b",
    "cerca_colegios": r"\bcolegios?\b|\bescuelas?\b|\buniversidad\b",
    "cerca_hospitales": r"\bhospital(?:es)?\b|\bclinicas?\b|\bposta medica\b",
    "cerca_comercial": r"\bcentros? comercial(?:es)?\b|\bmall\b|\bsupermercado\b|\bplaza vea\b",
    "transporte_publico": r"\btransporte publico\b|\bparadero\b|\bavenida principal\b|\bcombi\b",
}


def _a_numero(txt: str) -> float | None:
    s = re.sub(r"\s", "", txt)
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        partes = s.split(",")
        s = s.replace(",", "") if len(partes[-1]) == 3 and len(partes) > 1 else s.replace(",", ".")
    elif "." in s:
        partes = s.split(".")
        s = s.replace(".", "") if len(partes[-1]) == 3 and len(partes) > 1 else s
    try:
        return float(s)
    except ValueError:
        return None


def _entero(patron: str, t: str, maximo: int) -> int | None:
    m = re.search(patron, t)
    if not m:
        return None
    n = int(m.group(1))
    return n if 0 <= n <= maximo else None


def _precio(t: str, tipo_cambio: float) -> tuple[float | None, str]:
    """Devuelve (precio en soles, moneda detectada)."""
    for m in re.finditer(r"(s\s*/\s*\.?|soles|usd|us\s*\$|\$)\s*" + _NUM, t):
        valor = _a_numero(m.group(2))
        if valor is None or valor < 1000:
            continue
        return (round(valor * tipo_cambio) if m.group(1).startswith(("usd", "us", "$")) else round(valor)), (
            "USD" if m.group(1).startswith(("usd", "us", "$")) else "PEN")
    m = re.search(_NUM + r"\s*(soles|dolares|usd)", t)
    if m:
        valor = _a_numero(m.group(1))
        if valor and valor >= 1000:
            return (round(valor * tipo_cambio) if m.group(2) != "soles" else round(valor)), ("PEN" if m.group(2) == "soles" else "USD")
    return None, ""


def analizar_anuncio(texto: str, tipo_cambio: float = TIPO_CAMBIO_REFERENCIAL) -> dict:
    """Interpreta el texto de un anuncio.

    Devuelve ``{"valores": {...}, "precio": str, "encontrados": [...], "faltan": [...], "notas": [...]}``.
    ``valores`` usa las mismas claves que el formulario de vivienda (cadenas); lo que no se
    encontró queda vacío para que la persona lo complete.
    """
    t = norm(texto).replace("m²", "m2").replace("mt2", "m2").replace("mts2", "m2")
    v: dict[str, str] = {}
    encontrados: list[str] = []
    notas: list[str] = []

    # tipo
    for clave, tipo in sorted(_TIPOS.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(clave)}s?\b", t):
            v["tipo"] = tipo
            break
    # distrito (el nombre solo «Arequipa» no cuenta: casi todos los anuncios lo dicen)
    mejor = None
    for clave, distrito in sorted(_DISTRITOS.items(), key=lambda kv: -len(kv[0])):
        if clave == "arequipa":
            continue
        if re.search(rf"\b{re.escape(clave)}\b", t):
            mejor = distrito
            break
    if mejor:
        v["distrito"] = mejor

    # áreas
    m_const = re.search(rf"{_NUM}\s*m2\s*(?:de\s*)?(?:area\s*)?(?:construid|techad)", t) or re.search(
        rf"(?:area\s*)?(?:construid[ao]|techad[ao])\D{{0,15}}{_NUM}\s*m2?", t)
    m_terr = re.search(rf"{_NUM}\s*m2\s*(?:de\s*)?(?:area\s*)?(?:terreno|lote)", t) or re.search(
        rf"(?:area\s*(?:de\s*)?)?(?:terreno|lote)\D{{0,15}}{_NUM}\s*m2?", t)
    sueltos = [a for a in (_a_numero(x) for x in re.findall(_NUM + r"\s*m2", t)) if a]
    if m_const:
        v["area_construida"] = f"{_a_numero(m_const.group(1)):g}"
    if m_terr:
        v["area_terreno"] = f"{_a_numero(m_terr.group(1)):g}"
    if "area_construida" not in v and sueltos:
        v["area_construida"] = f"{sueltos[0]:g}"
        if len(sueltos) == 1 and not m_terr:
            notas.append("Solo se encontró un área; se usó como área construida. Revísala.")

    # distribución
    for campo, patron, maximo in (
        ("habitaciones", r"(\d+)\s*(?:dormitorios?|habitaciones?|hab\b|cuartos?|dorm\b)", 15),
        ("banos", r"(\d+)\s*(?:banos?|bano completo|sshh)", 12),
        ("pisos", r"(\d+)\s*(?:pisos?|niveles|plantas)", 8),
        ("cochera", r"(\d+)\s*(?:cocheras?|garajes?|estacionamientos?)", 8),
    ):
        n = _entero(patron, t, maximo)
        if n is not None:
            v[campo] = str(n)
    if "cochera" not in v and re.search(r"\b(cochera|garaje|estacionamiento)\b", t):
        v["cochera"] = "1"

    # antigüedad
    n = _entero(r"(\d+)\s*anos?\s*(?:de\s*)?antiguedad", t, 100)
    if n is None:
        n = _entero(r"antiguedad\D{0,12}(\d+)", t, 100)
    if n is not None:
        v["antiguedad"] = str(n)
    elif re.search(r"\b(a estrenar|estreno|nuevo|nueva)\b", t):
        v["antiguedad"] = "0"

    # calidad
    if re.search(r"\blujo\b|\bde lujo\b", t):
        v["acabados"] = "Lujo"
    elif re.search(r"alta calidad|primera calidad|premium|acabados finos", t):
        v["acabados"] = "Alta calidad"
    if re.search(r"excelente estado|impecable|a estrenar|recien remodelad", t):
        v["estado_conservacion"] = "Excelente"
    elif re.search(r"mal estado|por remodelar|para remodelar|deteriorad", t):
        v["estado_conservacion"] = "Malo"

    # extras
    for campo, patron in _EXTRAS.items():
        if re.search(patron, t):
            v[campo] = "1"

    precio, moneda = _precio(t, tipo_cambio)
    if moneda == "USD":
        notas.append(f"El precio estaba en dólares; se convirtió a soles con tipo de cambio {tipo_cambio:g}.")

    obligatorios = ["tipo", "distrito", "area_construida", "habitaciones", "banos", "antiguedad"]
    faltan = [c for c in obligatorios if c not in v] + ([] if precio else ["precio"])
    encontrados = [c for c in list(v) + (["precio"] if precio else [])]
    return {"valores": v, "precio": "" if precio is None else str(int(precio)), "encontrados": encontrados,
            "faltan": faltan, "notas": notas}
