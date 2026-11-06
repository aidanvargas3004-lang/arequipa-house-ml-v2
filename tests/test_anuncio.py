from app.extensions import db
from app.ml.anuncio import analizar_anuncio
from app.models import DatasetRegistro

from .conftest import cliente_con_rol

CASA = (
    "Venta de hermosa casa en Yanahuara, 3 pisos. Área de terreno 250 m², área construida 320 m2. "
    "5 dormitorios, 4 baños, 2 cocheras, jardín y seguridad 24h. Cerca de colegios y clínica. "
    "Acabados de lujo, 8 años de antigüedad. Precio: US$ 285,000"
)
DEPA = "Departamento en Cayma 95 m2, 3 hab, 2 baños, a estrenar, ascensor. S/ 1,250,000"


def test_analiza_una_casa_con_precio_en_dolares():
    r = analizar_anuncio(CASA, 3.8)
    v = r["valores"]
    assert (v["tipo"], v["distrito"]) == ("Casa", "Yanahuara")
    assert (v["area_construida"], v["area_terreno"], v["pisos"]) == ("320", "250", "3")
    assert (v["habitaciones"], v["banos"], v["cochera"], v["antiguedad"]) == ("5", "4", "2", "8")
    assert v["acabados"] == "Lujo" and v["jardin"] == "1" and v["seguridad"] == "1"
    assert r["precio"] == str(round(285000 * 3.8))
    assert r["faltan"] == [] and any("dólares" in n for n in r["notas"])


def test_analiza_un_departamento_en_soles():
    r = analizar_anuncio(DEPA)
    assert r["valores"]["tipo"] == "Departamento" and r["valores"]["antiguedad"] == "0"
    assert r["valores"]["ascensor"] == "1" and r["precio"] == "1250000"


def test_texto_sin_datos_lista_lo_que_falta():
    r = analizar_anuncio("Linda propiedad, llámame para más información sobre esta oportunidad única.")
    assert "precio" in r["faltan"] and "distrito" in r["faltan"] and r["valores"] == {}


def test_arequipa_solo_no_cuenta_como_distrito():
    assert "distrito" not in analizar_anuncio("Casa en Arequipa 120 m2, 3 dormitorios")["valores"]


def test_flujo_de_captura_asistida(app):
    admin = cliente_con_rol(app, "administrador")
    assert admin.get("/admin/dataset/anuncio").status_code == 200
    r = admin.post("/admin/dataset/anuncio", data={"accion": "analizar", "texto": CASA, "tc": "3.75"})
    assert r.status_code == 200 and b"Revisa los datos" in r.data and b"Yanahuara" in r.data
    # corto o con tipo de cambio inválido → se rechaza
    assert admin.post("/admin/dataset/anuncio", data={"accion": "analizar", "texto": "hola", "tc": "3.75"}).status_code == 422
    assert admin.post("/admin/dataset/anuncio", data={"accion": "analizar", "texto": CASA, "tc": "abc"}).status_code == 422
    datos = {"accion": "guardar", "tipo": "Casa", "distrito": "Yanahuara", "area_construida": "320", "area_terreno": "250",
             "habitaciones": "5", "banos": "4", "pisos": "3", "antiguedad": "8", "estado_conservacion": "Bueno",
             "acabados": "Lujo", "cochera": "2", "jardin": "1", "seguridad": "1", "precio": "1068750"}
    with app.app_context():
        antes = DatasetRegistro.query.count()
    assert admin.post("/admin/dataset/anuncio", data=datos, follow_redirects=True).status_code == 200
    with app.app_context():
        assert DatasetRegistro.query.count() == antes + 1
        assert DatasetRegistro.query.filter_by(origen="web").count() == 1
    # el mismo registro no se vuelve a guardar
    assert admin.post("/admin/dataset/anuncio", data=datos).status_code == 422
    with app.app_context():
        assert DatasetRegistro.query.count() == antes + 1
    # datos inválidos
    assert admin.post("/admin/dataset/anuncio", data={**datos, "precio": "5", "area_construida": "400"}).status_code == 422


def test_captura_asistida_solo_para_administradores(app):
    for rol in ("comprador", "vendedor"):
        assert cliente_con_rol(app, rol).get("/admin/dataset/anuncio").status_code in (302, 403)


def test_precio_y_area_juntos_no_se_mezclan():
    r = analizar_anuncio("Casa en Yanahuara S/ 950,000 220 m² construidos 4 dorm. 3 baños 10 años de antigüedad")
    assert r["precio"] == "950000" and r["valores"]["area_construida"] == "220"
    r = analizar_anuncio("Casa en Paucarpata S/. 430 000 140 m2, 4 dormitorios, 2 baños, 15 años de antigüedad")
    assert r["precio"] == "430000" and r["valores"]["area_construida"] == "140"
    assert analizar_anuncio("Casa en Cayma 120 m2 S/ 1 050 000")["precio"] == "1050000"
