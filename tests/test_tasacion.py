import re

from app.extensions import db
from app.models import Tasacion

from .conftest import VIVIENDA, cliente_con_rol


def test_tasar_requiere_login(client):
    assert client.get("/tasar").status_code == 302
    assert client.post("/tasar", data=VIVIENDA).status_code == 302


def test_flujo_completo_de_tasacion(app):
    c = cliente_con_rol(app, "comprador")
    assert c.get("/tasar?tipo=Casa&distrito=Yanahuara&area_construida=150").status_code == 200
    r = c.post("/tasar", data=VIVIENDA)
    assert r.status_code == 302
    detalle = c.get(r.headers["Location"])
    assert detalle.status_code == 200
    html = detalle.get_data(as_text=True)
    assert "Precio estimado de venta" in html and "S/" in html and "Factores que influyeron" in html
    assert "datos sintéticos" not in html and "bg-amber-50" not in html  # sin avisos amarillos
    with app.app_context():
        t = Tasacion.query.one()
        assert t.precio_min < t.precio_estimado < t.precio_max and t.precio_m2 > 0 and t.factores
    hist = c.get("/historial").get_data(as_text=True)
    assert "Cayma" in hist


def test_tasar_con_datos_invalidos(app):
    c = cliente_con_rol(app, "comprador")
    r = c.post("/tasar", data={**VIVIENDA, "area_construida": "-5", "distrito": "Marte"})
    assert r.status_code == 422
    with app.app_context():
        assert Tasacion.query.count() == 0


def test_tasacion_ajena_no_visible_pero_admin_si(app):
    a = cliente_con_rol(app, "comprador", "a@test.pe")
    b = cliente_con_rol(app, "comprador", "b@test.pe")
    admin = cliente_con_rol(app, "administrador")
    url = a.post("/tasar", data=VIVIENDA).headers["Location"]
    assert b.get(url).status_code == 403
    assert admin.get(url).status_code == 200
    tid = int(re.search(r"/tasacion/(\d+)", url).group(1))
    assert b.post(f"/tasacion/{tid}/eliminar").status_code == 403
    assert a.post(f"/tasacion/{tid}/eliminar").status_code == 302
    with app.app_context():
        assert db.session.get(Tasacion, tid) is None


def test_api_tasar(app):
    c = cliente_con_rol(app, "vendedor")
    r = c.post("/api/tasar", data=VIVIENDA)
    j = r.get_json()
    assert r.status_code == 200 and j["ok"] and j["precio_min"] < j["precio"] < j["precio_max"]
    bad = c.post("/api/tasar", data={**VIVIENDA, "tipo": "x"})
    assert bad.status_code == 422 and not bad.get_json()["ok"]
    with app.app_context():
        assert Tasacion.query.count() == 0  # la API no ensucia el historial


def test_mas_area_mas_precio(app):
    c = cliente_con_rol(app, "comprador")
    p1 = c.post("/api/tasar", data={**VIVIENDA, "area_construida": "70"}).get_json()["precio"]
    p2 = c.post("/api/tasar", data={**VIVIENDA, "area_construida": "180"}).get_json()["precio"]
    assert p2 > p1


def test_tasaciones_de_ejemplo_se_guardan_y_no_se_duplican(app):
    from app import services
    from app.models import Tasacion
    from tests.conftest import crear_usuario

    uid = crear_usuario(app, "administrador", email="ejemplo@test.pe")
    with app.app_context():
        assert services.cargar_tasaciones_ejemplo(uid) == 10
        assert services.cargar_tasaciones_ejemplo(uid) == 0  # repetirlo no duplica
        filas = Tasacion.query.filter_by(usuario_id=uid).all()
        assert len(filas) == 10
        assert all(t.precio_min < t.precio_estimado < t.precio_max for t in filas)


def test_pantallas_sin_avisos_amarillos_de_datos_sinteticos(app):
    from tests.conftest import cliente_con_rol

    admin = cliente_con_rol(app, "administrador", email="aviso@test.pe")
    for ruta in ("/admin/", "/tasar", "/admin/segmentos"):
        html = admin.get(ruta).get_data(as_text=True)
        assert "datos sintéticos" not in html and "bg-amber-50" not in html
