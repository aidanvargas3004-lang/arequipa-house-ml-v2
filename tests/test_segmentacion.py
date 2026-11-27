"""Segmentación sintética y aislamiento respecto de la tasación."""
import numpy as np
import pytest

from app import services
from app.extensions import db
from app.ml.datagen import generar_dataset
from app.ml.segmentacion import ajustar_segmentos, asignar_segmentos
from app.models import DatasetRegistro, ModeloML, Propiedad
from .conftest import cliente_con_rol, crear_usuario


def test_segmentacion_no_usa_precios_y_asigna():
    df = generar_dataset(80, 11)
    a = ajustar_segmentos(df)
    b = ajustar_segmentos(df.assign(precio=df.precio * 10))
    assert 2 <= a["k"] <= 6
    assert sum(g["n"] for g in a["grupos"]) == len(df)
    assert -1 <= a["silueta"] <= 1
    assert np.array_equal(asignar_segmentos(a, df), asignar_segmentos(b, df))
    assert len(asignar_segmentos(a, df.iloc[:0])) == 0
    with pytest.raises(ValueError, match="30 registros"):
        ajustar_segmentos(df.iloc[:10])
    repetidos = df.iloc[[0] * 30].reset_index(drop=True)
    with pytest.raises(ValueError, match="perfiles diferentes"):
        ajustar_segmentos(repetidos)


def test_panel_sintetico_persistencia_y_modelo_intacto(app):
    admin = cliente_con_rol(app, "administrador")
    vendedor = crear_usuario(app, "vendedor")
    assert "Todavía no hay" in admin.get("/admin/segmentos").get_data(as_text=True)
    assert admin.post("/admin/segmentos", follow_redirects=True).status_code == 200
    df = generar_dataset(60, 11)
    with app.app_context():
        services.cargar_sintetico(60, 11)
        # Estos registros reales no deben incorporarse al ajuste.
        for fila in df.iloc[:4].to_dict("records"):
            services.agregar_registro(fila, "csv")
        datos = df.iloc[0].to_dict()
        precio = datos.pop("precio")
        for estado, titulo in [("aprobada", "Publicada para segmento"), ("vendida", "No disponible")]:
            db.session.add(Propiedad(vendedor_id=vendedor, titulo=titulo, estado=estado, precio=precio, **datos))
        db.session.commit()
        anterior = [(m.id, m.activo, m.archivo) for m in ModeloML.query.all()]
    respuesta = admin.post("/admin/segmentos", follow_redirects=True)
    assert respuesta.status_code == 200
    html = respuesta.get_data(as_text=True)
    assert "60 registros analizados" in html
    assert "Publicada para segmento" in html and "No disponible" not in html
    with app.app_context():
        vista = services.segmentos_catalogo()
        assert vista["resultado"]["n"] == 60 and not vista["desactualizado"]
        assert anterior == [(m.id, m.activo, m.archivo) for m in ModeloML.query.all()]
        DatasetRegistro.query.filter_by(origen="sintetico").first().habitaciones += 1
        db.session.commit()
        assert services.segmentos_catalogo()["desactualizado"]
    assert "El dataset cambió" in admin.get("/admin/segmentos").get_data(as_text=True)


def test_segmentos_solo_admin(app, client):
    assert client.get("/admin/segmentos").status_code == 302
    for rol in ("comprador", "vendedor"):
        c = cliente_con_rol(app, rol)
        assert c.get("/admin/segmentos").status_code == 403
        assert c.post("/admin/segmentos").status_code == 403
