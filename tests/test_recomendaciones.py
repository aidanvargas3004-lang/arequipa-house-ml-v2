"""Relevancia, visibilidad y reutilización de las recomendaciones KNN."""
from app import services
from app.extensions import db
from app.ml.recomendaciones import vecinos_cercanos
from app.ml.schema import limpiar_vivienda
from app.models import Propiedad
from .conftest import VIVIENDA, cliente_con_rol, crear_usuario


def test_vecinos_comparan_mas_que_area():
    datos, errores = limpiar_vivienda(VIVIENDA)
    assert not errores
    # Con idéntica área, habitaciones y baños distinguen la mejor alternativa.
    distinto = {**datos, "habitaciones": 8, "banos": 6}
    assert vecinos_cercanos(datos, [distinto, datos]) == [1, 0]
    assert vecinos_cercanos(datos, []) == []
    assert vecinos_cercanos(datos, [datos], limite=0) == []
    assert vecinos_cercanos(datos, [datos, datos], limite=1) == [0]


def test_recomendaciones_en_ambos_flujos(app):
    vendedor = crear_usuario(app, "vendedor")
    cliente = cliente_con_rol(app, "comprador")
    datos, _ = limpiar_vivienda(VIVIENDA)
    with app.app_context():
        for titulo, estado in [("Origen", "aprobada"), ("Alternativa visible", "aprobada"),
                               ("Oculta pendiente", "pendiente"), ("Oculta vendida", "vendida"),
                               ("Oculta rechazada", "rechazada")]:
            db.session.add(Propiedad(vendedor_id=vendedor, titulo=titulo, precio=520000,
                                     estado=estado, **datos))
        db.session.commit()
        origen = Propiedad.query.filter_by(titulo="Origen").one().id
        resultados = services.recomendar_propiedades(datos, excluir_id=origen)
        assert [r["propiedad"].titulo for r in resultados] == ["Alternativa visible"]
        assert "Mismo distrito" in resultados[0]["motivos"]
    detalle = cliente.get(f"/propiedades/{origen}")
    tasacion = cliente.post("/tasar", data=VIVIENDA, follow_redirects=True)
    for respuesta in (detalle, tasacion):
        assert respuesta.status_code == 200
        html = respuesta.get_data(as_text=True)
        assert "Alternativa visible" in html and "Mismo distrito" in html
        assert "Oculta" not in html
    # Una venta se refleja de inmediato; no existe un índice antiguo en caché.
    with app.app_context():
        Propiedad.query.filter_by(titulo="Alternativa visible").one().estado = "vendida"
        db.session.commit()
        assert services.recomendar_propiedades(datos, excluir_id=origen) == []
    assert "Aún no hay otras viviendas" in cliente.get(f"/propiedades/{origen}").get_data(as_text=True)
