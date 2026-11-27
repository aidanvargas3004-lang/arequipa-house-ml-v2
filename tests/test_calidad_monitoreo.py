"""Control de calidad de datos, rango probable calibrado y monitoreo contra ventas reales."""
import io

import numpy as np

from app import services
from app.extensions import db
from app.ml import pipeline
from app.ml.calidad import detectar_atipicos, separar_atipicos
from app.ml.datagen import generar_dataset
from app.models import DatasetRegistro, EvaluacionVenta

from .conftest import VIVIENDA, cliente_con_rol, publicar


def _dataset_con_errores(n=1200, errores=24, semilla=3):
    df = generar_dataset(n, semilla)
    df["precio"] = df["precio"].astype(float)
    rng = np.random.default_rng(semilla)
    ids = rng.choice(len(df), errores, replace=False)
    df.loc[ids[: errores // 2], "precio"] *= 6   # precio mal escrito (de más)
    df.loc[ids[errores // 2:], "precio"] /= 6    # precio mal escrito (de menos)
    return df, set(int(i) for i in ids)


# ------------------------------------------------------------ calidad de datos
def test_detecta_los_errores_de_captura_y_casi_no_da_falsos_positivos():
    df, ids = _dataset_con_errores()
    det = set(int(i) for i in detectar_atipicos(df)["indice"])
    assert len(det & ids) >= 0.9 * len(ids)           # encuentra casi todos los errores inyectados
    assert len(det - ids) / len(df) < 0.02            # menos de 2 % de falsos positivos
    limpio = generar_dataset(1200, 3)
    assert len(detectar_atipicos(limpio)) / len(limpio) < 0.02


def test_separar_atipicos_respeta_el_tope_y_el_minimo():
    df, _ = _dataset_con_errores(errores=40)
    limpio, info = separar_atipicos(df, minimo_filas=60)
    assert info["excluidos"] == info["detectados"] and len(limpio) == len(df) - info["excluidos"]
    # con un mínimo imposible de respetar no se descarta nada
    _, info2 = separar_atipicos(df, minimo_filas=len(df))
    assert info2["excluidos"] == 0


def test_entrenar_registra_la_depuracion_y_se_puede_desactivar():
    df, _ = _dataset_con_errores(n=900, errores=20)
    con = pipeline.entrenar(df, excluir_atipicos=True)["metricas"]
    sin = pipeline.entrenar(df, excluir_atipicos=False)["metricas"]
    assert con["calidad"]["excluidos"] >= 18 and con["n_total"] == len(df) - con["calidad"]["excluidos"]
    assert sin["calidad"]["excluidos"] == 0 and sin["n_total"] == len(df)


def test_panel_de_calidad_y_eliminacion_de_atipicos(app):
    admin = cliente_con_rol(app, "administrador")
    with app.app_context():
        services.cargar_sintetico(300, semilla=4)
        base = DatasetRegistro.query.filter_by(distrito="Yanahuara", tipo="Casa").first()
        datos = {c: getattr(base, c) for c in services.COLUMNAS_DATASET}
        datos["precio"] = base.precio * 12                       # error de captura evidente
        services.agregar_registro(datos, "csv")
        db.session.commit()
        assert services.calidad_dataset()["detectados"] >= 1
    html = admin.get("/admin/dataset").get_data(as_text=True)
    assert "Calidad de los datos" in html and "Precio por m²" in html
    r = admin.post("/admin/dataset/atipicos/eliminar", follow_redirects=True)
    assert r.status_code == 200 and "atípicos" in r.get_data(as_text=True)
    with app.app_context():
        assert services.calidad_dataset()["detectados"] == 0


def test_entrenar_desde_el_panel_con_y_sin_depuracion(app):
    admin = cliente_con_rol(app, "administrador")
    with app.app_context():
        services.cargar_sintetico(300, semilla=9)
        m = services.lanzar_entrenamiento(None, activar=False, sincrono=True, excluir_atipicos=True)
        assert m.metricas["calidad"]["total_inicial"] >= 300 and "intervalo" in m.metricas
        mid = m.id
    html = admin.get(f"/admin/modelos/{mid}").get_data(as_text=True)
    assert "Depuración de datos y rango probable" in html and "Rango probable calibrado" in html
    assert "Excluir registros atípicos" in admin.get("/admin/modelos").get_data(as_text=True)


# --------------------------------------------------------------- rango calibrado
def test_el_rango_calibrado_cubre_cerca_del_nivel_pedido():
    df = generar_dataset(3000, 21)
    inf = pipeline.entrenar(df)["metricas"]["intervalo"]
    assert inf["nivel"] == 0.8 and 68 <= inf["cobertura_prueba"] <= 92   # meta 80 % ± error de muestreo
    assert inf["ancho_mediano_pct"] > 0


def test_estimar_usa_el_rango_calibrado_y_respeta_el_margen_extra(app):
    with app.app_context():
        datos = {"tipo": "Departamento", "distrito": "Cayma", "area_construida": 110.0, "area_terreno": 110.0,
                 "habitaciones": 3, "banos": 2, "pisos": 1, "antiguedad": 5, "estado_conservacion": "Bueno",
                 "acabados": "Alta calidad", "cochera": 1, "jardin": 0, "piscina": 0, "ascensor": 1, "seguridad": 0,
                 "cerca_colegios": 0, "cerca_hospitales": 0, "cerca_comercial": 0, "transporte_publico": 1}
        e = services.estimar(datos, con_factores=False)
        assert e["precio_min"] < e["precio"] < e["precio_max"]
        _, bundle = services.obtener_bundle()
        assert "intervalo" in bundle
        services.guardar_parametro("margen_extra", 0.2)
        e2 = services.estimar(datos, con_factores=False)
        assert e2["precio_min"] < e["precio_min"] and e2["precio_max"] > e["precio_max"]


# ----------------------------------------------------------------- monitoreo
def test_marcar_vendida_registra_la_evaluacion_y_el_panel_la_muestra(app):
    v = cliente_con_rol(app, "vendedor")
    admin = cliente_con_rol(app, "administrador")
    publicar(v)
    admin.post("/admin/publicaciones/1/moderar", data={"accion": "aprobar"})
    v.post("/vendedor/publicaciones/1/vendida", data={"precio_final": "500000"})   # sin aportar al dataset
    with app.app_context():
        ev = EvaluacionVenta.query.one()
        assert ev.origen == "plataforma" and ev.precio_real == 500000 and ev.propiedad_id == 1
        assert ev.precio_estimado > 0 and ev.precio_min < ev.precio_estimado < ev.precio_max
        assert DatasetRegistro.query.filter_by(origen="venta").count() == 0
        r = services.resumen_monitoreo()
        assert r["n"] == 1 and r["mape"] >= 0 and "por_version" in r
    html = admin.get("/admin/monitoreo").get_data(as_text=True)
    assert "Ventas evaluadas" in html and "Plataforma".lower() in html.lower()


def test_registrar_venta_manual_y_evaluar_csv(app):
    admin = cliente_con_rol(app, "administrador")
    vivienda = {**VIVIENDA, "precio": "480000"}
    r = admin.post("/admin/monitoreo/venta", data={**vivienda, "entrenar": "1"}, follow_redirects=True)
    assert r.status_code == 200 and "Evaluación registrada" in r.get_data(as_text=True)
    csv = ("tipo,distrito,area_construida,habitaciones,baños,antigüedad,precio\n"
           "Casa,Yanahuara,200,4,3,5,900000\nDepartamento,Cayma,90,3,2,8,430000\nCasa,Cayma,200,4,3,5,abc\n")
    r = admin.post("/admin/monitoreo/csv", data={"archivo": (io.BytesIO(csv.encode()), "ventas.csv")},
                   content_type="multipart/form-data", follow_redirects=True)
    assert "2 ventas evaluadas, 1 omitidas" in r.get_data(as_text=True)
    with app.app_context():
        assert EvaluacionVenta.query.count() == 3
        assert EvaluacionVenta.query.filter_by(origen="csv").count() == 2
        assert DatasetRegistro.query.filter_by(origen="venta").count() == 1   # solo la marcada «entrenar»
    export = admin.get("/admin/monitoreo/exportar")
    assert export.status_code == 200 and b"error_pct" in export.data


def test_monitoreo_solo_para_administradores(app):
    for rol in ("comprador", "vendedor"):
        c = cliente_con_rol(app, rol)
        assert c.get("/admin/monitoreo").status_code in (302, 403)
        assert c.post("/admin/monitoreo/csv").status_code in (302, 403)
    assert app.test_client().get("/admin/monitoreo").status_code in (302, 401, 403)


# ------------------------------------------------ compatibilidad con modelos antiguos
def test_un_modelo_entrenado_antes_de_las_mejoras_sigue_funcionando(app):
    """Los modelos ya guardados no tienen rango calibrado ni datos de depuración."""
    from app.models import ModeloML
    admin = cliente_con_rol(app, "administrador")
    with app.app_context():
        modelo, bundle = services.obtener_bundle()
        bundle.pop("intervalo", None)                       # bundle antiguo: sin rango calibrado
        m = db.session.get(ModeloML, modelo.id)
        m.metricas = {k: v for k, v in m.metricas.items() if k not in ("calidad", "intervalo")}
        db.session.commit()
        datos = {"tipo": "Casa", "distrito": "Yanahuara", "area_construida": 180.0, "area_terreno": 260.0,
                 "habitaciones": 4, "banos": 3, "pisos": 2, "antiguedad": 5, "estado_conservacion": "Bueno",
                 "acabados": "Lujo", "cochera": 2, "jardin": 1, "piscina": 0, "ascensor": 0, "seguridad": 1,
                 "cerca_colegios": 1, "cerca_hospitales": 0, "cerca_comercial": 0, "transporte_publico": 0}
        e = services.estimar(datos)
        assert e["precio_min"] < e["precio"] < e["precio_max"] and 0.05 <= e["margen"] < 0.5
        mid = m.id
    assert admin.get(f"/admin/modelos/{mid}").status_code == 200
    r = admin.post("/tasar", data={**VIVIENDA, "tipo": "Casa", "area_terreno": "150"})
    assert r.status_code in (200, 302)
