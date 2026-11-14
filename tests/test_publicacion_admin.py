import io
from datetime import timedelta

from app.extensions import db
from app.models import DatasetRegistro, Propiedad, Usuario, Visita
from app.utils import ahora_lima
from .conftest import cliente_con_rol, datos_publicacion, png_bytes, publicar


def test_admin_publica_directamente_y_atiende_sus_contactos(app, client):
    admin = cliente_con_rol(app, "administrador")
    comprador = cliente_con_rol(app, "comprador")
    otro_admin = cliente_con_rol(app, "administrador", "otro@test.pe")
    form = admin.get("/admin/publicaciones/nueva")
    assert form.status_code == 200 and b'Publicar en el cat' in form.data
    datos = datos_publicacion()
    datos["fotos"] = (io.BytesIO(png_bytes()), "vivienda.png")
    # Los campos sensibles enviados por el cliente no eligen propietario.
    datos["vendedor_id"] = "999999"
    respuesta = admin.post("/admin/publicaciones/nueva", data=datos, content_type="multipart/form-data")
    assert respuesta.status_code == 302
    with app.app_context():
        p = Propiedad.query.one()
        pid = p.id
        assert p.estado == "aprobada" and len(p.fotos) == 1
        assert p.vendedor.email == "administrador@test.pe"
        assert DatasetRegistro.query.count() == 0
        uid = Usuario.query.filter_by(rol="comprador").one().id
        visita = Visita(propiedad_id=pid, comprador_id=uid, fecha_hora=ahora_lima() + timedelta(days=1), modalidad="presencial")
        db.session.add(visita)
        db.session.commit()
        vid = visita.id
    assert "Departamento moderno" in client.get("/propiedades").get_data(as_text=True)
    detalle = admin.get(respuesta.headers["Location"])
    assert detalle.status_code == 200
    assert f"/admin/publicaciones/{pid}/editar" in detalle.get_data(as_text=True)
    contacto = comprador.post(f"/propiedades/{pid}/contactar", data={"contenido": "Me interesa conocer esta vivienda"})
    conversacion = contacto.headers["Location"]
    assert admin.get(conversacion).status_code == 200
    assert admin.post(conversacion, data={"contenido": "Coordinemos una visita"}).status_code == 302
    assert otro_admin.get(conversacion).status_code == 403
    assert admin.get("/mensajes").status_code == 200
    assert "Solicitudes recibidas" in admin.get("/visitas").get_data(as_text=True)
    assert otro_admin.post(f"/visitas/{vid}/responder", data={"accion": "confirmar"}).status_code == 403
    assert admin.post(f"/visitas/{vid}/responder", data={"accion": "confirmar"}).status_code == 302
    assert admin.get(f"/admin/publicaciones/{pid}/editar").status_code == 200


def test_permisos_fotos_y_moderacion_no_se_omiten(app, client):
    admin = cliente_con_rol(app, "administrador")
    assert client.get("/admin/publicaciones/nueva").status_code == 302
    for rol in ("comprador", "vendedor"):
        c = cliente_con_rol(app, rol)
        assert c.get("/admin/publicaciones/nueva").status_code == 403
        assert c.post("/admin/publicaciones/nueva", data=datos_publicacion()).status_code == 403
    assert admin.post("/admin/publicaciones/nueva", data=datos_publicacion()).status_code == 422
    datos = datos_publicacion()
    datos["fotos"] = (io.BytesIO(b"no es una imagen"), "falsa.png")
    assert admin.post("/admin/publicaciones/nueva", data=datos, content_type="multipart/form-data").status_code == 422
    with app.app_context():
        assert Propiedad.query.count() == 0
    vendedor = app.test_client()
    vendedor.post("/login", data={"email": "vendedor@test.pe", "password": "Clave12345"})
    assert publicar(vendedor, estado="aprobada", admin="1").status_code == 302
    with app.app_context():
        assert Propiedad.query.one().estado == "pendiente"
