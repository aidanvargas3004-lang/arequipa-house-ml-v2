# Arequipa House ML

Plataforma web para **estimar el precio de casas, departamentos y dúplex en Arequipa con Machine Learning** y para **publicar, buscar, comparar y contactar** viviendas. Combina un modelo de tasación (Random Forest vs. Gradient Boosting, scikit-learn) con un marketplace con tres roles: comprador, vendedor y administrador.

> **Las estimaciones son referenciales** y no reemplazan una tasación oficial. El modelo inicial se entrena con un **dataset sintético de demostración** (ver [Datos y Machine Learning](#datos-y-machine-learning)); el sistema está listo para cargar el histórico real de la empresa y reentrenar.

## Funcionalidades

| Rol | Qué puede hacer |
|---|---|
| **Cualquier visitante** | Ver el catálogo, buscar con filtros (tipo, distrito, precio, área, habitaciones, baños, antigüedad, conservación, extras), ver el detalle con galería de fotos, comparar hasta 4 viviendas. |
| **Comprador** | Todo lo anterior + tasar viviendas, guardar favoritos, contactar al vendedor (mensajes), solicitar visitas presenciales/virtuales, ver su historial de tasaciones. |
| **Vendedor** | Lo del comprador + publicar viviendas con fotos (con **precio sugerido por el modelo**), editar/eliminar sus publicaciones, responder mensajes, confirmar/rechazar visitas y marcar la vivienda como vendida (aportando opcionalmente el precio real al histórico). |
| **Administrador** | Panel con estadísticas, **moderación** (aprobar / rechazar con motivo / editar / eliminar), gestión de usuarios y roles, **dataset histórico** (CRUD, importar/exportar CSV), **modelos ML** (entrenar en segundo plano, comparar métricas, activar versiones) y **parámetros de tasación** (factor de mercado, margen y ajuste por distrito). |

Cada tasación muestra: precio estimado, rango probable, precio por m², **factores que subieron o bajaron el precio**, vivienda similares publicadas y la versión del modelo utilizada.

## Obtener el código

```bash
git clone https://github.com/aidanvargas3004-lang/arequipa-house-ml-v2
cd arequipa-house-ml-v2
```

Para colaborar (ramas, commits y Pull Requests), consulta la [guía de contribución](CONTRIBUTING.md).

## Inicio rápido (desarrollo)

Requisitos: Python 3.11 o superior (desarrollado y probado con 3.13).

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py                        # http://127.0.0.1:5000
```

El primer arranque (≈6 s) crea la base SQLite, 4 000 registros sintéticos, entrena el modelo inicial y siembra datos de demostración. Cuentas de prueba (`python run.py` activa `SEED_DEMO=1`):

| Rol | Correo | Contraseña |
|---|---|---|
| Administrador | `admin@arequipahouse.pe` | `Admin12345` |
| Vendedor | `vendedor@demo.pe` · `vendedor2@demo.pe` | `Demo12345` |
| Comprador | `comprador@demo.pe` | `Demo12345` |

> Cambia estas claves o no actives `SEED_DEMO` en un entorno real.

Copia `.env.example` a `.env` para ajustar variables (`SECRET_KEY`, `DATABASE_URL`, `ADMIN_PASSWORD`, …).

## Datos y Machine Learning

### Publicar viviendas como administrador

En **Administración → Moderación → + Publicar vivienda**, el administrador puede
crear un anuncio con características, precio y al menos una foto válida. Se
publica directamente como aprobado y queda asociado a la cuenta administrativa
que lo creó. Se reutilizan las validaciones, el procesamiento seguro de fotos y
el cálculo de precio sugerido del formulario del vendedor. El estado y el
responsable se determinan en el servidor.

La publicación se administra desde Moderación y el administrador puede atender
sus conversaciones y solicitudes desde Mensajes y Visitas. Esos accesos siguen
limitados a las conversaciones y viviendas donde participa su cuenta; no dan
acceso a conversaciones ajenas. Crear una publicación no agrega automáticamente
un registro al dataset histórico. Los vendedores siguen enviando sus anuncios
a revisión como antes.

**Variables del modelo (19):** tipo, distrito, área construida, área del terreno, pisos, habitaciones, baños, cocheras, antigüedad, estado de conservación, calidad de acabados, jardín, piscina, ascensor, seguridad, cercanía a colegios / hospitales / centros comerciales y acceso a transporte público.

**Entrenamiento (`app/ml/pipeline.py`):**
1. Los registros se dividen en 70 % entrenamiento · 15 % validación · 15 % prueba.
2. Se entrenan **Random Forest** y **Gradient Boosting** sobre `log(precio)` (One-Hot para tipo y distrito).
3. Se elige el de **menor RMSE en validación**; las métricas finales (**MAE, RMSE, R²**, error porcentual) se reportan sobre el conjunto de **prueba**, que no participó en la selección.
4. Se calcula la importancia global de variables por permutación.

**Explicación de cada tasación:** método de oclusión — se reemplaza cada característica por el valor típico del mercado (mediana/moda) y se mide cuánto cambia el precio. Es una aproximación, no SHAP.

**Rango de precio:** `precio × (1 ± error)` con el error porcentual mediano del modelo en validación (mínimo 5 %) más el margen extra configurable.

### Dataset sintético vs. datos reales

`app/ml/datagen.py` genera viviendas con un modelo hedónico (precio base por distrito × características × ruido). Los **precios base por distrito (`app/ml/constants.py`) son aproximaciones** que conviene ajustar. Como el simulador define la fórmula, el R² obtenido (≈ 0.96) **valida el pipeline, no la precisión sobre el mercado real**; la aplicación lo advierte en pantalla mientras el modelo se entrene solo con datos sintéticos.

Para usar datos reales: **Administración → Dataset → Importar CSV** (marca «Reemplazar los datos sintéticos») y luego **Modelos ML → Entrenar nueva versión**. Mientras entrena (en segundo plano), el modelo activo sigue atendiendo estimaciones.

**Captura asistida:** en **Dataset → Capturar desde un anuncio** pegas el texto de un anuncio real, el sistema propone los datos (tipo, distrito, áreas, habitaciones, precio, extras…) y tú los revisas antes de guardarlos con origen «Anuncio de portal». No visita ningún sitio web ni hace scraping; respeta los términos de uso del portal y no pegues datos personales.

**Script de scraping (opcional, fuera de la web):** `scripts/scraper_anuncios.py` lee un catálogo de anuncios y genera `ventas_reales_arequipa.csv` para importarlo en el panel. Consulta `robots.txt`, espera entre páginas y se detiene ante un bloqueo (403/429). También lee una página guardada (`--html`). Requiere `pip install requests beautifulsoup4 lxml`. Úsalo solo con pocas páginas y respetando los términos del portal.

Formato del CSV (separador `,` o `;`, UTF-8; descarga la plantilla desde el panel). Precios en **soles**.

| Obligatorias | Opcionales (valor por defecto) |
|---|---|
| `tipo` (Casa / Departamento / Dúplex), `distrito`, `area_construida`, `habitaciones`, `banos`, `antiguedad`, `precio` | `area_terreno`, `pisos`, `estado_conservacion` (Bueno), `acabados` (Estándar), `cochera` (0), `jardin`, `piscina`, `ascensor`, `seguridad`, `cerca_colegios`, `cerca_hospitales`, `cerca_comercial`, `transporte_publico` (sí/no, 1/0) |

Se aceptan alias comunes (`baños`, `dormitorios`, `garaje`, `m2`, …), tildes y comas decimales. Las filas inválidas se omiten y se informan con su número de fila.

Las ventas marcadas como «vendida» por un vendedor (con su consentimiento) se agregan al histórico con origen `venta`.

## Base de datos

### Segmentación K-Means (administrador)

En **Administración → Segmentos**, pulsa **Calcular / actualizar segmentos**.
También hay un acceso desde Dataset. El análisis utiliza exclusivamente los
registros con origen `sintetico` existentes en la base; no genera datos nuevos ni
mezcla registros reales. Requiere al menos 30 filas y diversidad de perfiles.
Si eliminaste los sintéticos al importar un CSV, no podrá calcularse con esa base.

`app/ml/segmentacion.py` implementa un aprendizaje no supervisado independiente:

1. Convierte conservación y acabados a niveles ordinales y estandariza las
   cantidades mediante `StandardScaler`. En departamentos anula terreno y pisos.
2. Codifica tipo y distrito con one-hot (peso 0.7); los extras booleanos tienen
   peso 0.35 para que su cantidad no domine. Son decisiones iniciales ajustables.
3. Prueba K-Means de 2 a 6 grupos, limitado por el tamaño y diversidad del dataset
   (como máximo un grupo por cada 10 filas, sin garantizar ese tamaño por grupo).
   Usa `init="k-means++"`, `n_init=10` y semilla 42.
4. Selecciona la mayor silueta, evaluada sobre una muestra reproducible de hasta
   1000 filas. La silueta mide separación de grupos entre −1 y 1, no precisión de
   precios ni validez sobre el mercado real. Puede ser baja aunque sea la mejor
   de las opciones probadas.
5. Resume cada grupo mediante medianas de área, habitaciones, baños, antigüedad,
   precio y precio por m², además del tipo y distrito más frecuentes. El precio
   no interviene en la formación de grupos. Los segmentos se numeran por área
   mediana y sus números pueden cambiar al recalcular.

El resultado se guarda aparte en `MODELS_FOLDER/segmentacion_sintetica.joblib`.
Una huella de los valores sintéticos detecta cambios y muestra un aviso para
recalcular; abrir el panel no vuelve a entrenar K-Means. Las publicaciones
aprobadas actuales se asignan al centro más cercano y se listan por segmento,
sin incorporarlas al entrenamiento. Una asignación no garantiza que una
publicación sea cercana al perfil: siempre se elige el centro disponible más
próximo. No se cambian versiones, métricas ni archivos del modelo de tasación.

Los resultados describen **perfiles simulados**, no demanda ni segmentos reales
del mercado. Si se pierde el archivo del análisis, vuelve a calcularlo desde el
panel. No requiere nuevas dependencias ni cambios al esquema de la base.

Referencias: [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
y [silueta](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html).

### Recomendaciones de viviendas por vecinos cercanos (KNN)

Al consultar una tasación o abrir el detalle de una propiedad, se muestran hasta
tres publicaciones aprobadas, ordenadas por similitud. Se excluye la propia
publicación y nunca se recomiendan viviendas pendientes, rechazadas o vendidas.
Si no hay candidatos se muestra un mensaje sin recomendaciones.

Ambos flujos usan `services.recomendar_propiedades`, que consulta el catálogo
actual y delega la búsqueda a `app/ml/recomendaciones.py`. Este módulo utiliza
`NearestNeighbors` de scikit-learn con distancia euclídea: convierte tipo y
distrito a one-hot, escala cantidades y codifica conservación, acabados y extras.
Las escalas fijas y pesos iniciales están comentados en el código; evitan que
metros cuadrados dominen sobre habitaciones o equipamiento. En departamentos
no se comparan terreno ni pisos. Los empates se resuelven por ID de publicación.

Las tarjetas explican coincidencias verificables, como mismo distrito o área
construida dentro de ±15 %. Son las alternativas más cercanas disponibles,
incluso si el catálogo es pequeño; no existe un umbral mínimo de similitud.
El precio no participa en la distancia y no se aplica un filtro de presupuesto.

Este algoritmo recomienda inmuebles; Random Forest y Gradient Boosting siguen
estimando el precio. No requiere reentrenamiento, migraciones ni dependencias
adicionales. El índice se reconstruye por consulta para reflejar cambios del
catálogo; para catálogos grandes convendrá medir latencia y añadir un índice
reutilizable con invalidación al publicar, editar, moderar o vender.

Referencia: [NearestNeighbors en scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html).

* **Desarrollo:** SQLite (`instance/arequipa_house.db`), sin configuración.
* **Producción / PostgreSQL:** define `DATABASE_URL` (acepta `postgres://` y `postgresql://`; se usa el driver `psycopg2`).

```bash
# 1) Crear la base (ejemplo local)
createdb -U postgres arequipa_house          # o desde pgAdmin 4
# 2) Variables
export DATABASE_URL=postgresql://usuario:clave@localhost:5432/arequipa_house
export SECRET_KEY=$(python -c "import secrets;print(secrets.token_urlsafe(48))")
export ADMIN_PASSWORD='UnaClaveSegura123'
# 3) Crear tablas, administrador, dataset y modelo inicial (idempotente)
flask --app wsgi bootstrap
# 4) Servir
gunicorn wsgi:app --workers 1 --threads 4 --timeout 120
```

**Creación manual con pgAdmin:** ejecuta [`docs/01_crear_base_postgresql.sql`](docs/01_crear_base_postgresql.sql) (usuario y base) y luego [`docs/02_tablas_postgresql.sql`](docs/02_tablas_postgresql.sql) (11 tablas, conectado a `arequipa_house`). También puedes omitirlos: la aplicación crea las tablas sola al iniciar.

El esquema SQL de PostgreSQL está en [`docs/schema_postgresql.sql`](docs/schema_postgresql.sql) y el diagrama ER + diccionario de datos en [`docs/modelo_datos.md`](docs/modelo_datos.md) (se regeneran con `PYTHONPATH=. python scripts/export_schema.py`).

También puedes levantar todo con PostgreSQL local: `docker compose up --build` → http://localhost:8000.

## Despliegue

* **Render:** el archivo `render.yaml` define el servicio web y una base PostgreSQL. Crea un *Blueprint* apuntando al repositorio y define `ADMIN_PASSWORD`.
* **Docker:** `docker build -t arequipa-house .` (usa `flask bootstrap` + `gunicorn`).
* **Importante — disco efímero:** las fotos subidas (`instance/uploads`) y los modelos (`instance/models`) viven en disco. En planes gratuitos se pierden al redesplegar. Los modelos se **reentrenan solos** desde la base si falta el archivo, pero las fotos requieren un disco persistente (variable `UPLOAD_FOLDER`) o migrar a almacenamiento de objetos (S3/Cloudinary).
* Mantén `--workers 1` con hilos si usas el entrenamiento en segundo plano en instancias pequeñas (memoria limitada por scikit-learn/pandas).

## Variables de entorno

| Variable | Descripción | Por defecto |
|---|---|---|
| `APP_ENV` | `development` / `production` (exige `SECRET_KEY`, cookies seguras, ProxyFix) | `development` |
| `SECRET_KEY` | Clave de sesiones/CSRF | solo dev |
| `DATABASE_URL` | URL de la base de datos | SQLite local |
| `ADMIN_EMAIL` / `ADMIN_PASSWORD` | Administrador inicial (si no existe ninguno) | `admin@arequipahouse.pe` / generada en producción |
| `SEED_DEMO` | `1` crea usuarios y publicaciones demo | `0` (`run.py`: `1`) |
| `AUTO_BOOTSTRAP` | `1` inicializa al arrancar la app | `1` |
| `UPLOAD_FOLDER` / `MODELS_FOLDER` | Carpetas de fotos y modelos | `instance/…` |
| `ML_SYNTHETIC_ROWS`, `ML_MIN_ROWS`, `ML_N_JOBS` | Tamaño del dataset sintético, mínimo para entrenar, hilos | 4000 · 60 · 1 |
| `COOKIE_SECURE` | Forzar cookies `Secure` (HTTPS) | `1` en producción |

## Comandos

```bash
flask --app wsgi bootstrap            # tablas + parámetros + admin + dataset + modelo (idempotente)
flask --app wsgi train                # entrena y activa una nueva versión (síncrono)
flask --app wsgi seed-demo            # usuarios y publicaciones de demostración
flask --app wsgi create-admin correo@dominio.pe
```

## Pruebas

```bash
pip install -r requirements-dev.txt
python -m pytest -q                                                    # SQLite en memoria
TEST_DATABASE_URL=postgresql://u:p@localhost:5432/prueba python -m pytest -q   # PostgreSQL
```

66 pruebas cubren el pipeline de ML, validación, autenticación y roles, bloqueo por intentos, CSRF, XSS, subida de archivos, moderación, mensajes, visitas, importación CSV, captura asistida de anuncios, recomendaciones KNN, segmentación K-Means, publicación del administrador, reentrenamiento (incluido en segundo plano), parámetros y autoreparación del modelo. El workflow de GitHub Actions las corre con SQLite y PostgreSQL.

El CSS de Tailwind ya está compilado en `app/static/css/tailwind.css`. Solo si cambias clases de las plantillas: `npm install && npm run build:css`.

## Estructura

```
app/
  __init__.py        fábrica de la app, filtros, errores, cabeceras
  config.py          configuración por variables de entorno
  models.py          modelos SQLAlchemy (usuarios, propiedades, tasaciones, dataset, modelos ML…)
  services.py        dataset, entrenamiento, modelo activo, estimación, parámetros
  bootstrap.py       arranque idempotente y datos demo
  blueprints/        auth · main · tasacion · propiedades · vendedor · interaccion · admin
  ml/                constants · schema (validación) · datagen · pipeline (entrenar/predecir/explicar)
  templates/ static/ interfaz (Jinja2 + Tailwind), JS mínimo, Chart.js local
tests/               pytest
docs/                esquema SQL y modelo de datos
```

## Seguridad implementada

Contraseñas con hash (Werkzeug) y política mínima · bloqueo temporal tras 5 intentos fallidos · CSRF en todos los formularios y llamadas AJAX · control de acceso por rol y por dueño del recurso · redirecciones `next` validadas · consultas parametrizadas (SQLAlchemy) · escape automático de Jinja (XSS) · subida de imágenes verificada con Pillow, reencodada a JPEG y con límites de tamaño · cabeceras `X-Frame-Options`, `nosniff`, `Referrer-Policy` · cookies `HttpOnly`/`SameSite` (y `Secure` en producción) · política de privacidad y términos (Ley 29733).

## Alcance y limitaciones

* Estima **precio de venta** de casas, departamentos y dúplex de la provincia de Arequipa; **no** terrenos ni alquileres.
* No hay app móvil nativa (la web es responsive) ni procesamiento de imágenes.
* El modelo es tan bueno como sus datos: con el histórico real hay que revisar las métricas antes de confiar en el resultado.
* Los mensajes son internos a la plataforma (no envían correos); no hay recuperación de contraseña por correo.
