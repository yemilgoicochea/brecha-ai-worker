# Brecha AI Worker

Worker Python que procesa trabajos de clasificación de proyectos desde Google Cloud Pub/Sub usando Vertex AI (Gemini). Parte del sistema **SNPMGI — Sistema Brecha** para el Perú.

## Arquitectura

```
Pub/Sub (brecha-classification-topic)
        │
        ▼
  [Worker recibe mensaje]
        │
        ▼
  Supabase: status → processing
        │
        ▼
  Vertex AI (Gemini 3.1 Pro Preview, location=global)
        │
        ▼
  Supabase: guarda clasificaciones + status → completed
```

## Prerequisitos

- Python 3.11+
- Docker
- Google Cloud Project con:
  - Pub/Sub habilitado
  - Vertex AI / Agent Platform API habilitada
  - Service Account con permisos `pubsub.subscriber` y `aiplatform.user`
- Proyecto Supabase con el schema DDL aplicado

## Configuración local

```bash
# 1. Crear entorno virtual
python -m venv venv
.\venv\Scripts\activate        # Windows
source venv/bin/activate       # Linux/Mac

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales

# 4. Autenticación GCP
set GOOGLE_APPLICATION_CREDENTIALS=C:\ruta\a\key.json   # Windows
export GOOGLE_APPLICATION_CREDENTIALS=/ruta/a/key.json  # Linux/Mac
```

## Variables de entorno

| Variable | Default | Descripción |
|---|---|---|
| `GCP_PROJECT_ID` | — | ID del proyecto GCP (requerido) |
| `GCP_LOCATION` | `global` | Ubicación Vertex AI (`global` para Gemini 3.x) |
| `PUBSUB_SUBSCRIPTION_ID` | `brecha-worker-sub` | Suscripción Pub/Sub |
| `PUBSUB_TOPIC_ID` | `brecha-classification-topic` | Topic Pub/Sub |
| `GEMINI_MODEL_NAME` | `gemini-3.1-pro-preview` | Modelo Vertex AI |
| `GEMINI_MAX_RETRIES` | `3` | Reintentos ante fallo |
| `GEMINI_RETRY_DELAY` | `2` | Segundos entre reintentos |
| `SUPABASE_URL` | — | URL del proyecto Supabase (requerido) |
| `SUPABASE_KEY` | — | Service role key de Supabase (requerido) |
| `ENVIRONMENT` | `development` | Entorno (`development` / `production`) |
| `LOG_LEVEL` | `INFO` | Nivel de logs |
| `WORKER_TIMEOUT` | `300` | Timeout por mensaje en segundos |

## Ejecutar

```bash
# Local
python -m app.main

# Docker
docker build -t brecha-ai-worker:latest .
docker run --rm --env-file .env brecha-ai-worker:latest
```

## Despliegue (Cloud Run)

El deploy se realiza automáticamente via GitHub Actions (`.github/workflows/build-and-deploy.yml`) al hacer push a `main`.

Las variables de entorno se configuran como variables del servicio en Cloud Run. La autenticación con GCP usa el Service Account vinculado al servicio.

## Estructura del proyecto

```
brecha-ai-worker/
├── app/
│   ├── main.py                  # Punto de entrada
│   ├── core/
│   │   ├── config.py            # Configuración (pydantic-settings)
│   │   └── logging_config.py    # Setup de logs
│   ├── models/
│   │   └── schemas.py           # Modelos Pydantic
│   └── services/
│       ├── pubsub_service.py    # Escucha Pub/Sub
│       ├── gemini_service.py    # Clasificación con Vertex AI
│       ├── supabase_service.py  # Persistencia en Supabase
│       └── worker_service.py    # Orquestación principal
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

## Tests

Los tests cubren las tres capas del worker sin necesidad de credenciales reales de GCP ni Supabase — todo está mockeado con `unittest.mock`.

### Cobertura

| Archivo | Qué se prueba |
|---|---|
| `test_supabase_service.py` | `update_query_status` (con/sin campos opcionales, excepción), `save_classifications` (éxito, lista vacía, fallo), `get_query` |
| `test_gemini_service.py` | `classify` (éxito, error JSON, reintentos, título vacío, catálogo no cargado), `cargar_o_actualizar_catalogo` (carga inicial, fallo, mantiene catálogo previo) |
| `test_worker_service.py` | Happy path completo, fallo Gemini, fallo al guardar en Supabase, filtro de labels `NO_CLASIFICADO`, payload inválido, múltiples clasificaciones en orden |

### Ejecutar

```bash
# 1. Activar entorno virtual (si no está activo)
.\venv\Scripts\activate        # Windows
source venv/bin/activate       # Linux/Mac

# 2. Instalar dependencias de test (ya incluidas en requirements.txt)
pip install pytest pytest-asyncio pytest-cov

# 3. Correr todos los tests
pytest tests/ -v

# 4. Con reporte de cobertura en terminal
pytest tests/ -v --cov=app --cov-report=term-missing

# 5. Generar reporte HTML (se abre htmlcov/index.html)
pytest tests/ --cov=app --cov-report=html
```

### Resultado esperado

```
tests/test_supabase_service.py ..........   10 passed
tests/test_gemini_service.py  ..............  14 passed
tests/test_worker_service.py  ............   12 passed
```

> Los tests **no requieren** `.env`, credenciales GCP ni conexión a Supabase.

## Troubleshooting

**404 Publisher Model not found**
- Verificar que `GCP_LOCATION=global` (los modelos Gemini 3.x usan ubicación lógica global)
- Verificar que la Agent Platform API esté habilitada en el proyecto GCP

**Subscription not found**
- Verificar `PUBSUB_SUBSCRIPTION_ID` en `.env`
- Confirmar que el Service Account tiene el rol `roles/pubsub.subscriber`

**Supabase connection failed**
- Verificar `SUPABASE_URL` y `SUPABASE_KEY`
- Confirmar que el schema DDL está aplicado en la base de datos
