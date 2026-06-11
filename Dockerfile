# ── Stage 1: Runtime dependencies (sin torch) ────────────────────────────
FROM python:3.11-slim AS deps

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# ── Stage 2: Conversión BETO → ONNX (torch solo aquí, no en runtime) ─────
FROM deps AS converter

WORKDIR /app

# Instalar torch + optimum solo para la conversión
RUN pip install torch==2.3.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install "optimum[exporters]<2.0" onnx

# Copiar modelo BETO (model.safetensors descargado de GCS en el paso de CI)
COPY modelo_beto/ modelo_beto/

# Convertir a ONNX y copiar label_mapping.json al directorio de salida
RUN python -c "\
from optimum.exporters.onnx import main_export; \
import shutil; \
main_export('modelo_beto', output='modelo_beto_onnx', task='text-classification'); \
shutil.copy('modelo_beto/label_mapping.json', 'modelo_beto_onnx/label_mapping.json'); \
print('ONNX conversion completed')"

# ── Stage 3: Runtime (solo onnxruntime, sin torch) ───────────────────────
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Venv limpio sin torch (de stage 1)
COPY --from=deps /opt/venv /opt/venv

# Modelo convertido a ONNX (de stage 2)
COPY --from=converter /app/modelo_beto_onnx /app/modelo_beto_onnx

# Código de la aplicación
COPY app/ app/

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "print('OK')" || exit 1

CMD ["python", "-m", "app.main"]
