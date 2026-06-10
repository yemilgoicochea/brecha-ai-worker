"""Main entry point for Brecha AI Worker."""

import asyncio
import http.server
import json
import logging
import threading
from typing import Any, Dict, Tuple

from google.cloud import pubsub_v1

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.services.gemini_service import GeminiService
from app.services.supabase_service import SupabaseService
from app.services.worker_service import WorkerService
from app.services.beto_service import BetoService

logger = logging.getLogger(__name__)


def _run_health_server(port: int) -> None:
    """Servidor HTTP mínimo requerido por Cloud Run para health checks."""

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"OK")
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            pass

    http.server.HTTPServer(("", port), _Handler).serve_forever()


class AsyncWorker:
    """
    Gestiona dos suscripciones Pub/Sub en paralelo con asyncio.

    - Suscripción de proyectos: clasifica cada mensaje con Gemini.
    - Suscripción de control:   recarga el catálogo en caliente sin reiniciar.

    Patrón: los callbacks de Pub/Sub corren en threads; usan
    loop.call_soon_threadsafe para depositar mensajes en asyncio.Queue,
    donde los consumers async los procesan sin bloquear el event loop.
    """

    def __init__(self):
        self._supabase = SupabaseService()
        self._gemini = GeminiService(self._supabase)
        self._beto = BetoService()
        self._beto.load()  # carga BETO una sola vez al arrancar
        self._worker = WorkerService(self._supabase, self._gemini, self._beto)
        self._project_queue: asyncio.Queue[Tuple[Any, Dict[str, Any]]] = asyncio.Queue()
        self._catalog_queue: asyncio.Queue[Any] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop

    # ------------------------------------------------------------------ #
    # Pub/Sub callbacks (ejecutan en threads del SDK)                     #
    # ------------------------------------------------------------------ #

    def _project_callback(self, message) -> None:
        try:
            payload = json.loads(message.data.decode("utf-8"))
            logger.debug("Mensaje Pub/Sub recibido: %s", json.dumps(payload, ensure_ascii=False))
            self._loop.call_soon_threadsafe(
                self._project_queue.put_nowait, (message, payload)
            )
        except Exception as e:
            logger.error(f"Error deserializando mensaje de proyecto: {e}")
            message.nack()

    def _catalog_callback(self, message) -> None:
        logger.info("Mensaje de refresco de catálogo recibido")
        self._loop.call_soon_threadsafe(self._catalog_queue.put_nowait, message)

    # ------------------------------------------------------------------ #
    # Consumers async                                                      #
    # ------------------------------------------------------------------ #

    async def _consume_projects(self) -> None:
        """Procesa mensajes de clasificación de forma asíncrona."""
        while True:
            message, payload = await self._project_queue.get()
            try:
                await self._worker.process_message(payload)
            except Exception as e:
                logger.error(f"Error inesperado en consumer de proyectos: {e}", exc_info=True)
            finally:
                # Siempre ack: process_message ya guardó el estado en Supabase.
                # nack() causaría reentrega infinita para errores no recuperables.
                message.ack()

    async def _consume_catalog_refresh(self) -> None:
        """Recarga el catálogo en memoria al recibir una señal de control."""
        while True:
            try:
                message = await self._catalog_queue.get()
                await self._gemini.cargar_o_actualizar_catalogo()
                message.ack()
                logger.info("Catálogo refrescado en memoria exitosamente")
            except Exception as e:
                logger.error(f"Error al refrescar catálogo: {e}", exc_info=True)
                message.nack()

    # ------------------------------------------------------------------ #
    # Entrypoint                                                           #
    # ------------------------------------------------------------------ #

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()

        # 1. Inicializar modelo con catálogo fresco antes de aceptar mensajes
        await self._gemini.cargar_o_actualizar_catalogo()

        # 2. Abrir las dos suscripciones streaming pull
        subscriber = pubsub_v1.SubscriberClient()

        project_path = subscriber.subscription_path(
            settings.GCP_PROJECT_ID, settings.PUBSUB_SUBSCRIPTION_ID
        )
        catalog_path = subscriber.subscription_path(
            settings.GCP_PROJECT_ID, settings.PUBSUB_CATALOG_REFRESH_SUBSCRIPTION_ID
        )

        future_projects = subscriber.subscribe(
            project_path,
            callback=self._project_callback,
            flow_control=pubsub_v1.types.FlowControl(max_messages=1),
        )
        future_catalog = subscriber.subscribe(
            catalog_path,
            callback=self._catalog_callback,
            flow_control=pubsub_v1.types.FlowControl(max_messages=1),
        )

        logger.info(f"Escuchando proyectos en:  {project_path}")
        logger.info(f"Escuchando control en:    {catalog_path}")

        # 3. Correr ambos consumers en paralelo indefinidamente
        try:
            await asyncio.gather(
                self._consume_projects(),
                self._consume_catalog_refresh(),
            )
        finally:
            future_projects.cancel()
            future_catalog.cancel()
            subscriber.close()


async def _main_async() -> None:
    worker = AsyncWorker()
    await worker.run()


def main() -> None:
    setup_logging()
    logger.info("Iniciando Brecha AI Worker...")

    threading.Thread(target=_run_health_server, args=(settings.PORT,), daemon=True).start()
    logger.info(f"Health server escuchando en puerto {settings.PORT}")

    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
