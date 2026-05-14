"""Main entry point for Brecha AI Worker."""

import http.server
import logging
import os
import threading

from app.core.logging_config import setup_logging
from app.services.worker_service import WorkerService


def _run_health_server(port: int) -> None:
    """Run a minimal HTTP health-check server required by Cloud Run."""

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
            pass  # suppress per-request HTTP logs

    server = http.server.HTTPServer(("", port), _Handler)
    server.serve_forever()


def main():
    """Main function to start the worker."""
    logger = setup_logging()
    logger.info("Starting Brecha AI Worker...")

    # Cloud Run sets PORT; also useful for local health checks
    port = int(os.environ.get("PORT", 8080))
    health_thread = threading.Thread(target=_run_health_server, args=(port,), daemon=True)
    health_thread.start()
    logger.info(f"Health server listening on port {port}")

    try:
        worker = WorkerService()
        worker.start()
    except KeyboardInterrupt:
        logger.info("Worker shutdown requested")
    except Exception as e:
        logger.error(f"Fatal error in worker: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
