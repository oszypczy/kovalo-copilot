"""Serwer usługi ui: FastAPI z /health i zamontowanym Chainlitem (kontrakt F0, sekcja 6).

`/health` jest rejestrowane przed montowaniem Chainlita, więc ma pierwszeństwo.
"""

from __future__ import annotations

from pathlib import Path

import uvicorn
from chainlit.utils import mount_chainlit
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from kovalo import __version__
from kovalo.config import get_settings
from kovalo.health import CheckResult, build_report, check_http, http_status

CHAINLIT_APP = Path(__file__).with_name("app.py")
CHAINLIT_PATH = "/"


def _create_app() -> FastAPI:
    app = FastAPI(title="Kovalo Decision Copilot UI", version=__version__)

    @app.get("/health", tags=["health"])
    async def health() -> JSONResponse:
        settings = get_settings()  # czytane w czasie żądania
        timeout_s = settings.services.health_timeout_s

        async def api() -> CheckResult:
            url = f"{settings.services.api_url.rstrip('/')}/health"
            return await check_http(url, timeout_s=timeout_s)

        report = await build_report("ui", {"api": api}, timeout_s=timeout_s)
        return JSONResponse(report.model_dump(mode="json"), status_code=http_status(report))

    mount_chainlit(app=app, target=str(CHAINLIT_APP), path=CHAINLIT_PATH)
    return app


app = _create_app()


def run() -> None:
    """Punkt wejścia `kovalo-ui`: uvicorn na 0.0.0.0:services.ui_port."""
    uvicorn.run("kovalo.ui.server:app", host="0.0.0.0", port=get_settings().services.ui_port)
