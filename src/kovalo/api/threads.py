"""Router /threads: cienka warstwa nad CopilotService (FR-6.1; kontrakt F3, 8.1; F4, 9;
ADR-0010, ADR-0012).

Bez reguł biznesowych: walidacja wejścia, wywołanie serwisu, mapowanie błędów serwisu na
kody HTTP. Endpointy są synchroniczne (`def`): graf i PostgresSaver są synchroniczne, więc
FastAPI uruchamia je w puli wątków. Ścieżki `/stream` zgłaszają błędy wykrywalne przed turą
jako zwykły JSON ApiError, a potem zwracają text/event-stream (tool_start, tool_end, final
albo error).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from kovalo.api.deps import get_copilot_service, get_settings
from kovalo.api.schemas import ApiError, MessageIn, ResumeIn, ThreadCreated, TraceOut
from kovalo.api.sse import SSE_HEADERS, sse_body
from kovalo.config import Settings
from kovalo.workflow import (
    CheckpointerUnavailableError,
    CopilotService,
    NotPausedError,
    PendingPause,
    StalePauseError,
    ThreadBusyError,
    ThreadPausedError,
    ThreadState,
    ThreadStuckError,
    TurnResult,
)

__all__ = ["ApiProblem", "router"]

_ERRORS: dict[int | str, dict[str, Any]] = {
    409: {"model": ApiError, "description": "Konflikt stanu wątku (pauza, inna tura, ...)"},
    422: {"model": ApiError, "description": "Niepoprawne żądanie"},
    500: {"model": ApiError, "description": "Nieoczekiwany błąd (szczegóły w logu)"},
    503: {"model": ApiError, "description": "Copilot niedostępny (klucz LLM-a, Postgres)"},
}

_SSE: dict[int | str, dict[str, Any]] = {
    200: {
        "content": {"text/event-stream": {}},
        "description": "Strumień SSE: tool_start*, tool_end*, final albo error",
    },
    **_ERRORS,
}

router = APIRouter(prefix="/threads", tags=["threads"])
Service = Annotated[CopilotService, Depends(get_copilot_service)]
AppSettings = Annotated[Settings, Depends(get_settings)]


class ApiProblem(Exception):
    """Błąd do zwrócenia jako ApiError (obsługuje go handler w main.py)."""

    def __init__(
        self, status: int, error: str, detail: str, pending: PendingPause | None = None
    ) -> None:
        super().__init__(detail)
        self.status = status
        self.body = ApiError(error=error, detail=detail, pending=pending)


def _call[T](fn: Callable[[], T]) -> T:
    """Wywołanie serwisu; jego błędy → ApiProblem z kodem z kontraktu (8.1)."""
    try:
        return fn()
    except ThreadPausedError as exc:
        raise ApiProblem(409, "thread_paused", str(exc), exc.pending) from exc
    except StalePauseError as exc:
        raise ApiProblem(409, "stale_pause", str(exc), exc.pending) from exc
    except NotPausedError as exc:
        raise ApiProblem(409, "not_paused", str(exc)) from exc
    except ThreadBusyError as exc:
        raise ApiProblem(409, "thread_busy", str(exc)) from exc
    except ThreadStuckError as exc:
        raise ApiProblem(
            409,
            "thread_stuck",
            f"{exc} Ten wątek nie przyjmie już wiadomości; zacznij nową rozmowę.",
        ) from exc
    except CheckpointerUnavailableError as exc:  # R1-2: utrata Postgresa w trakcie pracy
        raise ApiProblem(503, "copilot_unavailable", str(exc)) from exc
    except ValueError as exc:  # pusta / za długa wiadomość, value i text razem
        raise ApiProblem(422, "invalid_request", str(exc)) from exc


def _event_stream(stream: Any, settings: Settings) -> StreamingResponse:
    """text/event-stream z gotowych bajtów sse_body (StreamingResponse zamiast
    fastapi.sse.EventSourceResponse: ta druga oczekuje generatora w samym endpoincie)."""
    return StreamingResponse(
        sse_body(stream, ping_s=settings.services.sse_ping_s),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.post("", status_code=201, response_model=ThreadCreated)
def create_thread() -> ThreadCreated:
    """Nowy wątek (UUID4) bez rejestru: checkpoint powstaje przy pierwszej wiadomości."""
    return ThreadCreated(thread_id=str(uuid.uuid4()))


@router.post("/{thread_id}/messages", response_model=TurnResult, responses=_ERRORS)
def send_message(thread_id: uuid.UUID, body: MessageIn, service: Service) -> TurnResult:
    """Nowa wiadomość menedżera; wątek w pauzie → 409 thread_paused z ładunkiem pauzy."""
    return _call(lambda: service.send_message(str(thread_id), body.text))


@router.post("/{thread_id}/resume", response_model=TurnResult, responses=_ERRORS)
def resume(thread_id: uuid.UUID, body: ResumeIn, service: Service) -> TurnResult:
    """Odpowiedź na bieżącą pauzę: `value` z przycisku albo `text` wpisany w czacie."""
    return _call(
        lambda: service.resume(
            str(thread_id), value=body.value, text=body.text, pause_ref=body.pause_ref
        )
    )


@router.post("/{thread_id}/messages/stream", response_model=None, responses=_SSE)
def send_message_stream(
    thread_id: uuid.UUID, body: MessageIn, service: Service, settings: AppSettings
) -> StreamingResponse:
    """Nowa wiadomość jako strumień SSE kroków agenta i wyniku tury (ADR-0012)."""
    stream = _call(lambda: service.start_message(str(thread_id), body.text))
    return _event_stream(stream, settings)


@router.post("/{thread_id}/resume/stream", response_model=None, responses=_SSE)
def resume_stream(
    thread_id: uuid.UUID, body: ResumeIn, service: Service, settings: AppSettings
) -> StreamingResponse:
    """Odpowiedź na pauzę jako strumień SSE (zwykle sam `final`)."""
    stream = _call(
        lambda: service.start_resume(
            str(thread_id), value=body.value, text=body.text, pause_ref=body.pause_ref
        )
    )
    return _event_stream(stream, settings)


@router.get("/{thread_id}/trace", response_model=TraceOut, responses=_ERRORS)
def get_trace(thread_id: uuid.UUID, service: Service) -> TraceOut:
    """Ostatni trace agenta w wątku (z historii checkpointów); bez tury agenta → trace null."""
    return _call(lambda: service.get_trace(str(thread_id)))


@router.get("/{thread_id}", response_model=ThreadState, responses=_ERRORS)
def get_thread(thread_id: uuid.UUID, service: Service) -> ThreadState:
    """Stan wątku; wątek bez checkpointu → status null (nie 404)."""
    return _call(lambda: service.get_thread(str(thread_id)))
