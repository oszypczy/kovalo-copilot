"""Klient HTTP UI → api (kontrakt F3, 9.1; F4, 10.1; ADR-0010 pkt 6, ADR-0012 pkt 6).

UI nie importuje backendu: ma własne, minimalne modele widoku (`extra="ignore"`), a zgodność
z `TurnResult` pilnuje test kontraktu. Każda odpowiedź spoza 2xx → ApiCallError z ciała
ApiError; brak połączenia albo przekroczony limit → ApiCallError(status=0, 'api_unreachable').
F4: strumień tury (SSE, httpx-sse) i ostatni trace agenta w wątku.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, Literal

import httpx
from httpx_sse import SSEError, aconnect_sse
from pydantic import BaseModel, ConfigDict, ValidationError

__all__ = [
    "ApiCallError",
    "CopilotApiClient",
    "PendingView",
    "SourceView",
    "StreamItem",
    "ToolEventView",
    "TraceView",
    "TurnView",
    "pause_ref",
]


class ApiCallError(RuntimeError):
    """Błąd wywołania api: kod HTTP (0: api nie odpowiada), kod błędu, opis, pauza."""

    def __init__(
        self, status: int, error: str, detail: str, pending: dict[str, Any] | None = None
    ) -> None:
        super().__init__(f"{status} {error}: {detail}")
        self.status = status
        self.error = error
        self.detail = detail
        self.pending = pending


class _View(BaseModel):
    model_config = ConfigDict(extra="ignore")


class SourceView(_View):
    doc_id: str
    chunk_id: str
    text: str


class PendingView(_View):
    type: Literal["clarification", "approval"]
    field: str | None = None
    question: str | None = None
    role: str | None = None
    revision: int | None = None
    approvers: list[str] = []
    summary: str | None = None
    options: list[str] = []


class TurnView(_View):
    thread_id: str
    status: str | None
    replies: list[str] = []
    pending: PendingView | None = None
    resume_rejected: bool = False
    sources: list[SourceView] = []
    error: str | None = None
    evidence_path: str | None = None  # F4: ścieżka dowodowa tury z agentem
    trace: dict[str, Any] | None = None  # F4: nieprzezroczysty dla UI poza trace["status"]


class ToolEventView(_View):
    """Zdarzenie tool_start / tool_end strumienia tury."""

    call_id: str
    parent_call_id: str | None = None
    agent: str = "agent"
    tool: str
    arguments: dict[str, Any] = {}
    origin: str | None = None
    status: str | None = None
    brief: str | None = None
    error: str | None = None


class TraceView(_View):
    thread_id: str
    trace: dict[str, Any] | None = None
    evidence_path: str | None = None


StreamItem = (
    tuple[Literal["tool_start", "tool_end"], ToolEventView] | tuple[Literal["final"], TurnView]
)


def pause_ref(pending: PendingView) -> dict[str, Any]:
    """`pause_ref` dla /resume: typ, rola albo pole, wersja wniosku (jak PendingPause.ref)."""
    key = pending.role if pending.type == "approval" else pending.field
    return {"type": pending.type, "key": key or "", "revision": pending.revision}


class CopilotApiClient:
    """Asynchroniczny klient /threads (httpx). Nowy AsyncClient na każde żądanie."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.transport = transport

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url, timeout=self.timeout_s, transport=self.transport
        )

    @staticmethod
    def _error_from(response: httpx.Response, data: Any) -> ApiCallError:
        payload = data if isinstance(data, dict) else {}
        detail = payload.get("detail")
        return ApiCallError(
            response.status_code,
            str(payload.get("error") or f"http_{response.status_code}"),
            detail if isinstance(detail, str) else (response.text or response.reason_phrase),
            payload.get("pending") if isinstance(payload.get("pending"), dict) else None,
        )

    async def _request(
        self, method: str, path: str, *, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        try:
            async with self._client() as client:
                response = await client.request(method, path, json=body)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise ApiCallError(
                0, "api_unreachable", f"{type(exc).__name__}: {exc}".rstrip(": ")
            ) from exc
        try:
            data: Any = response.json()
        except ValueError:
            data = None
        if not response.is_success:
            raise self._error_from(response, data)
        if not isinstance(data, dict):
            raise ApiCallError(response.status_code, "bad_response", "Odpowiedź api nie jest JSON.")
        return data

    @staticmethod
    def _view(data: dict[str, Any]) -> TurnView:
        try:
            return TurnView.model_validate(data)
        except ValidationError as exc:
            raise ApiCallError(200, "bad_response", f"Nieoczekiwana odpowiedź api: {exc}") from exc

    async def create_thread(self) -> str:
        data = await self._request("POST", "/threads")
        return str(data["thread_id"])

    async def send_message(self, thread_id: str, text: str) -> TurnView:
        return self._view(
            await self._request("POST", f"/threads/{thread_id}/messages", body={"text": text})
        )

    async def resume(
        self,
        thread_id: str,
        *,
        value: float | str | None = None,
        text: str | None = None,
        pause_ref: dict[str, Any] | None = None,
    ) -> TurnView:
        body: dict[str, Any] = {"value": value, "text": text, "pause_ref": pause_ref}
        return self._view(await self._request("POST", f"/threads/{thread_id}/resume", body=body))

    async def get_thread(self, thread_id: str) -> TurnView:
        """Stan wątku (ThreadState) jako TurnView: bez odpowiedzi tej tury (replies puste)."""
        data = await self._request("GET", f"/threads/{thread_id}")
        return self._view({**data, "replies": []})

    async def get_trace(self, thread_id: str) -> TraceView:
        """Ostatni trace agenta w wątku (GET /threads/{id}/trace)."""
        data = await self._request("GET", f"/threads/{thread_id}/trace")
        try:
            return TraceView.model_validate(data)
        except ValidationError as exc:
            raise ApiCallError(200, "bad_response", f"Nieoczekiwana odpowiedź api: {exc}") from exc

    async def stream_message(self, thread_id: str, text: str) -> AsyncIterator[StreamItem]:
        """POST /threads/{id}/messages/stream (SSE): ('tool_start' | 'tool_end', ToolEventView)
        po kolei, na końcu ('final', TurnView).

        Odpowiedź spoza 2xx przed strumieniem → ApiCallError z ApiError; zdarzenie error →
        ApiCallError(200, error, detail); koniec strumienia bez final →
        ApiCallError(0, 'stream_interrupted'). Limit czasu odczytu (ui_api_timeout_s) liczy się
        między zdarzeniami.
        """
        path = f"/threads/{thread_id}/messages/stream"
        try:
            async with (
                self._client() as client,
                aconnect_sse(client, "POST", path, json={"text": text}) as source,
            ):
                response = source.response
                if not response.is_success:
                    await response.aread()
                    try:
                        data: Any = response.json()
                    except ValueError:
                        data = None
                    raise self._error_from(response, data)
                async for sse in source.aiter_sse():
                    try:
                        payload = json.loads(sse.data) if sse.data else {}
                    except ValueError as exc:
                        raise ApiCallError(
                            200, "bad_response", f"Niepoprawne zdarzenie SSE: {exc}"
                        ) from exc
                    if sse.event in ("tool_start", "tool_end"):
                        try:
                            event = ToolEventView.model_validate(payload)
                        except ValidationError as exc:
                            raise ApiCallError(
                                200, "bad_response", f"Nieoczekiwane zdarzenie api: {exc}"
                            ) from exc
                        yield (sse.event, event)  # type: ignore[misc]
                    elif sse.event == "final":
                        yield ("final", self._view(payload))
                        return
                    elif sse.event == "error":
                        raise ApiCallError(
                            200,
                            str(payload.get("error") or "stream_error"),
                            str(payload.get("detail") or ""),
                        )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise ApiCallError(
                0, "api_unreachable", f"{type(exc).__name__}: {exc}".rstrip(": ")
            ) from exc
        except SSEError as exc:
            raise ApiCallError(0, "stream_interrupted", f"Niepoprawny strumień api: {exc}") from exc
        raise ApiCallError(
            0, "stream_interrupted", "Strumień api skończył się bez wyniku tury. Spróbuj ponownie."
        )
