"""Format text/event-stream strumienia tury (FR-6.1; kontrakt F4, sekcja 9; ADR-0012).

`data` to `event.data` jako JSON w jednej linii (ensure_ascii=False), `id` rośnie od 1
w obrębie strumienia, komentarz `: ping`, gdy tura długo milczy. Rozłączenie klienta
(zamknięcie generatora) tylko przestaje czytać: tura biegnie dalej w wątku roboczym serwisu.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Protocol

from kovalo.workflow.service import TurnEvent

__all__ = ["PING", "SSE_HEADERS", "EventSource", "encode_event", "sse_body"]

PING: bytes = b": ping\n\n"
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


class EventSource(Protocol):
    """To, czego sse_body potrzebuje od TurnStream."""

    def next_event(self, timeout: float | None) -> TurnEvent | None: ...

    def close(self) -> None: ...


def encode_event(event: TurnEvent, *, event_id: int) -> bytes:
    """b'id: <n>\\nevent: <type>\\ndata: <json jednej linii>\\n\\n'."""
    data = json.dumps(event.data, ensure_ascii=False, separators=(",", ":"))
    return f"id: {event_id}\nevent: {event.type}\ndata: {data}\n\n".encode()


def sse_body(stream: EventSource, *, ping_s: float) -> Iterator[bytes]:
    """next_event(timeout=ping_s): None → PING; zdarzenie → encode_event; po final/error
    koniec. finally: stream.close() (rozłączenie klienta nie przerywa tury)."""
    event_id = 0
    try:
        while True:
            event = stream.next_event(timeout=ping_s)
            if event is None:
                yield PING
                continue
            event_id += 1
            yield encode_event(event, event_id=event_id)
            if event.type in ("final", "error"):
                return
    finally:
        stream.close()
