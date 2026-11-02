"""Polecenie `kovalo`. W F0 tylko `first-call` (kontrakt F0, sekcja 7)."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from typing import TextIO

from kovalo.config import ConfigError, Settings, get_settings

DEFAULT_PROMPT = "Przedstaw się jednym zdaniem po polsku."

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_PROVIDER = 3
EXIT_EMPTY = 4


def _format_seconds(seconds: float) -> str:
    return f"{seconds:.1f}".replace(".", ",")


def first_call(
    settings: Settings, *, prompt: str, use_fallbacks: bool, out: TextIO, err: TextIO
) -> int:
    """Jedno wywołanie modelu z configu; wynik po polsku na `out`, błędy na `err`."""
    from kovalo.llm import get_chat_model, get_resilient_chat_model

    try:
        model = (
            get_resilient_chat_model("default", settings=settings)
            if use_fallbacks
            else get_chat_model("default", settings=settings)
        )
    except ConfigError as exc:
        print(str(exc), file=err)
        return EXIT_CONFIG

    start = time.perf_counter()
    try:
        message = model.invoke(prompt)
    except ConfigError as exc:
        print(str(exc), file=err)
        return EXIT_CONFIG
    except Exception as exc:  # ModelError, httpx i inne wyjątki dostawcy
        print(f"Wywołanie LLM-a nie powiodło się: {type(exc).__name__}: {exc}", file=err)
        return EXIT_PROVIDER
    elapsed = time.perf_counter() - start

    text = message.text.strip()
    if not text:
        print("Model zwrócił pustą odpowiedź.", file=err)
        return EXIT_EMPTY

    usage = getattr(message, "usage_metadata", None)
    tokens = (
        f"wejście {usage['input_tokens']}, wyjście {usage['output_tokens']}, "
        f"razem {usage['total_tokens']}"
        if usage
        else "brak danych"
    )
    answered_by = message.response_metadata.get("model_name") or "nieznany"
    print(f"Dostawca: {settings.llm.provider}", file=out)
    print(f"Model z configu: {settings.llm.model}", file=out)
    print(f"Odpowiedział: {answered_by}", file=out)
    print(f"Czas: {_format_seconds(elapsed)} s", file=out)
    print(f"Tokeny: {tokens}", file=out)
    print("Odpowiedź:", file=out)
    print(text, file=out)
    return EXIT_OK


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kovalo", description="Kovalo Decision Copilot.")
    sub = parser.add_subparsers(dest="command", required=True)
    fc = sub.add_parser("first-call", help="jedno wywołanie LLM-a z configu (test klucza)")
    fc.add_argument("--prompt", default=DEFAULT_PROMPT, help="treść zapytania")
    fc.add_argument(
        "--no-fallback", action="store_true", help="tylko model główny, bez modeli zapasowych"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        settings = get_settings()  # klucz API nie jest tu wymagany; sprawdza go fabryka modeli
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_CONFIG
    if args.command == "first-call":
        return first_call(
            settings,
            prompt=args.prompt,
            use_fallbacks=not args.no_fallback,
            out=sys.stdout,
            err=sys.stderr,
        )
    return EXIT_CONFIG  # pragma: no cover (argparse wymusza polecenie)


def entrypoint() -> None:
    """Punkt wejścia `kovalo`."""
    sys.exit(main())
