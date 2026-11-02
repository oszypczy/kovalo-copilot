"""Polecenie `kovalo`: first-call (kontrakt F0, sekcja 7), ingest i eval (kontrakt F1, sekcja 7)."""

from __future__ import annotations

import argparse
import functools
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from kovalo.config import ConfigError, EmbeddingModelSpec, Settings, get_settings

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

    from kovalo.knowledge.ingest import IndexWriter

DEFAULT_PROMPT = "Przedstaw się jednym zdaniem po polsku."

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_PROVIDER = 3
EXIT_EMPTY = 4
EXIT_CORPUS = 5
EXIT_DB = 6


def _format_seconds(seconds: float) -> str:
    return f"{seconds:.1f}".replace(".", ",")


def _plural(n: int, one: str, few: str, many: str) -> str:
    if n == 1:
        return f"{n} {one}"
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return f"{n} {few}"
    return f"{n} {many}"


def _db_unavailable(settings: Settings, exc: BaseException, err: TextIO) -> int:
    reason = str(exc) or type(exc).__name__
    print(
        f"Postgres niedostępny ({settings.db.host}:{settings.db.port}): {reason}. Uruchom: make up",
        file=err,
    )
    return EXIT_DB


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


def _index_writer(settings: Settings, embeddings: Embeddings, table: str) -> IndexWriter:
    """Writer produkcyjny: replace_table na PGEngine z configu (testy go podmieniają)."""
    from kovalo.knowledge.pgvector import pg_engine, replace_table

    return functools.partial(replace_table, settings, pg_engine(settings), embeddings, table=table)


def ingest(settings: Settings, *, table: str | None, out: TextIO, err: TextIO) -> int:
    """Korpus → chunki MP1 → embeddingi (cache) → podmiana tabeli indeksu."""
    from kovalo.knowledge.corpus import CorpusError
    from kovalo.knowledge.ingest import ingest_mp1
    from kovalo.knowledge.pgvector import IndexUnavailableError
    from kovalo.llm import get_embeddings

    if table is not None and table != settings.db.vector_collection_mp1:
        try:
            db = settings.db.model_validate(
                {**settings.db.model_dump(), "vector_collection_mp1": table}
            )
        except ValueError as exc:
            print(f"Niepoprawna nazwa tabeli {table!r}: {exc}", file=err)
            return EXIT_CONFIG
        settings = settings.model_copy(update={"db": db})
    try:
        embeddings = get_embeddings(settings=settings)
        writer = _index_writer(settings, embeddings, settings.db.vector_collection_mp1)
        stats = ingest_mp1(settings, embeddings=embeddings, writer=writer)
    except ConfigError as exc:
        print(str(exc), file=err)
        return EXIT_CONFIG
    except CorpusError as exc:
        print(str(exc), file=err)
        return EXIT_CORPUS
    except IndexUnavailableError as exc:
        return _db_unavailable(settings, exc, err)

    corpus, chunks = stats.corpus, stats.chunks
    avg = f"{chunks.avg_chars:.1f}".replace(".", ",")
    print(
        f"Korpus: {_plural(corpus.n_documents, 'dokument', 'dokumenty', 'dokumentów')}, "
        f"{_plural(corpus.total_tokens, 'token', 'tokeny', 'tokenów')} "
        f"({settings.tokenizer.encoding})",
        file=out,
    )
    print(f"Chunki: {chunks.n_chunks} (średnio {avg} znaku, maks. {chunks.max_chars})", file=out)
    print(
        f"Embeddingi: {stats.embedding_provider} / {stats.embedding_model} ({stats.dimensions}), "
        f"policzone {stats.embedded}, z cache {stats.from_cache}",
        file=out,
    )
    print(
        f"Indeks: tabela {stats.table}, wierszy {stats.rows_after} (wcześniej {stats.rows_before})",
        file=out,
    )
    print(f"Czas: {_format_seconds(stats.seconds)} s", file=out)
    return EXIT_OK


def _e1_specs(
    settings: Settings, *, candidates: bool, models: Sequence[str]
) -> list[EmbeddingModelSpec]:
    """Modele E1: lokalny z configu (pierwszy), opcjonalnie kandydaci; --model zawęża listę."""
    emb = settings.embeddings
    local = emb.local.spec()
    known: dict[str, EmbeddingModelSpec] = {local.model: local}
    for spec in emb.candidates:
        known.setdefault(spec.model, spec)
    if models:
        unknown = [m for m in models if m not in known]
        if unknown:
            raise ConfigError(
                f"Nieznany model embeddingów: {', '.join(unknown)}. Dostępne (embeddings.local "
                f"i embeddings.candidates): {', '.join(known)}"
            )
        return [known[m] for m in dict.fromkeys(models)]
    if candidates:
        return list(known.values())
    # openai_cache: model z configu to referencja kursu, więc bez wiersza fastembed
    return [local] if emb.provider == "fastembed" else []


def eval_command(
    settings: Settings, *, stage: str, args: argparse.Namespace, out: TextIO, err: TextIO
) -> int:
    """`kovalo eval <stage>`: w F1 tylko E1 (raporty w paths.mini_project_1_output_dir)."""
    if stage != "e1":
        print(f"Etap {stage} jeszcze nie istnieje (F2).", file=err)
        return EXIT_CONFIG

    from kovalo.eval.retrieval.e1 import E1LlmCallError, run_e1
    from kovalo.eval.retrieval.report import write_reports
    from kovalo.knowledge.chunking import chunk_documents
    from kovalo.knowledge.corpus import CorpusError, load_corpus
    from kovalo.knowledge.pgvector import IndexUnavailableError

    out_dir = Path(args.out) if args.out else settings.paths.mini_project_1_output_dir
    llm_error: E1LlmCallError | None = None
    try:
        specs = _e1_specs(settings, candidates=args.candidates, models=args.model or [])
        try:
            report = run_e1(
                settings, specs=specs, include_reference=True, backend=args.backend,
                with_llm=args.with_llm,
            )  # fmt: skip
        except E1LlmCallError as exc:
            llm_error, report = exc, exc.report
        chunks = chunk_documents(load_corpus(settings), settings.chunking)
    except ConfigError as exc:
        print(str(exc), file=err)
        return EXIT_CONFIG
    except CorpusError as exc:
        print(str(exc), file=err)
        return EXIT_CORPUS
    except IndexUnavailableError as exc:
        return _db_unavailable(settings, exc, err)

    paths = write_reports(report, out_dir, chunks=chunks)
    for run in report.runs:
        rates = [run.mp1.hit_rate_at_1, run.mp1.hit_rate_at_3]
        rates += [run.mp2.hit_rate_at_1, run.mp2.hit_rate_at_3]
        mp1_h1, mp1_h3, mp2_h1, mp2_h3 = (str(r).replace(".", ",") for r in rates)
        print(
            f"{run.label}: MP1 hit@1 {mp1_h1}, hit@3 {mp1_h3}; MP2 hit@1 {mp2_h1}, hit@3 {mp2_h3}",
            file=out,
        )
    print(f"Rekomendacja reguły: {report.recommended_model or 'brak'}", file=out)
    print("Raporty:", file=out)
    for path in paths:
        print(f"  {path}", file=out)
    if llm_error is not None:
        cause = llm_error.cause
        print(f"Wywołanie LLM-a nie powiodło się: {type(cause).__name__}: {cause}", file=err)
        return EXIT_CONFIG if isinstance(cause, ConfigError) else EXIT_PROVIDER
    return EXIT_OK


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kovalo", description="Kovalo Decision Copilot.")
    sub = parser.add_subparsers(dest="command", required=True)
    fc = sub.add_parser("first-call", help="jedno wywołanie LLM-a z configu (test klucza)")
    fc.add_argument("--prompt", default=DEFAULT_PROMPT, help="treść zapytania")
    fc.add_argument(
        "--no-fallback", action="store_true", help="tylko model główny, bez modeli zapasowych"
    )
    ing = sub.add_parser("ingest", help="korpus → chunki MP1 → embeddingi → tabela pgvector")
    ing.add_argument(
        "--table", default=None, help="tabela indeksu (domyślnie db.vector_collection_mp1)"
    )
    ev = sub.add_parser("eval", help="ewaluacja etapu (F1: e1), raporty w reports/")
    ev.add_argument("stage", help="etap ewaluacji, np. e1")
    ev.add_argument("--candidates", action="store_true", help="także embeddings.candidates")
    ev.add_argument("--model", action="append", help="tylko wskazany model (można powtórzyć)")
    ev.add_argument("--backend", choices=["memory", "pgvector"], default="memory")
    ev.add_argument("--with-llm", action="store_true", help="jedno wywołanie LLM-a z promptem Q01")
    ev.add_argument("--out", default=None, help="katalog raportów")
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
    if args.command == "ingest":
        return ingest(settings, table=args.table, out=sys.stdout, err=sys.stderr)
    if args.command == "eval":
        return eval_command(settings, stage=args.stage, args=args, out=sys.stdout, err=sys.stderr)
    return EXIT_CONFIG  # pragma: no cover (argparse wymusza polecenie)


def entrypoint() -> None:
    """Punkt wejścia `kovalo`."""
    sys.exit(main())
