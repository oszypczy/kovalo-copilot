"""Polecenie `kovalo`: first-call (kontrakt F0, 7), ingest i eval e1 (F1, 7), eval e2 (F2, 8.7)."""

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


def _decimal(value: float) -> str:
    """Jedno miejsce po przecinku, zapis polski (sekundy, średnie tokeny)."""
    return f"{value:.1f}".replace(".", ",")


_format_seconds = _decimal


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


_VARIANT_LABELS = {"mp1": "chunki MP1", "ctx": "chunki z kontekstem"}


def _with_table(settings: Settings, variant: str, table: str) -> Settings:
    """Kopia ustawień z inną tabelą wariantu (--table); zła nazwa → ValueError."""
    field = "vector_collection_mp1" if variant == "mp1" else "vector_collection"
    db = settings.db.model_validate({**settings.db.model_dump(), field: table})
    return settings.model_copy(update={"db": db})


def ingest(
    settings: Settings, *, table: str | None, out: TextIO, err: TextIO, chunks: str = "all"
) -> int:
    """Korpus → chunki (MP1 i/lub z kontekstem) → embeddingi (cache) → podmiana tabel indeksu."""
    from kovalo.knowledge.corpus import CorpusError
    from kovalo.knowledge.ingest import IngestStats, index_table, ingest_chunks
    from kovalo.knowledge.pgvector import IndexUnavailableError
    from kovalo.llm import get_embeddings

    variants = ["mp1", "ctx"] if chunks == "all" else [chunks]
    if table is not None:
        if len(variants) != 1:
            print("--table wymaga --chunks mp1 albo ctx", file=err)
            return EXIT_CONFIG
        try:
            settings = _with_table(settings, variants[0], table)
        except ValueError as exc:
            print(f"Niepoprawna nazwa tabeli {table!r}: {exc}", file=err)
            return EXIT_CONFIG

    def report(stats: IngestStats, *, first: bool) -> None:
        corpus, chunk_stats = stats.corpus, stats.chunks
        if first:
            print(
                f"Korpus: {_plural(corpus.n_documents, 'dokument', 'dokumenty', 'dokumentów')}, "
                f"{_plural(corpus.total_tokens, 'token', 'tokeny', 'tokenów')} "
                f"({settings.tokenizer.encoding})",
                file=out,
            )
        avg = f"{chunk_stats.avg_chars:.1f}".replace(".", ",")
        print(f"Wariant {stats.variant} ({_VARIANT_LABELS[stats.variant]}):", file=out)
        print(
            f"  Chunki: {chunk_stats.n_chunks} (średnio {avg} znaku, maks. "
            f"{chunk_stats.max_chars})",
            file=out,
        )
        print(
            f"  Embeddingi: {stats.embedding_provider} / {stats.embedding_model} "
            f"({stats.dimensions}), policzone {stats.embedded}, z cache {stats.from_cache}",
            file=out,
        )
        print(
            f"  Indeks: tabela {stats.table}, wierszy {stats.rows_after} "
            f"(wcześniej {stats.rows_before})",
            file=out,
        )
        print(f"  Czas: {_format_seconds(stats.seconds)} s", file=out)

    try:
        embeddings = get_embeddings(settings=settings)  # wspólny cache dla obu wariantów
        for n, variant in enumerate(variants):
            writer = _index_writer(settings, embeddings, index_table(settings, variant))  # type: ignore[arg-type]
            stats = ingest_chunks(
                settings,
                variant=variant,  # type: ignore[arg-type]
                embeddings=embeddings,
                writer=writer,
            )
            report(stats, first=n == 0)
    except ConfigError as exc:
        print(str(exc), file=err)
        return EXIT_CONFIG
    except CorpusError as exc:
        print(str(exc), file=err)
        return EXIT_CORPUS
    except IndexUnavailableError as exc:
        return _db_unavailable(settings, exc, err)
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
    """`kovalo eval <stage>`: E1 (reports/mp1) i E2 (reports/mp2)."""
    if stage == "e2":
        return eval_e2(settings, args=args, out=out, err=err)
    if stage != "e1":
        print(f"Etap {stage} jeszcze nie istnieje (F3).", file=err)
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


def _split(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def eval_e2(settings: Settings, *, args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """`kovalo eval e2`: poziomy 1-2 zawsze (bez LLM-a), 3-4 z --with-llm (kontrakt F2, 8.7)."""
    from kovalo.eval.datasets import load_gold
    from kovalo.eval.llm_cache import JsonFileLlmCache
    from kovalo.eval.rag.e2 import (
        DEFAULT_LLM_VARIANTS,
        SIDES,
        VARIANTS,
        E2AllErrorsError,
        parse_variants,
        plan_llm_calls,
        run_e2,
        select_questions,
    )
    from kovalo.eval.rag.report import write_reports
    from kovalo.eval.review import REVIEW_FILE, check_review_keys, load_review
    from kovalo.knowledge.chunking import build_contextual_chunks
    from kovalo.knowledge.corpus import CorpusError, load_corpus
    from kovalo.knowledge.pgvector import IndexUnavailableError
    from kovalo.rag.records import Answer, Verdict

    if args.no_cache and args.cache_only:
        print("--no-cache i --cache-only wykluczają się.", file=err)
        return EXIT_CONFIG
    with_llm, judge = bool(args.with_llm), not args.no_judge
    try:
        variants = parse_variants(_split(args.variants) or list(DEFAULT_LLM_VARIANTS))
        ids = _split(args.ids) or None
        gold = load_gold(settings.paths.mini_project_2_questions)
        questions = select_questions(gold, ids)
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        print(str(exc), file=err)
        return EXIT_CONFIG

    # Komentarze przeglądu ręcznego przed planem, cache i modelami: błąd → kod 2, zero wywołań.
    out_dir = Path(args.out) if args.out else settings.paths.mini_project_2_output_dir
    review_path = out_dir / REVIEW_FILE
    try:
        review = load_review(review_path)
        if review is not None:
            labels = [f"{s}-{v}" for s in SIDES for v in VARIANTS]
            check_review_keys(review, variants=labels, ids=[q.id for q in gold])
    except ConfigError as exc:
        message = str(exc)
        print(message if str(review_path) in message else f"{review_path}: {message}", file=err)
        return EXIT_CONFIG

    planned = plan_llm_calls(len(questions), len(variants), judge=judge) if with_llm else 0
    if with_llm:
        print(
            f"Plan: ≤ {planned} wywołań LLM-a ({len(questions)} pytań x {len(variants)} "
            f"wariant(y) x {'odpowiedź + sędzia' if judge else 'odpowiedź'}; strona "
            f"{args.llm_side}; bez cache: {'tak' if args.no_cache else 'nie'})",
            file=out,
        )
    else:
        print("Plan: 0 wywołań LLM-a (poziomy 1–2; poziomy 3–4: --with-llm)", file=out)  # noqa: RUF001
    if args.plan:
        return EXIT_OK

    cache = None
    if with_llm and not args.no_cache:
        cache = JsonFileLlmCache(
            settings.paths.llm_cache_dir, mode="strict" if args.cache_only else "read_write"
        )

    # Modele budujemy przed czymkolwiek innym: brak klucza → kod 2, ale raport bez LLM-a
    # i tak powstaje (jak w E1).
    llm_error: ConfigError | None = None
    answer_model = judge_model = None
    if with_llm:
        from kovalo.llm import get_structured_model

        try:
            answer_model = get_structured_model(Answer, "answer", settings=settings, cache=cache)
            if judge:
                judge_model = get_structured_model(Verdict, "judge", settings=settings, cache=cache)
        except ConfigError as exc:
            llm_error = exc

    all_errors: E2AllErrorsError | None = None
    try:
        try:
            report = run_e2(
                settings, variants=variants, llm_side=args.llm_side,
                with_llm=with_llm and llm_error is None, judge=judge, ids=ids,
                backend=args.backend, cache=cache, answer_model=answer_model,
                judge_model=judge_model,
            )  # fmt: skip
        except E2AllErrorsError as exc:
            all_errors, report = exc, exc.report
        ctx_chunks = build_contextual_chunks(load_corpus(settings), settings.chunking)
    except ConfigError as exc:
        print(str(exc), file=err)
        return EXIT_CONFIG
    except CorpusError as exc:
        print(str(exc), file=err)
        return EXIT_CORPUS
    except IndexUnavailableError as exc:
        return _db_unavailable(settings, exc, err)

    paths = write_reports(report, out_dir, ctx_chunks=ctx_chunks, review=review)
    for r in report.retrieval:
        n = len(r.rows)
        print(
            f"{r.side} {r.variant}: hit@1 {r.hit_at_1}/{n}, hit@3 {r.hit_at_3}/{n}, "
            f"dowody {r.evidence_phrases_complete}/{n}, "
            f"tokeny {_decimal(r.avg_prompt_tokens)}",
            file=out,
        )
    for result in report.llm:
        s, c = result.summary, result.calls
        judge_text = f"{s.judge_correct[0]}/{s.judge_correct[1]}" if report.judge else "-"
        print(
            f"{result.side} {result.variant}: odpowiedzi {s.answered[0]}/{s.answered[1]}, "
            f"fakty {s.key_facts_complete[0]}/{s.key_facts_complete[1]}, sędzia {judge_text}, "
            f"odmowy {s.refusals_correct[0]}/{s.refusals_correct[1]}, błędy {s.errors}; "
            f"wywołania {c.answer + c.judge}, z cache {c.from_cache}",
            file=out,
        )
    print(f"Rekomendacja reguły: {report.recommended_mode or 'brak'}", file=out)
    print("Raporty:", file=out)
    for path in paths:
        print(f"  {path}", file=out)
    if llm_error is not None:
        print(str(llm_error), file=err)
        return EXIT_CONFIG
    if all_errors is not None:
        print(
            f"Żadna odpowiedź się nie udała (przebieg {report.run_id} zapisano jako nieudany), "
            f"np.: {all_errors.first_error}\nSprawdź klucz i model poleceniem: kovalo first-call "
            "(docs/06-evaluation.md, „Testy z prawdziwymi usługami”).",
            file=err,
        )
        return EXIT_PROVIDER
    return EXIT_OK


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kovalo", description="Kovalo Decision Copilot.")
    sub = parser.add_subparsers(dest="command", required=True)
    fc = sub.add_parser("first-call", help="jedno wywołanie LLM-a z configu (test klucza)")
    fc.add_argument("--prompt", default=DEFAULT_PROMPT, help="treść zapytania")
    fc.add_argument(
        "--no-fallback", action="store_true", help="tylko model główny, bez modeli zapasowych"
    )
    ing = sub.add_parser("ingest", help="korpus → chunki → embeddingi → tabele pgvector")
    ing.add_argument(
        "--chunks",
        choices=["mp1", "ctx", "all"],
        default="all",
        help="wariant chunków (domyślnie oba: mp1, potem ctx)",
    )
    ing.add_argument("--table", default=None, help="tabela indeksu (tylko z --chunks mp1 albo ctx)")
    ev = sub.add_parser("eval", help="ewaluacja etapu (e1, e2), raporty w reports/")
    ev.add_argument("stage", help="etap ewaluacji: e1 albo e2")
    ev.add_argument("--candidates", action="store_true", help="e1: także embeddings.candidates")
    ev.add_argument("--model", action="append", help="e1: tylko wskazany model (można powtórzyć)")
    ev.add_argument("--backend", choices=["memory", "pgvector"], default="memory")
    ev.add_argument(
        "--with-llm",
        action="store_true",
        help="e1: jedno wywołanie z promptem Q01; e2: poziomy 3–4 (odpowiedzi i sędzia)",  # noqa: RUF001
    )
    ev.add_argument("--variants", default=None, help="e2: warianty z LLM-em (domyślnie ctx,hybrid)")
    ev.add_argument("--llm-side", choices=["ours", "reference"], default="ours", help="e2")
    ev.add_argument("--ids", default=None, help="e2: tylko te pytania, np. M03,M10")
    ev.add_argument("--no-judge", action="store_true", help="e2: bez sędziego")
    ev.add_argument("--no-cache", action="store_true", help="e2: bez cache LLM-a")
    ev.add_argument("--cache-only", action="store_true", help="e2: tylko cache LLM-a, bez sieci")
    ev.add_argument("--plan", action="store_true", help="e2: wypisz plan wywołań i zakończ")
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
        return ingest(
            settings, table=args.table, chunks=args.chunks, out=sys.stdout, err=sys.stderr
        )
    if args.command == "eval":
        return eval_command(settings, stage=args.stage, args=args, out=sys.stdout, err=sys.stderr)
    return EXIT_CONFIG  # pragma: no cover (argparse wymusza polecenie)


def entrypoint() -> None:
    """Punkt wejścia `kovalo`."""
    sys.exit(main())
