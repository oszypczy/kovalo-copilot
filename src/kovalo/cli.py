"""Polecenie `kovalo`: first-call (F0, 7), ingest, eval e1 (F1, 7), e2 (F2, 8.7), e3 (F3, 10.6),
eval e4, ask, show-trace (F4, 11.6), eval e5 i eval e4 --transport (F5, 10.6),
eval dashboard (F6, 8.5)."""

from __future__ import annotations

import argparse
import functools
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TextIO

from kovalo.config import ConfigError, EmbeddingModelSpec, Settings, get_settings

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

    from kovalo.knowledge.ingest import IndexWriter

DEFAULT_PROMPT = "Przedstaw się jednym zdaniem po polsku."

EXIT_OK = 0
EXIT_CHECK_DIFF = 1  # eval dashboard --check: pliki różnią się od wygenerowanych
EXIT_CONFIG = 2
EXIT_PROVIDER = 3
EXIT_EMPTY = 4
EXIT_CORPUS = 5
EXIT_DB = 6
EXIT_MAX_CALLS = 7
EXIT_MCP_STARTUP = 8


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
    """`kovalo eval <stage>`: E1 (reports/mp1), E2 (reports/mp2), E3 (reports/mp3), E4 (mp4),
    E5 (mp5)."""
    if stage == "e2":
        return eval_e2(settings, args=args, out=out, err=err)
    if stage == "e3":
        return eval_e3(settings, args=args, out=out, err=err)
    if stage == "e4":
        return eval_e4(settings, args=args, out=out, err=err)
    if stage == "e5":
        return eval_e5(settings, args=args, out=out, err=err)
    if stage == "dashboard":
        return eval_dashboard(settings, args=args, out=out, err=err)
    if stage != "e1":
        phase = f"F{stage[1:]}" if stage[:1] == "e" and stage[1:].isdigit() else "?"
        print(f"Etap {stage} jeszcze nie istnieje ({phase}).", file=err)
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
    # i tak powstaje (jak w E1). --cache-only: offline, bez klucza (N3, kontrakt F3 7.2).
    llm_error: ConfigError | None = None
    answer_model = judge_model = None
    if with_llm:
        from kovalo.llm import get_structured_model

        offline = bool(args.cache_only)
        try:
            answer_model = get_structured_model(
                Answer, "answer", settings=settings, cache=cache, offline=offline
            )
            if judge:
                judge_model = get_structured_model(
                    Verdict, "judge", settings=settings, cache=cache, offline=offline
                )
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


_E3_STAGE_WORDS = (
    ("route", "kierowanie"),
    ("fields", "pola"),
    ("request", "wniosek"),
    ("approvers", "zatwierdzający"),
    ("pauses", "pauzy"),
    ("status", "status"),
)


def _e3_line(name: str, score: dict[str, object]) -> str:
    stages = ", ".join(
        f"{word} {score[key][0]}/{score[key][1]}"  # type: ignore[index]
        for key, word in _E3_STAGE_WORDS
    )
    return f"{name}: {stages}, błędy {score['errors']}"


def eval_e3(settings: Settings, *, args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """`kovalo eval e3`: odtworzenie (zawsze, 0 wywołań) i przebieg na żywo (--live)
    (kontrakt F3, 10.6). --cache-only: offline, bez klucza i bez sieci."""
    from kovalo.eval.datasets import load_dialogues
    from kovalo.eval.llm_cache import JsonFileLlmCache
    from kovalo.eval.review import REVIEW_FILE, check_review_keys, load_review
    from kovalo.eval.workflow.e3 import (
        E3AllErrorsError,
        E3Report,
        plan_llm_calls,
        replay_perfect,
        run_e3,
        select_dialogues,
    )
    from kovalo.eval.workflow.report import REVIEW_LABELS, turn_ids, write_reports
    from kovalo.knowledge.corpus import CorpusError
    from kovalo.knowledge.pgvector import IndexUnavailableError

    if args.no_cache and args.cache_only:
        print("--no-cache i --cache-only wykluczają się.", file=err)
        return EXIT_CONFIG
    live = bool(args.live)
    try:
        ids = _split(args.ids) or None
        dialogues = select_dialogues(load_dialogues(settings.paths.mini_project_3_dialogues), ids)
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        print(str(exc), file=err)
        return EXIT_CONFIG

    # Komentarze przeglądu ręcznego przed planem, cache i modelami: błąd → kod 2, zero wywołań.
    out_dir = Path(args.out) if args.out else settings.paths.mini_project_3_output_dir
    review_path = out_dir / REVIEW_FILE
    try:
        review = load_review(review_path)
        if review is not None:
            all_ids = turn_ids(
                [
                    {"dialogue": d.id}
                    for d in load_dialogues(settings.paths.mini_project_3_dialogues)
                    for _ in d.turns
                ]
            )
            check_review_keys(review, variants=REVIEW_LABELS, ids=all_ids)
    except ConfigError as exc:
        message = str(exc)
        print(message if str(review_path) in message else f"{review_path}: {message}", file=err)
        return EXIT_CONFIG

    plan = plan_llm_calls(dialogues)
    if live:
        print(
            f"Plan: understand {plan['understand']}, odpowiedzi {plan['answer']}, razem "
            f"{plan['total']} (górna granica bez fallbacku; bez cache: "
            f"{'tak' if args.no_cache else 'nie'}; tylko cache: "
            f"{'tak' if args.cache_only else 'nie'})",
            file=out,
        )
    else:
        print("Plan: 0 wywołań LLM-a (odtworzenie; przebieg na żywo: --live)", file=out)
    if args.plan:
        return EXIT_OK

    cache = None
    if live and not args.no_cache:
        cache = JsonFileLlmCache(
            settings.paths.llm_cache_dir, mode="strict" if args.cache_only else "read_write"
        )

    config_error: ConfigError | None = None
    all_errors: E3AllErrorsError | None = None
    report: E3Report
    try:
        try:
            report = run_e3(
                settings, live=live, ids=ids, cache=cache, offline=bool(args.cache_only),
                backend=args.backend,
            )  # fmt: skip
        except E3AllErrorsError as exc:
            all_errors, report = exc, exc.report
        except ConfigError as exc:
            if not live:
                raise
            # Brak klucza (albo inny błąd konfiguracji na żywo): odtworzenie i tak zapisujemy.
            config_error = exc
            report = run_e3(settings, ids=ids)
    except ConfigError as exc:
        print(str(exc), file=err)
        return EXIT_CONFIG
    except CorpusError as exc:
        print(str(exc), file=err)
        return EXIT_CORPUS
    except IndexUnavailableError as exc:
        return _db_unavailable(settings, exc, err)

    paths = write_reports(report, out_dir, review=review)
    print(_e3_line("odtworzenie", report.replay.score), file=out)
    if report.live is not None:
        c = report.live.calls
        models = ", ".join(f"{m}: {n}" for m, n in report.live.answered_by.items()) or "brak"
        print(
            f"{_e3_line('na żywo', report.live.score)}; wywołania {c.understand + c.answer}, "
            f"z cache {c.from_cache}, błędy wywołań {c.errors}, modele: {models}",
            file=out,
        )
    print("Raporty:", file=out)
    for path in paths:
        print(f"  {path}", file=out)

    if config_error is not None:
        print(f"{config_error}\nZapisano tylko odtworzenie (bez przebiegu na żywo).", file=err)
        return EXIT_CONFIG
    if not replay_perfect(report.replay):
        wrong = [d for d in report.replay.details if d["wrong"]]
        print(
            "Odtworzenie z idealnym understand nie daje 100%: błąd workflow, nie modelu "
            f"(np. {wrong[0] if wrong else 'błędy tur'}). Sprawdź raport.",
            file=err,
        )
        return EXIT_EMPTY
    if all_errors is not None:
        print(
            f"Żadna tura na żywo się nie udała (przebieg {all_errors.run_id} zapisano jako "
            f"nieudany), np.: {all_errors.first_error}\nSprawdź klucz i model "
            "poleceniem: kovalo first-call albo uzupełnij cache (bez --cache-only).",
            file=err,
        )
        return EXIT_PROVIDER
    return EXIT_OK


_E4_TESTS = (("tools", "narzędzia"), ("args", "argumenty"), ("facts", "fakty"))


def _e4_line(name: str, summary: dict[str, object]) -> str:
    parts = [f"{word} {summary[key][0]}/{summary[key][1]}" for key, word in _E4_TESTS]  # type: ignore[index]
    approval = summary["approval"]
    return (
        f"{name}: {', '.join(parts)}, bez źródła {summary['uncovered']}, twierdzenia o "
        f"zatwierdzeniu {approval[0]}/{approval[1]}, wywołania narzędzi {summary['tool_calls']} "  # type: ignore[index]
        f"(nieudane {summary['failed']})"
    )


def eval_e4(settings: Settings, *, args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """`kovalo eval e4`: przeliczenie przebiegów kursu i odtworzenie (zawsze, 0 wywołań),
    przebieg na żywo (--live, z cache i --max-calls), raporty w reports/mp4 (kontrakt F4, 11.6).
    Kody: 2 konfiguracja, 4 port ≠ kurs, 7 limit --max-calls, 3 każde pytanie na żywo z błędem."""
    from kovalo.eval.datasets import load_agent_gold
    from kovalo.eval.llm_cache import JsonFileLlmCache
    from kovalo.eval.review import REVIEW_FILE, check_review_keys, load_review
    from kovalo.eval.trace import reference
    from kovalo.eval.trace.e4 import (
        E4AllErrorsError,
        E4Report,
        plan_llm_calls,
        replay_problems,
        run_e4,
        select_questions,
    )
    from kovalo.eval.trace.report import REVIEW_LABELS, write_reports
    from kovalo.knowledge.corpus import CorpusError
    from kovalo.knowledge.pgvector import IndexUnavailableError
    from kovalo.mcp_client import McpStartupError

    if args.no_cache and args.cache_only:
        print("--no-cache i --cache-only wykluczają się.", file=err)
        return EXIT_CONFIG
    if args.max_calls is not None and args.max_calls < 0:
        print("--max-calls musi być liczbą ≥ 0 (0 = bez limitu).", file=err)
        return EXIT_CONFIG
    live = bool(args.live)
    max_calls = 20 if args.max_calls is None else (args.max_calls or None)
    transport: Literal["direct", "memory", "stdio", "http"] = args.transport or "direct"
    if transport != "direct" and not live:
        print(
            "--transport inny niż direct wymaga --live (odtworzenie kursu jest bezpośrednie).",
            file=err,
        )
        return EXIT_CONFIG
    try:
        ids = _split(args.ids) or None
        gold = load_agent_gold(settings.paths.mini_project_4_questions)
        questions = select_questions(gold, ids)
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        print(str(exc), file=err)
        return EXIT_CONFIG

    out_dir = Path(args.out) if args.out else settings.paths.mini_project_4_output_dir
    review_path = out_dir / REVIEW_FILE
    try:
        review = load_review(review_path)
        if review is not None:
            known = [g.id for g in gold] + [f"Q{n}" for n in range(1, 5)]
            check_review_keys(review, variants=REVIEW_LABELS, ids=known)
    except ConfigError as exc:
        message = str(exc)
        print(message if str(review_path) in message else f"{review_path}: {message}", file=err)
        return EXIT_CONFIG

    plan = plan_llm_calls(questions, max_steps=settings.agent.max_steps)
    if live:
        limit = str(max_calls) if max_calls is not None else "brak"
        print(
            f"Plan: szacunek {plan['estimate']}, górna granica {plan['upper_bound']}, limit "
            f"{limit} (wywołania sieciowe modelu; bez cache: {'tak' if args.no_cache else 'nie'}; "
            f"tylko cache: {'tak' if args.cache_only else 'nie'})",
            file=out,
        )
    else:
        print("Plan: 0 wywołań LLM-a (przeliczenie i odtworzenie; na żywo: --live)", file=out)
    if args.plan:
        return EXIT_OK

    cache = None
    if live and not args.no_cache:
        cache = JsonFileLlmCache(
            settings.paths.llm_cache_dir, mode="strict" if args.cache_only else "read_write"
        )
    config_error: ConfigError | None = None
    all_errors: E4AllErrorsError | None = None
    report: E4Report
    try:
        try:
            report = run_e4(
                settings, live=live, ids=ids, cache=cache, offline=bool(args.cache_only),
                max_calls=max_calls, backend=args.backend, variant=args.tools,
                transport=transport, count_errors=bool(args.count_errors),
            )  # fmt: skip
        except E4AllErrorsError as exc:
            all_errors, report = exc, exc.report
        except McpStartupError:
            raise
        except ConfigError as exc:
            if not live:
                raise
            config_error = exc  # brak klucza: odtworzenie i tak zapisujemy
            report = run_e4(settings, ids=ids, backend=args.backend)
    except reference.E4ReferenceMismatch as exc:
        print(f"Przeliczenie przebiegu kursu ≠ zapis kursu (błąd portu): {exc}", file=err)
        return EXIT_EMPTY
    except McpStartupError as exc:
        print(f"Serwer MCP nie wystartował: {exc}", file=err)
        return EXIT_MCP_STARTUP
    except ConfigError as exc:
        print(str(exc), file=err)
        return EXIT_CONFIG
    except CorpusError as exc:
        print(str(exc), file=err)
        return EXIT_CORPUS
    except IndexUnavailableError as exc:
        return _db_unavailable(settings, exc, err)

    paths = write_reports(report, out_dir, review=review)
    print(_e4_line("kurs, jeden agent", report.course.summary), file=out)
    print(_e4_line("kurs, zespół", report.course_supervisor.summary), file=out)
    print(_e4_line("odtworzenie", report.replay.summary), file=out)
    if report.live is not None:
        c = report.live.calls
        models = ", ".join(f"{m}: {n}" for m, n in report.live.answered_by.items()) or "brak"
        print(
            f"{_e4_line('na żywo', report.live.summary)}; wywołania modelu sieciowo {c.network}, "
            f"z cache {c.from_cache}, błędy {c.errors}, modele: {models}",
            file=out,
        )
    print("Raporty:", file=out)
    for path in paths:
        print(f"  {path}", file=out)

    if config_error is not None:
        print(f"{config_error}\nZapisano tylko przeliczenie i odtworzenie.", file=err)
        return EXIT_CONFIG
    problems = replay_problems(report)
    if problems:
        print(
            "Odtworzenie przebiegu kursu nie spełnia wymagań (błąd portu): " + "; ".join(problems),
            file=err,
        )
        return EXIT_EMPTY
    if report.stopped_by_max_calls:
        print(
            f"Przerwano: wyczerpany limit --max-calls {max_calls}. Raport częściowy zapisany; "
            "uruchom ponownie (np. następnego dnia), cache odtworzy wykonane pytania.",
            file=err,
        )
        return EXIT_MAX_CALLS
    if all_errors is not None:
        print(
            f"Żadne pytanie na żywo się nie udało (przebieg {all_errors.run_id}), np.: "
            f"{all_errors.first_error}\nSprawdź klucz i model (kovalo first-call) albo uzupełnij "
            "cache (bez --cache-only).",
            file=err,
        )
        return EXIT_PROVIDER
    return EXIT_OK


_E5_LINE = (
    ("route", "ścieżka"),
    ("status", "status"),
    ("fields", "pola"),
    ("pauses", "pauzy"),
    ("approvers", "zatwierdzający"),
    ("request", "wniosek"),
    ("agent_trace", "trace"),
    ("tool_coverage", "narzędzia"),
    ("arguments_correct", "argumenty"),
    ("answer_facts", "fakty"),
    ("numbers_backed", "liczby z pokryciem"),
    ("no_approval_claim", "bez twierdzenia o akceptacji"),
)


def _e5_line(name: str, totals: dict[str, list[int]]) -> str:
    parts = [
        f"{word} {totals[key][0]}/{totals[key][1]}" for key, word in _E5_LINE if totals[key][1]
    ]
    return f"{name}: {', '.join(parts)}"


def eval_e5(settings: Settings, *, args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """`kovalo eval e5`: zgodność MCP i odtworzenie rozmów (zawsze, 0 wywołań LLM-a), przebieg
    na żywo (--live, z cache i wspólnym --max-calls), raporty w reports/mp5 (kontrakt F5, 10.6).
    Kody (w tej kolejności): 2 konfiguracja, 8 serwer MCP nie wystartował, 4 zgodność albo
    odtworzenie < 100% (błąd portu), 7 limit --max-calls, 3 każda tura na żywo z błędem;
    5 CorpusError, 6 Postgres niedostępny."""
    from kovalo.eval.assistant.e5 import (
        E5AllErrorsError,
        E5Report,
        plan_llm_calls,
        replay_problems,
        run_e5,
        select_conversations,
    )
    from kovalo.eval.assistant.report import REVIEW_LABELS, turn_id, write_reports
    from kovalo.eval.datasets import load_conversations
    from kovalo.eval.llm_cache import JsonFileLlmCache
    from kovalo.eval.review import REVIEW_FILE, check_review_keys, load_review
    from kovalo.knowledge.corpus import CorpusError
    from kovalo.knowledge.pgvector import IndexUnavailableError
    from kovalo.mcp_client import McpStartupError

    if args.no_cache and args.cache_only:
        print("--no-cache i --cache-only wykluczają się.", file=err)
        return EXIT_CONFIG
    if args.max_calls is not None and args.max_calls < 0:
        print("--max-calls musi być liczbą ≥ 0 (0 = bez limitu).", file=err)
        return EXIT_CONFIG
    transport = args.transport or "stdio"
    if transport == "direct":
        print("E5 zawsze idzie przez MCP: --transport memory, stdio albo http.", file=err)
        return EXIT_CONFIG
    live = bool(args.live)
    max_calls = 40 if args.max_calls is None else (args.max_calls or None)
    paths = settings.paths
    try:
        ids = _split(args.ids) or None
        everything = load_conversations(
            paths.mini_project_5_conversations, paths.mini_project_3_dialogues
        )
        conversations = select_conversations(everything, ids)
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        print(str(exc), file=err)
        return EXIT_CONFIG

    out_dir = Path(args.out) if args.out else paths.mini_project_5_output_dir
    review_path = out_dir / REVIEW_FILE
    try:
        review = load_review(review_path)
        if review is not None:
            known = [turn_id(d.id, n) for d in everything for n in range(1, len(d.turns) + 1)]
            check_review_keys(review, variants=REVIEW_LABELS, ids=known)
    except ConfigError as exc:
        message = str(exc)
        print(message if str(review_path) in message else f"{review_path}: {message}", file=err)
        return EXIT_CONFIG

    plan = plan_llm_calls(conversations, max_steps=settings.agent.max_steps)
    if live:
        limit = str(max_calls) if max_calls is not None else "brak"
        print(
            f"Plan: understand {plan['understand']}, agent ≈ {plan['agent_estimate']}, szacunek "
            f"{plan['estimate']}, górna granica {plan['upper_bound']}, limit {limit} (jeden "
            f"licznik prób, także nieudanych; bez cache: {'tak' if args.no_cache else 'nie'}; "
            f"tylko cache: {'tak' if args.cache_only else 'nie'}; transport {transport})",
            file=out,
        )
    else:
        print(
            f"Plan: 0 wywołań LLM-a (zgodność MCP i odtworzenie, transport {transport}; "
            "na żywo: --live)",
            file=out,
        )
    if args.plan:
        return EXIT_OK

    cache = None
    if live and not args.no_cache:
        cache = JsonFileLlmCache(
            paths.llm_cache_dir, mode="strict" if args.cache_only else "read_write"
        )
    common = {"ids": ids, "transport": transport, "backend": args.backend}
    config_error: ConfigError | None = None
    all_errors: E5AllErrorsError | None = None
    report: E5Report
    try:
        try:
            report = run_e5(
                settings, live=live, cache=cache, offline=bool(args.cache_only),
                max_calls=max_calls, **common,
            )  # fmt: skip
        except E5AllErrorsError as exc:
            all_errors, report = exc, exc.report
        except McpStartupError:
            raise
        except ConfigError as exc:
            if not live:
                raise
            config_error = exc  # brak klucza: zgodność i odtworzenie i tak zapisujemy
            report = run_e5(settings, **common)
    except McpStartupError as exc:
        print(f"Serwer MCP nie wystartował albo nie ma wymaganych narzędzi: {exc}", file=err)
        return EXIT_MCP_STARTUP
    except ConfigError as exc:
        print(str(exc), file=err)
        return EXIT_CONFIG
    except CorpusError as exc:
        print(str(exc), file=err)
        return EXIT_CORPUS
    except IndexUnavailableError as exc:
        return _db_unavailable(settings, exc, err)

    written = write_reports(report, out_dir, review=review)
    same = sum(1 for row in report.parity if row.same)
    print(f"zgodność MCP: {same}/{len(report.parity)}", file=out)
    print(_e5_line("odtworzenie", report.replay.totals), file=out)
    if report.live is not None:
        c = report.live.calls
        models = ", ".join(f"{m}: {n}" for m, n in report.live.answered_by.items()) or "brak"
        print(
            f"{_e5_line('na żywo', report.live.totals)}; wywołania modelu sieciowo "
            f"{c.understand_network + c.agent_network} (understand {c.understand_network}, agent "
            f"{c.agent_network}), z cache {c.understand_cache + c.agent_cache}, błędy {c.errors}; "
            f"MCP {c.mcp_calls} (nieudane {sum(c.mcp_failures.values())}); modele: {models}",
            file=out,
        )
    print("Raporty:", file=out)
    for path in written:
        print(f"  {path}", file=out)

    if config_error is not None:
        print(f"{config_error}\nZapisano tylko zgodność i odtworzenie.", file=err)
        return EXIT_CONFIG
    problems = replay_problems(report)
    if problems:
        print(
            "Zgodność albo odtworzenie poniżej 100% (błąd portu): " + "; ".join(problems),
            file=err,
        )
        return EXIT_EMPTY
    if report.stopped_by_max_calls:
        print(
            f"Przerwano: wyczerpany limit --max-calls {max_calls}. Raport częściowy zapisany; "
            "uruchom ponownie (np. następnego dnia), cache odtworzy wykonane tury.",
            file=err,
        )
        return EXIT_MAX_CALLS
    if all_errors is not None:
        print(
            f"Żadna tura na żywo się nie udała (przebieg {all_errors.run_id}), np.: "
            f"{all_errors.first_error}\nSprawdź klucz i model (kovalo first-call) albo uzupełnij "
            "cache (bez --cache-only).",
            file=err,
        )
        return EXIT_PROVIDER
    return EXIT_OK


def eval_dashboard(
    settings: Settings, *, args: argparse.Namespace, out: TextIO, err: TextIO
) -> int:
    """`kovalo eval dashboard [--reports-dir DIR] [--check]` (F6, 8.5): zero wywołań LLM-a.

    Kody: 0 ok, 1 różnica w `--check`, 2 błąd specyfikacji albo danych raportu."""
    from kovalo.eval.dashboard import build_dashboard, write_dashboard
    from kovalo.eval.dashboard.build import (
        JSON_NAME,
        MARKDOWN_NAME,
        SPEC_NAME,
        check_dashboard,
        load_previous,
    )
    from kovalo.eval.dashboard.spec import load_spec
    from kovalo.eval.dashboard.stages import DashboardDataError

    reports_dir = Path(args.reports_dir) if args.reports_dir else settings.paths.reports_dir
    try:
        spec = load_spec(reports_dir / SPEC_NAME)
        previous = load_previous(reports_dir)
        dashboard = build_dashboard(
            spec, reports_dir, previous, warn=lambda message: print(message, file=err)
        )
    except (ConfigError, DashboardDataError) as exc:
        print(str(exc), file=err)
        return EXIT_CONFIG
    if args.check:
        diffs = check_dashboard(dashboard, reports_dir)
        if diffs:
            print(
                f"Dashboard nieaktualny w {reports_dir}: {', '.join(diffs)}. "
                "Wygeneruj go poleceniem `uv run kovalo eval dashboard`.",
                file=err,
            )
            return EXIT_CHECK_DIFF
        print(f"Dashboard aktualny ({reports_dir}).", file=out)
        return EXIT_OK
    write_dashboard(dashboard, reports_dir)
    print(f"Dashboard (stan na {dashboard.as_of.isoformat()}):", file=out)
    for name in (MARKDOWN_NAME, JSON_NAME):
        print(f"  {reports_dir / name}", file=out)
    return EXIT_OK


def ask(settings: Settings, *, args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """`kovalo ask "pytanie"`: jedno pytanie do agenta bez rozmowy (port ask_agent.py):
    odpowiedź i ścieżka dowodowa na stdout; 0 answered, 3 error/limit, 2 konfiguracja."""
    from kovalo.agents.show import format_trace
    from kovalo.eval.llm_cache import JsonFileLlmCache
    from kovalo.eval.trace.e4 import LazySearch
    from kovalo.workflow.factory import build_agent_runner

    question = args.question.strip()
    if not question:
        print("Pytanie jest puste.", file=err)
        return EXIT_CONFIG
    if args.no_cache and args.cache_only:
        print("--no-cache i --cache-only wykluczają się.", file=err)
        return EXIT_CONFIG
    cache = None
    if not args.no_cache:
        cache = JsonFileLlmCache(
            settings.paths.llm_cache_dir, mode="strict" if args.cache_only else "read_write"
        )
    try:
        runner = build_agent_runner(
            settings,
            retriever=LazySearch(settings, "memory"),
            cache=cache,
            offline=bool(args.cache_only),
            variant=args.tools,
        )
    except ConfigError as exc:
        print(str(exc).replace("\n", " "), file=err)
        return EXIT_CONFIG
    trace = runner(question)
    if trace.answer:
        print(trace.answer, file=out)
        print("", file=out)
    print(format_trace(trace), file=out)
    if trace.status != "answered":
        print(f"Agent zakończył pracę ze statusem {trace.status}: {trace.error}", file=err)
        return EXIT_PROVIDER
    return EXIT_OK


_SHOW_RUNS = {"live": "agent_run.live.json", "replay": "agent_run.replay.json"}


def show_trace(settings: Settings, *, args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """`kovalo show-trace ID --run ...`: ścieżka dowodowa jednego pytania z zapisanego przebiegu
    (port show_trace.py), bez API i bez LLM-a."""
    from kovalo.agents.show import architecture_label, format_trace
    from kovalo.agents.trace import load_traces
    from kovalo.eval.trace.reference import course_run_path

    if args.run == "course":
        path = course_run_path(settings, "single")
    elif args.run == "course-supervisor":
        path = course_run_path(settings, "supervisor")
    else:
        folder = Path(args.out) if args.out else settings.paths.mini_project_4_output_dir
        path = folder / _SHOW_RUNS[args.run]
    if not path.is_file():
        print(
            f"Nie znaleziono {path}: najpierw uruchom kovalo eval e4 (z --live dla --run live).",
            file=err,
        )
        return EXIT_CONFIG
    run, rows = load_traces(path)
    row = next((r for r in rows if r["trace"].question_id == args.question_id), None)
    if row is None:
        known = ", ".join(str(r["trace"].question_id) for r in rows)
        print(f"W {path.name} nie ma pytania {args.question_id!r}; są: {known}", file=err)
        return EXIT_CONFIG
    print(
        f"przebieg {run.get('run_id')}: {architecture_label(str(run.get('architecture')))}, "
        f"{run.get('provider')} / {run.get('model')}\n",
        file=out,
    )
    print(format_trace(row["trace"], row.get("score")), file=out)
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
    ev = sub.add_parser(
        "eval",
        help="ewaluacja etapu (e1–e5) albo dashboard, raporty w reports/",  # noqa: RUF001
    )
    ev.add_argument("stage", help="etap ewaluacji: e1, e2, e3, e4, e5 albo dashboard")
    ev.add_argument("--candidates", action="store_true", help="e1: także embeddings.candidates")
    ev.add_argument("--model", action="append", help="e1: tylko wskazany model (można powtórzyć)")
    ev.add_argument(
        "--backend",
        choices=["memory", "pgvector"],
        default="memory",
        help="wyszukiwanie w pamięci albo w Postgresie (e3: pytania o dokumenty na żywo)",
    )
    ev.add_argument(
        "--with-llm",
        action="store_true",
        help="e1: jedno wywołanie z promptem Q01; e2: poziomy 3–4 (odpowiedzi i sędzia)",  # noqa: RUF001
    )
    ev.add_argument(
        "--live",
        action="store_true",
        help="e3: także przebieg z modelem z configu (≤ 14 wywołań); e4: nasz agent na żywo",
    )
    ev.add_argument(
        "--max-calls",
        type=int,
        default=None,
        help="e4: limit wywołań sieciowych modelu na żywo (domyślnie 20; 0 = bez limitu); "
        "e5: wspólny limit prób understand + agent, także nieudanych (domyślnie 40)",
    )
    ev.add_argument(
        "--transport",
        choices=["direct", "memory", "stdio", "http"],
        default=None,
        help="e4: narzędzia agenta na żywo (domyślnie direct; inny = przez MCP); "
        "e5: połączenie z serwerem MCP (domyślnie stdio; http = services.mcp_url)",
    )
    ev.add_argument(
        "--count-errors",
        action="store_true",
        help="e4: nieudane próby modelu zjadają --max-calls (e5: zawsze)",
    )
    ev.add_argument(
        "--tools", choices=["course", "ours"], default=None, help="e4: wariant narzędzi na żywo"
    )
    ev.add_argument("--variants", default=None, help="e2: warianty z LLM-em (domyślnie ctx,hybrid)")
    ev.add_argument("--llm-side", choices=["ours", "reference"], default="ours", help="e2")
    ev.add_argument(
        "--ids", default=None, help="e2: tylko te pytania, np. M03,M10; e3: dialogi, np. D2,D7"
    )
    ev.add_argument("--no-judge", action="store_true", help="e2: bez sędziego")
    ev.add_argument("--no-cache", action="store_true", help="e2, e3: bez cache LLM-a")
    ev.add_argument(
        "--cache-only",
        action="store_true",
        help="e2, e3: tylko cache LLM-a, bez sieci i bez klucza",
    )
    ev.add_argument("--plan", action="store_true", help="e2, e3: wypisz plan wywołań i zakończ")
    ev.add_argument("--out", default=None, help="katalog raportów")
    ev.add_argument(
        "--reports-dir",
        default=None,
        help="dashboard: katalog z dashboard.yaml (domyślnie paths.reports_dir)",
    )
    ev.add_argument(
        "--check",
        action="store_true",
        help="dashboard: tylko porównaj z zapisanymi plikami (kod 1 przy różnicy)",
    )
    ask_p = sub.add_parser("ask", help="jedno pytanie do agenta z narzędziami (bez rozmowy)")
    ask_p.add_argument("question", help="pytanie po polsku")
    ask_p.add_argument("--tools", choices=["course", "ours"], default=None, help="wariant narzędzi")
    ask_p.add_argument("--no-cache", action="store_true", help="bez cache LLM-a")
    ask_p.add_argument(
        "--cache-only", action="store_true", help="tylko cache LLM-a, bez sieci i bez klucza"
    )
    st = sub.add_parser("show-trace", help="ścieżka dowodowa jednego pytania z zapisanego E4")
    st.add_argument("question_id", help="np. P1")
    st.add_argument(
        "--run",
        choices=["live", "replay", "course", "course-supervisor"],
        default="live",
        help="który przebieg (domyślnie live z reports/mp4)",
    )
    st.add_argument("--out", default=None, help="katalog raportów E4 (domyślnie reports/mp4)")
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
    if args.command == "ask":
        return ask(settings, args=args, out=sys.stdout, err=sys.stderr)
    if args.command == "show-trace":
        return show_trace(settings, args=args, out=sys.stdout, err=sys.stderr)
    return EXIT_CONFIG  # pragma: no cover (argparse wymusza polecenie)


def entrypoint() -> None:
    """Punkt wejścia `kovalo`."""
    sys.exit(main())
