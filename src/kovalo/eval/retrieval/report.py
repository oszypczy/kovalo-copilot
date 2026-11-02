"""Raporty E1: Markdown (commitowany dowód etapu), JSON-y, report.json w kształcie kursu.

Kontrakt F1, sekcja 7.4 (FR-6.4). Liczby w Markdownie w zapisie polskim (przecinek
dziesiętny), identyfikatory bez zmian. Nazwy modeli pochodzą z raportu (z configu).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from kovalo.eval.retrieval.e1 import REFERENCE_LABEL, E1Report, EmbeddingRun
from kovalo.eval.retrieval.metrics import QueryResult, RetrievalResult

__all__ = ["course_report", "render_markdown", "write_reports"]

HIT, MISS = "✓", "✗"
NONE = "\u2013"  # półpauza w pustych komórkach


def _num(value: float, digits: int | None = None) -> str:
    """Liczba w zapisie polskim; digits=None: tak, jak jest zaokrąglona w raporcie."""
    text = f"{value:.{digits}f}" if digits is not None else repr(float(value))
    return text.replace(".", ",")


def _usd(value: float) -> str:
    return f"${_num(value, 2)}"


def _plural(n: int, one: str, few: str, many: str) -> str:
    if n == 1:
        return f"{n} {one}"
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return f"{n} {few}"
    return f"{n} {many}"


def _label(run: EmbeddingRun) -> str:
    return f"{run.label} ({run.model})" if run.label == REFERENCE_LABEL else run.label


def _configured_run(report: E1Report) -> EmbeddingRun | None:
    for run in report.runs:
        if run.provider == "fastembed" and run.model == report.configured_model:
            return run
    return report.runs[0] if report.runs else None


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def _summary_section(report: E1Report) -> list[str]:
    header = [
        "Model", "Wymiar", "Rozmiar", "MP1 hit@1", "MP1 hit@3", "MP1 MRR",
        "MP2 hit@1", "MP2 hit@3", "Cel", "Czas 47 chunków", "Pytanie (ms)",
    ]  # fmt: skip
    header[9] = f"Czas {report.chunks.n_chunks} chunków"
    rows = [
        [
            _label(run),
            str(run.dimensions),
            f"{_num(run.size_gb, 2)} GB" if run.size_gb is not None else NONE,
            _num(run.mp1.hit_rate_at_1),
            _num(run.mp1.hit_rate_at_3),
            _num(run.mp1.mrr),
            _num(run.mp2.hit_rate_at_1),
            _num(run.mp2.hit_rate_at_3),
            "tak" if run.meets_target else "nie",
            f"{_num(run.embed_chunks_seconds, 1)} s"
            if run.embed_chunks_seconds is not None
            else NONE,
            _num(run.query_ms_avg, 1) if run.query_ms_avg is not None else NONE,
        ]
        for run in report.runs
    ]
    targets = report.targets
    lines = [f"## Wyszukiwanie (top-{report.k})", ""]
    lines += _table(header, rows)
    lines += [
        "",
        f"Cel: MP1 hit@1 ≥ {_num(targets['hit_rate_at_1'])} i hit@3 ≥ "
        f"{_num(targets['hit_rate_at_3'])}. MP2: tylko pytania z odpowiedzią w korpusie.",
    ]
    return lines


def _cell(result: QueryResult) -> str:
    mark = HIT if result.hit_at_3 else MISS
    return f"{mark} {', '.join(result.top_ids)}"


def _per_question_section(title: str, report: E1Report, pick: str) -> list[str]:
    lines = [title, ""]
    if not report.runs:
        return [*lines, "Brak wierszy.", ""]
    sets: list[RetrievalResult] = [getattr(run, pick) for run in report.runs]
    header = ["Pytanie", "Istotne", *(run.label for run in report.runs)]
    rows = []
    for i, first in enumerate(sets[0].results):
        rows.append(
            [first.query_id, ", ".join(first.relevant_docs), *(_cell(s.results[i]) for s in sets)]
        )
    lines += _table(header, rows)
    lines += ["", f"{HIT}/{MISS}: czy w top-{report.k} jest chunk istotnego dokumentu."]
    return lines


def _cost_section(report: E1Report) -> list[str]:
    scenario = report.cost_scenario
    lines = [
        f"## Koszty miesięczne ({scenario.questions_per_day} pytań/dzień, "
        f"{scenario.working_days_per_month} dni, {scenario.output_tokens_per_answer} tokenów "
        "odpowiedzi)",
        "",
    ]
    if not report.runs:
        return [*lines, "Brak wierszy."]
    first = report.runs[0]
    header = [
        "Model embeddingów",
        "Śr. tokeny promptu (retrieval / cały korpus)",
        f"{first.cost_reference.llm_model} (referencja)",
        f"{first.cost_ours.llm_model} (ceny z {report.our_prices_checked_on.isoformat()})",
    ]
    rows = []
    for run in report.runs:
        ref, ours = run.cost_reference, run.cost_ours
        rows.append(
            [
                _label(run),
                f"{_num(ref.rag_avg_input_tokens, 1)} / "
                f"{_num(ref.full_context_avg_input_tokens, 1)}",
                f"{_usd(ref.rag_monthly_usd)} / {_usd(ref.full_context_monthly_usd)} "
                f"(x{_num(ref.full_to_rag_cost_ratio, 1)})",
                f"{_usd(ours.rag_monthly_usd)} / {_usd(ours.full_context_monthly_usd)}",
            ]
        )
    lines += _table(header, rows)
    lines += [
        "",
        f"Uwagi: tokeny liczone tokenizerem kursu (PINNED, wspólny dla obu cenników); "
        f"{first.cost_ours.llm_model} liczy tokeny inaczej i dolicza tokeny myślenia, więc to "
        "dolna granica. Darmowy tier Gemini = $0. Ceny płatnego tieru sprawdzone "
        f"{report.our_prices_checked_on.isoformat()}; zapowiedzianą zmianę cen opisuje ADR-0007.",
    ]
    return lines


def _llm_section(report: E1Report) -> list[str]:
    rec = report.llm_call
    if rec is None:
        return []
    lines = ["## Wywołanie LLM-a (--with-llm)", ""]
    lines += [
        f"- Dostawca / model z configu: {rec.provider} / {rec.model}; odpowiedział: "
        f"{rec.answered_by}; czas: {_num(rec.seconds, 1)} s ({rec.date.isoformat()})",
        f"- Pytanie: {rec.question}",
        f"- Kontekst: {', '.join(rec.context_chunks)}",
        f"- Tokeny wejściowe: oszacowanie {rec.estimated_input_tokens} ({rec.estimate_tokenizer}), "
        f"dostawca policzył {rec.input_tokens}",
        f"- Tokeny wyjściowe: scenariusz zakłada {rec.assumed_output_tokens}, ta odpowiedź "
        f"zużyła {rec.output_tokens}"
        + (f" (w tym myślenie {rec.reasoning_tokens})" if rec.reasoning_tokens is not None else ""),
        "",
        "Odpowiedź:",
        "",
    ]
    lines += [f"> {line}" if line.strip() else ">" for line in rec.answer.splitlines()]
    return lines


def _pg_section(report: E1Report) -> list[str]:
    checked = [run for run in report.runs if run.pg_consistent is not None]
    if not checked:
        return []
    lines = ["## Zgodność z Postgresem (--backend pgvector)", ""]
    for run in checked:
        verdict = "tak" if run.pg_consistent else "nie"
        lines.append(
            f"- {_label(run)}: top-{report.k} z Postgresa = top-{report.k} z pamięci: {verdict}"
        )
    return lines


def _review_section(report: E1Report) -> list[str]:
    lines = ["## Rozbieżności przejrzane ręcznie", ""]
    wanted = {report.configured_model, report.recommended_model}
    runs = [r for r in report.runs if r.provider == "fastembed" and r.model in wanted]
    if not runs:
        runs = report.runs[:1]
    found = False
    for run in runs:
        for result_set in (run.mp1, run.mp2):
            for r in result_set.results:
                if r.hit_at_3:
                    continue
                found = True
                lines += [
                    f"- {_label(run)}, {r.query_id} ({result_set.set_name}): {r.question}",
                    f"  - top-{report.k}: {', '.join(r.top_ids)}; istotne: "
                    f"{', '.join(r.relevant_docs)}",
                    "  - Komentarz: (do uzupełnienia przed commitem)",
                ]
    if not found:
        lines.append("Brak pudeł w top-k dla modelu z configu i modelu rekomendowanego.")
    return lines


def render_markdown(report: E1Report) -> str:
    corpus, chunks = report.corpus, report.chunks
    recommended = report.recommended_model or "brak (brak wierszy fastembed)"
    lines = [
        "# E1: wyszukiwanie i koszty (MP1)",
        "",
        f"Przebieg: {report.run_id} · Model z configu: {report.configured_model} · "
        f"Rekomendacja reguły: {recommended}",
        f"Korpus: {_plural(corpus.n_documents, 'dokument', 'dokumenty', 'dokumentów')} / "
        f"{_plural(corpus.total_tokens, 'token', 'tokeny', 'tokenów')} · "
        f"Chunki: {chunks.n_chunks} (średnio {_num(chunks.avg_chars, 1)}, maks. "
        f"{chunks.max_chars}) · Chunki spoza cache kursu: {report.chunks_not_in_course_cache}",
        "",
    ]
    sections = [
        _summary_section(report),
        _per_question_section("## Wyniki pytanie po pytaniu (MP1)", report, "mp1"),
        _per_question_section("## Wyniki pytanie po pytaniu (MP2, M01–M10)", report, "mp2"),  # noqa: RUF001
        _cost_section(report),
        _llm_section(report),
        _pg_section(report),
        _review_section(report),
    ]
    for section in sections:
        if section:
            lines += [*section, ""]
    return "\n".join(lines).rstrip() + "\n"


def course_report(report: E1Report) -> dict[str, Any]:
    """Kształt report.json z run_pipeline.py kursu dla modelu z configu (+ cost_ours)."""
    run = _configured_run(report)
    data: dict[str, Any] = {
        "corpus": report.corpus.model_dump(mode="json"),
        "chunks": report.chunks.model_dump(mode="json"),
    }
    if run is None:
        return data
    data["embeddings"] = {
        "provider": run.provider,
        "model": run.model,
        "dimensions": run.dimensions,
        "chunks_not_cached": report.chunks_not_in_course_cache,
    }
    data["retrieval"] = {
        "k": report.k,
        "results": [
            {
                "query_id": r.query_id,
                "top_ids": r.top_ids,
                "top_scores": r.top_scores,
                "hit_at_1": r.hit_at_1,
                "hit_at_3": r.hit_at_3,
            }
            for r in run.mp1.results
        ],
        "hit_rate_at_1": run.mp1.hit_rate_at_1,
        "hit_rate_at_3": run.mp1.hit_rate_at_3,
    }
    data["cost"] = run.cost_reference.model_dump(mode="json")
    data["cost_ours"] = run.cost_ours.model_dump(mode="json")
    return data


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def write_reports(report: E1Report, out_dir: Path, *, chunks: Sequence[Document]) -> list[Path]:
    """<data>-e1.md, <data>-e1.json, report.json, chunks.jsonl i llm_call.json (gdy jest)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    day = report.created_at.date().isoformat()
    paths = [
        _write(out_dir / f"{day}-e1.md", render_markdown(report)),
        _write(out_dir / f"{day}-e1.json", report.model_dump_json(indent=2)),
        _write(
            out_dir / "report.json",
            json.dumps(course_report(report), indent=2, ensure_ascii=False),
        ),
        _write(
            out_dir / "chunks.jsonl",
            "".join(
                json.dumps(
                    {"metadata": c.metadata, "page_content": c.page_content}, ensure_ascii=False
                )
                + "\n"
                for c in chunks
            ),
        ),
    ]
    if report.llm_call is not None:
        paths.append(
            _write(
                out_dir / "llm_call.json",
                json.dumps(report.llm_call.model_dump(mode="json"), indent=2, ensure_ascii=False),
            )
        )
    return paths
