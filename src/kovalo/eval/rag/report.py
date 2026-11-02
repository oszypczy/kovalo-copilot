"""Raporty E2: Markdown (commitowany dowód etapu), JSON-y i pliki w kształcie kursu.

Kontrakt F2, sekcja 8.8 (FR-6.4). Liczby w Markdownie po polsku (przecinek dziesiętny),
identyfikatory bez zmian. Nazwy modeli pochodzą z raportu (z configu), nigdy ze stałych.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from kovalo.eval.rag.e2 import (
    E2Report,
    LlmVariantResult,
    RetrievalRow,
    RetrievalVariantResult,
)
from kovalo.eval.rag.reference import COURSE_LIVE_SUMMARY, COURSE_RETRIEVAL
from kovalo.eval.rag.summary import E2Summary, QuestionRow
from kovalo.eval.review import REVIEW_FILE, Review, ReviewItem, answer_digest
from kovalo.rag.records import RunRecord

__all__ = [
    "course_live_report",
    "course_retrieval_report",
    "load_run",
    "render_markdown",
    "save_run",
    "write_reports",
]

HIT, MISS = "✓", "✗"
NONE = "\u2013"  # półpauza w pustych komórkach
COMMENT = "  - Komentarz: (do uzupełnienia w T2)"
_COURSE_KEY = {"mp1": "mp1", "ctx": "contextual"}  # nazwy wariantów w plikach kursu


# ---------------------------------------------------------------------------
# Pliki w kształcie kursu
# ---------------------------------------------------------------------------


def _side_result(report: E2Report, side: str, variant: str) -> RetrievalVariantResult | None:
    return next((r for r in report.retrieval if r.side == side and r.variant == variant), None)


def _course_row(row: RetrievalRow) -> dict[str, Any]:
    return {
        "top_chunk_ids": row.top_chunk_ids,
        "hit_at_1": row.hit_at_1,
        "hit_at_3": row.hit_at_3,
        "all_required_found": row.all_required_found,
        "evidence_phrase_coverage": row.evidence_phrase_coverage,
        "prompt_tokens": row.prompt_tokens,
    }


def course_retrieval_report(report: E2Report, side: str) -> dict[str, Any]:
    """Kształt retrieval_report.json kursu: {k, n_answerable, summary{mp1, contextual}, results}.

    Wariant 'ctx' występuje pod kluczem 'contextual'; hybryda poza tym plikiem (kurs jej nie ma).
    """
    variants = {
        key: result
        for variant, key in _COURSE_KEY.items()
        if (result := _side_result(report, side, variant)) is not None
    }
    if not variants:
        raise ValueError(f"Raport E2 nie ma strony {side!r}.")
    first = next(iter(variants.values()))
    summary = {
        key: {
            "n_chunks": r.n_chunks,
            "hit_at_1": r.hit_at_1,
            "hit_at_3": r.hit_at_3,
            "evidence_phrases_complete": r.evidence_phrases_complete,
            "avg_prompt_tokens": r.avg_prompt_tokens,
        }
        for key, r in variants.items()
    }
    results = []
    for i, row in enumerate(first.rows):
        item: dict[str, Any] = {
            "id": row.id,
            "question": row.question,
            "relevant_docs": row.relevant_docs,
            "requires_all": row.requires_all,
        }
        for key, r in variants.items():
            item[key] = _course_row(r.rows[i])
        results.append(item)
    return {"k": report.k, "n_answerable": len(first.rows), "summary": summary, "results": results}


def _run_info(report: E2Report, result: LlmVariantResult) -> dict[str, Any]:
    return {
        "run_id": report.run_id,
        "date": report.created_at.date().isoformat(),
        "provider": report.provider,
        "model": report.model,
        "side": result.side,
        "variant": result.variant,
    }


def course_live_report(report: E2Report, result: LlmVariantResult) -> dict[str, Any]:
    """Kształt live_report.json kursu: run_id, date, provider, model, summary, results."""
    return {
        **_run_info(report, result),
        "judge_model": report.judge_model if report.judge else None,
        "summary": result.summary.model_dump(mode="json"),
        "results": [
            {
                "id": row.id,
                "question": row.question,
                "answerable": row.answerable,
                "retrieval_hit": row.retrieval_hit,
                "answer": row.answer,
                "citations": row.citations,
                "error": row.error,
                "score": row.score.model_dump(mode="json"),
                "verdict": row.verdict.course_dict(),
            }
            for row in result.rows
        ],
    }


def save_run(path: Path, run: dict[str, Any], records: Sequence[Mapping[str, Any]]) -> None:
    """Port helpers/records.py: jeden plik JSON {"run": ..., "records": [{"id", "record"}]}."""
    rows = [{**row, "record": row["record"].model_dump(mode="json")} for row in records]
    Path(path).write_text(
        json.dumps({"run": run, "records": rows}, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_run(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Odwrotność save_run: (run, records z RunRecord)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["run"], [
        {**row, "record": RunRecord.model_validate(row["record"])} for row in data["records"]
    ]


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _num(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _usd(value: float | None) -> str:
    return NONE if value is None else f"${_num(value, 2)}"


def _frac(pair: tuple[int, int]) -> str:
    return f"{pair[0]}/{pair[1]}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def _side_label(report: E2Report, side: str) -> str:
    model = report.reference_embedding_model if side == "reference" else report.embedding_model
    return f"{side} ({model})"


def _course_cell(result: RetrievalVariantResult) -> str:
    course = COURSE_RETRIEVAL.get(result.variant)
    if result.side != "reference" or course is None:
        return NONE
    ours = {
        "n_chunks": result.n_chunks,
        "hit_at_1": result.hit_at_1,
        "hit_at_3": result.hit_at_3,
        "evidence_phrases_complete": result.evidence_phrases_complete,
        "avg_prompt_tokens": result.avg_prompt_tokens,
    }
    match = "tak" if ours == course else "nie"
    return (
        f"{course['hit_at_1']} / {course['hit_at_3']} / {course['evidence_phrases_complete']} / "
        f"{_num(course['avg_prompt_tokens'])}; zgodne: {match}"
    )


def _header(report: E2Report) -> list[str]:
    mp1, ctx = report.chunk_stats["mp1"], report.chunk_stats["ctx"]
    missing = report.chunks_not_in_course_cache
    recommended = report.recommended_mode or "brak (brak strony ours)"
    judge = report.judge_model if report.judge else "bez sędziego (--no-judge)"
    if report.llm:
        calls = [r.calls for r in report.llm]
        network = sum(c.answer + c.judge for c in calls)
        llm = (
            f"Wywołania LLM-a: {network} (plan ≤ {report.llm_calls_planned}, z cache "
            f"{sum(c.from_cache for c in calls)}, błędy {sum(c.errors for c in calls)})"
        )
    else:
        llm = "Wywołania LLM-a: 0 (bez --with-llm)"
    return [
        "# E2: odpowiedzi ze źródłami (MP2)",
        "",
        f"Przebieg: {report.run_id} · Model odpowiedzi: {report.model} · Sędzia: {judge} · "
        f"Embeddingi: {report.embedding_model} · Tryb w configu: {report.configured_mode} · "
        f"Rekomendacja reguły: {recommended}",
        f"Chunki MP1: {mp1.n_chunks} · z kontekstem: {ctx.n_chunks} (średnio "
        f"{_num(ctx.avg_chars)}, maks. {ctx.max_chars}) · spoza cache kursu: "
        f"{missing.get('mp1', 0)} / {missing.get('ctx', 0)} · {llm}",
        "",
    ]


def _retrieval_section(report: E2Report) -> list[str]:
    header = [
        "Embeddingi", "Wariant", "Chunki", "hit@1", "hit@3", "Komplet dowodów",
        "M02: oba dokumenty", "Tokeny promptu", "Kurs (hit@1 / hit@3 / dowody / tokeny)",
    ]  # fmt: skip
    rows = []
    for r in report.retrieval:
        n = len(r.rows)
        found, required = r.all_required_found
        mark = NONE if required == 0 else (HIT if found == required else MISS)
        rows.append(
            [
                _side_label(report, r.side),
                r.variant,
                str(r.n_chunks),
                f"{r.hit_at_1}/{n}",
                f"{r.hit_at_3}/{n}",
                f"{r.evidence_phrases_complete}/{n}",
                f"{mark} {found}/{required}",
                _num(r.avg_prompt_tokens),
                _course_cell(r),
            ]
        )
    lines = ["## Wyszukiwanie i dowody (poziomy 1–2, bez LLM-a)", ""]  # noqa: RUF001
    lines += _table(header, rows)
    lines += [
        "",
        f"Top-{report.k}; pytania z odpowiedzią w korpusie. mp1: chunki MP1, wyszukiwanie "
        "wektorowe; ctx: chunki z kontekstem, wektorowe; hybrid: chunki z kontekstem, BM25 + "
        "wektory (fuzja RRF). Kolumna „Kurs”: wynik check.py kursu dla embeddingów referencyjnych.",
    ]
    pg = [r for r in report.retrieval if r.pg_consistent is not None]
    if pg:
        lines.append("")
        lines += [
            f"- {r.side} {r.variant}: top-{report.k} z Postgresa = top-{report.k} z pamięci: "
            f"{'tak' if r.pg_consistent else 'nie'}"
            for r in pg
        ]
    return lines


def _hit_cell(row: RetrievalRow) -> str:
    mark = f"{HIT}1" if row.hit_at_1 else (f"{HIT}3" if row.hit_at_3 else MISS)
    evidence = (
        NONE if row.evidence_phrase_coverage is None else _num(row.evidence_phrase_coverage, 2)
    )
    return f"{mark} {evidence} · {', '.join(row.top_chunk_ids)}"


def _per_question_retrieval(report: E2Report) -> list[str]:
    lines = ["## Wyszukiwanie pytanie po pytaniu", ""]
    if not report.retrieval:
        return [*lines, "Brak wierszy."]
    header = ["Pytanie", "Trafne", *(f"{r.side} {r.variant}" for r in report.retrieval)]
    rows = []
    for i, first in enumerate(report.retrieval[0].rows):
        rows.append(
            [
                first.id,
                ", ".join(first.relevant_docs),
                *(_hit_cell(r.rows[i]) for r in report.retrieval),
            ]
        )
    lines += _table(header, rows)
    lines += [
        "",
        f"{HIT}1 / {HIT}3: trafny dokument na 1. miejscu / w top-{report.k}; {MISS}: brak; "
        "liczba: udział fraz dowodowych w znalezionym tekście; potem top-k chunków.",
    ]
    return lines


_LEVELS: tuple[tuple[str, str], ...] = (
    ("1 Retrieval: znaleziono trafny dokument", "retrieval_hit"),
    ("2 Dowody: wszystkie frazy dowodowe w tekście", "evidence_phrases_complete"),
    ("3 Odpowiedź: udzielono odpowiedzi", "answered"),
    ("3 Odpowiedź: wszystkie kluczowe fakty", "key_facts_complete"),
    ("3 Sędzia: poprawna", "judge_correct"),
    ("3 Sędzia: oparta na kontekście", "judge_grounded"),
    ("4 Odwołania: prawidłowe (obecne, z kontekstu)", "citations_valid"),
    ("4 Odwołania: do trafnego dokumentu", "cites_relevant"),
    ("Odmowy: trafne (pytania spoza korpusu)", "refusals_correct"),
)


def _level_cell(summary: E2Summary | dict[str, Any], level: str, mark: bool | None) -> str:
    value = summary[level] if isinstance(summary, dict) else getattr(summary, level)
    text = _frac((int(value[0]), int(value[1])))
    if mark is None:
        return text
    return f"{text} {HIT if mark else MISS}"


def _llm_section(report: E2Report) -> list[str]:
    if not report.llm:
        return []
    header = [
        "Poziom",
        f"Kurs ({report.reference_model}, ctx)",
        *(f"{r.side} {r.variant}" for r in report.llm),
    ]
    rows = []
    for label, level in _LEVELS:
        cells = [label, _level_cell(COURSE_LIVE_SUMMARY, level, None)]
        for r in report.llm:
            if not report.judge and level.startswith("judge_"):
                cells.append(NONE)
            else:
                cells.append(_level_cell(r.summary, level, r.meets_target.get(level)))
        rows.append(cells)

    def refusals(summary: E2Summary | dict[str, Any]) -> str:
        data = (
            summary["refusals_on_answerable"]
            if isinstance(summary, dict)
            else summary.refusals_on_answerable
        )
        return f"bez fraz {data['phrases_missing']}, z frazami {data['phrases_present']}"

    rows.append(
        [
            "Odmowy przy pytaniach z odpowiedzią",
            refusals(COURSE_LIVE_SUMMARY),
            *(refusals(r.summary) for r in report.llm),
        ]
    )
    rows.append(
        ["Błędy", str(COURSE_LIVE_SUMMARY["errors"]), *(str(r.summary.errors) for r in report.llm)]
    )
    rows.append(
        [
            "Błędy sędziego",
            str(COURSE_LIVE_SUMMARY["judge_errors"]),
            *(str(r.summary.judge_errors) if report.judge else NONE for r in report.llm),
        ]
    )
    lines = ["## Odpowiedzi (poziomy 3–4, z LLM-em)", ""]  # noqa: RUF001
    lines += _table(header, rows)
    lines += [
        "",
        f"{HIT}/{MISS}: cel „wynik ≥ referencja − 1” (06-evaluation). Kolumna kursu to stałe "  # noqa: RUF001
        "z README MP2 (przebieg referencyjny, chunki z kontekstem); poziomów 3–4 kursu nie da się "  # noqa: RUF001
        "przeliczyć offline.",
    ]
    for r in report.llm:
        c = r.calls
        models = _model_counts(r.answered_by)
        judged = f"; ocenił: {_model_counts(r.judged_by)}" if report.judge else ""
        lines.append(
            f"- {r.side} {r.variant}: wywołania LLM-a: odpowiedzi {c.answer}, sędzia {c.judge}, "
            f"z cache {c.from_cache}, błędy {c.errors}; odpowiedział: {models}{judged}"
        )
        others = sorted(m for m in r.answered_by if m != report.model)
        if others:
            lines.append(
                f"  - Uwaga: część odpowiedzi z innego modelu niż {report.model} "
                f"({', '.join(others)}): powtórz te pytania przez --ids, gdy limit wróci."
            )
        other_judges = sorted(m for m in r.judged_by if m != report.judge_model)
        if report.judge and other_judges:
            lines.append(
                f"  - Uwaga: część ocen sędziego z innego modelu niż {report.judge_model} "
                f"({', '.join(other_judges)}): fallback; powtórz te pytania przez --ids, "
                "gdy limit wróci."
            )
    if report.judge:
        lines += _judge_identity(report)
    return lines


def _model_counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{m} {n}" for m, n in sorted(counts.items())) or NONE


def _judge_identity(report: E2Report) -> list[str]:
    """Czy sędzia to ten sam model co odpowiedzi: według modeli, które faktycznie odpowiedziały."""
    answer_models = {m for r in report.llm for m in r.answered_by}
    judge_models = {m for r in report.llm for m in r.judged_by}
    if not judge_models:
        if any(row.verdict.correct is not None for r in report.llm for row in r.rows):
            return ["- Sędzia: brak danych o modelu w odpowiedziach dostawcy."]
        return []
    if judge_models == answer_models and len(judge_models) == 1:
        return [
            f"- Sędzia to ten sam model co odpowiedzi ({next(iter(judge_models))}): ryzyko "
            "zawyżonej samooceny; uzasadnienia sędziego przegląda T2."
        ]
    shared = sorted(judge_models & answer_models)
    if shared:
        return [
            f"- Sędzia i odpowiedzi częściowo z tego samego modelu ({', '.join(shared)}): "
            "ryzyko zawyżonej samooceny dla tych ocen; uzasadnienia sędziego przegląda T2."
        ]
    return [
        f"- Sędzia to inny model niż odpowiedzi (sędzia: {', '.join(sorted(judge_models))}; "
        f"odpowiedzi: {', '.join(sorted(answer_models)) or NONE})."
    ]


def _verdict_text(row: QuestionRow) -> str:
    v = row.verdict
    if v.error is not None:
        return "błąd"
    if v.correct is None:
        return NONE
    return f"{'tak' if v.correct else 'NIE'}/{'tak' if v.grounded else 'NIE'}"


def _per_question_llm(report: E2Report) -> list[str]:
    if not report.llm:
        return []
    lines = ["## Odpowiedzi pytanie po pytaniu", ""]
    for r in report.llm:
        cached = set(r.from_cache)
        lines += [f"### {r.side} {r.variant}", ""]
        rows = []
        for row in r.rows:
            score = row.score
            outcome = score.outcome + (f" ({score.refusal})" if score.refusal else "")
            coverage = NONE if score.answer_coverage is None else _num(score.answer_coverage, 2)
            rows.append(
                [
                    row.id,
                    outcome,
                    coverage,
                    _verdict_text(row),
                    ", ".join(row.citations or []) or NONE,
                    row.record.answered_by or NONE,
                    row.verdict.judged_by or NONE,
                    "tak" if row.id in cached else "nie",
                ]
            )
        lines += _table(
            [
                "Pytanie",
                "Wynik",
                "Pokrycie",
                "Sędzia",
                "Odwołania",
                "Model",
                "Model sędziego",
                "Cache",
            ],
            rows,
        )
        lines.append("")
    lines.append(
        "Pokrycie: kluczowe fakty znalezione w odpowiedzi; sędzia: poprawna/oparta na kontekście; "
        "Model i Model sędziego: model, który faktycznie odpowiedział (fallback widoczny)."
    )
    return lines


def _cost_section(report: E2Report) -> list[str]:
    usage = {r.variant: r.avg_usage for r in report.llm}
    header = [
        "Wariant",
        "Tokeny promptu (o200k_base)",
        "Tokeny dostawcy: wejście / wyjście (w tym myślenie)",
        f"{report.reference_model} (350 tokenów)",
        f"{report.our_cost_model} (350 tokenów)",
        f"{report.our_cost_model} (faktyczne tokeny)",
    ]
    rows = []
    for c in report.costs:
        u = usage.get(c.variant)
        provider = NONE
        if u is not None:
            thinking = f" ({u.reasoning_tokens})" if u.reasoning_tokens is not None else ""
            provider = f"{u.input_tokens} / {u.output_tokens}{thinking}"
        rows.append(
            [
                c.variant,
                _num(c.avg_prompt_tokens),
                provider,
                _usd(c.reference_monthly_usd),
                _usd(c.ours_monthly_usd),
                _usd(c.ours_actual_monthly_usd),
            ]
        )
    lines = ["## Tokeny i koszt", ""]
    lines += _table(header, rows)
    lines += [
        "",
        "Uwagi: miesięczny koszt odpowiedzi w scenariuszu kursu (PINNED); tokeny promptu ze strony "
        "ours (bez niej reference). Sędzia nie wchodzi do kosztu produkcyjnego. Darmowy tier "
        f"Gemini = $0; ceny płatnego tieru sprawdzone {report.our_prices_checked_on} (ADR-0007).",
    ]
    return lines


def _recommendation_section(report: E2Report) -> list[str]:
    lines = ["## Rekomendacja trybu wyszukiwania", ""]
    ours = {r.variant: r for r in report.retrieval if r.side == "ours"}
    if "ctx" not in ours or "hybrid" not in ours:
        return [*lines, "Brak strony ours (embeddings.provider=openai_cache): brak rekomendacji."]
    lines.append(
        "Reguła (strona ours): hybrid, gdy hybryda nie jest gorsza od ctx w żadnej mierze "
        "i lepsza w co najmniej jednej; inaczej vector."
    )
    lines.append("")
    measures: list[tuple[str, int, int]] = [
        ("hit@1", ours["ctx"].hit_at_1, ours["hybrid"].hit_at_1),
        ("hit@3", ours["ctx"].hit_at_3, ours["hybrid"].hit_at_3),
        (
            "komplet dowodów",
            ours["ctx"].evidence_phrases_complete,
            ours["hybrid"].evidence_phrases_complete,
        ),
    ]
    llm = {r.variant: r.summary for r in report.llm if r.side == "ours"}
    if "ctx" in llm and "hybrid" in llm:
        for label, level in (
            ("udzielone", "answered"),
            ("kluczowe fakty", "key_facts_complete"),
            ("sędzia: poprawna", "judge_correct"),
        ):
            measures.append(
                (label, getattr(llm["ctx"], level)[0], getattr(llm["hybrid"], level)[0])
            )
    for label, ctx, hybrid in measures:
        verdict = "lepsza" if hybrid > ctx else ("gorsza" if hybrid < ctx else "remis")
        lines.append(f"- {label}: ctx {ctx}, hybrid {hybrid} → {verdict}")
    lines += [
        "",
        f"Wynik: {report.recommended_mode}. Tryb w configu: {report.configured_mode}. Rekomendacja "
        "nie zmienia configu: decyzję o retrieval.mode podejmuje właściciel projektu (pytanie U2).",
    ]
    return lines


@dataclass
class _ReviewEntry:
    """Pozycja wykryta automatycznie: etykieta wariantu, id pytania, treść i skrót odpowiedzi."""

    side: str
    variant: str
    id: str
    reasons: str
    details: list[str] = field(default_factory=list)
    digest: str | None = None

    @property
    def label(self) -> str:
        return f"{self.side}-{self.variant}"


def _retrieval_review_items(report: E2Report) -> list[_ReviewEntry]:
    """Bez LLM-a: pudła i niepełne dowody wariantów ctx i hybrid (strona ours albo reference)."""
    items: list[_ReviewEntry] = []
    side = "ours" if any(r.side == "ours" for r in report.retrieval) else "reference"
    for result in report.retrieval:
        if result.side != side or result.variant == "mp1":
            continue
        for row in result.rows:
            reasons = []
            if not row.hit_at_3:
                reasons.append(f"brak trafnego dokumentu w top-{report.k}")
            coverage = row.evidence_phrase_coverage
            if coverage is not None and coverage < 1:
                reasons.append(f"dowody {_num(coverage, 2)}")
            if reasons:
                items.append(
                    _ReviewEntry(
                        result.side,
                        result.variant,
                        row.id,
                        f"{'; '.join(reasons)} (top-{report.k}: {', '.join(row.top_chunk_ids)})",
                    )
                )
    return items


def _llm_review_items(report: E2Report) -> list[_ReviewEntry]:
    items: list[_ReviewEntry] = []
    for r in report.llm:
        for row in r.rows:
            reasons = []
            score, verdict = row.score, row.verdict
            if row.error:
                reasons.append(f"błąd odpowiedzi: {row.error}")
            if score.refusal in ("phrases_missing", "phrases_present"):
                reasons.append(f"odmowa przy pytaniu z odpowiedzią ({score.refusal})")
            if score.refusal == "missed":
                reasons.append("odpowiedź na pytanie spoza korpusu")
            coverage = score.answer_coverage
            if coverage is not None and coverage < 1:
                reasons.append(f"pokrycie kluczowych faktów {_num(coverage, 2)}")
            if verdict.correct is False and coverage == 1:
                reasons.append("sędzia: NIE przy pokryciu 1")
            if verdict.correct is True and coverage is not None and coverage < 1:
                reasons.append("sędzia: tak przy pokryciu < 1")
            if verdict.error:
                reasons.append(f"błąd sędziego: {verdict.error}")
            if reasons:
                details = [f"  - Uzasadnienie sędziego: {verdict.reason}"] if verdict.reason else []
                items.append(
                    _ReviewEntry(
                        r.side,
                        r.variant,
                        row.id,
                        "; ".join(reasons),
                        details,
                        answer_digest(row.answer, row.error),
                    )
                )
    return items


def _comment_lines(comment: str) -> list[str]:
    """„Komentarz:” jako punkt pod pozycją; kolejne wiersze wcięte jako ciąg dalszy punktu."""
    first, *rest = comment.splitlines()
    return [f"  - Komentarz: {first}", *(f"    {ln}" if ln.strip() else "" for ln in rest)]


def _stale_line(item: ReviewItem, label: str, digest: str | None) -> list[str]:
    old = item.answer_digest.get(label)
    if old is None or digest is None or old == digest:
        return []
    return [
        f"  - Uwaga: komentarz dotyczy innej odpowiedzi (#{old}, teraz #{digest}); "
        "przejrzyj ponownie."
    ]


def _current_rows(report: E2Report) -> dict[tuple[str, str], tuple[list[str], str | None]]:
    """Bieżące dane każdego wiersza raportu: (etykieta, id) → (opis, skrót odpowiedzi)."""
    rows: dict[tuple[str, str], tuple[list[str], str | None]] = {}
    if report.llm:
        for r in report.llm:
            for row in r.rows:
                score = row.score
                outcome = score.outcome + (f" ({score.refusal})" if score.refusal else "")
                coverage = NONE if score.answer_coverage is None else _num(score.answer_coverage, 2)
                digest = answer_digest(row.answer, row.error)
                text = [
                    f"  - {r.side}-{r.variant}: wynik {outcome}, pokrycie {coverage}, "
                    f"sędzia {_verdict_text(row)} · odpowiedź #{digest}"
                ]
                if row.verdict.reason:
                    text.append(f"  - Uzasadnienie sędziego: {row.verdict.reason}")
                rows[(f"{r.side}-{r.variant}", row.id)] = (text, digest)
        return rows
    for result in report.retrieval:
        for rrow in result.rows:
            evidence = (
                NONE
                if rrow.evidence_phrase_coverage is None
                else _num(rrow.evidence_phrase_coverage, 2)
            )
            rows[(f"{result.side}-{result.variant}", rrow.id)] = (
                [
                    f"  - {result.side}-{result.variant}: hit@1 "
                    f"{'tak' if rrow.hit_at_1 else 'nie'}, hit@{report.k} "
                    f"{'tak' if rrow.hit_at_3 else 'nie'}, dowody {evidence} "
                    f"(top-{report.k}: {', '.join(rrow.top_chunk_ids)})"
                ],
                None,
            )
    return rows


def _review_section(report: E2Report, review: Review | None = None) -> list[str]:
    lines = ["## Rozbieżności przejrzane ręcznie", ""]
    entries = _llm_review_items(report) if report.llm else _retrieval_review_items(report)
    if review is None:
        items: list[str] = []
        for e in entries:
            items += [f"- {e.side} {e.variant}, {e.id}: {e.reasons}", *e.details, COMMENT]
        if not items:
            items = ["- Brak rozbieżności wykrytych automatycznie.", COMMENT]
        return [*lines, *items]

    if review.intro.strip():
        lines += [review.intro.strip(), ""]
    lines += [f"Komentarze: {REVIEW_FILE}, przegląd {review.reviewed_on.isoformat()}", ""]
    covered: set[tuple[str, str]] = set()
    for e in entries:
        suffix = f" · odpowiedź #{e.digest}" if e.digest else ""
        lines += [f"- {e.label}, {e.id}: {e.reasons}{suffix}", *e.details]
        item = review.find(e.label, e.id)
        if item is None:
            lines.append(COMMENT)
            continue
        covered.add((e.label, e.id))
        lines += _stale_line(item, e.label, e.digest)
        lines += _comment_lines(item.comment)
    if not entries:
        lines += ["- Brak rozbieżności wykrytych automatycznie.", COMMENT]

    current = _current_rows(report)
    orphans: list[str] = []
    for item in review.items:
        rest = [v for v in item.variants if (v, item.id) not in covered]
        present = [v for v in rest if (v, item.id) in current]
        missing = [v for v in rest if (v, item.id) not in current]
        if present:
            lines.append(f"- {', '.join(present)}, {item.id} (dodane w przeglądzie ręcznym)")
            for label in present:
                text, digest = current[(label, item.id)]
                lines += [*text, *_stale_line(item, label, digest)]
            lines += _comment_lines(item.comment)
        if missing:
            orphans += [f"- {', '.join(missing)}, {item.id}", *_comment_lines(item.comment)]
    if orphans:
        lines += ["", "Komentarze bez pozycji w tym przebiegu:", "", *orphans]
    if review.notes:
        lines += ["", *(f"- {note.strip()}" for note in review.notes)]
    return lines


def render_markdown(report: E2Report, review: Review | None = None) -> str:
    """Markdown raportu; komentarze z `review` (bez walidacji kluczy: robi to CLI)."""
    lines = _header(report)
    sections = [
        _retrieval_section(report),
        _per_question_retrieval(report),
        _llm_section(report),
        _per_question_llm(report),
        _cost_section(report),
        _recommendation_section(report),
        _review_section(report, review),
    ]
    for section in sections:
        if section:
            lines += [*section, ""]
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Zapis
# ---------------------------------------------------------------------------


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def write_reports(
    report: E2Report,
    out_dir: Path,
    *,
    ctx_chunks: Sequence[Document],
    review: Review | None = None,
) -> list[Path]:
    """<data>-e2.md, <data>-e2.json, retrieval_report.json (strona ours, bez niej reference),
    chunks_ctx.jsonl oraz answers.<strona>-<wariant>.json i live_report.<strona>-<wariant>.json
    dla każdego wariantu z LLM-em. Zapis zawsze, także gdy wszystkie rekordy to błędy."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    day = report.created_at.date().isoformat()
    side = "ours" if any(r.side == "ours" for r in report.retrieval) else "reference"
    paths = [
        _write(out_dir / f"{day}-e2.md", render_markdown(report, review)),
        _write(out_dir / f"{day}-e2.json", report.model_dump_json(indent=2)),
        _write(out_dir / "retrieval_report.json", _json(course_retrieval_report(report, side))),
        _write(
            out_dir / "chunks_ctx.jsonl",
            "".join(
                json.dumps(
                    {"metadata": c.metadata, "page_content": c.page_content}, ensure_ascii=False
                )
                + "\n"
                for c in ctx_chunks
            ),
        ),
    ]
    for result in report.llm:
        name = f"{result.side}-{result.variant}"
        answers = out_dir / f"answers.{name}.json"
        save_run(
            answers,
            _run_info(report, result),
            [{"id": row.id, "record": row.record} for row in result.rows],
        )
        paths.append(answers)
        paths.append(
            _write(out_dir / f"live_report.{name}.json", _json(course_live_report(report, result)))
        )
    return paths
