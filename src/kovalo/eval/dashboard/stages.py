# ruff: noqa: RUF001, RUF002  (polska typografia: półpauza w zakresach)
"""Adaptery etapów E1–E5: JSON raportu → `StageSummary` (kontrakt F6, 8.3).

Każdy adapter czyta tylko pola zbiorcze JSON-a swojego etapu (struktura JSON-ów E1–E5 jest różna)
i nie liczy wyników od nowa: przepisuje je w zapisie raportów Markdown i porównuje z celami
z docs/06-evaluation.md (`TARGETS`). Brakujący klucz to `DashboardDataError` z etapem i ścieżką,
nie cicha pustka. Liczby kursu, których nie ma w JSON-ach (poziomy 3–4 E2, koszt miesięczny E1,
check.py MP2), są tu stałymi z README kursu, jak w generatorach raportów.
"""

from __future__ import annotations

import datetime
from collections.abc import Callable, Sequence
from typing import Any

from kovalo.config import get_settings
from kovalo.eval.dashboard.models import MetricRow, MonthlyCost, RunCalls, StageCost, StageSummary
from kovalo.eval.dashboard.spec import StageSpec
from kovalo.text_pl import number

__all__ = [
    "ADAPTERS",
    "TARGETS",
    "TITLES",
    "DashboardDataError",
    "calls_from_first_live",
    "monthly_cost_e1",
    "summarise_e1",
    "summarise_e2",
    "summarise_e3",
    "summarise_e4",
    "summarise_e5",
]


class DashboardDataError(ValueError):
    """Brak pola w JSON-ie raportu etapu (CLI: kod 2)."""

    def __init__(self, stage: str, path: str, detail: str = "brak pola {path} w JSON-ie raportu"):
        self.stage = stage
        self.path = path
        super().__init__(f"{stage.upper()} ({stage}): {detail.format(path=path)}")


TITLES: dict[str, str] = {
    "e1": "wyszukiwanie i koszty (MP1)",
    "e2": "odpowiedzi ze źródłami (MP2)",
    "e3": "workflow decyzyjny (MP3)",
    "e4": "agent z narzędziami (MP4)",
    "e5": "końcowy asystent przez MCP (MP5)",
}

# Cele z docs/06-evaluation.md („Metryki i cele”); reguły `met` są w adapterach niżej.
TARGETS: dict[str, dict[str, str]] = {
    "e1": {
        "hit_at_1": "≥ 0,8",
        "hit_at_3": "≥ 0,9",
        "m03": "trafione",
        "monthly_reference": "równy kursowi ($17,78 / $69,86)",
    },
    "e2": {
        "levels_1_2": "równe kursowi (check.py)",
        "levels_3_4": "≥ referencja − 1",
    },
    "e3": {
        "route": "odtworzenie 100%; na żywo ≥ 12/13",
        "default": "odtworzenie 100%",
        "errors": "0",
    },
    "e4": {
        "tools": "9/9",
        "args": "3/3",
        "facts": "9/9",
        "uncovered": "0",
        "approval": "0/9",
        "t1": "100% / 100%",
    },
    "e5": {
        "parity": "100%",
        "route": "odtworzenie 100%; na żywo ≥ 24/25",
        "default": "odtworzenie 100%; na żywo ≥ kurs − 1",
        "data": "≥ kurs − 1",
        "mcp": "0 nieudanych",
    },
}

# Liczby kursu spoza JSON-ów raportów (README kursu; te same stałe co generatory raportów E1, E2).
COURSE_E1_MONTHLY_USD = (17.78, 69.86)  # retrieval / cały korpus, model referencyjny kursu
COURSE_E2_LEVELS_1_2 = {  # check.py MP2: hit@1, hit@3, komplet dowodów, tokeny promptu
    "mp1": (9, 9, 9, 850.3),
    "ctx": (9, 9, 8, 610.4),
}
COURSE_E2_LEVELS_3_4 = {  # README MP2 (model referencyjny, chunki z kontekstem)
    "retrieval_hit": "9/10",
    "evidence_phrases_complete": "8/10",
    "answered": "8/10",
    "key_facts_complete": "8/8",
    "judge_correct": "8/8",
    "judge_grounded": "8/8",
    "citations_valid": "8/8",
    "cites_relevant": "8/8",
    "refusals_correct": "2/2",
}
E2_LEVEL_NAMES = {
    "retrieval_hit": "Poziom 1: znaleziono trafny dokument",
    "evidence_phrases_complete": "Poziom 2: komplet fraz dowodowych w kontekście",
    "answered": "Poziom 3: udzielone odpowiedzi",
    "key_facts_complete": "Poziom 3: wszystkie kluczowe fakty",
    "judge_correct": "Poziom 3: sędzia, odpowiedź poprawna",
    "judge_grounded": "Poziom 3: sędzia, odpowiedź oparta na kontekście",
    "citations_valid": "Poziom 4: odwołania prawidłowe",
    "cites_relevant": "Poziom 4: odwołania do trafnego dokumentu",
    "refusals_correct": "Odmowy trafne (pytania spoza korpusu)",
}
E3_NAMES = {
    "route": "Kierowanie (rodzaj wiadomości)",
    "fields": "Pola odczytane przez LLM",
    "request": "Stan wniosku po aktualizacji",
    "approvers": "Osoby zatwierdzające (polityka)",
    "pauses": "Pauzy (typ, rola, kolejność)",
    "status": "Status końcowy",
}
E5_NAMES = {
    "route": "Ścieżka (oczekiwany rodzaj wiadomości)",
    "status": "Status końcowy",
    "fields": "Pola wyodrębnione przez LLM",
    "pauses": "Pauzy (rodzaj, rola, kolejność)",
    "approvers": "Osoby zatwierdzające (polityka)",
    "request": "Stan wniosku",
    "agent_trace": "Trace agenta",
}
E5_DATA_NAMES = {
    "tool_coverage": "Pytania o dane: pokrycie narzędzi",
    "arguments_correct": "Pytania o dane: poprawne argumenty",
    "answer_facts": "Pytania o dane: fakty w odpowiedzi",
    "numbers_backed": "Pytania o dane: liczby z pokryciem źródłami",
    "no_approval_claim": "Pytania o dane: bez twierdzenia o akceptacji",
}


# ---------------------------------------------------------------------------
# Odczyt pól i zapis liczb
# ---------------------------------------------------------------------------


def _at(stage: str, obj: Any, *keys: str | int, base: str = "") -> Any:
    """Pole pod ścieżką kluczy; brak → `DashboardDataError(stage, "a.b[0].c")`."""
    path = base
    for key in keys:
        if isinstance(key, int):
            path = f"{path}[{key}]"
            ok = isinstance(obj, list) and -len(obj) <= key < len(obj)
        else:
            path = f"{path}.{key}" if path else key
            ok = isinstance(obj, dict) and key in obj
        if not ok:
            raise DashboardDataError(stage, path)
        obj = obj[key]
    return obj


def _pair(stage: str, obj: Any, *keys: str, base: str = "") -> tuple[int, int]:
    value = _at(stage, obj, *keys, base=base)
    if not (isinstance(value, list) and len(value) == 2 and all(isinstance(v, int) for v in value)):
        path = ".".join(filter(None, (base, *keys)))
        raise DashboardDataError(stage, path, "oczekiwano pary [trafione, razem] w polu {path}")
    return value[0], value[1]


def _frac(pair: tuple[int, int]) -> str:
    return f"{pair[0]}/{pair[1]}"


def _full(pair: tuple[int, int]) -> bool:
    return pair[0] == pair[1]


def _rate(value: float) -> str:
    """hit@k z jednym miejscem po przecinku, jak w raportach („1,0”, „0,9”)."""
    return f"{value:.1f}".replace(".", ",")


def _usd(value: float) -> str:
    return f"${value:.2f}".replace(".", ",")


def _pct(value: float) -> str:
    return f"{number(round(value * 100))}%"


def cents_text(value: float) -> str:
    return f"{number(round(value, 2))} centa"


def calls_text(n: int) -> str:
    if n == 1:
        return "1 wywołanie"
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return f"{number(n)} wywołania"
    return f"{number(n)} wywołań"


def _date(stage: str, data: dict[str, Any]) -> datetime.date:
    created = _at(stage, data, "created_at")
    try:
        return datetime.date.fromisoformat(str(created)[:10])
    except ValueError as exc:
        raise DashboardDataError(stage, "created_at", "niepoprawna data w polu {path}") from exc


def _cents(tokens_in: int, tokens_out: int, model: str | None) -> tuple[float | None, float]:
    """Koszt w centach po cenniku płatnego tieru Gemini (`cost.our_prices_per_1m_tokens`) i po
    cenach referencyjnych kursu (PINNED); tylko dla etapów bez kosztu w JSON-ie (E1, E2)."""
    cost = get_settings().cost
    reference = cost.prices_per_1m_tokens[cost.reference_model]
    ours = cost.our_prices_per_1m_tokens.get(model) if model else None

    def cents(price_in: float, price_out: float) -> float:
        return round((tokens_in * price_in + tokens_out * price_out) / 1_000_000 * 100, 4)

    return (
        cents(ours.input, ours.output) if ours is not None else None,
        cents(reference.input, reference.output),
    )


def _summary(
    spec: StageSpec,
    data: dict[str, Any],
    *,
    model: str | None,
    embedding_model: str | None,
    headline: str,
    metrics: list[MetricRow],
    calls: RunCalls | None,
    cost: StageCost | None,
) -> StageSummary:
    stage = spec.stage
    return StageSummary(
        stage=stage,
        title=TITLES[stage],
        date=_date(stage, data),
        model=model,
        embedding_model=embedding_model,
        report_md=spec.markdown,
        related_md=list(spec.related),
        headline=headline,
        metrics=metrics,
        calls=calls,
        cost=cost,
        notes=list(spec.notes),
    )


def calls_from_first_live(spec: StageSpec, planned: int | None) -> RunCalls | None:
    """Liczby oryginalnego przebiegu z `dashboard.yaml` (JSON raportu opisuje powtórkę z cache)."""
    first = spec.first_live
    if first is None:
        return None
    return RunCalls(
        planned=planned,
        network=first.network_calls,
        cached=first.cached_calls,
        failed=first.failed_attempts,
        seconds=first.seconds,
        source="dashboard.yaml",
        run_id=first.run_id,
        origin=first.source,
    )


def _with_first_live(spec: StageSpec, calls: RunCalls) -> RunCalls:
    return calls_from_first_live(spec, calls.planned) or calls


# ---------------------------------------------------------------------------
# E1: wyszukiwanie i koszty
# ---------------------------------------------------------------------------


def _e1_runs(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
    runs = _at("e1", data, "runs")
    if not isinstance(runs, list):
        raise DashboardDataError("e1", "runs", "oczekiwano listy w polu {path}")
    configured = str(_at("e1", data, "configured_model"))
    reference = next((r for r in runs if r.get("provider") == "openai_cache"), None)
    if reference is None:
        raise DashboardDataError("e1", "runs[provider=openai_cache]")
    ours = next((r for r in runs if r.get("model") == configured), None)
    if ours is None:
        raise DashboardDataError("e1", f"runs[model={configured}]")
    return reference, ours, configured


def monthly_cost_e1(data: dict[str, Any]) -> MonthlyCost:
    """Scenariusz miesięczny kursu z wiersza „referencja kursu” E1 (te same tokeny co kurs)."""
    reference, _, _ = _e1_runs(data)
    base = "runs[referencja kursu]"
    ours = _at("e1", reference, "cost_ours", base=base)
    return MonthlyCost(
        reference_rag_usd=_at("e1", reference, "cost_reference", "rag_monthly_usd", base=base),
        reference_full_usd=_at(
            "e1", reference, "cost_reference", "full_context_monthly_usd", base=base
        ),
        ours_rag_usd=ours.get("rag_monthly_usd") if isinstance(ours, dict) else None,
        ours_full_usd=ours.get("full_context_monthly_usd") if isinstance(ours, dict) else None,
        ours_model=ours.get("llm_model") if isinstance(ours, dict) else None,
    )


def _m03_hit(run: dict[str, Any], base: str) -> bool:
    results = _at("e1", run, "mp2", "results", base=base)
    for item in results:
        if isinstance(item, dict) and item.get("query_id") == "M03":
            return bool(_at("e1", item, "hit_at_3", base=f"{base}.mp2.results[M03]"))
    raise DashboardDataError("e1", f"{base}.mp2.results[M03]")


def summarise_e1(data: dict[str, Any], spec: StageSpec) -> StageSummary:
    reference, ours, configured = _e1_runs(data)
    ref_base, our_base = "runs[referencja kursu]", f"runs[{configured}]"
    targets = TARGETS["e1"]
    metrics: list[MetricRow] = []
    for set_name, label in (("mp1", "Q01–Q10"), ("mp2", "M01–M10")):
        for k in ("1", "3"):
            key = f"hit_rate_at_{k}"
            ref_value = float(_at("e1", reference, set_name, key, base=ref_base))
            our_value = float(_at("e1", ours, set_name, key, base=our_base))
            limit = 0.8 if k == "1" else 0.9
            metrics.append(
                MetricRow(
                    name=f"hit@{k} ({set_name.upper()}, {label})",
                    reference=_rate(ref_value),
                    replay=_rate(our_value),
                    live=None,
                    target=targets[f"hit_at_{k}"],
                    met=our_value >= limit,
                )
            )
    ref_m03, our_m03 = _m03_hit(reference, ref_base), _m03_hit(ours, our_base)
    metrics.append(
        MetricRow(
            name="M03 (SUP-114) w top-3",
            reference="trafione" if ref_m03 else "nietrafione",
            replay="trafione" if our_m03 else "nietrafione",
            live=None,
            target=targets["m03"],
            met=our_m03,
        )
    )
    monthly = monthly_cost_e1(data)
    reproduced = (monthly.reference_rag_usd, monthly.reference_full_usd)
    metrics.append(
        MetricRow(
            name="Koszt miesięczny: retrieval / cały korpus (ceny referencyjne kursu)",
            reference=f"{_usd(COURSE_E1_MONTHLY_USD[0])} / {_usd(COURSE_E1_MONTHLY_USD[1])}",
            replay=f"{_usd(reproduced[0])} / {_usd(reproduced[1])}",
            live=None,
            target=targets["monthly_reference"],
            met=reproduced == COURSE_E1_MONTHLY_USD,
        )
    )
    if monthly.ours_rag_usd is not None and monthly.ours_full_usd is not None:
        metrics.append(
            MetricRow(
                name=f"Koszt miesięczny po cenach {monthly.ours_model} (płatny tier Gemini)",
                reference=None,
                replay=f"{_usd(monthly.ours_rag_usd)} / {_usd(monthly.ours_full_usd)}",
                live=None,
                target=None,
                met=None,
            )
        )
    llm_call = _at("e1", data, "llm_call")
    calls: RunCalls | None = None
    cost: StageCost | None = None
    model: str | None = None
    if isinstance(llm_call, dict):
        model = str(_at("e1", llm_call, "answered_by", base="llm_call"))
        tokens_in = int(_at("e1", llm_call, "input_tokens", base="llm_call"))
        tokens_out = int(_at("e1", llm_call, "output_tokens", base="llm_call"))
        ours_cents, reference_cents = _cents(tokens_in, tokens_out, model)
        calls = RunCalls(
            planned=1, network=1, cached=0, failed=0,
            seconds=round(float(_at("e1", llm_call, "seconds", base="llm_call")), 1),
            source="report", run_id=str(_at("e1", data, "run_id")),
        )  # fmt: skip
        cost = StageCost(
            tokens_in=tokens_in, tokens_out=tokens_out, ours_cents=ours_cents,
            reference_cents=reference_cents, course_cents=None, includes_cached=False,
        )  # fmt: skip
        calls = _with_first_live(spec, calls)
    mp1 = [m for m in metrics if "MP1" in m.name]
    headline = (
        f"E1: hit@1 / hit@3 na Q01–Q10 {mp1[0].reference} / {mp1[1].reference} → "
        f"{mp1[0].replay} / {mp1[1].replay} ({configured}); "
        f"M03 {'trafione' if our_m03 else 'nietrafione'}; koszt kursu "
        f"{_usd(reproduced[0])} / {_usd(reproduced[1])} "
        f"{'odtworzony' if reproduced == COURSE_E1_MONTHLY_USD else 'inny niż w kursie'}"
    )
    return _summary(
        spec, data, model=model, embedding_model=configured, headline=headline,
        metrics=metrics, calls=calls, cost=cost,
    )  # fmt: skip


# ---------------------------------------------------------------------------
# E2: odpowiedzi ze źródłami
# ---------------------------------------------------------------------------


def _e2_retrieval(data: dict[str, Any], side: str, variant: str) -> dict[str, Any] | None:
    entries = _at("e2", data, "retrieval")
    if not isinstance(entries, list):
        raise DashboardDataError("e2", "retrieval", "oczekiwano listy w polu {path}")
    for entry in entries:
        if isinstance(entry, dict) and (entry.get("side"), entry.get("variant")) == (side, variant):
            return entry
    return None


def _e2_levels_1_2(entry: dict[str, Any], base: str) -> tuple[tuple[int, int, int, float], str]:
    total = len(_at("e2", entry, "rows", base=base))
    values = (
        int(_at("e2", entry, "hit_at_1", base=base)),
        int(_at("e2", entry, "hit_at_3", base=base)),
        int(_at("e2", entry, "evidence_phrases_complete", base=base)),
        float(_at("e2", entry, "avg_prompt_tokens", base=base)),
    )
    text = f"{values[0]}/{total}, {values[1]}/{total}, {values[2]}/{total}, {number(values[3])}"
    return values, text


def summarise_e2(data: dict[str, Any], spec: StageSpec) -> StageSummary:
    targets = TARGETS["e2"]
    metrics: list[MetricRow] = []
    _at("e2", data, "retrieval")
    for variant in ("mp1", "ctx", "hybrid"):
        reference = _e2_retrieval(data, "reference", variant)
        ours = _e2_retrieval(data, "ours", variant)
        if reference is None and ours is None:
            continue
        ref_values, ref_text = (
            _e2_levels_1_2(reference, f"retrieval[reference {variant}]")
            if reference is not None
            else (None, None)
        )
        our_text = (
            _e2_levels_1_2(ours, f"retrieval[ours {variant}]")[1] if ours is not None else None
        )
        course = COURSE_E2_LEVELS_1_2.get(variant)
        metrics.append(
            MetricRow(
                name=f"Poziomy 1–2, {variant}: hit@1, hit@3, komplet dowodów, tokeny promptu",
                # kolumna „Kurs” tylko dla wariantów, które kurs ma (hybrid jest nasz)
                reference=ref_text if course is not None else None,
                replay=our_text,
                live=None,
                target=targets["levels_1_2"] if course is not None else None,
                met=(ref_values == course) if course is not None else None,
            )
        )
    variants = _at("e2", data, "llm")
    if not isinstance(variants, list):
        raise DashboardDataError("e2", "llm", "oczekiwano listy w polu {path}")
    labels = [str(_at("e2", v, "variant", base=f"llm[{i}]")) for i, v in enumerate(variants)]
    label = " / ".join(labels)
    for key, name in E2_LEVEL_NAMES.items():
        if not variants:
            break
        pairs = [_pair("e2", v, "summary", key, base=f"llm[{i}]") for i, v in enumerate(variants)]
        met = all(
            bool(_at("e2", v, "meets_target", key, base=f"llm[{i}]"))
            for i, v in enumerate(variants)
        )
        metrics.append(
            MetricRow(
                name=f"{name} ({label})",
                reference=COURSE_E2_LEVELS_3_4[key],
                replay=None,
                live=" / ".join(_frac(p) for p in pairs),
                target=targets["levels_3_4"],
                met=met,
            )
        )
    model = str(_at("e2", data, "model"))
    cost_model = str(_at("e2", data, "our_cost_model"))
    network = failed = answers = tokens_in = tokens_out = 0
    for i, v in enumerate(variants):
        base = f"llm[{i}]"
        network += int(_at("e2", v, "calls", "answer", base=base))
        network += int(_at("e2", v, "calls", "judge", base=base))
        failed += int(_at("e2", v, "calls", "errors", base=base))
        answered = sum(int(n) for n in _at("e2", v, "answered_by", base=base).values())
        answers += answered
        tokens_in += answered * int(_at("e2", v, "avg_usage", "input_tokens", base=base))
        tokens_out += answered * int(_at("e2", v, "avg_usage", "output_tokens", base=base))
    calls = RunCalls(
        planned=int(_at("e2", data, "llm_calls_planned")),
        network=network,
        cached=int(_at("e2", data, "cache", "hits")),
        failed=failed,
        seconds=None,
        source="report",
        run_id=str(_at("e2", data, "run_id")),
    )
    cost: StageCost | None = None
    if answers:
        ours_cents, reference_cents = _cents(tokens_in, tokens_out, cost_model)
        cost = StageCost(
            tokens_in=tokens_in, tokens_out=tokens_out, ours_cents=ours_cents,
            reference_cents=reference_cents, course_cents=None, includes_cached=False,
        )  # fmt: skip
    by_key = {m.name.split(" (")[0]: m for m in metrics}
    answered_row = by_key.get(E2_LEVEL_NAMES["answered"])
    judge_row = by_key.get(E2_LEVEL_NAMES["judge_correct"])
    levels_1_2 = [m for m in metrics if m.name.startswith("Poziomy 1–2") and m.met is not None]
    parts = []
    if answered_row is not None and judge_row is not None:
        parts.append(
            f"odpowiedzi {answered_row.reference} → {answered_row.live}, sędzia "
            f"{judge_row.reference} → {judge_row.live} ({label})"
        )
    if levels_1_2:
        same = all(m.met for m in levels_1_2)
        parts.append("poziomy 1–2 jak kurs" if same else "poziomy 1–2 inne niż kurs")
    headline = "E2: " + "; ".join(parts or ["bez przebiegu z LLM-em"])
    return _summary(
        spec, data, model=model, embedding_model=str(_at("e2", data, "embedding_model")),
        headline=headline, metrics=metrics, calls=_with_first_live(spec, calls), cost=cost,
    )  # fmt: skip


# ---------------------------------------------------------------------------
# E3: workflow decyzyjny
# ---------------------------------------------------------------------------


def summarise_e3(data: dict[str, Any], spec: StageSpec) -> StageSummary:
    targets = TARGETS["e3"]
    reference = _at("e3", data, "reference")
    replay = _at("e3", data, "replay", "score")
    live_run = _at("e3", data, "live")
    live = _at("e3", live_run, "score", base="live") if live_run is not None else None
    metrics: list[MetricRow] = []
    for key, name in E3_NAMES.items():
        ref_pair = _pair("e3", reference, key, base="reference")
        replay_pair = _pair("e3", replay, key, base="replay.score")
        live_pair = _pair("e3", live, key, base="live.score") if live is not None else None
        met = _full(replay_pair)
        if key == "route":
            met = met and live_pair is not None and live_pair[0] >= live_pair[1] - 1
        metrics.append(
            MetricRow(
                name=name,
                reference=_frac(ref_pair),
                replay=_frac(replay_pair),
                live=_frac(live_pair) if live_pair is not None else None,
                target=targets.get(key, targets["default"]),
                met=met,
            )
        )
    replay_errors = int(_at("e3", replay, "errors", base="replay.score"))
    live_errors = int(_at("e3", live, "errors", base="live.score")) if live is not None else None
    metrics.append(
        MetricRow(
            name="Błędy (nieudane tury)",
            reference=str(_at("e3", reference, "errors", base="reference")),
            replay=str(replay_errors),
            live=str(live_errors) if live_errors is not None else None,
            target=targets["errors"],
            met=replay_errors == 0,
        )
    )
    calls: RunCalls | None = None
    cost: StageCost | None = None
    model: str | None = None
    if live_run is not None:
        model = _at("e3", live_run, "model", base="live")
        live_calls = _at("e3", live_run, "calls", base="live")
        calls = RunCalls(
            planned=int(_at("e3", data, "llm_calls_planned", "total")),
            network=int(_at("e3", live_calls, "understand", base="live.calls"))
            + int(_at("e3", live_calls, "answer", base="live.calls")),
            cached=int(_at("e3", live_calls, "from_cache", base="live.calls")),
            failed=int(_at("e3", live_calls, "errors", base="live.calls")),
            seconds=round(float(_at("e3", live_run, "seconds", base="live")), 1),
            source="report",
            run_id=str(_at("e3", live_run, "run_id", base="live")),
        )
        calls = _with_first_live(spec, calls)
        usage = _at("e3", live_run, "usage", base="live")
        cost = StageCost(
            tokens_in=int(_at("e3", usage, "input_tokens", base="live.usage")),
            tokens_out=int(_at("e3", usage, "output_tokens", base="live.usage")),
            ours_cents=float(_at("e3", data, "live_cost_cents", "ours")),
            reference_cents=float(_at("e3", data, "live_cost_cents", "reference")),
            course_cents=float(_at("e3", reference, "cost_cents", base="reference")),
            includes_cached=False,
        )
    route, fields = metrics[0], metrics[1]
    replay_full = replay_errors == 0 and all(
        _full(_pair("e3", replay, key, base="replay.score")) for key in E3_NAMES
    )
    headline = (
        f"E3: na żywo kierowanie {route.reference} → {route.live}, pola {fields.reference} → "
        f"{fields.live}; odtworzenie {'100%' if replay_full else 'poniżej 100%'}"
    )
    if cost is not None and cost.ours_cents is not None:
        headline += f"; {cents_text(cost.ours_cents)}"
    return _summary(
        spec, data, model=model, embedding_model=_at("e3", data, "embedding_model"),
        headline=headline, metrics=metrics, calls=calls, cost=cost,
    )  # fmt: skip


# ---------------------------------------------------------------------------
# E4: agent z narzędziami
# ---------------------------------------------------------------------------


def _e4_tests(stage_data: Any, base: str) -> dict[str, str]:
    """Pięć testów kursu w zapisie raportu: narzędzia, argumenty, fakty, bez źródła, twierdzenia."""
    return {
        "tools": _frac(_pair("e4", stage_data, "tools", base=base)),
        "args": _frac(_pair("e4", stage_data, "args", base=base)),
        "facts": _frac(_pair("e4", stage_data, "facts", base=base)),
        "uncovered": f"{_at('e4', stage_data, 'uncovered', base=base)} "
        f"({_at('e4', stage_data, 'user_stated', base=base)})",
        "approval": _frac(_pair("e4", stage_data, "approval", base=base)),
    }


def _e4_question(run: Any, question_id: str, base: str) -> dict[str, Any] | None:
    for i, row in enumerate(_at("e4", run, "rows", base=base)):
        trace = row.get("trace") if isinstance(row, dict) else None
        if isinstance(trace, dict) and trace.get("question_id") == question_id:
            score = _at("e4", row, "score", base=f"{base}.rows[{i}]")
            return score if isinstance(score, dict) else None
    return None


def _t1_text(score: dict[str, Any] | None) -> str | None:
    if score is None:
        return None
    tools, facts = score.get("tool_coverage"), score.get("answer_fact_coverage")
    if tools is None or facts is None:
        return None
    return f"{_pct(tools)} / {_pct(facts)}"


def summarise_e4(data: dict[str, Any], spec: StageSpec) -> StageSummary:
    targets = TARGETS["e4"]
    single = _at("e4", data, "reference", "single")
    supervisor = _at("e4", data, "reference", "supervisor")
    replay_run = _at("e4", data, "replay")
    live_run = _at("e4", data, "live")
    reference = _e4_tests(single, "reference.single")
    replay = (
        _e4_tests(_at("e4", replay_run, "summary", base="replay"), "replay.summary")
        if replay_run is not None
        else None
    )
    live_summary = _at("e4", live_run, "summary", base="live") if live_run is not None else None
    live = _e4_tests(live_summary, "live.summary") if live_summary is not None else None
    names = {
        "tools": "Test 1: pokrycie narzędzi (każde potrzebne wywołanie)",
        "args": "Test 2: poprawne argumenty (gdzie liczono karę)",
        "facts": "Test 3: fakty (każdy kluczowy fakt w odpowiedzi)",
        "uncovered": "Test 4: liczby bez źródła (podane przez użytkownika)",
        "approval": "Test 5: twierdzenia o zatwierdzeniu w odpowiedziach",
    }
    metrics: list[MetricRow] = []
    for key, name in names.items():
        live_value = live[key] if live is not None else None
        met: bool | None = None
        if live_summary is not None:
            if key == "uncovered":
                met = int(_at("e4", live_summary, "uncovered", base="live.summary")) == 0
            elif key == "approval":
                met = _pair("e4", live_summary, "approval", base="live.summary")[0] == 0
            else:
                met = _full(_pair("e4", live_summary, key, base="live.summary"))
        metrics.append(
            MetricRow(
                name=name,
                reference=reference[key],
                replay=replay[key] if replay is not None else None,
                live=live_value,
                target=targets[key],
                met=met,
            )
        )
    team = _e4_tests(supervisor, "reference.supervisor")
    metrics.append(
        MetricRow(
            name="Zespół kursu (supervisor + 3 specjalistów): testy 1–5",
            reference=", ".join(team.values()),
            replay=None,
            live=None,
            target=None,
            met=None,
        )
    )
    course_run = _at("e4", data, "course")
    t1_live = _e4_question(live_run, "T1", "live") if live_run is not None else None
    t1_reference = _e4_question(course_run, "T1", "course") if course_run is not None else None
    t1_replay = _e4_question(replay_run, "T1", "replay") if replay_run is not None else None
    if t1_reference is not None or t1_live is not None:
        live_ok = (
            t1_live.get("tool_coverage") == 1 and t1_live.get("answer_fact_coverage") == 1
            if t1_live is not None
            else None
        )
        metrics.append(
            MetricRow(
                name="T1: warunek rozwiązania umowy (narzędzia / fakty)",
                reference=_t1_text(t1_reference),
                replay=_t1_text(t1_replay),
                live=_t1_text(t1_live),
                target=targets["t1"],
                met=live_ok,
            )
        )

    def tool_calls(run: Any, base: str) -> str:
        return f"{_at('e4', run, 'tool_calls', base=base)} ({_at('e4', run, 'failed', base=base)})"

    metrics.append(
        MetricRow(
            name="Wywołania narzędzi (nieudane)",
            reference=tool_calls(single, "reference.single"),
            replay=tool_calls(_at("e4", replay_run, "summary", base="replay"), "replay.summary")
            if replay_run is not None
            else None,
            live=tool_calls(live_summary, "live.summary") if live_summary is not None else None,
            target=None,
            met=None,
        )
    )
    calls: RunCalls | None = None
    cost: StageCost | None = None
    model: str | None = None
    below: list[str] = []
    if live_run is not None and live_summary is not None:
        model = _at("e4", live_run, "run", "model", base="live")
        live_calls = _at("e4", live_run, "calls", base="live")
        calls = RunCalls(
            planned=int(_at("e4", data, "llm_calls_planned", "estimate")),
            network=int(_at("e4", live_calls, "network", base="live.calls")),
            cached=int(_at("e4", live_calls, "from_cache", base="live.calls")),
            failed=int(_at("e4", live_calls, "errors", base="live.calls")),
            seconds=round(float(_at("e4", live_run, "seconds", base="live")), 1),
            source="report",
            run_id=str(_at("e4", live_run, "run", "run_id", base="live")),
        )
        calls = _with_first_live(spec, calls)
        cost = StageCost(
            tokens_in=int(_at("e4", live_summary, "tokens_in", base="live.summary")),
            tokens_out=int(_at("e4", live_summary, "tokens_out", base="live.summary")),
            ours_cents=float(_at("e4", data, "live_cost_cents", "ours")),
            reference_cents=float(_at("e4", data, "live_cost_cents", "reference")),
            course_cents=float(_at("e4", data, "reference", "cost_cents", "single")),
            includes_cached=False,
        )
        for i, row in enumerate(_at("e4", live_run, "rows", base="live")):
            score = _at("e4", row, "score", base=f"live.rows[{i}]")
            trace = row.get("trace") or {}
            if score.get("tool_coverage", 1) < 1 or (score.get("answer_fact_coverage") or 1) < 1:
                below.append(str(trace.get("question_id", f"#{i + 1}")))
    tools_row, facts_row = metrics[0], metrics[2]
    headline = (
        f"E4: na żywo narzędzia {tools_row.reference} → {tools_row.live}, fakty "
        f"{facts_row.reference} → {facts_row.live}"
    )
    if tools_row.met is False or facts_row.met is False:
        headline += f" (cel 9/9 niespełniony: {', '.join(below) or 'brak danych'})"
    t1_row = next((m for m in metrics if m.name.startswith("T1")), None)
    if t1_row is not None and t1_row.live is not None:
        headline += f"; T1 {t1_row.reference} → {t1_row.live}"
    if cost is not None and cost.ours_cents is not None:
        headline += f"; {cents_text(cost.ours_cents)}"
    return _summary(
        spec, data, model=model, embedding_model=_at("e4", data, "embedding_model"),
        headline=headline, metrics=metrics, calls=calls, cost=cost,
    )  # fmt: skip


# ---------------------------------------------------------------------------
# E5: końcowy asystent przez MCP
# ---------------------------------------------------------------------------


def summarise_e5(data: dict[str, Any], spec: StageSpec) -> StageSummary:
    targets = TARGETS["e5"]
    reference = _at("e5", data, "reference")
    parity = _at("e5", data, "parity")
    replay_run = _at("e5", data, "replay")
    live_run = _at("e5", data, "live")
    replay = _at("e5", replay_run, "totals", base="replay") if replay_run is not None else None
    live = _at("e5", live_run, "totals", base="live") if live_run is not None else None
    metrics: list[MetricRow] = []
    if isinstance(parity, list):
        same = sum(1 for item in parity if isinstance(item, dict) and item.get("same") is True)
        metrics.append(
            MetricRow(
                name="Zgodność MCP z wywołaniem bezpośrednim",
                reference=None,
                replay=f"{same}/{len(parity)}",
                live=None,
                target=targets["parity"],
                met=same == len(parity),
            )
        )
    for key, name in E5_NAMES.items():
        ref_pair = _pair("e5", reference, key, base="reference")
        replay_pair = _pair("e5", replay, key, base="replay.totals") if replay is not None else None
        live_pair = _pair("e5", live, key, base="live.totals") if live is not None else None
        met: bool | None = None
        if replay_pair is not None and live_pair is not None:
            slack_ok = live_pair[0] >= ref_pair[0] - 1
            if key == "route":
                slack_ok = live_pair[0] >= live_pair[1] - 1
            met = _full(replay_pair) and slack_ok
        metrics.append(
            MetricRow(
                name=name,
                reference=_frac(ref_pair),
                replay=_frac(replay_pair) if replay_pair is not None else None,
                live=_frac(live_pair) if live_pair is not None else None,
                target=targets["route"] if key == "route" else targets["default"],
                met=met,
            )
        )
    for key, name in E5_DATA_NAMES.items():
        ref_pair = _pair("e5", reference, key, base="reference")
        replay_pair = _pair("e5", replay, key, base="replay.totals") if replay is not None else None
        live_pair = _pair("e5", live, key, base="live.totals") if live is not None else None
        metrics.append(
            MetricRow(
                name=name,
                reference=_frac(ref_pair),
                replay=_frac(replay_pair) if replay_pair and replay_pair[1] else None,
                live=_frac(live_pair) if live_pair is not None else None,
                target=targets["data"],
                met=live_pair[0] >= ref_pair[0] - 1 if live_pair is not None else None,
            )
        )
    ref_mcp = f"{_at('e5', reference, 'mcp_calls', base='reference')} "
    ref_mcp += f"({_at('e5', reference, 'mcp_failures', base='reference')})"
    live_mcp: str | None = None
    mcp_met: bool | None = None
    if live_run is not None:
        failures = _at("e5", live_run, "calls", "mcp_failures", base="live")
        failed = sum(int(v) for v in failures.values()) if isinstance(failures, dict) else 0
        live_mcp = f"{_at('e5', live_run, 'calls', 'mcp_calls', base='live')} ({failed})"
        mcp_met = failed == 0
    metrics.append(
        MetricRow(
            name="Wywołania MCP w turach (nieudane)",
            reference=ref_mcp,
            replay=None,
            live=live_mcp,
            target=targets["mcp"],
            met=mcp_met,
        )
    )
    calls: RunCalls | None = None
    cost: StageCost | None = None
    model: str | None = None
    if live_run is not None:
        model = _at("e5", live_run, "run", "model", base="live")
        c = _at("e5", live_run, "calls", base="live")
        calls = RunCalls(
            planned=int(_at("e5", data, "llm_calls_planned", "estimate")),
            network=int(_at("e5", c, "understand_network", base="live.calls"))
            + int(_at("e5", c, "agent_network", base="live.calls")),
            cached=int(_at("e5", c, "understand_cache", base="live.calls"))
            + int(_at("e5", c, "agent_cache", base="live.calls")),
            failed=int(_at("e5", c, "errors", base="live.calls")),
            seconds=round(float(_at("e5", live_run, "seconds", base="live")), 1),
            source="report",
            run_id=str(_at("e5", live_run, "run", "run_id", base="live")),
        )
        calls = _with_first_live(spec, calls)
        cost = StageCost(
            tokens_in=int(_at("e5", live_run, "tokens_in", base="live")),
            tokens_out=int(_at("e5", live_run, "tokens_out", base="live")),
            ours_cents=float(_at("e5", data, "live_cost_cents", "ours")),
            reference_cents=float(_at("e5", data, "live_cost_cents", "reference")),
            course_cents=float(_at("e5", reference, "cost_cents", base="reference")),
            includes_cached=True,  # R1-3 z F5: tokeny i koszt liczą też tury z cache
        )
    route = next(m for m in metrics if m.name == E5_NAMES["route"])
    compared = [m for m in metrics if m.reference is not None and m.live is not None]
    as_course = all(m.reference == m.live for m in compared)
    headline = f"E5: na żywo {route.live} ścieżki przez MCP, " + (
        "jak kurs" if as_course else f"kurs {route.reference}"
    )
    if calls is not None and calls.network is not None:
        headline += f"; {calls_text(calls.network)}"
        if cost is not None and cost.ours_cents is not None:
            headline += f", {cents_text(cost.ours_cents)}"
    return _summary(
        spec, data, model=model, embedding_model=_at("e5", data, "embedding_model"),
        headline=headline, metrics=metrics, calls=calls, cost=cost,
    )  # fmt: skip


Adapter = Callable[[dict[str, Any], StageSpec], StageSummary]

ADAPTERS: dict[str, Adapter] = {
    "e1": summarise_e1,
    "e2": summarise_e2,
    "e3": summarise_e3,
    "e4": summarise_e4,
    "e5": summarise_e5,
}


def met_overall(metrics: Sequence[MetricRow]) -> str:
    """„tak” / „nie” / „częściowo” / „–” dla kolumny „Cel spełniony” w podsumowaniu."""
    flags = [m.met for m in metrics if m.met is not None]
    if not flags:
        return "–"
    if all(flags):
        return "tak"
    if not any(flags):
        return "nie"
    return "częściowo"
