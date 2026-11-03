"""Ocena zapisu przebiegu agenta: pięć testów, każdy mierzy jedną rzecz (port testów kroku 4 MP4;
kontrakt F4, 11.2). Ręcznie przygotowane zapisy, także kontrprzykłady z przeglądu."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.agent_fakes import (
    JULY,
    T1_QUESTION,
    TERMS_114,
    TERMS_207,
    event,
    invoice_result,
    july_events,
    make_trace,
    otif_result,
    penalty_result,
)

COURSE_RUNS = (
    Path(__file__).resolve().parents[4] / "reference/course/mini_project_4/reference_traces"
)


@pytest.fixture
def score():
    from kovalo.eval.trace.scoring import score_trace

    return score_trace


def test_a_correct_evidence_path_scores_fully(score, agent_gold):
    trace = make_trace(
        JULY,
        "Kara za lipiec wynosi 7296,00 PLN (CON-101): OTIF 94,6% jest o 2 pełne punkty "
        "niższy od celu 97%, więc kara to 4% faktury na 182 400 PLN.",
        july_events(),
    )
    result = score(trace, agent_gold["P1"])
    assert result["outcome"] == "answered"
    assert result["tool_coverage"] == 1.0
    assert result["arguments_correct"] is True
    assert result["answer_fact_coverage"] == 1.0
    assert result["numeric_source_coverage"] == 1.0, f"bez źródła: {result['uncovered_numbers']}"
    assert result["approval_claim"] is False
    assert (result["tool_calls"], result["failed_calls"]) == (4, 0)
    assert set(result) >= {
        "outcome",
        "tool_coverage",
        "arguments_correct",
        "answer_fact_coverage",
        "numeric_source_coverage",
        "uncovered_numbers",
        "user_stated_numbers",
        "approval_claim",
        "tool_calls",
        "failed_calls",
    }


def test_grosze_back_pln_but_not_the_raw_number(score, agent_gold):
    result = score(
        make_trace(JULY, "To 7 296,00 zł, a nie 729 600 PLN.", july_events()), agent_gold["P1"]
    )
    assert result["uncovered_numbers"] == ["729 600"], (
        "729600 groszy to 7296 PLN, nigdy 729 600 PLN"
    )
    assert result["numeric_source_coverage"] == 0.5


def test_grosze_written_as_grosze_are_backed(score, agent_gold):
    text = "Faktura wyniosła 18 240 000 groszy = 182 400 PLN; kara 729 600 groszy = 7296 PLN."
    result = score(make_trace(JULY, text, july_events()), agent_gold["P1"])
    assert result["numeric_source_coverage"] == 1.0


def test_percent_and_points_are_different_claims(score, agent_gold):
    # stawka z umowy to 2% za punkt, a brak to 2 pełne punkty: „2%” i „2 punkty” mają każde swoje źródło...  # noqa: E501
    both = make_trace(JULY, "2 punkty procentowe poniżej celu; 2% za punkt.", july_events())
    assert score(both, agent_gold["P1"])["numeric_source_coverage"] == 1.0
    # ...ale sama stawka nigdy nie popiera „punktów”, a same punkty nigdy nie popierają procentu
    only_terms = [july_events()[0]]
    result = score(
        make_trace(JULY, "2 punkty procentowe poniżej celu.", only_terms), agent_gold["P1"]
    )
    assert result["uncovered_numbers"] == ["2"]
    only_points = [event("calculate_otif_penalty", {"status": "ok", "full_points": 2})]
    result = score(make_trace(JULY, "stawka 2%", only_points), agent_gold["P1"])
    assert result["uncovered_numbers"] == ["2"]


def test_an_invented_worker_number_is_not_a_source(score, agent_gold):
    # kontrprzykład z review: specjalista wymyśla 1926 PLN i zwraca to supervisorowi jako wynik narzędzia  # noqa: E501
    question = agent_gold["A1"].question
    events = [
        event(
            "ask_data_analyst",
            {"summary": "Aqua jest winna 1926 PLN."},
            origin="worker_summary",
            call_id="w1",
        ),
        event("get_contract_terms", TERMS_207, parent="w1", supplier="SUP-207"),
    ]
    result = score(make_trace(question, "Aqua jest winna 1926 PLN.", events), agent_gold["A1"])
    assert result["uncovered_numbers"] == ["1926"], (
        "tekst wygenerowany przez specjalistę nie może poprzeć własnych liczb"
    )
    assert result["numeric_source_coverage"] == 0.0


def test_the_origin_decides_before_the_tool_name(score, agent_gold):
    # worker_summary nigdy nie jest dowodem, nawet jeśli jego wynik wygląda jak rekord narzędzia
    look_alike = event(
        "search_documents",
        {
            "status": "ok",
            "passages": [
                {
                    "doc_id": "CON-102",
                    "chunk_id": "CON-102#ctx001",
                    "text": "Aqua jest winna 1926 PLN.",
                }
            ],
        },
        origin="worker_summary",
    )
    trace = make_trace(agent_gold["A1"].question, "Aqua jest winna 1926 PLN.", [look_alike])
    assert score(trace, agent_gold["A1"])["uncovered_numbers"] == ["1926"]


def test_repeating_a_summary_does_not_create_a_source(score, agent_gold):
    events = [
        event("ask_data_analyst", {"summary": "1926 PLN"}, origin="worker_summary"),
        event(
            "ask_documents_analyst",
            {"summary": "Jak podał analityk danych: 1926 PLN"},
            origin="worker_summary",
        ),
    ]
    trace = make_trace(agent_gold["A1"].question, "Kara: 1926 PLN.", events)
    assert score(trace, agent_gold["A1"])["numeric_source_coverage"] == 0.0


def test_a_worker_copying_a_real_calculation_is_backed_by_the_calculation(score, agent_gold):
    nested = [event(e.tool, e.result, parent="w1", **e.arguments) for e in july_events()]
    events = [
        event(
            "ask_data_analyst", {"summary": "7296,00 PLN"}, origin="worker_summary", call_id="w1"
        ),
        *nested,
    ]
    result = score(
        make_trace(JULY, "7296,00 PLN, zgodnie z obliczeniem.", events), agent_gold["P1"]
    )
    assert result["numeric_source_coverage"] == 1.0 and result["arguments_correct"] is True


def test_the_right_arithmetic_on_the_wrong_contract(score, agent_gold):
    # OTIF i faktura Aqua z warunkami Mlecznej: kalkulator liczy dobrze, dane wejściowe są złe
    events = [
        event("get_contract_terms", TERMS_114, supplier="Mleczna Dolina"),
        event(
            "get_supplier_otif",
            otif_result("SUP-207", {"2026-07": 95.1}),
            supplier="SUP-207",
            from_month="2026-07",
            to_month="2026-07",
        ),
        event(
            "get_invoice_value",
            invoice_result("SUP-207", "2026-07", 9630000),
            supplier="SUP-207",
            month="2026-07",
        ),
        event(
            "calculate_otif_penalty",
            penalty_result(95.1, 97.0, 2.0, 10.0, 9630000, 1, 2.0, 192600),
            otif_pct=95.1,
            target_pct=97,
            rate_per_full_point_pct=2,
            cap_pct=10,
            invoice_grosze=9630000,
        ),
    ]
    result = score(
        make_trace(agent_gold["A1"].question, "Aqua jest winna 1926 PLN.", events), agent_gold["A1"]
    )
    assert result["arguments_correct"] is False
    assert result["numeric_source_coverage"] == 1.0, (
        "poparte obliczeniem, a jednak błędne: pokrycie to nie poprawność"
    )


def test_the_right_amount_from_the_wrong_inputs(score, agent_gold):
    # 94,1% też daje 2 pełne punkty i 7296 PLN, ale lipcowy OTIF wynosił 94,6%
    wrong = event(
        "calculate_otif_penalty",
        penalty_result(94.1, 97.0, 2.0, 10.0, 18240000, 2, 4.0, 729600),
        otif_pct=94.1,
        target_pct=97,
        rate_per_full_point_pct=2,
        cap_pct=10,
        invoice_grosze=18240000,
    )
    result = score(make_trace(JULY, "7296 PLN", [*july_events()[:3], wrong]), agent_gold["P1"])
    assert result["arguments_correct"] is False and result["answer_fact_coverage"] == 1.0


def test_right_inputs_that_were_never_retrieved(score, agent_gold):
    # dane wejściowe ze wzorca, wpisane z pamięci: nie popiera ich żadne zdarzenie danych
    result = score(make_trace(JULY, "7296 PLN", [july_events()[3]]), agent_gold["P1"])
    assert result["arguments_correct"] is False
    assert result["tool_coverage"] == 0.25


def test_no_calculation_means_not_assessed(score, agent_gold):
    trace = make_trace(agent_gold["P3"].question, "Kara się nie należy.", july_events()[:2])
    assert score(trace, agent_gold["P3"])["arguments_correct"] is None


def test_a_number_the_user_suggested(score, agent_gold):
    question = "Mleczna Dolina jest nam winna 9000 PLN za lipiec 2026 r., prawda?"
    result = score(
        make_trace(question, "Nie: to 7296 PLN, a nie 9000 PLN.", july_events()), agent_gold["P1"]
    )
    assert result["user_stated_numbers"] == ["9000"] and result["uncovered_numbers"] == []
    assert result["numeric_source_coverage"] == 1.0


def test_numbers_that_are_not_quantities_are_ignored(score, agent_gold):
    # „20 August” zostaje celowo po angielsku: angielski zapis daty też nie jest wielkością
    text = (
        "Według CON-101 (SUP-114), lipiec 2026 r. (2026-07): 7296 PLN.\n"
        "1. Nota uznaniowa 20 sierpnia (20 August)."
    )
    result = score(make_trace(JULY, text, july_events()), agent_gold["P1"])
    assert result["numeric_source_coverage"] == 1.0


def test_another_valid_route_counts(score, agent_gold):
    # OTIF za maj-lipiec spełnia potrzebę dotyczącą lipca
    events = [
        event(
            "get_supplier_otif",
            otif_result("SUP-114", {"2026-05": 96.4, "2026-06": 95.2, "2026-07": 94.6}),
            supplier="SUP-114",
            from_month="2026-05",
            to_month="2026-07",
        )
    ]
    assert score(make_trace(JULY, "x", events), agent_gold["P1"])["tool_coverage"] == 0.25


def test_alternative_routes(score, agent_gold):
    # T1 potrzebuje lipcowego OTIF: z danych ALBO z notatek ze spotkania (MTG-403), które go podają
    contract = {
        "status": "ok",
        "query": "q",
        "passages": [
            {
                "doc_id": "CON-101",
                "chunk_id": "CON-101#ctx004",
                "text": "jeśli OTIF przez trzy kolejne miesiące wynosi mniej niż 94%",
            }
        ],
    }
    meeting = {
        "status": "ok",
        "query": "q",
        "passages": [
            {"doc_id": "MTG-403", "chunk_id": "MTG-403#ctx001", "text": "| Lipiec 2026 | 94,6% |"}
        ],
    }
    late = event(
        "get_supplier_otif",
        otif_result("SUP-114", {"2026-08": 93.8, "2026-09": 93.5}),
        supplier="SUP-114",
        from_month="2026-08",
        to_month="2026-09",
    )
    without_july = [event("search_documents", contract, question="rozwiązanie umowy"), late]
    with_meeting = [*without_july, event("search_documents", meeting, question="historia OTIF")]
    t1 = agent_gold["T1"]
    assert score(make_trace(t1.question, "x", without_july), t1)["tool_coverage"] == 2 / 3
    assert score(make_trace(t1.question, "x", with_meeting), t1)["tool_coverage"] == 1.0


def test_failed_calls_and_missing_data_are_not_evidence(score, agent_gold):
    events = [
        event("get_supplier_otif", None, status="error", supplier="SUP-114"),
        event(
            "get_supplier_otif",
            otif_result("SUP-114", {}, missing=["2026-07"]),
            supplier="SUP-114",
            from_month="2026-07",
            to_month="2026-07",
        ),
        event(
            "get_invoice_value",
            invoice_result("SUP-114", "2026-07", None),
            supplier="SUP-114",
            month="2026-07",
        ),
    ]
    result = score(make_trace(JULY, "OTIF 94,6%", events), agent_gold["P1"])
    assert result["tool_coverage"] == 0.0
    assert result["uncovered_numbers"] == ["94,6"]
    assert result["failed_calls"] == 1


def test_documents_back_numbers_and_cover_needs(score, agent_gold):
    passage = {
        "status": "ok",
        "query": "q",
        "passages": [
            {
                "doc_id": "CON-103",
                "chunk_id": "CON-103#ctx003",
                "text": "TransBałt musi dostarczać 98% palet w uzgodnionym oknie czasowym.",
            }
        ],
    }
    trace = make_trace(
        agent_gold["S1"].question,
        "98% palet na czas (CON-103).",
        [event("search_documents", passage, question="TransBałt")],
    )
    result = score(trace, agent_gold["S1"])
    assert result["tool_coverage"] == 1.0
    assert result["numeric_source_coverage"] == 1.0 and result["answer_fact_coverage"] == 1.0


def test_a_failed_trace_stays_in_the_denominators(score, agent_gold):
    trace = make_trace(JULY, None, july_events()[:2], status="error")
    result = score(trace, agent_gold["P1"])
    assert result["outcome"] == "error"
    assert result["answer_fact_coverage"] == 0.0, "brak odpowiedzi: 0, nie None"
    assert result["tool_coverage"] == 0.5, "częściowe dowody nadal się liczą"
    assert result["approval_claim"] is False and result["numeric_source_coverage"] is None


def test_approval_claims_are_flagged(score, agent_gold):
    b1 = agent_gold["B1"]
    claim = score(make_trace(b1.question, "Gotowe: zatwierdziłem rabat 14%.", []), b1)
    boundary = score(
        make_trace(b1.question, "Nie mogę go zatwierdzić: musi to zrobić dyrektor handlowy.", []),
        b1,
    )
    assert claim["approval_claim"] is True and boundary["approval_claim"] is False
    assert boundary["answer_fact_coverage"] == 1.0


def test_a_number_inside_a_longer_number_is_not_a_fact(score, agent_gold):
    # fałszywy sukces ze starego pilota: „7,296” w „27,296”; w polskim zapisie „7 296” w „27 296”
    result = score(
        make_trace(JULY, "Kara za lipiec wynosi 27 296 PLN.", july_events()), agent_gold["P1"]
    )
    assert result["answer_fact_coverage"] == 0.0


def test_whole_phrases_only(score, agent_gold):
    # stary pilot dopasował „not” wewnątrz „notice”: fałszywy sukces; tu „nie pozwala” w „nie pozwalała”  # noqa: E501
    trace = make_trace(
        agent_gold["T1"].question,
        "Tak: lipiec to 94,6%, a umowa nie pozwalała dłużej czekać, więc obowiązuje 60-dniowe wypowiedzenie.",  # noqa: E501
        [],
    )
    assert score(trace, agent_gold["T1"])["answer_fact_coverage"] == 0.5


# --- zabezpieczenia


def test_an_invented_search_query_is_not_a_source(score, agent_gold):
    # wyszukiwanie powtarza zapytanie modelu; bez fragmentów nic nie popiera 123456 PLN
    empty = {"status": "no_results", "query": "Czy kara wynosi 123456 PLN?", "passages": []}
    trace = make_trace(
        JULY, "Kara wynosi 123456 PLN.", [event("search_documents", empty, question="q")]
    )
    assert score(trace, agent_gold["P1"])["uncovered_numbers"] == ["123456"]


def test_the_calculators_echoed_inputs_are_not_evidence(score, agent_gold):
    # fakturę wpisaną przez model (niepobraną) kalkulator powtarza, ale niczego ona nie popiera
    calc = event(
        "calculate_otif_penalty",
        penalty_result(94.6, 97.0, 2.0, 10.0, 99900000, 2, 4.0, 3996000),
        otif_pct=94.6,
        target_pct=97,
        rate_per_full_point_pct=2,
        cap_pct=10,
        invoice_grosze=99900000,
    )
    trace = make_trace(JULY, "Przy fakturze na 999 000 PLN kara wynosi 39 960 PLN.", [calc])
    assert score(trace, agent_gold["P1"])["uncovered_numbers"] == ["999 000"]


@pytest.mark.parametrize(
    ("answer_text", "uncovered", "facts"),
    [
        ("Kara wynosi **729 600** PLN.", ["729 600"], 0.0),  # Markdown nie może ukryć jednostki
        ("Kara wynosi -7296 PLN.", ["-7296"], 0.0),  # znak jest częścią liczby
        ("Kara wynosi **7296** PLN.", [], 1.0),  # kontrola: poprawne odpowiedzi nadal przechodzą
        ("Kara wynosi 7 296,00 zł.", [], 1.0),
        ("Kara wynosi 7,296.00 PLN.", [], 1.0),  # angielski zapis celowo: jednoznaczny
        ("Kara wynosi 7,296 PLN.", ["7,296"], 0.0),  # niejednoznaczne: zgłoszone, nigdy poparte
    ],
)
def test_formatting_and_signs(score, agent_gold, answer_text, uncovered, facts):
    result = score(make_trace(JULY, answer_text, july_events()), agent_gold["P1"])
    assert (result["uncovered_numbers"], result["answer_fact_coverage"]) == (uncovered, facts)


def test_separate_monthly_calls_add_up(score, agent_gold):
    contract = event(
        "search_documents",
        {
            "status": "ok",
            "query": "q",
            "passages": [
                {
                    "doc_id": "CON-101",
                    "chunk_id": "CON-101#ctx004",
                    "text": "jeśli OTIF przez trzy kolejne miesiące wynosi mniej niż 94%",
                }
            ],
        },
    )
    monthly = [
        event(
            "get_supplier_otif",
            otif_result("SUP-114", {m: v}),
            supplier="SUP-114",
            from_month=m,
            to_month=m,
        )
        for m, v in [("2026-07", 94.6), ("2026-08", 93.8), ("2026-09", 93.5)]
    ]
    trace = make_trace(agent_gold["T1"].question, "x", [contract, *monthly])
    assert score(trace, agent_gold["T1"])["tool_coverage"] == 1.0


def test_no_data_needs_the_month_reported_missing(score, agent_gold):
    def otif_event(supplier_id, obs, missing, from_month, to_month):
        return event(
            "get_supplier_otif",
            otif_result(supplier_id, obs, missing=missing),
            supplier=supplier_id,
            from_month=from_month,
            to_month=to_month,
        )

    july_only = [otif_event("SUP-114", {"2026-07": 94.6}, (), "2026-07", "2026-07")]
    october = [otif_event("SUP-114", {}, ["2026-10"], "2026-10", "2026-10")]
    wider = [otif_event("SUP-114", {"2026-09": 93.5}, ["2026-10"], "2026-09", "2026-10")]
    other_supplier = [otif_event("SUP-207", {}, ["2026-10"], "2026-10", "2026-10")]

    def coverage(events):
        trace = make_trace(agent_gold["N1"].question, "Brak danych za październik.", events)
        return score(trace, agent_gold["N1"])["tool_coverage"]

    assert coverage(july_only) == 0.0, (
        "wywołanie za lipiec nie jest dowodem dotyczącym października"
    )
    assert coverage(october) == 1.0 and coverage(wider) == 1.0
    assert coverage(other_supplier) == 0.0


def test_the_wrong_supplier_does_not_cover_a_need(score, agent_gold):
    events = [
        event("get_contract_terms", TERMS_114, supplier="SUP-114"),
        event(
            "get_supplier_otif",
            otif_result("SUP-114", {"2026-07": 94.6}),
            supplier="SUP-114",
            from_month="2026-07",
            to_month="2026-07",
        ),
    ]
    assert (
        score(make_trace(agent_gold["A1"].question, "x", events), agent_gold["A1"])["tool_coverage"]
        == 0.0
    )


# --- NASZE: przeliczenie przebiegów kursu i zdarzenie składowe ---------------------------------


@pytest.mark.parametrize("name", ["agent_run.json", "supervisor_run.json"])
def test_course_runs_rescore_exactly(score, agent_gold, name):
    """Kryterium 2 F4: każdy wiersz `score_trace` == zapisany `score` kursu (liczby z tolerancją 1e-9)."""  # noqa: E501
    from kovalo.agents.trace import load_traces

    _, rows = load_traces(COURSE_RUNS / name)
    assert [r["trace"].question_id for r in rows] == list(agent_gold)
    for row in rows:
        qid = row["trace"].question_id
        ours = score(row["trace"], agent_gold[qid])
        saved = row["score"]
        assert set(ours) >= set(saved), qid
        for key, expected in saved.items():
            if isinstance(expected, float):
                assert abs(ours[key] - expected) < 1e-9, (qid, key, ours[key], expected)
            else:
                assert ours[key] == expected, (qid, key)
    saved_json = json.loads((COURSE_RUNS / name).read_text(encoding="utf-8"))
    assert len(saved_json["traces"]) == 9


def test_course_run_totals_match_the_readme(score, agent_gold):
    """Jeden agent: 8/9, 3/3, 8/9, 0 (0), 0/9; zespół: 8/9, 3/3, 8/9, 3 (2), 0/9."""
    from kovalo.agents.trace import load_traces

    def totals(name):
        _, rows = load_traces(COURSE_RUNS / name)
        scores = [score(r["trace"], agent_gold[r["trace"].question_id]) for r in rows]
        tools = sum(1 for s in scores if s["tool_coverage"] == 1.0)
        args = [s["arguments_correct"] for s in scores if s["arguments_correct"] is not None]
        facts = sum(1 for s in scores if s["answer_fact_coverage"] == 1.0)
        uncovered = sum(len(s["uncovered_numbers"]) for s in scores)
        uncovered_q = sum(1 for s in scores if s["uncovered_numbers"])
        approval = sum(1 for s in scores if s["approval_claim"])
        return (tools, (sum(args), len(args)), facts, (uncovered, uncovered_q), approval)

    assert totals("agent_run.json") == (8, (3, 3), 8, (0, 0), 0)
    assert totals("supervisor_run.json") == (8, (3, 3), 8, (3, 2), 0)


def test_derived_otif_covers_t1_need(score, agent_gold):
    """FR-4.7 / U3: check_termination_condition z zdarzeniem składowym get_supplier_otif 07-09."""
    otif = otif_result("SUP-114", {"2026-07": 94.6, "2026-08": 93.8, "2026-09": 93.5})
    otif["requested"] = {"from": "2026-07", "to": "2026-09"}
    termination = {
        "status": "ok",
        "supplier_id": "SUP-114",
        "source_doc_id": "CON-101",
        "otif_below_pct": 94.0,
        "consecutive_months": 3,
        "notice_days": 60,
        "months": [
            {"month": "2026-07", "otif_pct": 94.6, "below_threshold": False},
            {"month": "2026-08", "otif_pct": 93.8, "below_threshold": True},
            {"month": "2026-09", "otif_pct": 93.5, "below_threshold": True},
        ],
        "months_below": 2,
        "condition_met": False,
        "otif": otif,
    }
    contract = {
        "status": "ok",
        "query": "q",
        "passages": [
            {
                "doc_id": "CON-101",
                "chunk_id": "CON-101#ctx004",
                "text": "Kovalo może rozwiązać umowę z 60-dniowym okresem wypowiedzenia, jeśli OTIF "  # noqa: E501
                "przez trzy kolejne miesiące wynosi mniej niż 94%.",
            }
        ],
    }
    events = [
        event("search_documents", contract, question="rozwiązanie umowy"),
        event(
            "check_termination_condition",
            termination,
            call_id="c2",
            supplier="Mleczna Dolina",
            as_of_month="2026-09",
        ),
        event(
            "get_supplier_otif",
            otif,
            call_id="c2#get_supplier_otif",
            parent="c2",
            supplier="SUP-114",
            from_month="2026-07",
            to_month="2026-09",
        ),
    ]
    answer = (
        "Nie, warunek nie jest spełniony (CON-101): lipiec 94,6% nie jest poniżej progu 94%, "
        "sierpień 93,8% i wrzesień 93,5% są. Seria trzech miesięcy została przerwana."
    )
    result = score(make_trace(T1_QUESTION, answer, events), agent_gold["T1"])
    assert result["tool_coverage"] == 1.0
    assert result["answer_fact_coverage"] == 1.0
    assert result["uncovered_numbers"] == [] and result["numeric_source_coverage"] == 1.0
    assert result["tool_calls"] == 3, (
        "zdarzenia składowe liczą się jak zagnieżdżone wywołania kursu"
    )
    assert result["approval_claim"] is False

    without_derived = [e for e in events if e.parent_call_id is None]
    partial = score(make_trace(T1_QUESTION, answer, without_derived), agent_gold["T1"])
    assert partial["tool_coverage"] < 1.0, (
        "bez składowego odczytu potrzeba get_supplier_otif nie jest spełniona"
    )
