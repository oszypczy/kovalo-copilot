from __future__ import annotations

import pytest


def test_load_mp1_and_mp2_sets(settings):
    from kovalo.eval.datasets import load_eval_queries, load_mp2_retrieval_queries

    mp1 = load_eval_queries(settings.paths.eval_queries)
    assert [q.query_id for q in mp1] == [f"Q{i:02d}" for i in range(1, 11)]
    by_id = {q.query_id: q for q in mp1}
    assert by_id["Q03"].relevant_docs == ("MTG-403", "CON-101")
    assert isinstance(by_id["Q01"].relevant_docs, tuple)

    mp2 = load_mp2_retrieval_queries(settings.paths.mini_project_2_questions)
    assert [q.query_id for q in mp2] == [f"M{i:02d}" for i in range(1, 11)]  # bez M11 i M12
    assert all(isinstance(q.relevant_docs, tuple) and q.relevant_docs for q in mp2)
    assert {q.query_id: q.relevant_docs for q in mp2}["M02"] == ("CON-101", "MTG-403")


def test_load_gold(settings, tmp_path):
    import pytest

    from kovalo.eval.datasets import GoldQuestion, load_gold

    gold = load_gold(settings.paths.mini_project_2_questions)
    assert [q.id for q in gold] == [f"M{i:02d}" for i in range(1, 13)]
    assert all(isinstance(q, GoldQuestion) for q in gold)
    by_id = {q.id: q for q in gold}
    assert not by_id["M11"].answerable and not by_id["M12"].answerable
    assert by_id["M11"].relevant_docs == [] and by_id["M11"].evidence_facts == []
    assert sum(q.answerable for q in gold) == 10
    assert by_id["M02"].requires_all and by_id["M02"].relevant_docs == ["CON-101", "MTG-403"]
    assert by_id["M03"].evidence_facts == [
        ["za każdy pełny punkt procentowy"],
        ["nie więcej jednak niż 10%"],
    ]
    assert by_id["M01"].key_facts == [["dyrektor handlowy", "dyrektora handlowego"]]
    assert by_id["M01"].reference.startswith("Rabat 15%")

    with pytest.raises(FileNotFoundError):
        load_gold(tmp_path / "brak.yaml")
    bad = tmp_path / "zle.yaml"
    bad.write_text("questions:\n  - id: X\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"zle\.yaml"):
        load_gold(bad)
    empty = tmp_path / "puste.yaml"
    empty.write_text("inne: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"puste\.yaml"):
        load_gold(empty)


def test_load_dialogues(settings, tmp_path):
    import pytest

    from kovalo.eval.datasets import Dialogue, DialogueTurn, load_dialogues
    from kovalo.workflow.schemas import Turn

    dialogues = load_dialogues(settings.paths.mini_project_3_dialogues)
    assert [d.id for d in dialogues] == [f"D{i}" for i in range(1, 9)]
    assert all(isinstance(d, Dialogue) and d.title for d in dialogues)
    assert sum(len(d.turns) for d in dialogues) == 13
    assert [len(d.turns) for d in dialogues] == [3, 2, 1, 2, 1, 1, 2, 1]
    assert all(isinstance(t, DialogueTurn) for d in dialogues for t in d.turns)
    for dialogue in dialogues:
        for t in dialogue.turns:
            Turn.model_validate(t.understand)  # idealny Turn waliduje się
    d2 = {d.id: d for d in dialogues}["D2"]
    assert d2.turns[1].resume == ["approved", "rejected"]
    assert d2.turns[0].expect["status"] == "approved"
    assert d2.turns[1].expect["pauses"] == [["approval", "CEO"], ["approval", "CFO"]]

    with pytest.raises(FileNotFoundError):
        load_dialogues(tmp_path / "brak.yaml")


def test_load_dialogues_rejects_invalid_understand(tmp_path):
    import pytest

    from kovalo.eval.datasets import load_dialogues

    bad = tmp_path / "zle.yaml"
    bad.write_text(
        "dialogues:\n"
        "  - id: D1\n    title: x\n    turns:\n"
        "      - user: Cześć\n        understand: {kind: nieznany}\n"
        "        resume: []\n        expect: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(Exception, match=r"kind|zle\.yaml"):
        load_dialogues(bad)


# ---------------------------------------------------------------------------
# F4: wzorzec agenta (kontrakt F4, 11.1; port load_agent_gold z helpers/agent_trace.py)
# ---------------------------------------------------------------------------


def test_load_agent_gold(settings):
    from kovalo.eval.datasets import AgentGold, GoldCalculation, Need, load_agent_gold

    gold = load_agent_gold(settings.paths.mini_project_4_questions)
    assert [g.id for g in gold] == ["P1", "P2", "P3", "T1", "A1", "D1", "S1", "N1", "B1"]
    assert all(isinstance(g, AgentGold) for g in gold)
    by_id = {g.id: g for g in gold}
    assert len(by_id["P1"].calculations) == 1 and isinstance(
        by_id["P1"].calculations[0], GoldCalculation
    )
    assert by_id["P1"].calculations[0].invoice_grosze == 18240000
    t1_needs = by_id["T1"].needs
    assert all(isinstance(n, Need) for n in t1_needs)
    assert [n.tool for n in t1_needs[:2]] == ["search_documents", "get_supplier_otif"]
    assert t1_needs[2].tool is None and len(t1_needs[2].any_of) == 2, "lipiec: dane albo MTG-403"
    assert {n.tool for n in t1_needs[2].any_of} == {"get_supplier_otif", "search_documents"}
    for g in gold:
        assert len(g.approval_claim_phrases) == 24, g.id
    assert by_id["P1"].kind == "penalty" and by_id["T1"].kind == "termination"


def test_load_agent_gold_missing_file(tmp_path):
    import pytest

    from kovalo.eval.datasets import load_agent_gold

    with pytest.raises(FileNotFoundError):
        load_agent_gold(tmp_path / "nie-ma.yaml")


# ---------------------------------------------------------------------------
# F5: rozmowy E5 (kontrakt F5, 10.1 i 13.2)
# ---------------------------------------------------------------------------


def test_load_conversations(settings):
    from kovalo.eval.datasets import Dialogue, load_conversations, load_dialogues

    mp5, mp3 = (
        settings.paths.mini_project_5_conversations,
        settings.paths.mini_project_3_dialogues,
    )
    conversations = load_conversations(mp5, mp3)
    assert [d.id for d in conversations] == ["C1", "C2", "C3", "C4", "C5"] + [
        f"D{i}" for i in range(1, 9)
    ]
    assert all(isinstance(d, Dialogue) for d in conversations)
    assert [len(d.turns) for d in conversations] == [4, 2, 2, 2, 2, 3, 2, 1, 2, 1, 1, 2, 1]
    assert sum(len(d.turns) for d in conversations) == 25

    d7 = {d.id: d for d in conversations}["D7"]
    assert d7.turns[0].expect["agent"]["needs"][0]["doc_id"] == "MTG-402"
    # plik MP3 nietknięty: nakładka dotyczy tylko wyniku
    original = {d.id: d for d in load_dialogues(mp3)}
    assert "agent" not in original["D7"].turns[0].expect
    again = {d.id: d for d in load_conversations(mp5, mp3)}
    assert "agent" not in original["D7"].turns[0].expect
    assert again["D7"].turns[0].expect["agent"] == d7.turns[0].expect["agent"]
    again["D7"].turns[0].expect["agent"]["needs"].clear()
    assert d7.turns[0].expect["agent"]["needs"], "głębokie kopie: wyniki wywołań są niezależne"

    data_turns = [
        (d.id, n)
        for d in conversations
        for n, t in enumerate(d.turns, start=1)
        if t.expect["kind"] == "document_question" and "agent" in t.expect
    ]
    assert data_turns == [("C1", 1), ("C1", 3), ("C5", 1), ("D7", 1)]
    # pozostałe tury D1-D8 nie dostały wzorca agenta
    assert all(
        "agent" not in t.expect
        for d in conversations
        if d.id.startswith("D") and d.id != "D7"
        for t in d.turns
    )


def test_load_conversations_validates_agent_gold(settings, tmp_path):
    """`expect.agent` (needs, calculations, key_facts) walidowany przy wczytaniu jako AgentGold
    bez id; zły kształt → ValueError z nazwą pliku."""
    import yaml

    from kovalo.eval.datasets import load_conversations

    mp5 = yaml.safe_load(settings.paths.mini_project_5_conversations.read_text(encoding="utf-8"))
    mp5["conversations"][0]["turns"][0]["expect"]["agent"]["needs"] = [
        {"tool": "x", "months": "2026-07"}
    ]
    bad = tmp_path / "zle_rozmowy.yaml"
    bad.write_text(yaml.safe_dump(mp5, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match=r"zle_rozmowy\.yaml"):
        load_conversations(bad, settings.paths.mini_project_3_dialogues)


def test_load_conversations_missing_file(settings, tmp_path):
    from kovalo.eval.datasets import load_conversations

    with pytest.raises(FileNotFoundError):
        load_conversations(tmp_path / "brak.yaml", settings.paths.mini_project_3_dialogues)
