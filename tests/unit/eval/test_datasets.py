from __future__ import annotations


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
