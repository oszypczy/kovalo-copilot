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
