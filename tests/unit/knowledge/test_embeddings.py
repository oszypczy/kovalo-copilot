"""Embeddingi: cache, tryb kursu, adapter fastembed (FR-1.4; kontrakt F1, 5.3 i 10.2)."""

from __future__ import annotations

import re

import pytest
import yaml


def _spec(
    model="intfloat/multilingual-e5-large", dimensions=1024, query="query: ", passage="passage: "
):
    from kovalo.config import EmbeddingModelSpec

    return EmbeddingModelSpec(
        model=model, dimensions=dimensions, query_prefix=query, passage_prefix=passage
    )


def _fastembed(spec, tmp_path, **kwargs):
    from kovalo.knowledge.embeddings import FastEmbedEmbeddings

    return FastEmbedEmbeddings(spec, models_dir=tmp_path / "models", **kwargs)


# --- cache (port kroku 4 kursu) --------------------------------------------------------------


def test_texts_embedded_once_then_cached(tmp_path, fake_embeddings):
    from kovalo.knowledge.embeddings import build_cached_embeddings

    emb = build_cached_embeddings(fake_embeddings, cache_dir=tmp_path, model="m/x")
    first = emb.embed_documents(["alpha", "beta"])
    second = emb.embed_documents(["alpha", "beta"])
    assert fake_embeddings.calls == ["alpha", "beta"]
    assert first == second


def test_queries_cached_separately_from_documents(tmp_path, fake_embeddings):
    from kovalo.knowledge.embeddings import build_cached_embeddings

    emb = build_cached_embeddings(fake_embeddings, cache_dir=tmp_path, model="m/x")
    emb.embed_query("tekst")
    emb.embed_documents(["tekst"])
    assert fake_embeddings.calls == ["tekst", "tekst"]  # dwa wywołania modelu
    emb.embed_query("tekst")
    emb.embed_documents(["tekst"])
    assert fake_embeddings.calls == ["tekst", "tekst"]  # powtórka z cache


def test_cache_layout(tmp_path, fake_embeddings):
    from kovalo.knowledge.embeddings import build_cached_embeddings, cache_namespace

    model = "intfloat/multilingual-e5-large"
    ns = cache_namespace(model)
    assert ns == "intfloat_multilingual-e5-large"
    emb = build_cached_embeddings(fake_embeddings, cache_dir=tmp_path, model=model)
    emb.embed_documents(["alpha"])
    emb.embed_query("beta")
    for kind in ("passage", "query"):
        names = [p.name for p in (tmp_path / ns / kind).iterdir()]
        assert len(names) == 1
        assert re.fullmatch(re.escape(ns) + r"[0-9a-f]{64}", names[0])


def test_cache_namespace_replaces_unsafe_characters():
    from kovalo.knowledge.embeddings import cache_namespace

    assert cache_namespace("a b:c/d.e-f_g") == "a_b_c_d.e-f_g"


# --- tryb kursu (openai_cache) ---------------------------------------------------------------


def test_course_cache_answers_every_question(settings, forbidden_embeddings):
    from kovalo.knowledge.embeddings import build_course_cache_embeddings

    emb = build_course_cache_embeddings(
        settings.paths.embedding_cache,
        model=settings.embeddings.model,
        underlying=forbidden_embeddings,
    )
    mp1 = yaml.safe_load(settings.paths.eval_queries.read_text(encoding="utf-8"))["queries"]
    mp2 = yaml.safe_load(settings.paths.mini_project_2_questions.read_text(encoding="utf-8"))[
        "questions"
    ]
    assert len(mp1) == 10 and len(mp2) == 12
    for item in [*mp1, *mp2]:
        assert len(emb.embed_query(item["question"])) == settings.embeddings.dimensions == 1536


def test_course_cache_miss_raises_and_writes_nothing(settings):
    from kovalo.config import ConfigError
    from kovalo.knowledge.embeddings import CourseCacheMissError, build_course_cache_embeddings

    root = settings.paths.embedding_cache
    before = sorted(p.name for p in root.rglob("*"))
    emb = build_course_cache_embeddings(root, model=settings.embeddings.model)
    assert issubclass(CourseCacheMissError, ConfigError)
    with pytest.raises(CourseCacheMissError, match="Zupełnie nowe pytanie"):
        emb.embed_query("Zupełnie nowe pytanie, którego nie ma w cache kursu?")
    with pytest.raises(CourseCacheMissError):
        emb.embed_documents(["Zupełnie nowy chunk, którego nie ma w cache kursu."])
    assert sorted(p.name for p in root.rglob("*")) == before


def test_course_cache_only_embeddings_always_raises():
    from kovalo.knowledge.embeddings import CourseCacheMissError, CourseCacheOnlyEmbeddings

    emb = CourseCacheOnlyEmbeddings()
    with pytest.raises(CourseCacheMissError):
        emb.embed_documents(["x"])
    with pytest.raises(CourseCacheMissError):
        emb.embed_query("x")


def test_course_cache_file_names_are_model_plus_sha256(tmp_path, fake_embeddings):
    from kovalo.knowledge.embeddings import build_course_cache_embeddings

    cache_dir = tmp_path / "cache"  # tmp_path zawiera też "cwd" z isolated_env
    emb = build_course_cache_embeddings(cache_dir, model="mod", underlying=fake_embeddings)
    emb.embed_documents(["alpha"])
    names = [p.name for p in cache_dir.iterdir()]
    assert len(names) == 1 and re.fullmatch(r"mod[0-9a-f]{64}", names[0])


def test_missing_from_course_cache(settings):
    from kovalo.knowledge.embeddings import missing_from_course_cache

    assert missing_from_course_cache(
        settings.paths.embedding_cache, settings.embeddings.model, ["tego tekstu nie ma w cache"]
    ) == [0]


# --- adapter fastembed (atrapa TextEmbedding) -------------------------------------------------


def test_fastembed_adapter_prefixes(tmp_path, fake_text_embedding):
    emb = _fastembed(_spec(), tmp_path)
    emb.embed_documents(["a"])
    emb.embed_query("b")
    assert fake_text_embedding.calls == [("passage", "passage: a"), ("query", "query: b")]


def test_fastembed_adapter_no_prefix_for_model_without_prefixes(tmp_path, fake_text_embedding):
    spec = _spec("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 384, "", "")
    emb = _fastembed(spec, tmp_path)
    emb.embed_documents(["a"])
    emb.embed_query("b")
    assert fake_text_embedding.calls == [("passage", "a"), ("query", "b")]


def test_fastembed_adapter_passes_init_arguments(tmp_path, fake_text_embedding):
    _fastembed(_spec(), tmp_path, threads=2, local_files_only=True)
    init = fake_text_embedding.inits[-1]
    assert init["model_name"] == "intfloat/multilingual-e5-large"
    assert init["cache_dir"] == str(tmp_path / "models")
    assert init["threads"] == 2 and init["local_files_only"] is True


def test_fastembed_adapter_returns_python_floats(tmp_path, fake_text_embedding):
    emb = _fastembed(_spec(), tmp_path)
    vectors = emb.embed_documents(["a", "b"])
    query = emb.embed_query("c")
    assert isinstance(vectors, list) and all(isinstance(v, list) for v in vectors)
    assert isinstance(query, list)
    assert all(type(x) is float for v in [*vectors, query] for x in v)
    assert len(vectors[0]) == len(query) == 1024


def test_fastembed_adapter_empty_inputs(tmp_path, fake_text_embedding):
    emb = _fastembed(_spec(), tmp_path)
    assert emb.embed_documents([]) == []
    assert fake_text_embedding.calls == []
    vector = emb.embed_query("")  # pusty tekst to wektor, nie błąd
    assert len(vector) == 1024


def test_fastembed_dimension_mismatch(tmp_path, fake_text_embedding):
    from kovalo.config import ConfigError

    with pytest.raises(ConfigError, match=r"1024.*999|999.*1024"):
        _fastembed(_spec(dimensions=999), tmp_path)


def test_fastembed_unknown_model(tmp_path, fake_text_embedding):
    from kovalo.config import ConfigError

    with pytest.raises(ConfigError, match="x/unknown-model"):
        _fastembed(_spec("x/unknown-model", 8), tmp_path)


def test_counting_embeddings(fake_embeddings):
    from kovalo.knowledge.embeddings import CountingEmbeddings

    counting = CountingEmbeddings(fake_embeddings)
    counting.embed_documents(["a", "b", "c"])
    counting.embed_documents([])
    counting.embed_query("q")
    assert counting.documents_embedded == 3
    assert counting.queries_embedded == 1
    assert fake_embeddings.calls == ["a", "b", "c", "q"]


def test_counting_embeddings_counts_only_cache_misses(tmp_path, fake_embeddings):
    from kovalo.knowledge.embeddings import CountingEmbeddings, build_cached_embeddings

    counting = CountingEmbeddings(fake_embeddings)
    emb = build_cached_embeddings(counting, cache_dir=tmp_path, model="m/x")
    emb.embed_documents(["a", "b"])
    emb.embed_documents(["a", "b", "c"])
    assert counting.documents_embedded == 3


def test_model_size_gb(fake_text_embedding):
    from kovalo.knowledge.embeddings import model_size_gb

    assert model_size_gb("intfloat/multilingual-e5-large") == pytest.approx(2.24)
    assert model_size_gb(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ) == pytest.approx(0.22)
    assert model_size_gb("x/unknown-model") is None
