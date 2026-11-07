"""Testy kovalo.config (kontrakt F0, sekcje 4 i 8)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.conftest import COURSE_CONFIG, REPO_CONFIG, REPO_ROOT

PINNED_KEYS = [
    "tokenizer.encoding",
    "chunking.chunk_size",
    "chunking.chunk_overlap",
    "chunking.separators",
    "embeddings.model",
    "embeddings.dimensions",
    "retrieval.top_k",
    "reference_llm",
    "cost.reference_model",
    "cost.prices_per_1m_tokens",
    "cost.scenario",
    "policy",
    "agent",
    "mcp",
    "prompts.system",
    "prompts.answer_system",
    "prompts.understand_system",
    "prompts.agent_system",
    "prompts.assistant_understand_system",
]


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _get(data: dict[str, Any], dotted: str) -> Any:
    node: Any = data
    for key in dotted.split("."):
        node = node[key]
    return node


def test_loads_repo_config_yaml():
    from kovalo.config import load_settings

    s = load_settings()
    assert s.root == REPO_ROOT


@pytest.mark.parametrize("key", PINNED_KEYS)
def test_pinned_values_match_course(key: str):
    assert REPO_CONFIG.is_file(), "brak config.yaml w katalogu głównym repo"
    assert _get(_load(REPO_CONFIG), key) == _get(_load(COURSE_CONFIG), key)


def test_course_sections_kept():
    ours, course = _load(REPO_CONFIG), _load(COURSE_CONFIG)
    assert ours["corpus"]["required_metadata"] == course["corpus"]["required_metadata"]
    assert ours["embeddings"]["batch_size"] == course["embeddings"]["batch_size"]
    assert set(course["paths"]) <= set(ours["paths"])
    for n in range(1, 6):
        assert ours["paths"][f"mini_project_{n}_output_dir"] == f"reports/mp{n}"


def test_paths_are_absolute_and_relative_to_config_dir():
    from kovalo.config import load_settings

    s = load_settings()
    assert s.paths.corpus_dir == s.root / "data" / "corpus"
    assert s.paths.corpus_dir.is_absolute()
    assert s.paths.mini_project_1_output_dir == s.root / "reports" / "mp1"


def test_env_overrides_yaml(monkeypatch: pytest.MonkeyPatch):
    from kovalo.config import load_settings

    assert load_settings().db.host == "localhost"
    monkeypatch.setenv("KOVALO_LLM__MODEL", "x")
    monkeypatch.setenv("KOVALO_DB__HOST", "postgres")
    s = load_settings()
    assert s.llm.model == "x"
    assert s.db.host == "postgres"


def test_init_overrides_env(monkeypatch: pytest.MonkeyPatch):
    from kovalo.config import DbConfig, load_settings

    monkeypatch.setenv("KOVALO_DB__HOST", "envhost")
    monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
    s = load_settings(db=DbConfig(host="inithost"), google_api_key="init-key")
    assert s.db.host == "inithost"
    assert s.google_api_key is not None
    assert s.google_api_key.get_secret_value() == "init-key"


def test_google_key_from_env_and_gemini_alias(monkeypatch: pytest.MonkeyPatch):
    from kovalo.config import load_settings

    monkeypatch.setenv("GOOGLE_API_KEY", "g-key")
    s = load_settings()
    assert s.google_api_key is not None
    assert s.google_api_key.get_secret_value() == "g-key"
    assert s.api_key_for("google_genai") == s.google_api_key

    monkeypatch.delenv("GOOGLE_API_KEY")
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    s = load_settings()
    assert s.google_api_key is not None
    assert s.google_api_key.get_secret_value() == "gem-key"

    # interpretacja: GOOGLE_API_KEY ma pierwszeństwo, GEMINI_API_KEY jest zapasowy
    monkeypatch.setenv("GOOGLE_API_KEY", "g-key")
    s = load_settings()
    assert s.google_api_key is not None
    assert s.google_api_key.get_secret_value() == "g-key"


def test_missing_google_key_is_none(monkeypatch: pytest.MonkeyPatch):
    from kovalo.config import load_settings

    assert load_settings().google_api_key is None
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    assert load_settings().google_api_key is None


def test_secrets_not_in_repr(monkeypatch: pytest.MonkeyPatch):
    from kovalo.config import load_settings

    monkeypatch.setenv("POSTGRES_PASSWORD", "haslo-pg-xyz")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-sekret-789")
    s = load_settings(google_api_key="sekret-klucz-123")
    text = repr(s) + str(s)
    for secret in ("sekret-klucz-123", "haslo-pg-xyz", "openai-sekret-789"):
        assert secret not in text


def test_unknown_key_rejected(tmp_config: Path, patch_config):
    from kovalo.config import ConfigError, load_settings

    patch_config(tmp_config, "llm.nieznany_klucz", 1)
    with pytest.raises(ConfigError, match="nieznany_klucz"):
        load_settings(tmp_config)


def test_unknown_top_level_yaml_key_rejected(tmp_config: Path, patch_config):
    from kovalo.config import ConfigError, load_settings

    patch_config(tmp_config, "nieznany_klucz_glowny", 1)
    with pytest.raises(ConfigError, match="nieznany_klucz_glowny"):
        load_settings(tmp_config)


def test_dotenv_unknown_keys_ignored(tmp_config: Path):
    from kovalo.config import load_settings

    # isolated_env robi chdir do pustego tmp; `.env` jest czytany z katalogu bieżącego
    Path(".env").write_text(
        "API_PORT=9090\nKOVALO_CONFIG_FILE=/tmp/x.yaml\nOBCA_ZMIENNA=abc\n", encoding="utf-8"
    )
    settings = load_settings(tmp_config)
    assert settings is not None


def test_invalid_provider_rejected(tmp_config: Path, patch_config):
    from kovalo.config import ConfigError, load_settings

    patch_config(tmp_config, "llm.provider", "groq")
    with pytest.raises(ConfigError):
        load_settings(tmp_config)


def test_missing_file_raises_config_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from kovalo.config import ConfigError, load_settings

    missing = tmp_path / "nie-ma.yaml"
    monkeypatch.setenv("KOVALO_CONFIG_FILE", str(missing))
    with pytest.raises(ConfigError, match=r"nie-ma\.yaml"):
        load_settings()


def test_get_settings_is_cached():
    from kovalo.config import get_settings

    first = get_settings()
    assert get_settings() is first
    get_settings.cache_clear()
    assert get_settings() is not first


def test_postgres_dsn_and_sqlalchemy_url(monkeypatch: pytest.MonkeyPatch):
    from kovalo.config import load_settings

    monkeypatch.setenv("POSTGRES_USER", "u")
    monkeypatch.setenv("POSTGRES_PASSWORD", "pw")
    monkeypatch.setenv("POSTGRES_DB", "d")
    monkeypatch.setenv("KOVALO_DB__HOST", "h")
    s = load_settings()
    assert s.postgres_dsn() == f"postgresql://u:pw@h:{s.db.port}/d"
    assert s.sqlalchemy_url() == f"postgresql+psycopg://u:pw@h:{s.db.port}/d"


def test_unknown_override_rejected():
    # extra="ignore" dotyczy tylko .env/env; literówka w argumencie load_settings ma dawać błąd.
    from kovalo.config import ConfigError, DbConfig, load_settings

    with pytest.raises(ConfigError, match="dbb"):
        load_settings(dbb=DbConfig(host="x"))


# ---------------------------------------------------------------------------
# F1 (kontrakt: docs/contracts/f1.md, sekcja 4 i 10.2)
# ---------------------------------------------------------------------------


def _f1_baseline() -> None:
    """Config repo ma klucze F1 (bez tego testy poniżej przechodziłyby przez extra=forbid)."""
    from kovalo.config import load_settings

    s = load_settings(google_api_key="test-key")
    assert s.cost.our_model and s.embeddings.candidates and s.db.index_meta_table


def _f1_settings(path: Path, **overrides: Any):
    from kovalo.config import load_settings

    return load_settings(path, google_api_key="test-key", **overrides)


def test_f1_config_keys_load():
    import datetime

    from kovalo.config import load_settings

    s = load_settings(google_api_key="test-key")
    local = s.embeddings.local
    # model z configu może zmienić się po E1 (A7); sprawdzamy spójność, nie nazwę
    assert s.embeddings.provider == "fastembed"
    assert local.dimensions > 0 and local.model
    assert local.local_files_only is False
    assert local.threads is None
    assert local.models_dir.is_absolute() and local.models_dir.parts[-2:] == (".cache", "fastembed")
    assert local.cache_dir.is_absolute() and local.cache_dir.parts[-2:] == (".cache", "embeddings")
    names = [c.model for c in s.embeddings.candidates]
    assert len(names) == 4 and len(set(names)) == 4
    assert local.model in names
    assert "jinaai/jina-embeddings-v3" in names
    by_name = {c.model: c for c in s.embeddings.candidates}
    assert by_name["intfloat/multilingual-e5-large"].query_prefix == "query: "
    assert by_name["intfloat/multilingual-e5-large"].passage_prefix == "passage: "
    assert by_name["sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"].dimensions == 384
    assert by_name["sentence-transformers/paraphrase-multilingual-mpnet-base-v2"].dimensions == 768
    cost = s.cost
    assert cost.our_model == "gemini-3.5-flash-lite" == s.llm.model  # koszt = model odpowiedzi
    assert cost.our_prices_checked_on == datetime.date(2026, 11, 2)
    assert cost.our_prices_per_1m_tokens["gemini-3.6-flash"].input == 0.75
    assert cost.our_prices_per_1m_tokens["gemini-3.6-flash"].output == 3.75
    assert cost.our_prices_per_1m_tokens["gemini-3.5-flash-lite"].input == 0.3
    assert cost.our_prices_per_1m_tokens["gemini-3.5-flash-lite"].output == 2.5
    assert [f.model for f in s.llm.fallbacks] == ["gemini-3.6-flash"]  # ADR-0006, aneks 2026-11-02
    assert s.llm.rate_limit.requests_per_second == 0.25
    assert s.llm.fallbacks[0].rate_limit.requests_per_second == 0.15
    assert s.db.index_meta_table == "kovalo_index_meta"
    assert s.db.vector_collection_mp1 == "kovalo_mp1"


def test_our_model_must_have_price(tmp_config: Path, patch_config):
    _f1_baseline()
    from kovalo.config import ConfigError

    patch_config(tmp_config, "cost.our_model", "model-bez-ceny")
    with pytest.raises(ConfigError, match="our_model"):
        _f1_settings(tmp_config)


def test_reference_model_must_have_price(tmp_config: Path, patch_config):
    _f1_baseline()
    from kovalo.config import ConfigError

    patch_config(tmp_config, "cost.reference_model", "model-bez-ceny")
    with pytest.raises(ConfigError, match="reference_model"):
        _f1_settings(tmp_config)


def test_candidate_conflicting_with_local_rejected(tmp_config: Path, patch_config):
    _f1_baseline()
    from kovalo.config import ConfigError

    data = yaml.safe_load(tmp_config.read_text(encoding="utf-8"))
    local_model = data["embeddings"]["local"]["model"]
    patch_config(tmp_config, "embeddings.candidates", [{"model": local_model, "dimensions": 7}])
    with pytest.raises(ConfigError):
        _f1_settings(tmp_config)


def test_duplicate_candidates_rejected(tmp_config: Path, patch_config):
    _f1_baseline()
    from kovalo.config import ConfigError

    spec = {"model": "x/y", "dimensions": 8}
    patch_config(tmp_config, "embeddings.candidates", [spec, dict(spec)])
    with pytest.raises(ConfigError):
        _f1_settings(tmp_config)


def test_invalid_table_name_rejected(tmp_config: Path, patch_config):
    _f1_baseline()
    from kovalo.config import ConfigError

    patch_config(tmp_config, "db.vector_collection_mp1", "kovalo-mp1")
    with pytest.raises(ConfigError, match="vector_collection_mp1"):
        _f1_settings(tmp_config)


def test_negative_price_rejected(tmp_config: Path, patch_config):
    _f1_baseline()
    from kovalo.config import ConfigError

    patch_config(
        tmp_config,
        "cost.our_prices_per_1m_tokens.gemini-3.5-flash-lite",
        {"input": -1, "output": 1},
    )
    with pytest.raises(ConfigError):
        _f1_settings(tmp_config)


def test_embeddings_env_overrides(monkeypatch: pytest.MonkeyPatch):
    from kovalo.config import load_settings

    monkeypatch.setenv("KOVALO_EMBEDDINGS__LOCAL__LOCAL_FILES_ONLY", "true")
    monkeypatch.setenv("KOVALO_EMBEDDINGS__PROVIDER", "openai_cache")
    s = load_settings(google_api_key="test-key")
    assert s.embeddings.local.local_files_only is True
    assert s.embeddings.provider == "openai_cache"


@pytest.mark.parametrize("length", [52, 53])
def test_table_name_within_limit_accepted(tmp_config: Path, patch_config, length: int):
    # P1: regex {0,52} (kontrakt F1, 4.2) = maks. 53 znaki; zapas na `__new_pkey` w 63 bajtach
    _f1_baseline()
    name = "t" * length
    patch_config(tmp_config, "db.vector_collection_mp1", name)
    assert _f1_settings(tmp_config).db.vector_collection_mp1 == name


TABLE_FIELDS = ["vector_collection", "vector_collection_mp1", "index_meta_table"]


@pytest.mark.parametrize("field", TABLE_FIELDS)
@pytest.mark.parametrize("length", [54, 63])
def test_table_name_too_long_rejected(tmp_config: Path, patch_config, field: str, length: int):
    _f1_baseline()
    from kovalo.config import ConfigError

    patch_config(tmp_config, f"db.{field}", "t" * length)
    with pytest.raises(ConfigError, match=field):
        _f1_settings(tmp_config)


# ---------------------------------------------------------------------------
# F2 (kontrakt: docs/contracts/f2.md, sekcja 4)
# ---------------------------------------------------------------------------


def test_f2_config_keys_load():
    from kovalo.config import load_settings

    s = load_settings(google_api_key="test-key")
    assert s.retrieval.top_k == 3
    assert s.retrieval.mode == "vector"
    hybrid = s.retrieval.hybrid
    assert (hybrid.fetch_k, hybrid.rrf_c) == (10, 60)
    assert (hybrid.weights.vector, hybrid.weights.bm25) == (0.5, 0.5)
    assert (hybrid.bm25.k1, hybrid.bm25.b, hybrid.bm25.epsilon) == (1.5, 0.75, 0.25)
    assert s.paths.llm_cache_dir.is_absolute()
    assert s.paths.llm_cache_dir == s.root / ".cache" / "llm"
    assert s.prompts.judge_system.strip()
    assert "correct" in s.prompts.judge_system and "grounded" in s.prompts.judge_system


def test_retrieval_mode_from_env(monkeypatch: pytest.MonkeyPatch):
    from kovalo.config import load_settings

    monkeypatch.setenv("KOVALO_RETRIEVAL__MODE", "hybrid")
    assert load_settings().retrieval.mode == "hybrid"
    monkeypatch.setenv("KOVALO_RETRIEVAL__MODE", "bm25")
    from kovalo.config import ConfigError

    with pytest.raises(ConfigError, match="mode"):
        load_settings()


def test_fetch_k_below_top_k_rejected(tmp_config: Path, patch_config):
    from kovalo.config import ConfigError, load_settings

    patch_config(tmp_config, "retrieval.hybrid.fetch_k", 2)
    with pytest.raises(ConfigError, match="fetch_k") as info:
        load_settings(tmp_config)
    assert "top_k" in str(info.value) or "3" in str(info.value)  # komunikat z obiema wartościami


@pytest.mark.parametrize(
    "weights",
    [{"vector": 0, "bm25": 0}, {"vector": -0.1, "bm25": 0.5}, {"vector": 0.5, "bm25": -1}],
    ids=["both_zero", "negative_vector", "negative_bm25"],
)
def test_hybrid_weights_validation(tmp_config: Path, patch_config, weights):
    from kovalo.config import ConfigError, load_settings

    patch_config(tmp_config, "retrieval.hybrid.weights", weights)
    with pytest.raises(ConfigError, match=r"weights|vector|bm25"):
        load_settings(tmp_config)


def test_one_zero_weight_is_allowed(tmp_config: Path, patch_config):
    from kovalo.config import load_settings

    patch_config(tmp_config, "retrieval.hybrid.weights", {"vector": 1.0, "bm25": 0})
    assert load_settings(tmp_config).retrieval.hybrid.weights.bm25 == 0


def test_purpose_params_merge(tmp_config: Path, patch_config):
    from kovalo.config import load_settings

    patch_config(tmp_config, "llm.purposes", {"answer": {"params": {"thinking_level": "minimal"}}})
    s = load_settings(tmp_config)
    assert s.llm.purposes["answer"].params == {"thinking_level": "minimal"}
    assert "judge" not in s.llm.purposes
    assert s.llm.params["thinking_level"] == "low"  # llm.params bez zmian


def test_default_purposes_have_no_params():
    from kovalo.config import load_settings

    s = load_settings()
    assert all(not override.params for override in s.llm.purposes.values())


# ---------------------------------------------------------------------------
# F3 (kontrakt: docs/contracts/f3.md, sekcja 4)
# ---------------------------------------------------------------------------


def test_f3_config_keys_load():
    from kovalo.config import load_settings

    s = load_settings(google_api_key="test-key")
    assert s.workflow.checkpointer == "postgres"
    assert s.workflow.max_message_chars == 4000
    assert s.services.ui_api_timeout_s == 180


def test_policy_decimal_and_ranks():
    from decimal import Decimal

    from kovalo.config import load_settings

    policy = load_settings(google_api_key="test-key").policy
    assert policy.bands[0].up_to == Decimal(5)
    assert [b.up_to for b in policy.bands] == [Decimal(5), Decimal(12), Decimal(20), Decimal(100)]
    assert all(isinstance(b.up_to, Decimal) for b in policy.bands)
    assert policy.margin_floor == Decimal(9) and isinstance(policy.margin_floor, Decimal)
    assert policy.exceptions["clearance"].cap == Decimal(35)
    assert policy.band_rank(["Dyrektor handlowy"]) == 2
    assert policy.band_rank(["CEO", "CFO"]) == 3
    with pytest.raises(ValueError):
        policy.band_rank(["Nieznana rola"])
    with pytest.raises(ValueError):
        policy.band_rank(["CFO", "CEO"])  # kolejność ma znaczenie


def test_policy_matches_course_numerically():
    from kovalo.config import load_settings

    policy = load_settings(google_api_key="test-key").policy
    course = _load(COURSE_CONFIG)["policy"]
    assert [float(b.up_to) for b in policy.bands] == [b["up_to"] for b in course["bands"]]
    assert float(policy.margin_floor) == course["margin_floor"]
    assert {k: float(v.cap) for k, v in policy.exceptions.items()} == {
        k: v["cap"] for k, v in course["exceptions"].items()
    }


def _policy_with(tmp_config: Path, patch_config, mutate):
    from kovalo.config import load_settings

    data = _load(tmp_config)["policy"]
    mutate(data)
    patch_config(tmp_config, "policy", data)
    return load_settings(tmp_config, google_api_key="test-key")


_POLICY_BREAKS = {
    "decreasing_up_to": (
        lambda p: p["bands"].__setitem__(
            1, {"up_to": 4, "roles": ["Regionalny kierownik sprzedaży"]}
        ),
        "up_to",
    ),
    "last_band_not_100": (lambda p: p["bands"][-1].__setitem__("up_to", 90), "up_to"),
    "exception_roles_outside_bands": (
        lambda p: p["exceptions"]["clearance"].__setitem__("roles", ["Prezes"]),
        "clearance",
    ),
    "unknown_exception_name": (
        lambda p: p["exceptions"].__setitem__("vip", {"cap": 10, "roles": ["CEO", "CFO"]}),
        "vip",
    ),
    "cap_zero": (lambda p: p["exceptions"]["clearance"].__setitem__("cap", 0), "cap"),
    "cap_over_100": (lambda p: p["exceptions"]["clearance"].__setitem__("cap", 101), "cap"),
}


@pytest.mark.parametrize("name", list(_POLICY_BREAKS))
def test_policy_validation(tmp_config: Path, patch_config, name):
    from kovalo.config import ConfigError

    mutate, key = _POLICY_BREAKS[name]
    with pytest.raises(ConfigError, match=key):
        _policy_with(tmp_config, patch_config, mutate)


@pytest.mark.parametrize(
    "workflow",
    [{"checkpointer": "sqlite"}, {"max_message_chars": 0}, {"max_message_chars": -5}],
    ids=["unknown_checkpointer", "zero_chars", "negative_chars"],
)
def test_workflow_config_rejected(tmp_config: Path, patch_config, workflow):
    from kovalo.config import ConfigError, load_settings

    patch_config(tmp_config, "workflow", workflow)
    with pytest.raises(ConfigError):
        load_settings(tmp_config, google_api_key="test-key")


# ---------------------------------------------------------------------------
# F4 (kontrakt: docs/contracts/f4.md, sekcja 4)
# ---------------------------------------------------------------------------


def test_f4_config_keys_load():
    from decimal import Decimal

    from kovalo.config import load_settings

    s = load_settings(google_api_key="test-key")
    assert s.tools.variant == "ours"
    assert s.workflow.answerer == "agent"
    assert s.workflow.understand_prompt == "assistant_understand_system"
    assert s.services.sse_ping_s == 15
    terms = s.company.termination_terms["SUP-114"]
    assert terms.otif_below_pct == Decimal(94) and terms.consecutive_months == 3
    assert terms.notice_days == 60 and terms.source_doc_id == "CON-101"
    assert s.company.termination_version == "termination-v1"
    assert set(s.company.termination_terms) == {"SUP-114"}
    assert (s.agent.max_steps, s.agent.timeout_s) == (24, 180), "PINNED"
    assert s.paths.company_data_dir == s.root / "data" / "company"


def test_termination_terms_match_con101():
    """Wartości z config.yaml są przepisane z tekstu umowy: test pilnuje zgodności (ADR-0011)."""
    from kovalo.config import load_settings

    s = load_settings(google_api_key="test-key")
    terms = s.company.termination_terms["SUP-114"]
    path = next((s.paths.corpus_dir).glob("CON-101*.md"))
    text = path.read_text(encoding="utf-8")
    assert "trzy kolejne miesiące wynosi mniej niż 94%" in text
    assert "60-dniowym" in text
    assert terms.consecutive_months == 3 and terms.notice_days == 60
    assert terms.source_doc_id == "CON-101" and terms.otif_below_pct == 94


def test_prompts_by_name():
    from kovalo.config import load_settings

    prompts = load_settings(google_api_key="test-key").prompts
    assert prompts.by_name("assistant_understand_system") == prompts.assistant_understand_system
    assert prompts.by_name("understand_system") == prompts.understand_system
    assert prompts.by_name("understand_system") != prompts.by_name("assistant_understand_system")
    with pytest.raises(KeyError):
        prompts.by_name("nie_ma_takiego")


_TERMS = {
    "SUP-114": {
        "source_doc_id": "CON-101",
        "otif_below_pct": 94,
        "consecutive_months": 3,
        "notice_days": 60,
    }
}


@pytest.mark.parametrize(
    ("dotted", "good", "bad"),
    [
        ("tools.variant", "course", "all"),
        ("workflow.answerer", "rag", "mcp"),
        ("workflow.understand_prompt", "understand_system", "answer_system"),
        ("services.sse_ping_s", 5, 0),
        ("company.termination_terms", _TERMS, {"SUP-114": {**_TERMS["SUP-114"], "otif_below_pct": 0}}),  # noqa: E501
        (
            "company.termination_terms",
            _TERMS,
            {"SUP-114": {**_TERMS["SUP-114"], "consecutive_months": 0}},
        ),
        ("company.termination_terms", _TERMS, {"SUP-114": {**_TERMS["SUP-114"], "notice_days": -1}}),  # noqa: E501
    ],
)  # fmt: skip
def test_f4_config_validation(tmp_config: Path, patch_config, dotted, good, bad):
    from kovalo.config import ConfigError

    patch_config(tmp_config, dotted, good)
    _f1_settings(tmp_config)  # poprawna wartość się wczytuje (klucz istnieje)
    patch_config(tmp_config, dotted, bad)
    with pytest.raises(ConfigError):
        _f1_settings(tmp_config)


def test_f4_env_overrides(monkeypatch: pytest.MonkeyPatch):
    from kovalo.config import load_settings

    monkeypatch.setenv("KOVALO_WORKFLOW__ANSWERER", "rag")
    monkeypatch.setenv("KOVALO_TOOLS__VARIANT", "course")
    s = load_settings(google_api_key="test-key")
    assert (s.workflow.answerer, s.tools.variant) == ("rag", "course")


# ---------------------------------------------------------------------------
# F5 (kontrakt F5, 4 i 13.2)
# ---------------------------------------------------------------------------


def test_f5_config_keys_load():
    from kovalo.config import load_settings

    s = load_settings()
    assert (s.tools.variant, s.tools.source, s.tools.mcp_transport) == ("ours", "mcp", "http")
    assert s.mcp_server.search == "pgvector"
    assert (s.mcp.startup_timeout_s, s.mcp.call_timeout_s, s.mcp.cleanup_timeout_s) == (
        60,
        120,
        10,
    ), "sekcja mcp PINNED bez zmian"
    assert s.services.mcp_url == "http://localhost:8765/mcp"


def test_f5_config_env_overrides_and_validation(monkeypatch: pytest.MonkeyPatch):
    from kovalo.config import ConfigError, load_settings

    monkeypatch.setenv("KOVALO_TOOLS__SOURCE", "direct")
    monkeypatch.setenv("KOVALO_TOOLS__MCP_TRANSPORT", "stdio")
    monkeypatch.setenv("KOVALO_MCP_SERVER__SEARCH", "memory")
    s = load_settings()
    assert (s.tools.source, s.tools.mcp_transport, s.mcp_server.search) == (
        "direct",
        "stdio",
        "memory",
    )
    monkeypatch.setenv("KOVALO_TOOLS__SOURCE", "grpc")
    with pytest.raises(ConfigError, match=r"tools\.source"):
        load_settings()
    monkeypatch.setenv("KOVALO_TOOLS__SOURCE", "mcp")
    monkeypatch.setenv("KOVALO_MCP_SERVER__SEARCH", "bm25")
    with pytest.raises(ConfigError, match=r"mcp_server\.search"):
        load_settings()


def test_f5_config_yaml_text():
    """config.yaml ma nowe klucze (NASZE) z komentarzem, a sekcja `mcp` zostaje PINNED."""
    ours = _load(REPO_CONFIG)
    assert ours["tools"]["source"] == "mcp" and ours["tools"]["mcp_transport"] == "http"
    assert ours["mcp_server"] == {"search": "pgvector"}
    assert ours["mcp"] == _load(COURSE_CONFIG)["mcp"]


# ---------------------------------------------------------------------------
# F6: paths.reports_dir (kontrakt F6, 4); PINNED bez zmian
# ---------------------------------------------------------------------------


def test_reports_dir():
    from kovalo.config import load_settings

    s = load_settings()
    assert s.paths.reports_dir == s.root / "reports", "względna wobec root jak pozostałe ścieżki"
    assert s.paths.reports_dir.is_absolute()
    assert s.paths.mini_project_5_output_dir.parent == s.paths.reports_dir, (
        "katalogi raportów MP1-MP5 leżą w reports_dir"
    )
    assert _load(REPO_CONFIG)["paths"]["reports_dir"] == "reports", "klucz NASZ jest w config.yaml"
    assert "reports_dir" not in _load(COURSE_CONFIG)["paths"], "to nie jest klucz kursu"


def test_reports_dir_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from kovalo.config import load_settings

    monkeypatch.setenv("KOVALO_PATHS__REPORTS_DIR", str(tmp_path / "inne"))
    assert load_settings().paths.reports_dir == tmp_path / "inne"
