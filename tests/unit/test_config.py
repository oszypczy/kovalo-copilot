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
    assert cost.our_model == "gemini-3.6-flash"
    assert cost.our_prices_checked_on == datetime.date(2026, 11, 2)
    assert cost.our_prices_per_1m_tokens["gemini-3.6-flash"].input == 0.75
    assert cost.our_prices_per_1m_tokens["gemini-3.6-flash"].output == 3.75
    assert cost.our_prices_per_1m_tokens["gemini-3.5-flash-lite"].output == 2.5
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
        tmp_config, "cost.our_prices_per_1m_tokens.gemini-3.6-flash", {"input": -1, "output": 1}
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
