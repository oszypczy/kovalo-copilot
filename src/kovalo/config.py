"""Ustawienia aplikacji: config.yaml (sekcje PINNED z kursu) + zmienne środowiskowe i .env.

Kontrakt: docs/contracts/f0.md, sekcja 4.3; klucze F1: docs/contracts/f1.md, sekcja 4.
Moduł nie importuje LangChaina (musi być lekki).
Priorytet źródeł (od najwyższego): argumenty load_settings → env → .env (katalog bieżący)
→ config.yaml → wartości domyślne.
"""

from __future__ import annotations

import datetime
import functools
import os
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import yaml
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class ConfigError(RuntimeError):
    """Zła albo brakująca wartość w config.yaml lub w zmiennych środowiskowych."""


class MissingConfigError(ConfigError):
    """Brak wymaganej zmiennej środowiskowej (np. klucza API)."""


ModelProvider = Literal["google_genai", "openai"]
ModelPurpose = Literal["default", "understand", "answer", "judge", "agent"]

CONFIG_FILE_ENV = "KOVALO_CONFIG_FILE"
_PACKAGE_REPO_ROOT = Path(__file__).resolve().parents[2]

# Plik YAML bieżącego ładowania (settings_customise_sources jest metodą klasy).
_yaml_file: ContextVar[Path | None] = ContextVar("kovalo_yaml_file", default=None)


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsConfig(_Section):
    corpus_dir: Path
    corpus_catalog: Path
    eval_queries: Path
    embedding_cache: Path
    mini_project_1_output_dir: Path
    mini_project_2_questions: Path
    mini_project_2_output_dir: Path
    mini_project_3_dialogues: Path
    mini_project_3_output_dir: Path
    company_data_dir: Path
    mini_project_4_questions: Path
    mini_project_4_output_dir: Path
    mini_project_5_conversations: Path
    mini_project_5_output_dir: Path


class CorpusConfig(_Section):
    required_metadata: list[str]


class TokenizerConfig(_Section):
    encoding: str


class ChunkingConfig(_Section):
    chunk_size: int
    chunk_overlap: int
    separators: list[str]


EmbeddingProvider = Literal["fastembed", "openai_cache"]


class EmbeddingModelSpec(_Section):
    """Model embeddingów fastembed z prefiksami pytań i dokumentów (ADR-0007)."""

    model: str = Field(min_length=1)
    dimensions: int = Field(gt=0)
    query_prefix: str = ""
    passage_prefix: str = ""


class LocalEmbeddingsConfig(EmbeddingModelSpec):
    threads: int | None = None
    local_files_only: bool = False
    models_dir: Path
    cache_dir: Path

    def spec(self) -> EmbeddingModelSpec:
        """Sam opis modelu (bez ścieżek i ustawień onnxruntime)."""
        return EmbeddingModelSpec(
            model=self.model,
            dimensions=self.dimensions,
            query_prefix=self.query_prefix,
            passage_prefix=self.passage_prefix,
        )


class EmbeddingsConfig(_Section):
    model: str
    dimensions: int
    batch_size: int = Field(gt=0)
    provider: EmbeddingProvider
    local: LocalEmbeddingsConfig
    candidates: list[EmbeddingModelSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_candidates(self) -> EmbeddingsConfig:
        names = [c.model for c in self.candidates]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"candidates: powtórzone modele: {', '.join(duplicates)}")
        local = self.local.spec()
        for candidate in self.candidates:
            if candidate.model == local.model and candidate != local:
                raise ValueError(
                    f"candidates: model {local.model} ma inny wymiar albo prefiksy niż "
                    "embeddings.local"
                )
        return self


class RetrievalConfig(_Section):
    top_k: int


class ReferenceLlmConfig(_Section):
    provider: str
    model: str


class RateLimitConfig(_Section):
    requests_per_second: float = Field(gt=0)
    check_every_n_seconds: float = 0.1
    max_bucket_size: float = 1


class ModelSpec(_Section):
    provider: ModelProvider
    model: str
    rate_limit: RateLimitConfig | None = None  # None → jak model główny


class PurposeOverride(_Section):
    model: str | None = None
    temperature: float | None = None


class LlmConfig(_Section):
    provider: ModelProvider
    model: str
    temperature: float | None
    max_retries: int = 2
    timeout_s: float = 60
    params: dict[str, Any] = Field(default_factory=dict)
    rate_limit: RateLimitConfig
    fallbacks: list[ModelSpec] = Field(default_factory=list)
    purposes: dict[ModelPurpose, PurposeOverride] = Field(default_factory=dict)


class TokenPrice(_Section):
    input: float = Field(ge=0)
    output: float = Field(ge=0)


class CostScenario(_Section):
    questions_per_day: int
    working_days_per_month: int
    output_tokens_per_answer: int


class CostConfig(_Section):
    reference_model: str
    prices_per_1m_tokens: dict[str, TokenPrice]
    scenario: CostScenario
    our_model: str
    our_prices_checked_on: datetime.date
    our_prices_per_1m_tokens: dict[str, TokenPrice]

    @model_validator(mode="after")
    def _check_models_have_prices(self) -> CostConfig:
        if self.reference_model not in self.prices_per_1m_tokens:
            raise ValueError(
                f"reference_model {self.reference_model!r} nie ma ceny w prices_per_1m_tokens"
            )
        if self.our_model not in self.our_prices_per_1m_tokens:
            raise ValueError(f"our_model {self.our_model!r} nie ma ceny w our_prices_per_1m_tokens")
        return self


class PolicyBand(_Section):
    up_to: int
    roles: list[str]


class PolicyException(_Section):
    cap: int
    roles: list[str]


class PolicyConfig(_Section):
    version: str
    bands: list[PolicyBand]
    exceptions: dict[str, PolicyException]
    margin_floor: int
    floor_role: str


class AgentConfig(_Section):
    max_steps: int
    timeout_s: int


class McpConfig(_Section):
    startup_timeout_s: int
    call_timeout_s: int
    cleanup_timeout_s: int


class PromptsConfig(_Section):
    system: str
    answer_system: str
    understand_system: str
    agent_system: str
    assistant_understand_system: str


# Maks. 53 znaki: replace_table dokleja "__new" i "__new_pkey", a Postgres obcina nazwy do 63.
_TABLE_NAME = r"^[a-z_][a-z0-9_]{0,52}$"


class DbConfig(_Section):
    host: str = "localhost"
    port: int = 5432
    connect_timeout_s: float = 5
    pool_min_size: int = 1
    pool_max_size: int = 5
    # nazwy tabel trafiają do SQL jako identyfikatory, więc tylko bezpieczne znaki
    vector_collection: str = Field(default="kovalo_ctx", pattern=_TABLE_NAME)
    vector_collection_mp1: str = Field(default="kovalo_mp1", pattern=_TABLE_NAME)
    index_meta_table: str = Field(default="kovalo_index_meta", pattern=_TABLE_NAME)


class ServicesConfig(_Section):
    api_port: int = 8080
    ui_port: int = 8000
    mcp_port: int = 8765
    api_url: str = "http://localhost:8080"
    mcp_url: str = "http://localhost:8765/mcp"
    health_timeout_s: float = 2


_API_KEY_ENV: dict[str, str] = {"google_genai": "GOOGLE_API_KEY", "openai": "OPENAI_API_KEY"}


def api_key_env_name(provider: ModelProvider) -> str:
    """Nazwa zmiennej środowiskowej z kluczem danego dostawcy."""
    return _API_KEY_ENV[provider]


def _resolve(root: Path, value: Any) -> Any:
    if isinstance(value, str | Path):
        path = Path(value)
        return path if path.is_absolute() else root / path
    return value


class Settings(BaseSettings):
    """Pełna konfiguracja procesu. Sekrety jako SecretStr (nie trafiają do repr ani logów)."""

    model_config = SettingsConfigDict(
        env_prefix="KOVALO_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        # obce zmienne w .env (API_PORT itp.) są ignorowane; nieznane klucze główne
        # config.yaml odrzuca _check_yaml_file, a sekcje mają własne extra="forbid"
        extra="ignore",
        populate_by_name=True,
        validate_default=True,
    )

    # z config.yaml
    paths: PathsConfig
    corpus: CorpusConfig
    tokenizer: TokenizerConfig
    chunking: ChunkingConfig
    embeddings: EmbeddingsConfig
    retrieval: RetrievalConfig
    reference_llm: ReferenceLlmConfig
    llm: LlmConfig
    cost: CostConfig
    policy: PolicyConfig
    agent: AgentConfig
    mcp: McpConfig
    prompts: PromptsConfig
    db: DbConfig = DbConfig()
    services: ServicesConfig = ServicesConfig()
    root: Path

    # sekrety i zmienne z env / .env (nazwy bez prefiksu, jak w .env.example)
    google_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("google_api_key", "GOOGLE_API_KEY", "GEMINI_API_KEY"),
    )
    openai_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("openai_api_key", "OPENAI_API_KEY")
    )
    langsmith_tracing: bool = Field(
        default=False, validation_alias=AliasChoices("langsmith_tracing", "LANGSMITH_TRACING")
    )
    langsmith_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("langsmith_api_key", "LANGSMITH_API_KEY")
    )
    langsmith_project: str = Field(
        default="kovalo-copilot",
        validation_alias=AliasChoices("langsmith_project", "LANGSMITH_PROJECT"),
    )
    postgres_user: str = Field(
        default="kovalo", validation_alias=AliasChoices("postgres_user", "POSTGRES_USER")
    )
    postgres_password: SecretStr = Field(
        default=SecretStr("kovalo"),
        validation_alias=AliasChoices("postgres_password", "POSTGRES_PASSWORD"),
    )
    postgres_db: str = Field(
        default="kovalo", validation_alias=AliasChoices("postgres_db", "POSTGRES_DB")
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_file = _yaml_file.get()
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings, dotenv_settings]
        if yaml_file is not None:
            sources.append(
                YamlConfigSettingsSource(
                    settings_cls, yaml_file=yaml_file, yaml_file_encoding="utf-8"
                )
            )
        return tuple(sources)

    @field_validator("google_api_key", "openai_api_key", "langsmith_api_key", mode="before")
    @classmethod
    def _empty_secret_is_none(cls, value: Any) -> Any:
        if isinstance(value, SecretStr):
            return value if value.get_secret_value() else None
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="before")
    @classmethod
    def _resolve_relative_paths(cls, data: Any) -> Any:
        """Ścieżki z configu są względne wobec katalogu config.yaml (jak w kursie)."""
        if not isinstance(data, dict) or "root" not in data:
            return data
        root = Path(data["root"])
        paths = data.get("paths")
        if isinstance(paths, dict):
            data["paths"] = {k: _resolve(root, v) for k, v in paths.items()}
        embeddings = data.get("embeddings")
        if isinstance(embeddings, dict) and isinstance(embeddings.get("local"), dict):
            local = dict(embeddings["local"])
            for key in ("models_dir", "cache_dir"):
                if key in local:
                    local[key] = _resolve(root, local[key])
            data["embeddings"] = {**embeddings, "local": local}
        return data

    def api_key_for(self, provider: ModelProvider) -> SecretStr | None:
        """Klucz API dostawcy modeli albo None, gdy nie ustawiony."""
        if provider == "google_genai":
            return self.google_api_key
        if provider == "openai":
            return self.openai_api_key
        raise ConfigError(f"Nieznany dostawca modeli: {provider}")

    def _dsn(self, scheme: str) -> str:
        user = quote(self.postgres_user, safe="")
        password = quote(self.postgres_password.get_secret_value(), safe="")
        return (
            f"{scheme}://{user}:{password}@{self.db.host}:{self.db.port}/"
            f"{quote(self.postgres_db, safe='')}"
        )

    def postgres_dsn(self) -> str:
        """DSN dla psycopg i PostgresSaver: postgresql://user:pass@host:port/db."""
        return self._dsn("postgresql")

    def sqlalchemy_url(self) -> str:
        """URL dla langchain-postgres: postgresql+psycopg://user:pass@host:port/db."""
        return self._dsn("postgresql+psycopg")


def default_config_file() -> Path:
    """KOVALO_CONFIG_FILE, potem config.yaml w katalogu głównym repo, potem w katalogu bieżącym."""
    from_env = os.environ.get(CONFIG_FILE_ENV)
    if from_env:
        return Path(from_env)
    repo_file = _PACKAGE_REPO_ROOT / "config.yaml"
    if repo_file.is_file():
        return repo_file
    return Path.cwd() / "config.yaml"  # w obrazie: WORKDIR /app → /app/config.yaml


def _check_yaml_file(path: Path) -> None:
    if not path.is_file():
        raise ConfigError(f"Nie znaleziono pliku konfiguracji: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Niepoprawny YAML w {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Plik konfiguracji {path} musi zawierać mapowanie (klucz: wartość).")
    if "root" in data:
        raise ConfigError(f"Klucz 'root' w {path} jest niedozwolony (ustawiany przy ładowaniu).")
    unknown = sorted(str(key) for key in data if key not in Settings.model_fields)
    if unknown:
        raise ConfigError(f"Nieznane klucze główne w {path}: {', '.join(unknown)}")


def _check_overrides(overrides: dict[str, Any]) -> None:
    """Settings ma extra="ignore" (obce zmienne w .env), więc literówki w argumentach łapiemy tu."""
    allowed: set[str] = set()
    for name, field in Settings.model_fields.items():
        allowed.add(name)
        if isinstance(field.alias, str):
            allowed.add(field.alias)
        choices = getattr(field.validation_alias, "choices", None)
        if choices:
            allowed.update(c for c in choices if isinstance(c, str))
        elif isinstance(field.validation_alias, str):
            allowed.add(field.validation_alias)
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ConfigError(f"Nieznane argumenty load_settings: {', '.join(unknown)}")


def load_settings(config_file: Path | None = None, **overrides: Any) -> Settings:
    """Ładuje ustawienia bez cache (testy, CLI). Błędy: ConfigError z kluczem i plikiem."""
    path = Path(config_file) if config_file is not None else default_config_file()
    path = path.expanduser().resolve()
    _check_yaml_file(path)
    _check_overrides(overrides)
    token = _yaml_file.set(path)
    try:
        return Settings(root=path.parent, **overrides)
    except ValidationError as exc:
        raise ConfigError(f"Błędna konfiguracja (plik {path}):\n{exc}") from exc
    finally:
        _yaml_file.reset(token)


@functools.cache
def get_settings() -> Settings:
    """Jedna instancja ustawień na proces; testy czyszczą przez get_settings.cache_clear()."""
    return load_settings()
