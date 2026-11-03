# syntax=docker/dockerfile:1.7
# Jeden obraz Pythona dla api i mcp-server (target `app`; różne `command` w compose,
# ten sam obraz uruchamia jednorazowo `kovalo ingest|eval|first-call`) oraz target `ui` (Chainlit).

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.9.18

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# --- deps: zależności (osobna warstwa, przebudowywana tylko po zmianie uv.lock) ---
FROM python:${PYTHON_VERSION}-slim-bookworm AS deps
COPY --from=uv /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# --- models: model embeddingów i tokenizer (z deps, nie zależy od src/: zmiana kodu nie pobiera ich ponownie) ---
FROM deps AS models
# musi równać się embeddings.local.model w config.yaml (pilnuje tego tests/unit/test_docker_config.py)
ARG FASTEMBED_MODEL=intfloat/multilingual-e5-large
RUN /opt/venv/bin/python -c "from fastembed import TextEmbedding as T; T('${FASTEMBED_MODEL}', cache_dir='/models/fastembed')" \
    && TIKTOKEN_CACHE_DIR=/models/tiktoken /opt/venv/bin/python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"

# --- builder: kod projektu ---
FROM deps AS builder
# kod projektu (README.md jest wymagany przez metadane pakietu)
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# --- runtime: wspólna baza ---
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime-base
RUN groupadd --gid 1000 kovalo \
    && useradd --uid 1000 --gid kovalo --create-home --shell /usr/sbin/nologin kovalo
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    KOVALO_CONFIG_FILE=/app/config.yaml \
    TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken \
    HF_HUB_DISABLE_TELEMETRY=1
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY config.yaml ./
# embeddings: cache wektorów, llm: cache odpowiedzi LLM-a w ewaluacjach (wolumeny w compose); reports: wyniki (wolumen/bind mount)
RUN mkdir -p /app/.cache/embeddings /app/.cache/llm /app/reports && chown -R kovalo:kovalo /app
USER kovalo

# --- app: api, mcp-server, kovalo ingest|eval|first-call ---
FROM runtime-base AS app
COPY --from=models --chown=kovalo:kovalo /models/fastembed /app/.cache/fastembed
COPY --from=models --chown=kovalo:kovalo /models/tiktoken /app/.cache/tiktoken
EXPOSE 8080 8765
CMD ["kovalo-api"]

# --- ui: Chainlit ---
FROM runtime-base AS ui
# konfiguracja Chainlita (language = "pl-PL") i polskie tłumaczenie UI; Chainlit czyta
# .chainlit/ z katalogu roboczego (/app), brakujące tłumaczenia dogeneruje przy starcie
# (stąd katalog tworzony przez użytkownika kovalo, z prawem zapisu)
RUN mkdir -p /app/.chainlit/translations
COPY --chown=kovalo:kovalo .chainlit/config.toml ./.chainlit/config.toml
COPY --chown=kovalo:kovalo .chainlit/translations/pl-PL.json ./.chainlit/translations/pl-PL.json
EXPOSE 8000
CMD ["kovalo-ui"]
