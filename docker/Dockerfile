# syntax=docker/dockerfile:1.7
# Jeden obraz Pythona dla api i mcp-server (target `app`; różne `command` w compose,
# ten sam obraz uruchamia jednorazowo `kovalo ingest|eval|first-call`) oraz target `ui` (Chainlit).

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.9.18

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# --- builder: zależności (osobna warstwa, przebudowywana tylko po zmianie uv.lock) ---
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder
COPY --from=uv /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
# kod projektu (README.md jest wymagany przez metadane pakietu)
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# --- runtime: wspólna baza ---
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime
RUN groupadd --gid 1000 kovalo \
    && useradd --uid 1000 --gid kovalo --create-home --shell /usr/sbin/nologin kovalo
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    KOVALO_CONFIG_FILE=/app/config.yaml
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY config.yaml ./
# .cache: modele fastembed i cache embeddingów; reports: wyniki (w compose montowane jako wolumen)
RUN mkdir -p /app/.cache /app/reports && chown -R kovalo:kovalo /app
# TODO(F1): pobranie modelu fastembed przy budowaniu (po dodaniu fastembed do zależności),
#           np. RUN python -c "from fastembed import TextEmbedding; TextEmbedding('<model>', cache_dir='/app/.cache/fastembed')"
USER kovalo

# --- app: api, mcp-server, kovalo ingest|eval|first-call ---
FROM runtime AS app
EXPOSE 8080 8765
CMD ["kovalo-api"]

# --- ui: Chainlit ---
FROM runtime AS ui
EXPOSE 8000
CMD ["kovalo-ui"]
