# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# 🍵 Matxa-TTS v2 (langtech-bsc Matcha-TTS fork, v2-graphemes branch) + 🥑 alVoCat
# served over HTTP with FastAPI.
#
# Build (CPU, default):
#   docker build -t matcha-tts-api .
# Build with the model weights baked into the image:
#   docker build --build-arg PREFETCH_MODELS="matxa_v2_multiaccent" -t matcha-tts-api .
# Build for CUDA:
#   docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 \
#                --build-arg BASE_IMAGE=python:3.11-slim -t matcha-tts-api:cuda .
# ---------------------------------------------------------------------------

ARG BASE_IMAGE=python:3.11-slim

# --------------------------------------------------------------------- base --
FROM ${BASE_IMAGE} AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MATCHA_HOME=/models \
    HF_HOME=/models/hf \
    PATH=/opt/venv/bin:$PATH

# libsndfile1: soundfile (wav/flac/ogg encoding)
# libgomp1: torchaudio (imported by vocos)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libsndfile1 \
      libgomp1 \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------ builder --
FROM base AS builder

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential git \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv \
 && pip install --upgrade pip wheel "setuptools==69.5.1"

# torch first, from the PyTorch index, so CPU-only wheels are used instead of
# the multi-gigabyte CUDA build pulled in from PyPI.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --index-url ${TORCH_INDEX_URL} --extra-index-url https://pypi.org/simple torch==2.2.2 torchaudio==2.2.2

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Matcha-TTS itself, --no-deps because its requirements.txt is training-oriented
# (gradio 3.43 / notebook / optuna) and conflicts with the API stack above.
# The sdist builds a Cython extension (matcha.utils.monotonic_align.core).
ARG MATCHA_REPO=https://github.com/langtech-bsc/Matcha-TTS.git
# head of the v2-graphemes branch; it also ships the vocos package it decodes with
ARG MATCHA_REF=2e15c908f0acce72d4a8f0c0f8627c54fbe62fb2
RUN pip install --no-deps "matcha-tts @ git+${MATCHA_REPO}@${MATCHA_REF}"

# Fail the build here rather than at the first request if the Cython extension
# or the text front-end are not usable.
# (python -c rather than a heredoc: heredocs need BuildKit, which Cloud Build's
# docker builder does not use.)
RUN python -c 'from matcha.models.matcha_tts import MatchaTTS; \
from matcha.text import sequence_to_text, text_to_sequence; \
from vocos import Vocos; \
print("graphemes ->", sequence_to_text(text_to_sequence("L'\''àvia Pepeta col·lecciona cançons.", ["basic_cleaners"])))'

# ------------------------------------------------------------------ runtime --
FROM base AS runtime

COPY --from=builder /opt/venv /opt/venv

WORKDIR /srv
COPY app ./app
COPY scripts ./scripts

RUN useradd --create-home --uid 1000 matcha \
 && mkdir -p ${MATCHA_HOME}/custom \
 && chown -R matcha:matcha ${MATCHA_HOME} /srv
USER matcha

# Optionally bake checkpoints in: space-separated model names, or "all".
# Note: these live in /models, so mounting a *bind* mount there hides them;
# a named volume (as in docker-compose.yml) keeps them.
ARG PREFETCH_MODELS="matxa_v2_multiaccent"
RUN if [ -n "${PREFETCH_MODELS}" ]; then python scripts/prefetch_models.py ${PREFETCH_MODELS}; fi

ENV MATCHA_DEFAULT_MODEL=matxa_v2_multiaccent \
    MATCHA_DEVICE=auto \
    MATCHA_PRELOAD=1 \
    MATCHA_MAX_CONCURRENCY=1 \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
  CMD python -c "import urllib.request,os;urllib.request.urlopen(f\"http://127.0.0.1:{os.environ.get('PORT','8000')}/healthz\").read()"

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
