FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt update && apt install -y --no-install-recommends \
    gcc build-essential \
    ca-certificates \
    python3.9-dev python3.9-distutils python3-pip python3.9-venv \
    vim wget tzdata curl \
 && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*
RUN python3.9 -m pip install --upgrade pip setuptools wheel \
    && ln -s $(which python3.9) /bin/python \
    && python3.9 -m pip install "poetry==1.5.0"

COPY pyproject.toml poetry.lock ./
# install poetry deps
RUN poetry export -f requirements.txt | python3.9 -m pip install -r /dev/stdin
# install llama-cpp-python compiled for GPU support
RUN python3.9 -m pip install 'llama-cpp-python[server]==0.1.77' \
    --prefer-binary --extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX/cu122

COPY llama2_server ./llama2_server

