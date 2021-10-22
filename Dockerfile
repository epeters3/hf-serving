FROM python:3.9.7-slim-buster as base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.1.11 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv"
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

FROM base as builder-base
RUN apt-get update \
    && apt-get install --no-install-recommends -y curl build-essential
RUN curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python
WORKDIR $PYSETUP_PATH
COPY poetry.lock pyproject.toml ./
RUN poetry install --no-dev

FROM base as api
COPY --from=builder-base $PYSETUP_PATH $PYSETUP_PATH
RUN pip install torch==1.9.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
WORKDIR /app
COPY ./hf_serving ./hf_serving
ENV PORT=8000
CMD uvicorn hf_serving.__main__:app --host 0.0.0.0 --port $PORT