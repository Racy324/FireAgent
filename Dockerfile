FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN python -m pip install --upgrade pip

COPY pyproject.toml README.md ./
COPY configs ./configs
COPY fireagent ./fireagent
COPY scripts ./scripts

RUN python -m pip install -e .

EXPOSE 8000

CMD ["uvicorn", "fireagent.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
