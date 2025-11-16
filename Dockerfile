FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app app
COPY configs configs
COPY models models
COPY scripts scripts

ENTRYPOINT ["python"]
CMD ["-m", "app.cli", "--config", "configs/sample-config.yaml"]
