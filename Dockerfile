FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY src ./src

ARG MODEL_DIR
ENV MODEL_DIR=${MODEL_DIR}

ARG APP_PORT
ENV APP_PORT=${APP_PORT}

EXPOSE ${APP_PORT}

CMD uvicorn src.api.main:app --host 0.0.0.0 --port ${APP_PORT}
