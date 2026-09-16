FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY src ./src

ENV MODEL_DIR=""
ENV APP_PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn src.api.main:app --host 0.0.0.0 --port ${APP_PORT}"]
