FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential vim iputils-ping git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

RUN pip install -U pip
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.9.0
RUN pip install --no-cache-dir git+https://github.com/Feinst69/RAG_time.git@qdrant_rag
COPY search_api/main.py .

EXPOSE 8000

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
