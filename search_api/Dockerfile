FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential vim iputils-ping git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

RUN pip install -U pip
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.9.0

# Copy the local project so the container runs the same code as the workspace.
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY search_api ./search_api

RUN pip install --no-cache-dir .

WORKDIR /app/search_api

EXPOSE 8000

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
