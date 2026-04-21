FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV RAG_JSONL_PATH=data/aa_embeddings_hg.jsonl
ENV RAG_QUERY_MODEL=intfloat/multilingual-e5-small
ENV RAG_DEVICE=cpu

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip
RUN python -m pip install --no-cache-dir \
    numpy \
    "sentence-transformers>=3.0.1" \
    "streamlit>=1.45.1" \
    "langchain-huggingface>=1.2.1" \
    "torch>=2.9.0,<2.10.0" \
    "torchvision>=0.24.0,<0.25.0"

COPY app.py /app/app.py
COPY src /app/src
COPY data /app/data

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
