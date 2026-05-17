# RAG Time

## Ingestion Airflow

L'ingestion Airflow s'appuie sur deux fichiers Compose:

- `compose.yml` pour `qdrant` et l'API.
- `compose.airflow.yml` pour `postgres`, `airflow-webserver`, `airflow-scheduler` et `airflow-init`.

Lancement reproductible:

```bash
docker compose -f compose.yml -f compose.airflow.yml up --build
```

Services attendus:

- Airflow UI: `http://localhost:8080`
- Qdrant: `http://localhost:6333`
- Search API: `http://localhost:8000`

Le DAG `rag_ingestion_pipeline`:

- lit `data/aa_dataset-tickets-multi-lang-5-2-50-version.csv`
- génère `data/output.jsonl`
- recrée la collection Qdrant `support_tickets`
- pousse les chunks et embeddings dans Qdrant

Points importants:

- Les services Airflow et Qdrant partagent maintenant le réseau Docker `rag-time-net`.
- Le cache Hugging Face `hf_cache/` est monté dans Airflow pour éviter de retélécharger les modèles à chaque run.
- Le DAG n'est plus créé en pause.
- L'image Airflow embarque `tqdm` et `pydantic-settings`, nécessaires au script `scripts/embed_data_small.py`.
