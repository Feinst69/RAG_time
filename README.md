## Running the evaluation

### 1. Start Qdrant
```bash
docker start qdrant
```

### 2. Add IDs to the dataset CSV (first time only)
```bash
uv run python scripts/add_ids_to_csv.py -i data/aa_sample.csv
```
This adds a stable `id` column to the CSV. Existing IDs are never overwritten.

### 3. Configure eval pairs in `eval_config.yaml`
Look up the IDs of the relevant tickets in the CSV and fill in `q_&_a_pairs`:
```yaml
q_&_a_pairs:
  - question: "My phone won't start"
    relevant_ids:
      - "3f2a1b4c-0000-0000-0000-000000000001"
```

### 4. Run the evaluation
```bash
uv run python src/rag_time/eval/rag_eval.py
```
Results are saved to `eval_results.json`. The pipeline:
- Creates one Qdrant collection per (dense, sparse) model pair
- Evaluates every combination of dense × sparse × reranker × rephrasing × k × candidates
- Ranks combinations by best answer quality across LLM models
