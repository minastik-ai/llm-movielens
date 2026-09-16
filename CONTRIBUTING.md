# Contributing to LLM-MovieLens

We welcome contributions! Here's how to get involved.

## Adding a New Recommendation Model

1. Create a new file in `src/benchmark/models/your_model.py`
2. Implement a class with the standard interface:
   ```python
   class YourModel(nn.Module):
       def __init__(self, n_users, n_items, embedding_dim, ...):
           ...
       def forward(self, users, pos_items, neg_items):
           """Return (pos_scores, neg_scores) for BPR loss."""
           ...
       def predict(self, users):
           """Return (n_users, n_items) score matrix."""
           ...
   ```
3. Register it in `run_experiment.py`'s model factory
4. Add a config entry in `config.py` FEATURE_CONFIGS
5. Run: `python3 src/benchmark/run_experiment.py --config YOUR_CONFIG --seed 42`

## Adding a New Embedding Encoder

1. The embedding generator supports any `sentence-transformers` model:
   ```bash
   python3 src/embedding_generator/main.py --model your-model-name
   ```
2. Outputs are saved to `output/{model-name}-nopca/`
3. Set `EMBEDDING_DIR` to point to your new embeddings when running the benchmark

## Adding a New Feature Type

1. Generate your feature as a `.npy` file with shape `(10381, dim)`, aligned with `movie_id_index.json`
2. Add loading logic in `src/benchmark/features/loader.py`
3. Add a config entry mapping to your feature

## Code Style

- Python 3.10+
- Format with `ruff format`
- Lint with `ruff check`
- Type hints encouraged

```bash
ruff check .                          # Check style
ruff format .                         # Auto-format
python3 tools/verify_paper_numbers.py # Re-check the paper's numbers (CPU, seconds)
```

## Submitting Results

If you run experiments with a new model or feature, we welcome result submissions:

1. Run all 5 seeds (42, 123, 456, 789, 2026)
2. Report mean ± std for NDCG@{10,20,50}, Recall@{10,20,50}, HR@{10,20,50}, MRR
3. Open a PR or issue with your results and configuration
