# Recommendation Benchmark: LLM-Generated Features for MovieLens 20M

A rigorous benchmark for evaluating LLM-generated movie embeddings as item side features in collaborative filtering. Designed for reproducibility: a fixed temporal split, five seeds, full ranking, and paired significance tests, so a new feature can be compared against the same baselines under the same protocol.

## Research Question

> Does having an LLM read genome tags, plot metadata, aggregates and user tags, and write a profile, produce a better item representation for recommendation than (a) pure collaborative filtering, (b) the genome tag vectors that ship with ML-20M, or (c) the same sentence encoder applied to titles?

## Dataset

**MovieLens 20M** (Harper & Konstan, 2015) — 20 million ratings from 138,493 users on 27,278 movies. We restrict to the **10,381 genome-covered movies** (99% of ratings preserved), which are the movies that have both LLM-generated features and genome tag annotations.

### LLM-Generated Features (from `embedding_generator/`)

| Feature | Dims (default) | Source | Description |
|---------|---------------|--------|-------------|
| **Profile embedding** | 1024 | LLM profile text → bge-large-en-v1.5 | Semantic fingerprint of each movie's thematic essence, tone, and style |
| **Mood vector** | 10 | Direct extraction from LLM output | 10-axis continuous mood (dark/light, serious/playful, slow/fast, etc.) |
| **Key themes** | 528 | Multi-hot over filtered LLM themes | Categorical theme labels (redemption, obsession, identity, ...) |
| **Genome PCA** | 128 | Raw 1128-dim genome scores → PCA | Baseline: existing ML-20M content features |
| **Title embedding** | 1024 | "Title \| Genres" → bge-large-en-v1.5 | Control: the same encoder without LLM synthesis |

## Baseline Models

### Why These Models?

We select baselines to isolate the contribution of LLM features:

1. **Pure CF baselines** — establish what collaborative filtering achieves alone, without any content features. Includes classic (BPR-MF), graph-based (LightGCN), and contrastive learning (SimGCL, XSimGCL, LightGCL) approaches.

2. **Content-augmented CF** — the primary ablation host. LightGCN-SF injects different feature types into the same architecture, enabling controlled comparison: genome vs. title embedding vs. LLM profile vs. mood vs. themes vs. combinations.

The selection forms a **controlled hierarchy**, the same one the paper's configuration
table describes:

| # | Comparison | Research Question |
|---|-----------|-------------------|
| Q1 | M4 vs M1/M1c | Does LLM content improve over pure CF (including strong contrastive methods)? |
| Q2 | M4 vs M2/M2b | Does LLM-synthesized content beat genome tags, regardless of dimensionality? **(Central claim)** |
| Q3 | M4 vs M3 | Does LLM synthesis matter, or does the same encoder applied to titles suffice? |
| Q4 | M7 vs M4 | Do structured mood features add value beyond semantic embeddings? |

Supporting comparisons (not primary research questions, but informative):

| Comparison | What It Tests |
|---|---|
| M0 (BPR-MF) vs M1 (LightGCN) | Does graph structure help? (sanity check) |
| M1 vs M1b/M1c/M1d | Does a contrastive objective improve pure CF on this split? |
| M7 vs M8 | Do categorical themes further help, or add noise? |
| M2 vs M9 | Do structured LLM features complement genome tags? |

For a **resource paper**, the goal is not to propose a new model — it is to demonstrate that the LLM-generated features (profiles, moods, themes) are a **useful resource**. We need baselines at multiple strength levels so that improvements cannot be dismissed as "you just used a weak baseline." On this split the strongest purely collaborative model is LightGCN (M1); the three contrastive objectives, each at its own settings selected on validation, score below it. LightGCN-SF is the controlled testbed that keeps architecture constant while varying only the features.

### Tier 1: Pure Collaborative Filtering

#### BPR-MF (Rendle et al., UAI 2009)

**Algorithm.** Each user *u* and item *i* gets a learned embedding vector (**e**_u, **e**_i ∈ ℝ^128). The predicted preference score is a dot product:

```
score(u, i) = eᵤ · eᵢ
```

Training uses **Bayesian Personalized Ranking**: for each observed interaction (u, i⁺), sample an unobserved item i⁻, then maximize the margin between observed and unobserved items:

```
L_BPR = −log σ(score(u, i⁺) − score(u, i⁻))  +  λ‖embeddings‖²
```

This learns to rank observed items above unobserved ones — no pointwise rating prediction, just pairwise ordering. No content features, no graph structure — purely learned from interaction patterns.

**Why included.** The simplest CF baseline. Isolates the value of collaborative signal alone with no graph structure and no content features. If LightGCN beats BPR-MF, the graph structure helps. If LightGCN-SF beats both, the side features help *on top of* graph structure. Also serves as a sanity check that the evaluation pipeline is correct (NDCG@10 0.1137 on this split).

#### LightGCN (He et al., SIGIR 2020)

**Algorithm.** Constructs a bipartite user-item interaction graph and computes a **normalized adjacency matrix** Â. User/item embeddings are iteratively propagated through *L* = 3 layers of graph convolution:

```
e⁽ˡ⁾ = Â · e⁽ˡ⁻¹⁾       (neighborhood averaging — no weight matrices, no nonlinearities)
```

The key simplification over standard GCNs: **no learned weight matrices, no nonlinear activations** at each layer. The final representation is the **mean pool** across all layers (including layer 0):

```
e_final = (1/(L+1)) · Σₗ₌₀ᴸ e⁽ˡ⁾
```

This captures multi-hop collaborative signals: layer 1 aggregates direct neighbors (users who rated the same item), layer 2 captures 2-hop patterns (items rated by similar users), layer 3 extends to 3-hop neighborhoods. BPR loss is applied on the final embeddings, with L2 regularization on the **initial** (non-propagated) embeddings only.

**Why included.** The standard GNN CF baseline expected at all top venues. Consistently strong on ML-20M. Our LightGCN-SF extends this architecture with side features, making LightGCN the natural ID-only comparison point.

#### SimGCL (Yu et al., SIGIR 2022)

**Algorithm.** Builds on LightGCN by adding **contrastive self-supervised learning** as regularization. The key insight: you don't need expensive graph augmentation (edge dropout, node dropout) — adding **random noise** to embeddings creates effective contrastive views. The implementation follows the authors' own code (SELFRec):

- The final embedding is the mean of propagation layers 1..L (the input embedding is not part of it).
- A perturbed pass adds, after each layer, noise with norm ε that keeps each coordinate's sign:
  ```
  e' = e + sign(e) · ε · normalize(U(0, 1))
  ```
- The ranking (BPR) loss uses the clean pass. The contrastive loss is InfoNCE between **two independently perturbed passes**, over the batch's unique users and unique positive items, with in-batch negatives, at temperature τ:

```
L = L_BPR  +  wd · L_reg  +  λ · (L_CL_user + L_CL_item)
```

The contrastive term forces the model to learn representations that are **invariant to small perturbations**, which improves embedding uniformity.

**Settings used here:** λ = 0.1, ε = 0.1, τ = 0.2, Xavier-uniform initialisation, selected on validation (`contrastive_settings.json`, applied by `run_experiment.py`).

**Why included.** Represents the **contrastive CF paradigm**: it tests whether self-supervised contrastive learning, without content features, can match content-augmented LightGCN-SF.

#### XSimGCL (Yu et al., TKDE 2023)

**Algorithm.** Simplifies SimGCL to **one perturbed pass** that serves both losses, and replaces same-layer contrast with **cross-layer** contrast:

- Each layer's propagation is followed by the same sign-preserving noise; the noisy embedding is what the next layer propagates.
- The ranking (BPR) loss uses the final embedding (mean of layers 1..L) of that perturbed pass.
- The contrastive loss is InfoNCE between the final embedding and the embedding after layer l* of the **same** pass, over the batch's unique users and unique positive items:

```
L_CL = InfoNCE(final[batch], layer_l*[batch])
```

- Prediction uses a clean (unperturbed) pass.

**Settings used here:** λ = 0.02, ε = 0.1, τ = 0.2, l* = 1, Xavier-uniform initialisation, selected on validation (`contrastive_settings.json`).

**Why included.** The cross-layer variant of the contrastive paradigm, cheaper than SimGCL (one pass instead of three).

#### LightGCL (Cai et al., ICLR 2023)

**Algorithm.** Replaces the random noise perturbation used by SimGCL/XSimGCL with **SVD-based augmentation**. A truncated SVD (rank *q* = 5) of the normalised user-item matrix gives the contrastive view; the implementation follows the authors' code (HKUDS/LightGCL):

1. **Main view**: LightGCN propagation on the normalised graph.
2. **Contrastive view**: its layer *l* is built from the **main view's** layer *l*−1 through the rank-*q* factors: `G_u^l = U S Vᵀ E_i^(l−1)`, `G_i^l = V S Uᵀ E_u^(l−1)`.
3. Both views are **summed** over layers 0..L.

```
L = L_BPR + wd · L_reg + λ₁ · (InfoNCE(G_u, E_u) + InfoNCE(G_i, E_i))
```

The InfoNCE is the authors' (no cosine normalisation), with every user (for users) or every item (for items) as a negative.

**Settings used here:** λ₁ = 0.2, τ = 1.0, q = 5, Xavier-uniform initialisation, τ selected on validation (`contrastive_settings.json`).

**Why included.** The SVD-augmentation member of the contrastive family: a structurally motivated view rather than random noise. If LLM features (M4, M7) improve on every contrastive objective as well as on LightGCN, content adds information that self-supervision over the interaction graph does not recover.

### Tier 2: Content-Augmented CF (Ablation Host)

#### LightGCN-SF (LightGCN with Side Features)

**Algorithm.** Extends LightGCN by injecting pre-computed content features into item embeddings **before** graph propagation, so content information flows through the graph alongside collaborative signals.

**Full forward pass (example: M4 with 1024-dim bge-large profile embeddings):**

```
Constructor: LightGCNSF(n_users=127371, n_items=9906, embed_dim=128, n_layers=3, feature_dim=1024)

Learnable parameters:
  user_emb:     Embedding(127371, 128)               — learned from interactions
  item_emb:     Embedding(9906, 128)                  — learned from interactions
  feature_proj: Linear(1024→128) → ReLU → Linear(128→128)  — learned projection

Step 1: Project content features into embedding space
  content_features (9906, 1024)
      → Linear(1024 → 128)     # compress to embedding space
      → ReLU                    # nonlinear transformation
      → Linear(128 → 128)      # refine
      = projected (9906, 128)

Step 2: Additive injection
  item_repr = item_emb(128) + projected(128)
  = (9906, 128)

Step 3: LightGCN graph propagation (3 layers)
  all_emb = [user_emb(127371, 128) ; item_repr(9906, 128)]   # concat users + items
  = (137277, 128)

  Layer 0: all_emb              (initial embeddings)
  Layer 1: Â · all_emb          (1-hop neighborhood averaging)
  Layer 2: Â · layer1           (2-hop patterns)
  Layer 3: Â · layer2           (3-hop patterns)

  final = mean(layer0, layer1, layer2, layer3)   # (137277, 128)

Step 4: BPR scoring
  user_final = final[:127371]    # (127371, 128)
  item_final = final[127371:]    # (9906, 128)
  score(u, i) = user_final[u] · item_final[i]    # dot product
```

The MLP compresses arbitrary-dimensional features (10-dim mood, 128-dim genome, 1024-dim bge-large) into the fixed 128-dim embedding space. Everything after Step 1 is identical regardless of input feature dimensionality — the graph propagation always operates in 128-dim space.

**Why this injection strategy:**
- **Addition** (not concatenation) keeps the item embedding dimension fixed at *d*=128, avoiding dimensionality confounds across experiments. A 1024-dim input and a 10-dim input both produce the same 128-dim output after the MLP.
- The **2-layer MLP with ReLU** allows nonlinear transformation — a single linear layer would be a linear projection that may not capture complex relationships between feature dimensions.
- Features are injected **before propagation** (not after) so content information can propagate to neighbors — a user's representation benefits from the content features of all items they interacted with. After 3 layers, content signal has reached 3-hop neighborhoods.

**Why included.** The **primary ablation host** and the most critical model in this benchmark. By varying only the content features while keeping the model architecture identical, we isolate the effect of each feature type. This is the core experimental design that makes the resource paper's contribution clear.


## Experimental Settings

### Data Preprocessing

| Step | Result | Rationale |
|------|--------|-----------|
| Genome filter | the 10,381 genome-covered films | Only genome-covered movies have LLM features |
| Implicit conversion (rating ≥ 3.5) | 19.8M → 12.1M interactions | Ratings below 3.5 are not positive signals — a user rating 1.5 doesn't mean "recommend this." We keep only positive interactions and sample negatives during training. |
| 10-core filtering | 12.1M → 12.05M, 127,371 users, 9,906 items | Remove users and items with fewer than 10 interactions iteratively until convergence. Standard practice to ensure sufficient collaborative signal. |
| Temporal split | Train: 11.5M (99.0%) / Val: 50K (0.4%) / Test: 67K (0.6%) | Split by timestamp, not randomly (see below). |

### Why Temporal Split (Not Random)

Random splits leak future information into training. If a user rates Movie C in 2014 (influenced by having seen Movies A and B in 2012), a random split might place Movie C in training and Movie A in test — the model trains on a preference shaped by experiences it hasn't "seen."

Temporal split prevents this by enforcing chronological order:

```
Train: all interactions before 2014-01-01     (99.0%,  11,499,778)
Val:   interactions 2014-01-01 to 2014-07-01  ( 0.4%,      49,668)
Test:  interactions after 2014-07-01          ( 0.6%,      67,466)
```

This mirrors real deployment (train on history, recommend for the future), and it keeps information from after the test period out of training.

The heavily skewed split reflects two factors: (1) ML-20M ratings are concentrated in earlier years (most activity is pre-2014), and (2) after remapping, val/test users must also appear in train, which further reduces val/test size. This is standard for temporal splits on ML-20M — the val/test sets still contain 50K/67K interactions, which is sufficient for reliable metric estimation across 127K users.

### Evaluation Protocol: Full Ranking

For each test user:
1. Compute predicted scores for **all 9,906 items**
2. Mask items the user interacted with during training (no re-recommendation)
3. Rank all remaining items by predicted score
4. Compute metrics on the full ranking

**Why full ranking, not sampled negatives?** Krichene & Rendle (KDD 2020) proved that sampled evaluation (e.g., scoring against 100 random negatives) produces inconsistent model orderings — Model A can beat Model B under sampling but lose under full ranking. The "winner" depends on the sample, not the model quality. With only 9,906 items, full ranking is computationally trivial (one matrix multiply per user batch).

### Metrics

| Metric | Formula | What It Measures |
|--------|---------|------------------|
| **NDCG@K** | Normalized Discounted Cumulative Gain | Ranking quality, position-aware. Higher-ranked relevant items contribute more. **Headline metric.** |
| **Recall@K** | \|relevant ∩ top-K\| / \|relevant\| | What fraction of relevant items appear in top-K. Measures coverage of user interests. |
| **HR@K** | 1 if any relevant item in top-K, else 0 | Binary: did the model surface anything useful in top-K? Most lenient metric. |
| **MRR** | 1 / rank_of_first_relevant_item, over the full ranking | How quickly the model finds the first relevant item. Sensitive to top-1 performance. |

Reported at K = {10, 20, 50}. All metrics averaged across test users.

### Statistical Rigor

- **5 random seeds** (42, 123, 456, 789, 2026) controlling parameter initialization and negative sampling
- Report **mean ± standard deviation** for all metrics
- **Paired t-test** over the five seeds, two-sided, for every comparison; the paper's four confirmatory hypotheses are corrected with Holm–Bonferroni (`scripts/analyze_contrastive.py`)
- Significance threshold: p < 0.05
- One shared training protocol for every configuration (learning rate, weight decay, batch, epoch budget, patience). The three contrastive baselines' own hyperparameters are selected on validation NDCG@10 at seed 42 and frozen before any reported run (`contrastive_settings.json`)

### Cold-Start Analysis

Content features are most valuable when collaborative signals are sparse. We report metrics separately by item training interaction count:

| Bucket | Items | Train Interactions |
|--------|-------|--------------------|
| Cold | 112 | < 10 |
| Medium | 1,872 | 10 - 49 |
| Warm | 7,922 | ≥ 50 |

We expect LLM features to show the largest lift on cold and medium items, where pure CF has insufficient signal.

## Complete Model Reference

Every model configuration in this benchmark, with full architectural details.

### Tier 1: Pure Collaborative Filtering (no content features)

These models use **only** interaction data (user-item pairs). No `.npy` embedding files are loaded.

| ID | Model | Architecture | Graph? | Contrastive? | Embed Dim | Key Mechanism |
|----|-------|-------------|--------|-------------|-----------|---------------|
| M0 | BPR-MF | Matrix factorization | No | No | 128 | `score = user_emb · item_emb`, BPR pairwise loss |
| M1 | LightGCN | Graph convolution | Yes (3 layers) | No | 128 | `final = mean(layer0..layer3)`, neighborhood averaging |
| M1b | SimGCL | LightGCN + noise | Yes (3 layers) | Yes (two noisy passes) | 128 | InfoNCE(noisy_final, noisy_final′), λ=0.1, ε=0.1 |
| M1c | XSimGCL | LightGCN + noise | Yes (3 layers) | Yes (cross-layer) | 128 | InfoNCE(noisy_final, noisy_layer1), λ=0.02, ε=0.1 |
| M1d | LightGCL | LightGCN + SVD | Yes (3 layers) | Yes (SVD view) | 128 | InfoNCE(main_view, svd_view), λ₁=0.2, τ=1.0, q=5 |

**No content features. No MLP. No `.npy` files loaded.** These establish what CF alone can achieve.

### Tier 2: LightGCN-SF — Content-Augmented (primary ablation)

All Tier 2 models use the **identical LightGCN-SF architecture**. Only the input features change.

```
Architecture (same for all):
  item_repr = item_emb(128) + MLP(features → 128)
  MLP = Linear(feature_dim → 128) → ReLU → Linear(128 → 128)
  Then: 3 layers of LightGCN graph propagation on the augmented embeddings
```

| ID | Side Features | Feature Source | Input Dim | After MLP | Purpose |
|----|--------------|----------------|-----------|-----------|---------|
| M2 | Genome PCA | `genome_embeddings.npy` | 128 | 128 | Existing content baseline |
| M2b | Genome raw | built from `genome-scores.csv` | 1128 | 128 | Dimensionality control |
| M3 | Title + genres | `bert_title_embeddings.npy` | 1024 | 128 | Same-encoder text control |
| **M4** | **LLM profile** | **`profile_embeddings.npy`** | **1024** | **128** | **Core contribution** |
| M5 | LLM mood | `mood_vectors.npy` | 10 | 128 | Structured signal only |
| M6 | LLM themes | `theme_matrix.npy` | 528 | 128 | Categorical themes only |
| **M7** | **LLM profile + mood** | **profile + mood concatenated** | **1034** | **128** | **Recommended combo** |
| M8 | LLM all | profile + mood + themes | 1562 | 128 | Full LLM feature set (sensitivity; not in the paper's table) |
| M9 | Genome + mood + themes | genome + mood + themes | 666 | 128 | Genome + structured LLM (sensitivity; not in the paper's table) |

Default dimensions shown for bge-large-en-v1.5 (1024-dim, no PCA). With `--pca-dims 128`: M3/M4=128, M7=138, M8=666. Dimensions are read dynamically from `.npy` files — no code changes needed.

**Key point:** Regardless of input feature dimensionality (10, 128, 528, 1024, 1562), the MLP always outputs 128-dim, which is added to the 128-dim learned embedding. The graph propagation always operates in 128-dim space. This makes all Tier 2 experiments directly comparable — the only variable is feature quality, not architecture.


### Key Comparisons

The comparisons the harness is built around, matching the hierarchy table above:

| # | Comparison | Research Question |
|---|-----------|-------------------|
| Q1 | M4 vs M1/M1c | Does LLM content improve over pure CF (including strong contrastive methods)? |
| Q2 | M4 vs M2/M2b | Does LLM-synthesized content beat genome tags, regardless of dimensionality? **(Central claim)** |
| Q3 | M4 vs M3 | Does LLM synthesis matter, or does the same encoder applied to titles suffice? |
| Q4 | M7 vs M4 | Do structured mood features add value beyond semantic embeddings? |

Supporting comparisons (not primary research questions, but informative):

| Comparison | What It Tests |
|---|---|
| M0 (BPR-MF) vs M1 (LightGCN) | Does graph structure help? (sanity check) |
| M1 vs M1b/M1c/M1d | Does a contrastive objective improve pure CF on this split? |
| M7 vs M8 | Do categorical themes further help, or add noise? |
| M2 vs M9 | Do structured LLM features complement genome tags? |

## Experiment Status

The ML-20M sweep is complete: 5 seeds x every configuration listed above, under two
sentence encoders, plus the GPT-4o-mini provider arm for M4 and M7. For headline
results see the paper's main results table. The protocol (200-epoch budget,
patience 20 epochs) is in `config.py` and recorded in every `results.json`.

## Output Layout

**`results/`** holds one directory per configuration — the five-seed aggregate and the
per-seed leaves the verifier reads:

```
results/<config>/metrics.json                    # 5-seed aggregate (NDCG@10/Recall@10/MRR)
results/<config>/<encoder>/seed-<N>/results.json # per-seed leaf
results/analysis/                                # NON-per-config: cold_start_*, mood_analysis/, verify_*
results_gpt4omini/                               # the provider-sensitivity arm (M4, M7)
```

- `<config>` ∈ the fourteen configurations: m0, m1, m1b, m1c, m1d, m2, m2b, m3 … m9. The paper's main table prints twelve; m8 and m9 are sensitivity configurations.
- `<encoder>` ∈ {`bge-large-en-v1.5`, `e5-large-v2`}; the headline numbers are bge.
- **`metrics.json` is the per-config headline aggregate.**
- **Every configuration has a `metrics.json` and full per-seed leaves** (all-K metrics).
  Each leaf is the record `train.py` writes for a run. `python3 eval_checkpoints.py
  [--configs ...]` re-scores saved checkpoints on the test split through the same
  `evaluate_model`; the checkpoints themselves are not part of the public release.

## How to Run

### Prerequisites

```bash

# 2. Generate the title embedding M3 reads (if not already done)
python3 features/bert_baseline.py
```

### Single Experiment

```bash
# Basic usage (uses default paths from config.py / env vars)
python3 run_experiment.py --model lightgcn_sf --features llm_profile --seed 42

# With custom paths via CLI arguments
python3 run_experiment.py --model lightgcn_sf --features llm_profile --seed 42 \
    --embedding-dir /path/to/embeddings \
    --data-dir /path/to/processed/data \
    --results-dir /path/to/results \
    --checkpoint-dir /path/to/checkpoints

# Override training hyperparameters
python3 run_experiment.py --model lightgcn_sf --features llm_profile --seed 42 \
    --lr 5e-4 --wd 1e-4 --epochs 100 --device cuda
```

**All `run_experiment.py` arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | none | Paper label (`M0`, `M4`, `M7`, ...); sets `--model` and `--features` together. Give this **or** `--model`. |
| `--model` | none | Model: `bpr_mf`, `lightgcn`, `lightgcn_sf`, `simgcl`, `xsimgcl`, `lightgcl` |
| `--features` | `none` | Feature config: `none`, `genome`, `genome_raw`, `bert_title`, `llm_profile`, `llm_mood`, `llm_themes`, `llm_prof_mood`, `llm_all`, `genome_llm` |
| `--seed` | `42` | Random seed |
| `--lr` | `1e-3` | Learning rate |
| `--wd` | `1e-5` | Weight decay (L2 regularization) |
| `--epochs` | `200` | Max training epochs |
| `--patience` | `20` | Early-stopping patience, in epochs |
| `--batch-size` | `32768` | BPR training batch size. The loader drops the final partial batch, so a dataset with fewer pairs than this yields **no** batches — lower it on a small or subsampled dataset. |
| `--min-epochs` | `0` (`45` for M1b–M1d) | Early stopping may not end the run before this epoch |
| `--dense-eval-until` | `0` (`20` for M1b–M1d) | Also validate every epoch up to this one (otherwise epoch 1 and every 5th) |
| `--no-test` | off | Skip the test evaluation (selection runs) |
| `--cl-weight` | from `contrastive_settings.json` | Contrastive weight λ (λ₁ for LightGCL); M1b–M1d only |
| `--cl-eps` | from `contrastive_settings.json` | Noise magnitude ε (SimGCL, XSimGCL) |
| `--cl-temp` | from `contrastive_settings.json` | InfoNCE temperature τ |
| `--layer-cl` | from `contrastive_settings.json` | XSimGCL: the layer l* contrasted with the final layer |
| `--init` | `normal` (`uniform` for M1b–M1d) | Embedding initialisation of the contrastive models |
| `--device` | `auto` | Device: `auto`, `cuda`, `mps`, `cpu` |
| `--embedding-dir` | from config | Path to embedding `.npy` files |
| `--data-dir` | from config | Path to processed train/val/test CSVs |
| `--results-dir` | from config | Where to save experiment results |
| `--checkpoint-dir` | from config | Where to save model checkpoints |
| `--resume` / `--no-resume` | resume | Continue from `training_state.pt` if one exists, or start clean |
| `--ignore-early-stop` | off | Keep training past the early-stopping trigger, up to `--epochs` |

Neither `--model` nor `--config` has a default: one of the two is required, and
giving both is an error. Every published number was produced at the defaults
above, which for M1b–M1d include their frozen settings from
`contrastive_settings.json` (its SHA-256 is checked before a run starts). `--lr`,
`--wd`, `--epochs`, `--patience`, `--batch-size` and the contrastive flags change
the optimisation, so a run that overrides one of them is not reproducing the
paper. `results.json` records every one of them, so a result always says what
produced it.

### Quick Ablation (BPR-MF + all LightGCN-SF configs, 1 seed)

```bash
# Default (10 epochs, seed 42)
bash run_quick.sh

# Custom epochs and seed
EPOCHS=50 SEED=123 bash run_quick.sh

# Custom paths via env vars
EMBEDDING_DIR=/path/to/embeddings \
RESULTS_DIR=/path/to/results \
CHECKPOINT_DIR=/path/to/checkpoints \
    bash run_quick.sh
```

### Full Ablation (all models × all features × 5 seeds)

```bash
# Run all experiments
python3 run_ablation.py

# Filter to a specific model
python3 run_ablation.py --model lightgcn_sf

# Quick test (1 seed, 20 epochs)
python3 run_ablation.py --quick

# With custom paths
python3 run_ablation.py \
    --embedding-dir /path/to/embeddings \
    --results-dir /path/to/results \
    --checkpoint-dir /path/to/checkpoints

# Collect and print results summary (no training)
python3 run_ablation.py --collect-only
python3 run_ablation.py --collect-only --results-dir /path/to/results
```


### Path Configuration Priority

Paths are resolved in this order (highest priority first):

1. **CLI arguments** (`--embedding-dir`, `--results-dir`, etc.)
2. **Environment variables** (`EMBEDDING_DIR`, `RESULTS_DIR`, etc.)
3. **Defaults** from `config.py` (relative to project root)

| Path | CLI Arg | Env Var | Default |
|------|---------|---------|---------|
| Embeddings | `--embedding-dir` | `EMBEDDING_DIR` | `../embedding_generator/output/bge-large-en-v1.5` |
| Processed data | `--data-dir` | `DATA_DIR` | `data/processed/` |
| Results | `--results-dir` | `RESULTS_DIR` | `results/` |
| Checkpoints | `--checkpoint-dir` | `CHECKPOINT_DIR` | `checkpoints/` |

## Running with Different Embeddings

The benchmark is **dimension-agnostic**: all feature dimensions are read dynamically from `.npy` file shapes at runtime. You can swap in embeddings from any encoder without changing benchmark code.

### How It Works

Override the embedding directory via `--embedding-dir` CLI argument or the `EMBEDDING_DIR` environment variable. By default it points to `../embedding_generator/output/bge-large-en-v1.5` (1024-dim, no PCA).

The pipeline:

```
--embedding-dir /path/to/embeddings  (or EMBEDDING_DIR env var)
    → features/loader.py reads .npy files, infers dimensions from file shapes
    → run_experiment.py gets feature_dim dynamically via loader.get_feature_dim()
    → model MLP auto-adapts: Linear(feature_dim → 128) regardless of input size
```

### Available Embedding Sets

Embedding sets live under `../embedding_generator/output/`, one directory per encoder:

```
embedding_generator/output/
├── bge-large-en-v1.5/              # the default EMBEDDING_DIR
│   ├── profile_embeddings.npy      # (10381, 1024) — bge-large-en-v1.5, no PCA
│   ├── mood_vectors.npy            # (10381, 10)   — always 10-dim
│   ├── theme_matrix.npy            # (10381, 528)  — always 528-dim
│   ├── genome_embeddings.npy       # (10381, 128)  — genome PCA-128 (built by Stage 2)
│   ├── bert_title_embeddings.npy   # (10381, 1024) — title + genres, bge-large (features/bert_baseline.py)
│   ├── combined_features.npy       # (10381, 1034) — profile(1024) + mood(10)
│   ├── combined_full.npy           # (10381, 1562) — profile(1024) + mood(10) + themes(528)
│   ├── movie_id_index.json         # row-to-movieId mapping (shared across all sets)
│   ├── theme_vocabulary.json
│   └── embedding_metadata.json
│
└── minilm-pca128/                  # Example: lightweight encoder with PCA
    ├── profile_embeddings.npy      # (10381, 128)
    ├── mood_vectors.npy            # (10381, 10)
    ├── theme_matrix.npy            # (10381, 528)
    ├── genome_embeddings.npy       # (10381, 128)
    ├── combined_features.npy       # (10381, 138)
    ├── combined_full.npy           # (10381, 666)
    ├── movie_id_index.json
    ├── theme_vocabulary.json
    └── embedding_metadata.json
```

### Step-by-Step: Run Experiments with a Different Encoder

#### 1. Generate new embeddings

From the `embedding_generator/` directory:

```bash
cd ../embedding_generator

# Default: bge-large-en-v1.5, no PCA (1024-dim) — outputs to output/bge-large-en-v1.5/
python3 main.py

# With PCA reduction
python3 main.py --pca-dims 128 --output-dir output/bge-large-pca128

# PCA sweep: generates full-dim + 128/256/512 variants in subdirectories
python3 main.py --pca-sweep 128 256 512

# Different encoder
python3 main.py --model Alibaba-NLP/gte-large-en-v1.5 \
    --output-dir output/gte-large-v1.5

# Lightweight encoder with PCA
python3 main.py --model all-MiniLM-L6-v2 --pca-dims 128 \
    --output-dir output/minilm-pca128
```

Each invocation creates a self-contained directory with all required `.npy` files, `movie_id_index.json`, and `embedding_metadata.json`.

PCA sweep (`--pca-sweep`) creates subdirectories for each dimension:

```
output/gte-large-v1.5/
├── profile_embeddings.npy      # full 1024-dim (no PCA)
├── ...                         # other files at full dim
├── pca128/
│   ├── profile_embeddings.npy  # (10381, 128)
│   ├── combined_features.npy   # (10381, 138)
│   └── ...
├── pca256/
│   └── ...
└── pca512/
    └── ...
```

#### 2. Generate the title embedding for the new encoder (optional)

M3's title embedding uses the same encoder as the profile embeddings and is read from the same directory (`EMBEDDING_DIR/bert_title_embeddings.npy`). For a matched comparison, encode the titles with your encoder into your new directory:

```bash
cd ../benchmark
# bge-large-en-v1.5 (the default directory):
python3 features/bert_baseline.py
# e5-large-v2: python3 ../../scripts/generate_e5_embeddings.py writes it with the e5 profiles.
# For another encoder, encode "Title | Genres" in movie_id_index.json order and save it there.
```

Without that file, `--features bert_title` (M3) stops with a message naming the command; the other configurations do not need it.

#### 3. Run benchmark experiments

```bash
cd ../benchmark

# Default embeddings (bge-large, no PCA) — no extra args needed
python3 run_experiment.py --model lightgcn_sf --features llm_profile --seed 42

# Use a different embedding set via CLI arg
python3 run_experiment.py --model lightgcn_sf --features llm_profile --seed 42 \
    --embedding-dir ../embedding_generator/output/bge-large-pca128

# Full ablation with a different encoder
python3 run_ablation.py --embedding-dir ../embedding_generator/output/gte-large-v1.5

# Same thing via env var
EMBEDDING_DIR=../embedding_generator/output/gte-large-v1.5 python3 run_ablation.py

# Quick test (1 seed, 20 epochs)
python3 run_ablation.py --quick --embedding-dir ../embedding_generator/output/gte-large-v1.5
```

#### 4. Output layout (config-first, encoder-second)

Each run writes to `{checkpoints,results}/{config}/{encoder}/seed-{seed}/`, where:

- `{config}` is the paper-style experiment label (e.g. `m0`, `m4`, `m7`), looked up from `CONFIG_NAME_MAP` in `config.py` by the `(model, features)` pair
- `{encoder}` is inferred from the trailing component of `EMBEDDING_DIR` (e.g. `bge-large-en-v1.5`, `e5-large-v2`, `gte-large-v1.5`)
- `{seed}` is the integer random seed

This layout keeps every encoder variant of a given config adjacent on disk — ideal for the encoder-sensitivity analysis the paper reports. Comparing bge vs e5 for M4 is a single glob: `results/m4/*/seed-42/results.json`.

Example tree after a full bge run plus an e5 sensitivity sweep (M3, M4, M7, M8). The summary files at the bottom are written by `run_ablation.py`; the shipped `results/` does not include them:

```
results/
├── m0/bge-large-en-v1.5/seed-42/results.json         # encoder-independent configs
├── m1/bge-large-en-v1.5/seed-42/results.json         # only get one encoder subtree
├── m1c/bge-large-en-v1.5/seed-42/results.json
├── m2/bge-large-en-v1.5/seed-42/results.json
├── m3/
│   ├── bge-large-en-v1.5/seed-42/results.json        # encoder-dependent configs
│   └── e5-large-v2/seed-42/results.json           # live side-by-side
├── m4/
│   ├── bge-large-en-v1.5/
│   │   ├── seed-42/results.json
│   │   ├── seed-123/results.json
│   │   └── ...                                    # all 5 seeds
│   └── e5-large-v2/
│       ├── seed-42/results.json
│       └── ...
├── m7/
│   ├── bge-large-en-v1.5/...
│   └── e5-large-v2/...
├── m7/bge-large-en-v1.5/seed-42/results.json
├── ablation_summary_bge-large-en-v1.5.json           # per-encoder summaries
├── ablation_summary_e5-large-v2.json
└── encoder_sensitivity_summary.json               # cross-encoder merge
```

`checkpoints/` mirrors the same layout with `best_model.pt` per run.

Runs under different encoders never collide, so keep the default `RESULTS_DIR` / `CHECKPOINT_DIR` for every encoder sweep:

```bash
# bge-large (default)
python3 run_ablation.py

# e5-large — same output roots; lands under results/{config}/e5-large-v2/
python3 run_ablation.py --embedding-dir ../embedding_generator/output/e5-large-v2

# Collect a specific encoder's summary (also refreshes encoder_sensitivity_summary.json)
python3 run_ablation.py --collect-only --embedding-dir ../embedding_generator/output/e5-large-v2
```

Each invocation writes `results/ablation_summary_{encoder}.json` for that encoder, and also refreshes `results/encoder_sensitivity_summary.json`, which merges every per-encoder summary it can find. The merged file is keyed by config label with one entry per encoder, so plotting bge-vs-e5 deltas per config needs no further scripting.

The `(model, features) → paper label` mapping lives in `CONFIG_NAME_MAP` in `config.py`; adding a new row there is how you introduce a new experiment whose output folder should match a paper-style name.

### Required Files per Embedding Directory

The benchmark expects these files in `EMBEDDING_DIR`. Missing optional files cause specific feature configs to be unavailable, not a crash.

| File | Required by | Notes |
|------|------------|-------|
| `profile_embeddings.npy` | M4, M7, M8 | Any dimension; shape `(10381, D)` |
| `mood_vectors.npy` | M5, M7, M8, M9 | Always `(10381, 10)` |
| `theme_matrix.npy` | M6, M8, M9 | Always `(10381, 528)` |
| `genome_embeddings.npy` | M2, M9 | PCA-128 from genome-scores; encoder-independent |
| `bert_title_embeddings.npy` | M3 | Same encoder as the profiles; M3 stops with a message if missing |
| `movie_id_index.json` | All | Row-to-movieId mapping; must match `.npy` row order |
| `combined_features.npy` | — | Not used by benchmark (convenience file) |
| `combined_full.npy` | — | Not used by benchmark (convenience file) |
| `theme_vocabulary.json` | — | Not used by benchmark (reference only) |
| `embedding_metadata.json` | — | Not used by benchmark (reference only) |

The `combined_*.npy` files are pre-concatenated convenience files from the embedding generator. The benchmark concatenates features itself via `features/loader.py`, so these are not loaded.

### Feature Dimension Reference

Dimensions that change with the encoder (profile-derived):

| Config | Features | bge-large (default, 1024d) | bge-large + PCA-128 | MiniLM + PCA-128 |
|--------|----------|---------------------------|---------------------|-----------------|
| M3 | title embedding | 1024 | 128 | 128 |
| M4 | profile | 1024 | 128 | 128 |
| M7 | profile + mood | 1034 | 138 | 138 |
| M8 | profile + mood + themes | 1562 | 666 | 666 |

Dimensions that stay fixed regardless of encoder:

| Config | Features | Dimension |
|--------|----------|-----------|
| M2 | genome PCA | 128 |
| M5 | mood | 10 |
| M6 | themes | 528 |
| M9 | genome + mood + themes | 666 |

### Verifying an Embedding Directory

Quick sanity check before running experiments:

```bash
# Check shapes of all .npy files in an embedding directory
python3 -c "
import numpy as np, json
from pathlib import Path
d = Path('../embedding_generator/output/bge-large-en-v1.5')  # or any EMBEDDING_DIR path
for f in sorted(d.glob('*.npy')):
    a = np.load(f)
    print(f'{f.name:35s} {str(a.shape):>15s}  dtype={a.dtype}')
ids = json.load(open(d / 'movie_id_index.json'))
print(f\"{'movie_id_index.json':35s} {len(ids)} IDs\")
meta = json.load(open(d / 'embedding_metadata.json'))
print(f\"{'Encoder':35s} {meta.get('sentence_model', 'N/A')}\")
print(f\"{'PCA applied':35s} {meta.get('profile_pca_applied', 'N/A')}\")
"
```

Expected output (default bge-large-en-v1.5, no PCA, after Stage 2 and `features/bert_baseline.py`; a download alone has no `genome_embeddings.npy` or `bert_title_embeddings.npy`):

```
bert_title_embeddings.npy               (10381, 1024)  dtype=float32
combined_features.npy                   (10381, 1034)  dtype=float32
combined_full.npy                       (10381, 1562)  dtype=float32
genome_embeddings.npy                   (10381, 128)   dtype=float32
mood_vectors.npy                        (10381, 10)    dtype=float32
profile_embeddings.npy                  (10381, 1024)  dtype=float32
theme_matrix.npy                        (10381, 528)   dtype=float32
Encoder                                 BAAI/bge-large-en-v1.5
PCA applied                             False
```

## Project Structure

```
benchmark/
├── config.py                      # Paths, hyperparameters, experiment configs
├── evaluate.py                    # Full-ranking metrics (NDCG, Recall, HR, MRR)
├── train.py                       # BPR training loop with early stopping
├── run_experiment.py              # Single experiment runner
├── run_ablation.py                # Full ablation table (all configs × 5 seeds)
├── run_quick.sh                   # Quick run: BPR-MF + LightGCN-SF, 10 epochs
├── data/
│   ├── preprocess.py              # ML-20M → filtered train/val/test splits
│   ├── dataset.py                 # PyTorch Dataset + BPR negative sampling
│   └── processed/                 # Train/val/test CSVs + ID mappings
├── features/
│   ├── loader.py                  # Load .npy embeddings, align to benchmark IDs
│   └── bert_baseline.py           # Encode "Title | Genres" for M3 (bge-large-en-v1.5)
├── models/
│   ├── bpr_mf.py                  # BPR Matrix Factorization
│   ├── lightgcn.py                # LightGCN + LightGCN-SF (with side features)
│   ├── simgcl.py                  # SimGCL (noise-based contrastive)
│   ├── xsimgcl.py                 # XSimGCL (cross-layer contrastive)
│   ├── lightgcl.py                # LightGCL (SVD-based contrastive)
├── external/
└── requirements.txt
```

## References

- Rendle et al. "BPR: Bayesian Personalized Ranking from Implicit Feedback." UAI 2009.
- He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." SIGIR 2020.
- Yu et al. "Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation." SIGIR 2022. (SimGCL)
- Yu et al. "XSimGCL: Towards Extremely Simple Graph Contrastive Learning for Recommendation." TKDE 2023.
- Cai et al. "LightGCL: Simple Yet Effective Graph Contrastive Learning for Recommendation." ICLR 2023.
- Krichene & Rendle. "On Sampled Metrics for Item Recommendation." KDD 2020.
- Harper & Konstan. "The MovieLens Datasets: History and Context." ACM TIIS 2015.
