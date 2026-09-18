# Reading the benchmark

The paper reports fourteen configurations in two tiers. This document says what
each one is, what it controls for, and how to read a difference between two
rows — the things a page limit does not leave room for.

If you only read one thing: **the configuration list is not a leaderboard.** It
is a set of matched pairs, each removing one alternative explanation for the
same result. A row is interesting for what it rules out, not for its rank.

## The two tiers, and why the boundary is there

**Purely collaborative (M0, M1, M1b, M1c, M1d).** No configuration in this tier
sees any content. Together they fix the ceiling that collaborative signal alone
reaches on this split — a band roughly one percent wide. That band is what any
content feature has to clear, and it is why the paper's three-percent headline
is not a small effect: it is about three times the spread of the methods it is
compared with.

**Content-augmented, identical backbone (M2 … M9).** Every configuration here
runs the same model with the same optimiser, seeds and split, and differs only
in the content feature vector fed to it. That is what licenses reading a
difference between two rows as a difference in the feature.

Best-in-tier is underlined in the paper's tables rather than best overall,
deliberately: a cross-tier comparison is a claim that needs a paired test and a
*p*-value, not bold type.

## The backbone: LightGCN, LightGCN-SF, and why LightGCL is neither

**LightGCN** is the architecture — neighbourhood aggregation over the
interaction graph plus a weighted layer combination, with no feature
transformation and no nonlinearity. M1 runs it on ID embeddings alone.

**LightGCN-SF** is that same architecture with one addition: the item's initial
embedding becomes `e_i = e_learned_i + MLP(f_i)`, where `f_i` is the content
feature. It is not a different model. Every content configuration is
LightGCN-SF with a different `f_i`, and the MLP is shared, so changing a 10-d
feature for a 1,024-d one changes the input and nothing else. Without that
shared projection, comparing features of different widths would confound feature
quality with input dimensionality — which is exactly what the M2 / M2b pair
exists to rule out.

Two consequences worth knowing before you interpret a row:

- The feature is **added** to the learned embedding, not concatenated and not
  gated. A feature carrying no signal is therefore not neutral: it perturbs an
  embedding that was already adequate. This is how a content configuration can
  score *below* the ID-only baseline.
- LightGCN's minimalism is deliberate here. Fewer places for a feature to
  interact means a gain is attributable to the feature rather than to an
  interaction with the architecture.

**LightGCL is not a LightGCN variant**, despite the name. It is a *training
objective* — graph contrastive learning, with the contrastive view built by
low-rank reconstruction of the adjacency matrix. It sits in the collaborative
tier because it uses no content, not because it is a form of the backbone. The
three contrastive baselines differ in how that view is built: SimGCL perturbs
the embedding space with uniform noise, XSimGCL contrasts across propagation
layers, LightGCL augments the graph. Three mechanisms, which is why three of
them rather than one.

## The ladder

| Config | Feature | What it rules out |
|---|---|---|
| M0 | none (BPR-MF) | that a pre-GNN model would have done as well — and it calibrates the scale |
| M1 | none (LightGCN) | the ID-only reference every content row is read against |
| M1b/M1c/M1d | none (contrastive) | that a *better-trained* collaborative model closes the gap |
| M2 | genome PCA, 128-d | that the human-curated genome is already enough |
| M2b | genome raw, 1,128-d | that the gain came from having more dimensions |
| M3 | title+genres, same encoder | that the gain came from using a sentence encoder at all |
| M4 | LLM profile | — the core configuration |
| M5 | mood only, 10-d | isolates the mood primitive |
| M6 | themes only, 528-d | isolates the theme primitive |
| M7 | profile + mood | whether mood adds ranking signal on top of the profile |
| M8 | profile + mood + themes | whether stacking primitives keeps helping (it does not) |
| M9 | genome + mood + themes | M8 with the profile swapped out, isolating the profile's own contribution |

## Feature diagnostics: why themes-only underperforms

The measure that matters for an additive side feature is **discriminability** —
how many items the feature can tell apart — not dimensionality. Items a feature
cannot separate are items it cannot help rank, and because the feature is added
to the learned embedding, it actively pulls them together.

<!-- BEGIN GENERATED feature-diagnostics -->
<!-- Generated block. Regenerated in the paper repository, which is not shipped here; the numbers are computed from the released arrays under embeddings/ml20m/. -->

| Primitive | Dim | Distinct vectors (of 10,381 items) | All-zero rows |
|---|---|---|---|
| profile (M4) | 1,024 | 10,381 | 0 |
| mood (M5) | 10 | 10,352 | 0 |
| themes (M6) | 528 | 6,446 | 1,130 |

The theme matrix separates 6,446 of 10,381 items. 10.9% receive an all-zero vector, because the vocabulary keeps only themes appearing in at least ten films, leaving a mean of 1.99 themes per item; 45.9% share their theme set with another film, and the largest identical group holds 1,130. Mood is the contrast: 10 continuous dimensions separate 10,352 items, more than 528 sparse binary ones manage.

Computed from the released arrays under `embeddings/ml20m/`. The genome and title-embedding baselines derive from upstream MovieLens data this release does not redistribute, so no released file exists to compute them from.

<!-- END GENERATED feature-diagnostics -->

Themes are released because they are useful inside a composite and for tasks
other than ranking. They are not advocated as a standalone ranking feature, and
M6 is the evidence for that restraint.

## Adding your own feature

Any directory of item vectors in the catalogue's index order substitutes for a
released one:

```bash
EMBEDDING_DIR=path/to/your/vectors \
    python3 src/benchmark/run_experiment.py --config m4 --seed 42
```

Your feature is then evaluated under the identical split, seeds and metrics as
everything in the paper, which is the point of shipping the harness rather than
only the files. See `docs/REPRODUCIBILITY.md` for the protocol and
`docs/DATASHEET.md` for how the released features were produced.
