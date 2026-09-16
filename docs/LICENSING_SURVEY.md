# Survey — how released datasets handle ML-20M and TMDb

**Compiled 2026-08-24.** Every row verified against a primary source (the dataset's own
README, licence page, or record) rather than secondary description. Purpose: settle what
this project may publish, and give §2 and the datasheet a defensible precedent base.

---

## 1. What GroupLens itself does and permits

| Fact | Source |
|---|---|
| Ships `links.csv` — `movieId,imdbId,tmdbId` for all 27,278 films | ML-20M distribution |
| *"Use of the resources listed above is subject to the terms of each provider."* | ML-20M README, links section |
| *"The user may not redistribute the data without separate permission."* | ML-20M README, Usage License |
| *"…may not use this information for any commercial or revenue-bearing purposes without first obtaining permission…"* | ML-20M README, Usage License |
| *"We typically do not permit public redistribution (see Kaggle for an alternative download location if you are concerned about availability)."* | grouplens.org/datasets/movielens/ |
| **A formal permission request form exists** — [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSdS6ZdesxmgOHPdO9PjUd31vB2_5CC-KxfaE825qTJhOsQ6Fg/viewform) | same page |

**Two things follow.** GroupLens's own design is the **ID-not-content** model: they hand you a
TMDb identifier and explicitly disclaim TMDb's terms rather than shipping TMDb's data. And
permission is a defined process with a form, not an ad-hoc email — so "ask GroupLens" is a
concrete, ten-minute action.

## 2. How the major platforms redistribute ML-20M

| Platform | Behaviour | Significance |
|---|---|---|
| **TensorFlow Datasets** (Google) | **Downloads from GroupLens at runtime; does not host the files.** Cites Harper & Konstan | The strongest precedent available. Google, with every incentive to host and the legal resources to decide, chose fetch-at-runtime |
| Kaggle | Hosts a full mirror | GroupLens points at it as *"an alternative download location"* — tacit tolerance, not a licence |
| HuggingFace | Many MovieLens variants (100K/1M/latest); ML-20M present (`TGB-Seq/ML-20M`, `yilmazerhakan/ml20m`) | Redistribution is common and unpoliced |
| Mendeley Data | Hosts ML-20M | ditto |

## 3. TMDb in released datasets

| Resource | What it ships | Notes |
|---|---|---|
| GroupLens `links.csv` | **TMDb IDs only** | the reference model |
| The Movies Dataset (Kaggle, rounakbanik) | 45k films: overview, keywords, cast, crew, budget, revenue, vote counts/averages **+ 26M ratings** | Carries the attribution notice *"This product uses the TMDb API but is not endorsed or certified by TMDb."* Very widely used |
| TMDB 5000 Movie Dataset (Kaggle) | 5k films of TMDb metadata | long-standing |
| Full TMDB Movies Dataset 2024 (Kaggle) | ~1M films | scraped at scale |

TMDb *metadata* redistribution on Kaggle is widespread, long-standing, and carries the
attribution notice. That is precedent, not permission — but it establishes that the
attribution notice is the field's expected artifact.

## 4. ML-20M content augmentations — what each actually releases

| Resource | Ships | Redistributes source data? |
|---|---|---|
| MMTF-14K | extracted audio/visual descriptors | **No** — features, not trailers |
| KB4Rec | entity linkage (IDs) | **No** — a mapping |
| PosterLens | poster embeddings (ML-**25M**) | **No** — embeddings, not posters |
| ViLLA-MMBench | embeddings + configs (ML-1M + MMTF-14K) | No |
| LLMRec | augmented text + CLIP features (ML-10M) | partially |
| **M3L-20M / Binge Watch** | features **+ train/val/test splits in MMRec format** + raw plot text and poster/trailer URLs | **Yes** — and released under **CC BY 4.0** |

**The M3L-20M case is worth noting carefully.** It is the closest competitor, it ships
MovieLens-derived interaction splits, and it applies CC BY 4.0 to the record — a licence it
cannot validly grant over data GroupLens restricts. Its own record says original rating files
are *"available in GroupLens or our repository."* This is what the field does; it is not what
the field is licensed to do.

## 5. The two camps, and where to sit

1. **ID / derived-feature only** — GroupLens `links.csv`, KB4Rec, MMTF-14K, PosterLens,
   TensorFlow Datasets. Ship what you made or an identifier; let the user fetch the source.
2. **Full redistribution** — Kaggle and HuggingFace mirrors, M3L-20M's splits. Common,
   tolerated, unlicensed.

**Camp 1 is the defensible position and it costs almost nothing**, because our contribution is
generated text, not MovieLens ratings. It is also what Google chose for TFDS.

## 6. Consequences for this release

- **Keep** the generated profiles, embeddings, human-eval annotations, harness, verifier,
  datasheet, and the SHA-256 prompt manifest. All ours.
- **Do not ship** `benchmark_splits/*.csv` (11.5M MovieLens triples) or the TMDb cache.
  Replace the former with a deterministic rebuild script — camp 1.
- **Submit the GroupLens form.** With the restructure, permission becomes upside rather than
  a precondition; without the restructure, it is a blocker.
- **Carry the TMDb attribution notice** on the paper, both repos and the datasheet.
- **Do not cite "everyone does it"** as a defence in the paper. M3L-20M's CC BY over
  MovieLens-derived splits shows the field's practice outruns its licences, and a resource
  paper that leans on that has picked the wrong precedent.
