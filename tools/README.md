# `tools/` - what each one is

Utilities you run directly, as opposed to the pipeline stages in `src/`
and the per-result run scripts in `scripts/`. Each description is the
tool's own first docstring line, and this file is generated from the
shipped directory, so it cannot name a tool that is not here.

`verify_paper_numbers.py` is the one to run first: it needs no GPU, no
download and no API key.

| Tool | What it does |
|---|---|
| `analyze_steam_arms.py` | Measure the world-knowledge share in Steam profiles from the paired arms. |
| `build_prompt_manifest.py` | Build the prompt-provenance manifest for the ML-20M profile pack. |
| `fix_derived_fields.py` | Repair derived fields in the released profile files. |
| `make_pipeline_resource.py` | Draw Figure 1, the three-stage pipeline diagram, as a vector PDF. |
| `prepare_steam_metadata.py` | Build the Steam item-metadata file the profile generator consumes. |
| `rebuild_splits.py` | Rebuild the LLM-MovieLens benchmark splits from your own copy of MovieLens 20M. |
| `recon_amazon.py` | Stage-0 recon for adding an Amazon Reviews 2023 category as a third dataset. |
| `recon_steam.py` | Stage-0 recon for adding the Steam dataset (McAuley) as a third dataset. |
| `verify_generator_e2e.py` | End-to-end check of the RELEASED generator against the RELEASED artifact. |
| `verify_paper_numbers.py` | One-command reproduction of every headline number in the ECIR paper. |
