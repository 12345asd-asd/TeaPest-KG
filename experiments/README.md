# Reproduction package — Evidence Organization and Pre-LLM Trust Gating for Tea Pest Diagnostic QA

This folder contains the data, raw predictions, and scripts needed to reproduce every
experimental number in the paper. All paths in the scripts are relative to **this folder**
(`experiments/`), so run commands from inside it.

## 1. Contents

```
experiments/
├── release_kg/                 Final knowledge graph
│   ├── kg_triples.json         413 triples (279 entities, 28 organisms)
│   ├── entities.csv            279 entities with types
│   └── dtu/                    banned-substance registry, source tiers
├── release_qa/                 Benchmark question sets
│   ├── qa_v2.json              TeaPest-QA v2, 158 questions (70/58/30)
│   ├── qa_external.json        independent manual audit, 55 questions
│   ├── qa.json                 106-question v1 set (import-time dependency)
│   └── *_manifest.txt          provenance manifests
├── rag_results/                All raw predictions and computed metrics
│   ├── release_v2/             14B main results (Tables 3–4, Figs 4–6)
│   ├── release_v2_7b/          7B replication
│   ├── release_v2_deepseek/    DeepSeek-Chat API replication (Table 9, §4.6)
│   ├── release_v2_dtu/         DTU gates, pollution, external audit (14B)
│   ├── release_v2_dtu_7b/      DTU gates, pollution, external audit (7B)
│   ├── external_manual/        independent-manual audit predictions (14B)
│   └── external_manual_7b/     independent-manual audit predictions (7B)
├── trustkg_rag/experiments/    Local library modules (metrics, local LLM client)
├── figures/                    Regenerated data figures (fig2/fig3/fig4)
└── *.py                        Experiment, evaluation, and plotting scripts
```

## 2. Environment

- Python 3.10+; standard scientific stack (`numpy`); no GPU libraries required for
  **re-scoring** the released predictions.
- Re-running the local 14B/7B generation requires an **Ollama** server serving
  `qwen2.5:14b-instruct-q4_K_M` / `qwen2.5:7b-instruct-q4_K_M` on localhost
  (the `local_model.py` client); raw predictions are already provided, so this is
  only needed to regenerate answers from scratch.
- Re-running the DeepSeek replication requires an API key:
  ```bash
  export DEEPSEEK_API_KEY="your-key"     # Windows PowerShell: $env:DEEPSEEK_API_KEY="your-key"
  ```
  No key is stored in this package; scripts read it from the environment.

## 3. Reproduce the paper tables from the released predictions (no LLM needed)

Run from `experiments/`:

```bash
# Tables 3–4 and the stratum decomposition (14B)
TEAPEST_RUN_DIR=rag_results/release_v2      python eval_release_v2.py
# 7B replication
TEAPEST_RUN_DIR=rag_results/release_v2_7b   python eval_release_v2.py
# DeepSeek-Chat replication (Table 9)
python eval_deepseek.py
# DTU gates, pollution stress, independent-manual audit
python eval_dtu_v2.py
python eval_dtu_pollution.py
python eval_dtu_external.py
# Cross-backbone table and Holm-corrected significance tests
python cross_backbone_v2.py
python stat_holm_release.py
# Cost table
python cost_benchmark_v2.py
# Data figures (crossover, gating, mechanism curves) -> figures/
python make_paper_figures.py
```

(Windows `cmd`: use `set TEAPEST_RUN_DIR=rag_results\release_v2` instead of the
inline env var; PowerShell: `$env:TEAPEST_RUN_DIR="rag_results/release_v2"`.)

## 4. Regenerate everything from scratch (requires local LLMs / API key)

```bash
python run_release_v2.py                 # 14B all methods on qa_v2 (run dir: rag_results/release_v2)
TEAPEST_RUN_DIR=rag_results/release_v2_7b TEAPEST_NO_V1_SEED=1 python run_release_v2.py
python run_dtu_methods.py                # DTU-gated variants
python inject_pollution.py               # synthetic contamination views (k=5/10/20%, 3 seeds)
python eval_dtu_pollution_llm.py         # LLM adoption under contamination
python run_deepseek_api.py               # DeepSeek API replication
```

Evaluation scripts cache per-question predictions and skip completed questions,
so interrupted runs can be resumed.

## 5. Notes

- In-distribution scores measure faithful reproduction of the curated graph
  (gold derives from the same 413 triples); generalisation claims rest on the
  55-question independent-manual set and the 30 OOD items — see §3.3 and §6.4.
- The banned-substance registry encodes GCWZ/T 4—2015 appendix B.3 (51 substances,
  52 printed names, 126 names/aliases); matching is exact-component only.
- Random seeds: contamination uses seeds 20261017–20261019; bootstrap/permutation
  tests use fixed seeds in the evaluation scripts.
