# TeaPest-KG · Tea Pest Diagnostic QA with Provenance Graphs and Pre-LLM Trust Gates

**Evidence Organization and Pre-LLM Trust Gating for Safe Diagnostic QA:
A Provenance-Traced Knowledge Graph, Three-Tier Benchmark, and Retrieval-Augmented
Generation for Tea Pest Management**

Open, reproducible artifacts behind the paper: a fully source-traced Chinese
knowledge graph for tea pests and diseases, a three-tier diagnostic QA benchmark,
controlled retrieval-augmented generation (RAG) experiments across a local 14B/7B
open-weight model **and a frontier API model (DeepSeek-Chat)**, plus deterministic
pre-LLM safety gates.

> **Headline finding:** no single retrieval organization wins everywhere.
> Fine-grained triples reach **99.3 %** recall on *named lookup*, while per-pest
> profiles reach **91.8 %** diagnosis recall and **88.7 %** control coverage on
> *symptom-only diagnosis*. The crossover follows from graph topology
> (a symptom is one hop from the organism, two from controls) and **replicates
> on three backbones** (Qwen2.5-14B, Qwen2.5-7B, DeepSeek-Chat).

---

## Why this project

Tea diagnosis is a high-stakes, multi-hop reasoning task — observe symptoms →
identify the organism → choose a safe treatment. Fluent LLM advice is not enough:
models hallucinate controls, confidently misdiagnose pests they do not know,
and happily repeat banned pesticides even when the source URL is forged.

We show that **how evidence is organised matters more than model size** for this
task, and that **deterministic gates before the LLM**, rather than the model's
self-reported confidence, are what make abstention and pesticide safety reliable —
with **zero added model calls**.

## Artifacts

| Artifact | Size | Description |
|---|---|---|
| **TeaPest-KG** | 279 entities / 413 triples / 28 organisms / 48 sources | Fully provenance-traced; 6 entity types, 5 relations; every triple carries source URL(s); zero dangling entities, zero type conflicts |
| **TeaPest-QA v2** | **158 questions** | 70 named lookup · 58 symptom-only diagnosis (28 identify / 20 open / 10 differential) · 30 out-of-KG abstention |
| **Independent manual audit** | **55 questions** | Authored from documents *excluded* from KG construction (46 from the 2015 green-control standard GCWZ/T 4–2015; 9 from a 2025 Tea Research Institute chapter), breaking gold–graph circularity |
| **Predictions & metrics** | 7 methods × 158 questions × 3 backbones | Raw answers, per-question scores, bootstrap CIs, Holm-adjusted p-values |
| **Trust-gating layer** | Data Trust Units | Source tiers, exact-match banned-pesticide registry (51 substances / 126 aliases), schema whitelist, numeric-slot completeness |

## Main result — the organization × grounding crossover (macro recall, %)

| Method | Named lookup (n=70) | Symptom diagnosis (n=58) | Overall (n=128) |
|---|---:|---:|---:|
| KG-RAG 1-hop (14B) | **99.3** | 57.9 | 80.5 |
| Profile-RAG (14B) | 77.0 | **91.8** | 83.7 |
| KG-RAG 1-hop (7B) | **95.4** | 57.0 | 78.0 |
| Profile-RAG (7B) | 89.3 | **88.7** | 89.0 |
| KG-RAG 1-hop (DeepSeek) | **99.8** | 68.2 | 85.5 |
| Profile-RAG (DeepSeek) | 87.0 | **94.5** | 90.4 |

The two organizations are tied overall on every backbone (Holm-adjusted, n.s.);
the stratum crossover is significant on all three
(14B: −22.3 lookup / +33.9 diagnosis; DeepSeek: −12.8 / +26.3, p<0.001).

## Safety results

- **33/55 → 0/55** unsafe banned-pesticide recommendations under injected
  contamination (forged tier-1 government URLs), on **both** 14B and 7B.
- Proper beyond-graph abstention on the independent audit: **62.5 % → 87.5 %**
  (72.5 % → 87.5 % on 7B); fabricated numeric advice **15.0 % → 2.5 %**.
- Cost of strict gating: at most **3.5 macro-recall points** in-distribution,
  with zero additional LLM calls.
- A source-tier score alone **cannot** veto a banned chemical carrying a forged
  authoritative URL — motivating the hard registry rule.

## Repository layout

```
experiments/
├── release_kg/      # final knowledge graph + DTU registries
├── release_qa/      # 158-question benchmark + 55-question external audit
├── rag_results/     # raw predictions and computed metrics, per backbone
│   ├── release_v2/            # 14B main results
│   ├── release_v2_7b/         # 7B replication
│   ├── release_v2_deepseek/   # DeepSeek-Chat replication
│   ├── release_v2_dtu*/       # gates, contamination, manual audit
│   └── external_manual*/      # independent audit predictions
├── trustkg_rag/     # local evaluation library
├── figures/         # regenerated data figures
└── *.py             # build, run, evaluation, and plotting scripts
```

## Quick start

Re-scoring the released predictions (reproduces every table) needs only Python
with NumPy — no GPU and no LLM:

```bash
cd experiments
TEAPEST_RUN_DIR=rag_results/release_v2  python eval_release_v2.py   # Tables 3–4 (14B)
TEAPEST_RUN_DIR=rag_results/release_v2_7b python eval_release_v2.py # 7B
python eval_deepseek.py                                            # DeepSeek (Table 9)
python eval_dtu_v2.py                                              # trust gates
python make_paper_figures.py                                       # data figures
```

Regenerating answers from scratch requires a local Ollama server
(`qwen2.5:14b/7b-instruct-q4_K_M`) for the local models, or a
`DEEPSEEK_API_KEY` environment variable for the API replication —
see [`README.md`](./README.md) for the full reproduction guide. **No API keys
are stored in this repository.**

## Methodology in one sentence

All methods share one lexical retrieval scorer and identical retrieved evidence;
only the **organisation of the evidence unit** changes (top-8 sentences,
1-/2-hop triples capped at 25/40 edges, or top-3 pre-aggregated per-pest
profiles), so measured differences isolate evidence organisation from
embeddings, prompt engineering, and model scale.

## Scope and honesty

- In-distribution gold derives from the same curated graph, so those scores are
  **retrieval-fidelity measures**, not open-world accuracy; generalisation claims
  rest on the 55-question independent audit and the 30 OOD items.
- The graph covers 28 Chinese priority tea pests/diseases and Chinese sources;
  no dense-embedding retriever or cross-family LLM is benchmarked (stated limits).
- Pesticide names and the banned-substance registry are reproduced for safety
  research from the cited local standard; this prototype must not replace
  official plant-protection guidance.

## Citation

```bibtex
@article{lin2026teapestkg,
  title   = {Evidence Organization and Pre-{LLM} Trust Gating for Safe Diagnostic {QA}:
             A Provenance-Traced Knowledge Graph, Three-Tier Benchmark, and
             Retrieval-Augmented Generation for Tea Pest Management},
  author  = {Lin, Weikeduo and Yang, Jun and Xiang, Pengjun and Lv, Jialin and Yu, Chengkai},
  journal = {Agriculture},
  year    = {2026}
}
```

## License

Code: [MIT](LICENSE). Data and figures: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Source manuals under `_research/` remain the property of their original publishers.
