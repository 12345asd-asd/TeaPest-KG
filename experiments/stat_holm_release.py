# -*- coding: utf-8 -*-
"""
Holm step-down multiplicity correction for the confirmatory recall contrasts
asserted in Sections 4.2 / 4.4 of the manuscript.

Raw p-values come from significance_release_v2.py (paired within-question label
permutation, 2000 permutations, seed 20261016); this script only re-reads
rag_results/release_v2{,_7b}/significance_v2.json and applies Holm's correction
to two pre-specified families:

  family A (14B, 10 contrasts): RQ1 lift (fusion vs zero/fewshot/bm25 overall),
      central crossover (profile vs kgrag1hop overall/lookup/diagnosis),
      fusion-vs-profile overall/lookup/diagnosis, fusion-vs-kgrag1hop overall.
  family B (7B, 10 contrasts): the same comparisons, treated as a replication
      family and corrected separately.

All other pairwise contrasts in significance_v2.json are exploratory and left
unadjusted. Outputs:
  rag_results/release_v2/holm_adjusted.json / .txt
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

CONTRASTS = [
    # (key in significance_v2.json, short label used in the paper)
    ("fused_parallel_vs_zeroshot_all_recall",        "Fusion vs Zero-Shot, overall (RQ1 lift)"),
    ("fused_parallel_vs_fewshot_all_recall",         "Fusion vs Few-Shot, overall (RQ1 lift)"),
    ("fused_parallel_vs_linear_bm25_all_recall",     "Fusion vs Flat-BM25, overall (RQ1 lift)"),
    ("graphrag_vs_kgrag1hop_all_recall",             "Profile vs KG-RAG-1hop, overall (tie)"),
    ("graphrag_vs_kgrag1hop_lookup_recall",          "Profile vs KG-RAG-1hop, lookup (crossover)"),
    ("graphrag_vs_kgrag1hop_diagnosis_recall",       "Profile vs KG-RAG-1hop, diagnosis (crossover)"),
    ("fused_parallel_vs_graphrag_all_recall",        "Fusion vs Profile, overall"),
    ("fused_parallel_vs_graphrag_lookup_recall",     "Fusion vs Profile, lookup"),
    ("fused_parallel_vs_graphrag_diagnosis_recall",  "Fusion vs Profile, diagnosis"),
    ("fused_parallel_vs_kgrag1hop_all_recall",       "Fusion vs KG-RAG-1hop, overall"),
]


def holm(pvals):
    """Holm step-down adjusted p-values (same order as input)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adjusted[i] = min(running, 1.0)
    return adjusted


def family(run_dir, name):
    sig = json.load(open(os.path.join(run_dir, "significance_v2.json"),
                         encoding="utf-8"))
    rows = []
    for key, label in CONTRASTS:
        r = sig[key]
        rows.append({"key": key, "label": label,
                     "diff": r["diff"], "ci95": r["ci95"],
                     "p_raw": r["p_perm"]})
    adj = holm([r["p_raw"] for r in rows])
    for r, a in zip(rows, adj):
        r["p_holm"] = round(a, 4)
        r["p_raw"] = round(r["p_raw"], 4)
        r["diff"] = round(100 * r["diff"], 1)
        r["ci95"] = [round(100 * r["ci95"][0], 1),
                     round(100 * r["ci95"][1], 1)]
        r["sig05_holm"] = bool(a < 0.05)
    return rows


def main():
    out = {}
    lines = []
    for tag, sub in (("14B", os.path.join(ROOT, "rag_results", "release_v2")),
                     ("7B", os.path.join(ROOT, "rag_results", "release_v2_7b"))):
        rows = family(sub, tag)
        out[tag] = rows
        lines.append("=== confirmatory recall family, Qwen2.5-%s (Holm, m=10) ===" % tag)
        lines.append("%-48s %7s %9s %s" % ("contrast", "diff", "p(raw)", "p(Holm)"))
        for r in rows:
            mark = "sig" if r["sig05_holm"] else "ns "
            lines.append("%-48s %+6.1f %9.4f %8.4f  %s" %
                         (r["label"], r["diff"], r["p_raw"], r["p_holm"], mark))
        lines.append("")
    dst = os.path.join(ROOT, "rag_results", "release_v2")
    json.dump(out, open(os.path.join(dst, "holm_adjusted.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2)
    text = "\n".join(lines)
    open(os.path.join(dst, "holm_adjusted.txt"), "w",
         encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
