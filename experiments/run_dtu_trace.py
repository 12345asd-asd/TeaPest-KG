# -*- coding: utf-8 -*-
"""
DTU 证据检索 trace 落盘（零 LLM）。

每题记录：链接结果（含科级泛化）、top1 档案病虫、候选单元全集及其静态信任信号
（域名/层级/R/安全否决/注入标记）。**不记录门控决策**——决策由 eval_dtu_retrieval
以任意 τ/消融配置离线重放，保证消融只改配置、永不重跑检索。

用法：
  python run_dtu_trace.py --qa release_qa/qa_v2.json
  python run_dtu_trace.py --qa release_qa/qa_external.json
  python run_dtu_trace.py --qa release_qa/qa_v2.json \
      --kg-view rag_results/release_v2_dtu/pollution/kg_polluted_k0.10_seed20261017.json
输出：rag_results/dtu_traces/<qa名>__<kg_tag>.jsonl
"""
import argparse
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import dtu_pipeline as P  # noqa: E402
import dtu_trust as D  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
TRACE_DIR = os.path.join(ROOT, "rag_results", "dtu_traces")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="release_qa/qa_v2.json")
    ap.add_argument("--kg-view", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    qa_path = os.path.join(ROOT, *args.qa.split("/"))
    qa = json.load(open(qa_path, encoding="utf-8"))
    ret, raw = P.load_view(os.path.join(ROOT, args.kg_view) if args.kg_view else None)
    tier_reg = D.load_source_registry()
    safety_reg = D.load_safety_registry()

    if args.tag:
        kg_tag = args.tag
    elif args.kg_view:
        kg_tag = os.path.splitext(os.path.basename(args.kg_view))[0]
    else:
        kg_tag = "clean"
    qa_stem = os.path.splitext(os.path.basename(args.qa))[0]
    out_path = os.path.join(TRACE_DIR, "%s__%s.jsonl" % (qa_stem, kg_tag))
    os.makedirs(TRACE_DIR, exist_ok=True)

    qs = qa[:args.limit] if args.limit else qa
    with open(out_path, "w", encoding="utf-8") as f:
        for q in qs:
            units, meta = P.candidate_units(q, ret, raw, tier_reg, safety_reg)
            rec = P.trace_record(q, units, meta, kg_tag)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("wrote %d traces -> %s" % (len(qs), out_path))


if __name__ == "__main__":
    main()
