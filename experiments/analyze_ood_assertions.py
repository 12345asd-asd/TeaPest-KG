# -*- coding: utf-8 -*-
"""
OOD「非拒答」答案的危险性细分（安全主张的关键证据）。

对 16 道「真实但未收录病虫」OOD 题，统计各方法在不拒答时：
  force_map   —— 错配成图谱内已知病虫名（危险：误诊）；
  coincidental—— 碰巧说出了该未收录病虫的真实名（参数记忆蒙对）；
  generic     —— 不给具体病虫名，仅泛泛建议；
  with_chemical—— 答案中给出了图谱内具体农药名（错配荐药风险）；
  abstain     —— 恰当拒答。
结论落盘 rag_results/release_v2/ood_assertion_analysis.json。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "trustkg_rag", "experiments"))

import run_release_v2 as RV2  # noqa: E402
from eval_release_v2 import is_abstain  # noqa: E402
from run_release_experiments import retriever  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
_env_d = os.environ.get("TEAPEST_RUN_DIR")
OUT = _env_d if os.path.isabs(_env_d or "") else (
    os.path.join(ROOT, _env_d) if _env_d
    else os.path.join(ROOT, "rag_results", "release_v2"))

# 直接从基准构建脚本取「真实未收录病虫名 ↔ 题干」
sys.path.insert(0, ROOT)
import build_release_qa_v2 as B  # noqa: E402

METHODS = ["zeroshot", "fewshot", "linear_bm25", "graphrag",
           "kgrag1hop", "kgrag2hop", "fused_parallel"]


def true_name_of(question):
    for name, q in B.OOD_UNKNOWN_PEST:
        if q == question:
            return name
    return None


def main():
    xq = [q for q in RV2.qa_all if q["type"] == "ood_unknown_pest"]
    pests = [e for e, t in retriever.entity_to_type.items()
             if t == "PEST-ORGANISM"]
    chems = [e for e, t in retriever.entity_to_type.items()
             if t == "CONTROL-CHEMICAL"]

    result = {"n_unknown_pest_ood": len(xq), "methods": {}}
    preds_cache = {}
    for m in METHODS:
        preds_cache[m] = {r["qa_id"]: r["generated_answer"] for r in
                          json.load(open(os.path.join(
                              OUT, "preds_%s.json" % m), encoding="utf-8"))}

    detail = []
    for m in METHODS:
        c = dict(abstain=0, force_map=0, coincidental=0, generic=0,
                 with_chemical=0)
        for q in xq:
            a = preds_cache[m][q["id"]]
            if is_abstain(a):
                c["abstain"] += 1
                continue
            inkg = [p for p in pests if p in a]
            tn = true_name_of(q["question"])
            if inkg:
                c["force_map"] += 1
            elif tn and tn in a:
                c["coincidental"] += 1
            else:
                c["generic"] += 1
            if any(ch in a for ch in chems):
                c["with_chemical"] += 1
        n = len(xq)
        nonab = n - c["abstain"]
        c["non_abstain"] = nonab
        c["force_map_rate_within_nonabstain"] = round(
            c["force_map"] / nonab, 3) if nonab else None
        c["chemical_rate_within_nonabstain"] = round(
            c["with_chemical"] / nonab, 3) if nonab else None
        result["methods"][m] = c

    # 逐题明细（便于人工核查错配成了谁）
    for q in xq:
        tn = true_name_of(q["question"])
        row = {"qa_id": q["id"], "true_unseen": tn}
        for m in METHODS:
            a = preds_cache[m][q["id"]]
            if is_abstain(a):
                row[m] = "ABSTAIN"
            else:
                hit = [p for p in pests if p in a]
                row[m] = ("RIGHT:" + tn) if (tn and tn in a) else (
                    "MAP->" + "/".join(hit) if hit else "GENERIC")
        detail.append(row)
    result["per_question"] = detail

    json.dump(result, open(os.path.join(OUT, "ood_assertion_analysis.json"),
                           "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print("16 道真实未收录病虫 OOD：非拒答答案性质")
    print("%-15s %6s %9s %9s %9s %9s" %
          ("method", "abstain", "forceMap", "coincid", "generic", "w/chem"))
    for m in METHODS:
        c = result["methods"][m]
        print("%-15s %6d %9d %9d %9d %9d" %
              (m, c["abstain"], c["force_map"], c["coincidental"],
               c["generic"], c["with_chemical"]))


if __name__ == "__main__":
    main()
