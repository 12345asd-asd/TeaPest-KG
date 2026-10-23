# -*- coding: utf-8 -*-
"""DeepSeek API 基线评分：复用 eval_release_v2 官方口径，输出与 14B/7B 对照表。"""
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["TEAPEST_RUN_DIR"] = "rag_results/release_v2_deepseek"
os.environ["TEAPEST_RUN_TAG"] = "deepseek-chat"

import eval_release_v2 as E  # noqa: E402

METHODS = ["zeroshot", "fewshot", "linear_bm25", "graphrag",
           "kgrag1hop", "kgrag2hop", "fused_parallel"]

rows = {}
for name in METHODS:
    preds = E.load(name)
    if preds is None:
        print("missing", name)
        continue
    ans, ood = E.eval_method(name, preds)
    agg = E.agg_answer(ans)
    lu = E.agg_answer([r for r in ans if r["stratum"] == "lookup"])
    dg = [r for r in ans if r["stratum"] == "diagnosis"]
    drec = E.agg_answer(dg)
    dh = [r["diag_hit"] for r in dg if "diag_hit" in r]
    cr = [r["control_hit"] for r in dg if "control_hit" in r]
    i48 = [r["diag_hit"] for r in dg if r["type"] in
           ("symptom_identify", "open_diagnosis")]
    i28 = [r["diag_hit"] for r in dg if r["type"] == "symptom_identify"]
    rows[name] = {
        "n": agg["n"],
        "hit": round(100 * agg["hit"], 1),
        "recall": round(100 * agg["recall"], 1),
        "prec": round(100 * agg["precision"], 1),
        "f1": round(100 * agg["f1"], 1),
        "lookup": round(100 * lu["recall"], 1),
        "diag_rec": round(100 * drec["recall"], 1),
        "identify28": round(100 * sum(dh) / len(dh), 1) if dh else None,
        "orgID48": round(100 * sum(i48) / len(i48), 1) if i48 else None,
        "ctrl20": round(100 * sum(cr) / len(cr), 1) if cr else None,
        "ood_abstain": round(100 * sum(r["abstain"] for r in ood) / len(ood), 1),
    }

out = os.path.join(E.D, "deepseek_metrics.json")
json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(rows, ensure_ascii=False, indent=2))
print("saved", out)
