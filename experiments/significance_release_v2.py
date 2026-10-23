# -*- coding: utf-8 -*-
"""
TeaPest-QA v2 配对显著性（修正版）。

- 配对 bootstrap（10000 次重采样）逐题差值的 95% 百分位置信区间；
- 双侧 p 用配对置换检验（2000 次，题内交换两方法标签）；
- 参照：fused_parallel（总最优）/ graphrag（诊断最优）/ kgrag1hop（事实最优）；
- 指标：answer(120) 的 recall/hit，并给诊断(50)与事实(70)分层 recall。
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "trustkg_rag", "experiments"))
from run_release_experiments import score_answer  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
_env_d = os.environ.get("TEAPEST_RUN_DIR")
D = _env_d if os.path.isabs(_env_d or "") else (
    os.path.join(ROOT, _env_d) if _env_d
    else os.path.join(ROOT, "rag_results", "release_v2"))
qa_all = json.load(open(os.path.join(ROOT, "release_qa", "qa_v2.json"),
                        encoding="utf-8"))
qa_map = {q["id"]: q for q in qa_all}
ANS = [q for q in qa_all if q["expected"] == "answer"]
METHODS = ["zeroshot", "fewshot", "linear_bm25", "kg_only", "graphrag",
           "kgrag1hop", "kgrag2hop", "fused_parallel"]
REFS = ["fused_parallel", "graphrag", "kgrag1hop"]
RNG = np.random.RandomState(20261016)


def per_question(method):
    preds = {r["qa_id"]: r["generated_answer"]
             for r in json.load(open(os.path.join(D, "preds_%s.json" % method),
                                     encoding="utf-8"))}
    out = {}
    for q in ANS:
        m = score_answer(preds.get(q["id"], ""), q)
        if m:
            out[q["id"]] = {"recall": m["recall"], "hit": m["hit"],
                            "stratum": q["stratum"]}
    return out


def paired_test(refv, basev, ids, metric):
    a = np.array([basev[i][metric] for i in ids])
    b = np.array([refv[i][metric] for i in ids])
    d = b - a
    n = len(d)
    boots = [d[RNG.randint(0, n, n)].mean() for _ in range(10000)]
    ci = np.percentile(boots, [2.5, 97.5])
    # 配对置换：题内随机交换
    stack = np.stack([a, b], axis=1)
    cnt = 0; nperm = 2000
    obs = d.mean()
    for _ in range(nperm):
        flip = RNG.rand(n) < 0.5
        perm = stack.copy()
        perm[flip] = perm[flip][:, [1, 0]]
        pd_ = perm[:, 1] - perm[:, 0]
        if abs(pd_.mean()) >= abs(obs) - 1e-12:
            cnt += 1
    p = (cnt + 1) / (nperm + 1)
    return obs, ci, p


def main():
    P = {m: per_question(m) for m in METHODS}
    res = {}
    for ref in REFS:
        for base in METHODS:
            if base == ref:
                continue
            for strat, ids in (
                    ("all", [q["id"] for q in ANS]),
                    ("lookup", [q["id"] for q in ANS if q["stratum"] == "lookup"]),
                    ("diagnosis", [q["id"] for q in ANS
                                   if q["stratum"] == "diagnosis"])):
                for metric in ("recall", "hit"):
                    key = "%s_vs_%s_%s_%s" % (ref, base, strat, metric)
                    obs, ci, p = paired_test(P[ref], P[base], ids, metric)
                    res[key] = {"diff": round(float(obs), 4),
                                "ci95": [round(float(ci[0]), 4),
                                         round(float(ci[1]), 4)],
                                "p_perm": round(float(p), 4)}
    json.dump(res, open(os.path.join(D, "significance_v2.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2)

    # 文本：只印整体 recall/hit 与分层 recall
    def line(ref, base, strat, metric):
        r = res["%s_vs_%s_%s_%s" % (ref, base, strat, metric)]
        sig = "**" if (r["ci95"][0] > 0 or r["ci95"][1] < 0) else "ns"
        return "%+6.1f CI[%+5.1f,%+5.1f] p=%.3f %s" % (
            100 * r["diff"], 100 * r["ci95"][0], 100 * r["ci95"][1],
            r["p_perm"], sig)

    for ref in REFS:
        print("\n=== reference: %s (正=参照更优) ===" % ref)
        for base in METHODS:
            if base == ref:
                continue
            print("%-14s all-R[%s]  diag-R[%s]  look-R[%s]" % (
                base,
                line(ref, base, "all", "recall"),
                line(ref, base, "diagnosis", "recall"),
                line(ref, base, "lookup", "recall")))


if __name__ == "__main__":
    main()
