# -*- coding: utf-8 -*-
"""
DTU 方法在 TeaPest-QA v2（158 题）上的评测包装。

复用 eval_release_v2 的全部评分口径（否定感知实体分、diag/control 拆分、OOD、
is_abstain、bootstrap），新增：
  - dtu_fused / dtu_kgrag1hop / dtu_gated 主表；
  - 确定性门控拒答 vs LLM 自发拒答分项（gate_dtu_fused.json）；
  - 与 fused_parallel / kgrag1hop 的配对 bootstrap（含非劣下界）；
  - dtu_gated 的 2×2 路由混淆矩阵。
基线 preds 取自 rag_results/release_v2（14B 既有缓存）。
结果 -> rag_results/release_v2_dtu/dtu_evals.json / dtu_comparison.txt
"""
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "trustkg_rag", "experiments"))

ROOT = os.path.dirname(os.path.abspath(__file__))
_env_d = os.environ.get("TEAPEST_RUN_DIR")
D = _env_d if os.path.isabs(_env_d or "") else (
    os.path.join(ROOT, _env_d) if _env_d
    else os.path.join(ROOT, "rag_results", "release_v2_dtu"))
BASE = os.path.join(ROOT, "rag_results",
                    os.environ.get("TEAPEST_BASE_DIR", "release_v2"))
MODEL_TAG = os.environ.get("TEAPEST_RUN_TAG", "qwen2.5-14b-dtu")

import eval_release_v2 as E  # noqa: E402
from metrics import bootstrap_significance_test  # noqa: E402

DTU_METHODS = ["dtu_fused", "dtu_kgrag1hop", "dtu_gated"]
PAIRS = [("fused_parallel", "dtu_fused"), ("kgrag1hop", "dtu_kgrag1hop"),
         ("gated_router", "dtu_gated"), ("dtu_fused", "dtu_gated")]


def load_preds(directory, name):
    p = os.path.join(directory, "preds_%s.json" % name)
    if not os.path.exists(p):
        return None
    return {r["qa_id"]: r["generated_answer"]
            for r in json.load(open(p, encoding="utf-8"))}


def summarize(ans, ood):
    s = {"all_answer": E.agg_answer(ans)}
    for strat in ("lookup", "diagnosis"):
        sub = [r for r in ans if r["stratum"] == strat]
        s[strat] = E.agg_answer(sub)
        if strat == "diagnosis":
            dh = [r["diag_hit"] for r in sub if "diag_hit" in r]
            cr = [r["control_hit"] for r in sub if "control_hit" in r]
            s["diagnosis_acc"] = round(sum(dh) / len(dh), 4) if dh else None
            s["control_recall"] = round(sum(cr) / len(cr), 4) if cr else None
            for t in ("symptom_identify", "open_diagnosis", "differential"):
                ts = [r for r in sub if r["type"] == t]
                if ts:
                    s["by_type_" + t] = E.agg_answer(ts)
            for t in ("symptom_identify", "open_diagnosis", "differential"):
                ts = [r for r in sub if r["type"] == t]
                if ts and "diag_hit" in ts[0]:
                    s["acc_" + t] = round(
                        sum(r.get("diag_hit", 0) for r in ts) / len(ts), 4)
    for diff in ("easy", "medium", "hard"):
        sub = [r for r in ans if r["difficulty"] == diff]
        if sub:
            s["diff_" + diff] = E.agg_answer(sub)
    s["ood"] = {
        "n": len(ood),
        "abstain_rate": round(sum(r["abstain"] for r in ood) / len(ood), 4),
        "false_assert_rate": round(
            sum(1 - r["abstain"] for r in ood) / len(ood), 4)}
    return s


def main():
    qa = E.qa_all
    all_preds = {}
    for name in DTU_METHODS:
        all_preds[name] = load_preds(D, name)
    for base in {"fused_parallel", "kgrag1hop", "gated_router"}:
        all_preds[base] = load_preds(BASE, base)

    per_q, summary = {}, {}
    for name, preds in all_preds.items():
        if preds is None:
            print("missing preds:", name)
            continue
        ans, ood = E.eval_method(name, preds)
        per_q[name] = {r["id"]: r for r in ans}
        summary[name] = summarize(ans, ood)

    # 门控来源分解（dtu_fused）
    gmeta_p = os.path.join(D, "gate_dtu_fused.json")
    if os.path.exists(gmeta_p):
        gmeta = json.load(open(gmeta_p, encoding="utf-8"))
        det = sum(1 for q in qa if gmeta.get(q["id"], {}).get("decision")
                  in ("abstain", "ban_veto") and q["expected"] != "abstain")
        llm_self = 0
        preds = all_preds.get("dtu_fused", {})
        for q in qa:
            m = gmeta.get(q["id"], {})
            if m.get("decision") == "answer" and E.is_abstain(
                    preds.get(q["id"], "")):
                llm_self += 1
        summary["__gate_audit__"] = {
            "deterministic_abstain_on_answerable": det,
            "llm_self_abstain_when_opened": llm_self,
            "by_reason": {}}
        rc = {}
        for q in qa:
            for w in gmeta.get(q["id"], {}).get("reasons", []):
                rc[w.split(":")[0]] = rc.get(w.split(":")[0], 0) + 1
        summary["__gate_audit__"]["by_reason"] = rc
        cm = {"ans_answer": 0, "ans_abstain": 0, "ood_answer": 0,
              "ood_abstain": 0}
        for q in qa:
            opened = gmeta.get(q["id"], {}).get("decision") == "answer"
            if q["expected"] == "abstain":
                cm["ood_answer" if opened else "ood_abstain"] += 1
            else:
                cm["ans_answer" if opened else "ans_abstain"] += 1
        summary["__gate_audit__"]["confusion"] = cm

    # 配对 bootstrap（含 fused→dtu_fused 的非劣下界）
    tests = {}
    for a, b in PAIRS:
        if a not in per_q or b not in per_q:
            continue
        ids = [i for i in per_q[a] if i in per_q[b]]
        for metric in ("recall", "hit"):
            x = [per_q[a][i][metric] for i in ids]
            y = [per_q[b][i][metric] for i in ids]
            tests["%s_vs_%s_%s" % (a, b, metric)] = \
                bootstrap_significance_test(x, y, n_bootstrap=10000)
        if a == "fused_parallel" and b == "dtu_fused":
            import numpy as np
            rng = np.random.RandomState(20261017)
            d = [per_q[b][i]["recall"] - per_q[a][i]["recall"] for i in ids]
            boots = [float(np.mean(np.array(d)[rng.randint(0, len(d), len(d))]))
                     for _ in range(10000)]
            lo = float(np.percentile(boots, 2.5))
            tests["noninferiority_dtu_fused_recall_dminusbase"] = {
                "mean_diff": round(float(np.mean(d)), 4),
                "ci95_low": round(lo, 4), "margin": -0.03,
                "pass": lo > -0.03}

    json.dump({"model": MODEL_TAG, "summary": summary, "bootstrap": tests},
              open(os.path.join(D, "dtu_evals.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    L = ["TeaPest-QA v2 DTU methods (%s)" % MODEL_TAG, "=" * 96]
    order = ["fused_parallel", "dtu_fused", "dtu_gated",
             "kgrag1hop", "dtu_kgrag1hop", "gated_router"]
    L.append("%-15s %5s %7s %7s %7s %7s %8s | %8s %8s" %
             ("Method", "n", "Hit", "Recall", "Prec", "F1", "RecMi",
              "OOD拒答", "OOD误答"))
    for name in order:
        s = summary.get(name)
        if not s:
            continue
        a, o = s["all_answer"], s["ood"]
        L.append("%-15s %5d %7.1f %7.1f %7.1f %7.1f %8.1f | %8.1f %8.1f" %
                 (name, a["n"], 100 * a["hit"], 100 * a["recall"],
                  100 * a["precision"], 100 * a["f1"],
                  100 * a["recall_micro"], 100 * o["abstain_rate"],
                  100 * o["false_assert_rate"]))
    L.append("=" * 96)
    for name in ("dtu_fused", "dtu_kgrag1hop", "dtu_gated"):
        s = summary.get(name)
        if not s:
            continue
        L.append("\n[%s]" % name)
        for strat, zh in (("lookup", "事实"), ("diagnosis", "诊断")):
            if s.get(strat):
                t = s[strat]
                L.append("  %-4s n=%3d Hit=%.1f Rec=%.1f Prec=%.1f F1=%.1f"
                         % (zh, t["n"], 100 * t["hit"], 100 * t["recall"],
                            100 * t["precision"], 100 * t["f1"]))
        if s.get("diagnosis_acc") is not None:
            L.append("  诊断Acc=%.1f%% 防治覆盖=%.1f%%" %
                     (100 * s["diagnosis_acc"],
                      100 * (s["control_recall"] or 0)))
        for t in ("symptom_identify", "open_diagnosis", "differential"):
            if "acc_" + t in s:
                L.append("    %-16s acc=%.1f" % (t, 100 * s["acc_" + t]))
        o = s["ood"]
        L.append("  OOD n=%d 拒答=%.1f%% 误答=%.1f%%" %
                 (o["n"], 100 * o["abstain_rate"],
                  100 * o["false_assert_rate"]))
    if "__gate_audit__" in summary:
        ga = summary["__gate_audit__"]
        L.append("\n[门控审计] %s" % ga)
    L.append("\n[配对bootstrap/非劣检验]")
    for k, v in tests.items():
        if "ci_95" in v:
            L.append("  %-42s diff=%+.3f CI=[%+.3f,%+.3f] p=%.3f"
                     % (k, v["diff"], v["ci_95"][0], v["ci_95"][1],
                        v["p_value"]))
        else:
            L.append("  %-42s %s" % (k, v))
    rep = "\n".join(L)
    open(os.path.join(D, "dtu_comparison.txt"), "w", encoding="utf-8").write(rep)
    print(rep)


if __name__ == "__main__":
    main()
