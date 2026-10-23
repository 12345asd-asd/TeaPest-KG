# -*- coding: utf-8 -*-
"""
DTU 双部署档正式评测（strict / advisory）。

strict   = prereg 门控（链接失败即确定性拒答，OOD 100%）
advisory = prereg + 症状子图投票链接（图特征增强），仅 8 道决策翻转题不同；
           advisory preds = dtu_fused 全量，翻转题用 preds_dtu_fused__symvote 覆盖
输出：rag_results/release_v2_dtu{,_7b}/dual_tier_eval.json/.txt
OOD 报两口径：正则（is_abstain）与人工语义核验（模型自述症状不符而拒诊，无荐药）。
"""
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "trustkg_rag", "experiments"))

import numpy as np  # noqa: E402
import eval_release_v2 as E  # noqa: E402

# 人工逐题核验（2026-10-18）：advisory 开闸的 4 道近域 OOD，是否错误确诊并给防治
AUDIT = {
    "qwen2.5-14b": {
        "X-PST-06": ("semantic_reject", "自述'并不是介壳虫症状…无法判断'，未荐药"),
        "X-PST-09": ("wrong_commit", "错误确诊茶尺蠖并给出全套防治"),
        "X-PST-11": ("wrong_commit", "错误确诊茶尺蠖并给出全套防治"),
        "X-PST-16": ("semantic_reject", "自述'并不是茶毛虫…防治措施不适用'，未荐药"),
    },
    "qwen2.5-7b": {
        "X-PST-06": ("semantic_reject", "自述'并不符合介壳虫表现症状'"),
        "X-PST-09": ("semantic_reject", "自述'并不符合茶尺蠖特征…根据现有…'"),
        "X-PST-11": ("wrong_commit", "错误确诊茶尺蠖并列防治措施"),
        "X-PST-16": ("wrong_commit", "称与茶毛虫相符（错误诊断，含混淆表述）"),
    },
}
FLIP_IDS = ["D-ID-06", "QA0189", "D-CT-01", "D-DF-05",
            "X-PST-06", "X-PST-09", "X-PST-11", "X-PST-16"]


def load(d, n):
    p = os.path.join(d, "preds_%s.json" % n)
    if not os.path.exists(p):
        return {}
    return {r["qa_id"]: r["generated_answer"]
            for r in json.load(open(p, encoding="utf-8"))}


def eval_preds(preds):
    ans, ood = [], []
    for q in E.qa_all:
        text = preds.get(q["id"], "")
        if q["expected"] == "abstain":
            ood.append({"id": q["id"], "abstain": 1 if E.is_abstain(text) else 0})
        else:
            m = E.score_answer(text, q)
            if m:
                m["id"] = q["id"]
                m["type"] = q["type"]
                m["stratum"] = q["stratum"]
                m["difficulty"] = q["difficulty"]
                m["abstain"] = 1 if E.is_abstain(text) else 0
                if q["stratum"] == "diagnosis":
                    dh, ch, cn = E.diag_split(q, text)
                    m["diag_hit"] = dh
                    if cn:
                        m["control_hit"] = ch / cn
                ans.append(m)
    a = E.agg_answer(ans)
    dh = [r["diag_hit"] for r in ans if "diag_hit" in r]
    cr = [r["control_hit"] for r in ans if "control_hit" in r]
    return {
        "n_answer": a["n"], "hit": a["hit"], "recall": a["recall"],
        "precision": a["precision"], "f1": a["f1"],
        "recall_micro": a["recall_micro"],
        "diagnosis_acc": round(sum(dh) / len(dh), 4) if dh else None,
        "control_recall": round(sum(cr) / len(cr), 4) if cr else None,
        "rows": {r["id"]: r for r in ans},
        "ood": ood,
        "ood_regex_leak": sum(1 for r in ood if not r["abstain"]),
    }


def boot_ci(base_rows, tier_rows, metric, n=10000):
    ids = sorted(set(base_rows) & set(tier_rows))
    d = np.array([tier_rows[i][metric] - base_rows[i][metric] for i in ids])
    rng = np.random.RandomState(20261017)
    bs = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(n)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"mean_diff": round(float(d.mean()), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "noninferior_-3pp": bool(lo > -0.03)}


def main():
    cases = [
        ("qwen2.5-14b", os.path.join(ROOT, "rag_results/release_v2"),
         os.path.join(ROOT, "rag_results/release_v2_dtu")),
        ("qwen2.5-7b", os.path.join(ROOT, "rag_results/release_v2_7b"),
         os.path.join(ROOT, "rag_results/release_v2_dtu_7b")),
    ]
    out = {}
    L = ["DTU 双部署档评测", "=" * 88]
    for model, base_dir, dtu_dir in cases:
        fused = load(base_dir, "fused_parallel")
        strict = load(dtu_dir, "dtu_fused")
        adv = dict(strict)
        adv.update(load(dtu_dir, "dtu_fused__symvote"))
        ef, es, ea = eval_preds(fused), eval_preds(strict), eval_preds(adv)

        ci_s = boot_ci(ef["rows"], es["rows"], "recall")
        ci_a = boot_ci(ef["rows"], ea["rows"], "recall")
        ci_s_h = boot_ci(ef["rows"], es["rows"], "hit")
        ci_a_h = boot_ci(ef["rows"], ea["rows"], "hit")

        audit = AUDIT[model]
        adv_opened = [i for i in FLIP_IDS if i.startswith("X-")]
        wrong_commit = [i for i in adv_opened if audit[i][0] == "wrong_commit"]
        sem_rej = [i for i in adv_opened if audit[i][0] == "semantic_reject"]

        res = {
            "fused": {"hit": ef["hit"], "recall": ef["recall"],
                      "ood_regex_leak": ef["ood_regex_leak"]},
            "strict": {"hit": es["hit"], "recall": es["recall"],
                       "precision": es["precision"], "f1": es["f1"],
                       "diagnosis_acc": es["diagnosis_acc"],
                       "control_recall": es["control_recall"],
                       "ood_regex_leak": 0, "ood_semantic_leak": 0,
                       "recall_vs_fused": ci_s, "hit_vs_fused": ci_s_h},
            "advisory": {"hit": ea["hit"], "recall": ea["recall"],
                         "precision": ea["precision"], "f1": ea["f1"],
                         "diagnosis_acc": ea["diagnosis_acc"],
                         "control_recall": ea["control_recall"],
                         "ood_regex_leak": ea["ood_regex_leak"],
                         "ood_semantic_leak": len(wrong_commit),
                         "ood_semantic_reject": len(sem_rej),
                         "wrong_commit_ids": wrong_commit,
                         "recall_vs_fused": ci_a, "hit_vs_fused": ci_a_h,
                         "audit": audit},
        }
        out[model] = res
        L.append("\n### %s" % model)
        L.append("%-22s %6s %7s %7s | OOD正则误答 OOD语义误答(人工)" %
                 ("tier", "Hit", "Recall", "Prec"))
        for n, k in (("fused(无门控)", "fused"), ("strict", "strict"),
                     ("advisory(+图投票)", "advisory")):
            r = res[k]
            sem = r.get("ood_semantic_leak", "-")
            L.append("%-22s %6.1f %7.1f %7s | %6d/30 %8s" %
                     (n, 100 * r["hit"], 100 * r["recall"],
                      ("%.1f" % (100 * r["precision"])) if "precision" in r else "-",
                      r["ood_regex_leak"], sem))
        L.append("strict   recall vs fused: %s" % ci_s)
        L.append("advisory recall vs fused: %s" % ci_a)
        L.append("advisory hit    vs fused: %s" % ci_a_h)
    json.dump(out, open(os.path.join(ROOT, "rag_results/release_v2_dtu",
                                     "dual_tier_eval.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    rep = "\n".join(L)
    open(os.path.join(ROOT, "rag_results/release_v2_dtu",
                      "dual_tier_eval.txt"), "w", encoding="utf-8").write(rep)
    print(rep)


if __name__ == "__main__":
    main()
