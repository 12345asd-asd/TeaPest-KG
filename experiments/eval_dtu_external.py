# -*- coding: utf-8 -*-
"""
DTU 方法在外部手册集（55 题）上的评测包装。
复用 eval_external 的 normalize/score_ban/score_groups 全部评分口径；
DTU preds 取自 TEAPEST_RUN_DIR（默认 rag_results/external_dtu），
基线 preds 取自 rag_results/external_manual（14B 既有缓存）。
结果 -> <dir>/dtu_external_eval.json / .txt
"""
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "trustkg_rag", "experiments"))

_env_d = os.environ.get("TEAPEST_RUN_DIR")
D = _env_d if os.path.isabs(_env_d or "") else (
    os.path.join(ROOT, _env_d) if _env_d
    else os.path.join(ROOT, "rag_results", "external_dtu"))
BASE = os.path.join(ROOT, "rag_results",
                    os.environ.get("TEAPEST_BASE_DIR", "external_manual"))
MODEL_TAG = os.environ.get("TEAPEST_RUN_TAG", "qwen2.5-14b-dtu")

import eval_external as X  # noqa: E402

QA = X.QA
METHODS = ["fused_parallel", "gated_router", "dtu_fused", "dtu_kgrag1hop",
           "dtu_gated"]


def load_preds(directory, name):
    p = os.path.join(directory, "preds_%s.json" % name)
    if not os.path.exists(p):
        return None
    return {r["qa_id"]: r["generated_answer"]
            for r in json.load(open(p, encoding="utf-8"))}


def score_all(preds):
    rows = []
    for q in QA:
        text = preds.get(q["id"], "")
        nt = X.normalize(text)
        ab = X.is_abstain(text)
        if q["type"] == "ban":
            label, det = X.score_ban(q, text, nt)
            rows.append({"id": q["id"], "type": "ban",
                         "kg_answerable": q["kg_answerable"], "abstain": ab,
                         "label": label, "detail": det,
                         "score": 1.0 if label == "safe" else 0.0})
        else:
            matched, need, total = X.score_groups(q, nt)
            sc = min(1.0, matched / need) if need else 0.0
            if ab and matched == 0:
                label = "abstain"
            elif matched >= need:
                label = "correct"
            else:
                label = "wrong"
            has_num = bool(re.search(r"\d", re.sub(r"\d{4}", "", nt)))
            rows.append({"id": q["id"], "type": q["type"],
                         "kg_answerable": q["kg_answerable"], "abstain": ab,
                         "label": label, "matched": matched, "need": need,
                         "numeric_assertion": has_num, "score": sc})
    return rows


def summarize(rows):
    def agg(filt, key=lambda r: r["score"]):
        rs = [r for r in rows if filt(r)]
        return round(sum(key(r) for r in rs) / len(rs), 4) if rs else None

    nonban = [r for r in rows if r["type"] != "ban"]
    beyond = [r for r in nonban if not r["kg_answerable"]]
    kga = [r for r in nonban if r["kg_answerable"]]
    ban = [r for r in rows if r["type"] == "ban"]
    n_ab = sum(r["label"] == "abstain" for r in beyond)
    n_numwrong = sum(r["label"] == "wrong" and r["numeric_assertion"]
                     for r in beyond)
    by_type = {}
    for t in ("threshold", "interval", "measure", "chemical", "shared"):
        trs = [r for r in nonban if r["type"] == t]
        if trs:
            by_type[t] = {
                "n": len(trs),
                "correct": sum(r["label"] == "correct" for r in trs),
                "abstain": sum(r["label"] == "abstain" for r in trs)}
    return {
        "exam_correct_all": agg(lambda r: True),
        "by_type": by_type,
        "kg_answerable_n": len(kga),
        "kg_answerable_answered": sum(r["label"] != "abstain" for r in kga),
        "kg_answerable_correct": agg(lambda r: r["kg_answerable"]),
        "beyond_n": len(beyond),
        "beyond_abstain_rate": round(n_ab / len(beyond), 4),
        "beyond_answer_n": sum(r["label"] != "abstain" for r in beyond),
        "beyond_numeric_wrong_rate": round(n_numwrong / len(beyond), 4),
        "ban_n": len(ban),
        "ban_safe_rate": round(sum(r["label"] == "safe" for r in ban)
                               / max(1, len(ban)), 4),
        "ban_unsafe_n": sum(r["label"] == "unsafe" for r in ban),
    }


def main():
    audits = {}
    for name in METHODS:
        preds = load_preds(D, name) or load_preds(BASE, name)
        if preds is None:
            print("missing:", name)
            continue
        audits[name] = score_all(preds)
    summary = {n: summarize(rows) for n, rows in audits.items()}

    import numpy as np
    rng = np.random.RandomState(20261017)

    def paired(a, b, keyf, subset=lambda r: True, n_boot=10000):
        ra, rb = {r["id"]: r for r in audits[a]}, {r["id"]: r for r in audits[b]}
        ids = [i for i in ra if i in rb and subset(ra[i])]
        da = np.array([1.0 if keyf(ra[i]) else 0.0 for i in ids])
        db = np.array([1.0 if keyf(rb[i]) else 0.0 for i in ids])
        diff = db - da
        boots = [diff[rng.randint(0, len(diff), len(diff))].mean()
                 for _ in range(n_boot)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return {"n": len(ids), "diff": round(float(diff.mean()), 4),
                "ci95": [round(float(lo), 4), round(float(hi), 4)]}

    beyond = lambda r: r["type"] != "ban" and not r["kg_answerable"]
    kga = lambda r: r["type"] != "ban" and r["kg_answerable"]
    tests = {}
    if "fused_parallel" in audits and "dtu_fused" in audits:
        tests["abstain_dtu_minus_fused_beyond"] = paired(
            "fused_parallel", "dtu_fused",
            lambda r: r["label"] == "abstain", beyond)
        tests["numwrong_dtu_minus_fused_beyond"] = paired(
            "fused_parallel", "dtu_fused",
            lambda r: r["label"] == "wrong" and r["numeric_assertion"], beyond)
        tests["answered_dtu_minus_fused_kga"] = paired(
            "fused_parallel", "dtu_fused",
            lambda r: r["label"] != "abstain", kga)
    if "dtu_fused" in audits:
        tests["bansafe_dtu_n6"] = {
            "rate": summary["dtu_fused"]["ban_safe_rate"]}

    out = {"model": MODEL_TAG, "methods": summary, "bootstrap": tests}
    json.dump(out, open(os.path.join(D, "dtu_external_eval.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(audits, open(os.path.join(D, "dtu_external_audit.json"), "w",
                           encoding="utf-8"), ensure_ascii=False, indent=2)

    L = ["TeaPest-QA-External DTU (%s; n=55)" % MODEL_TAG, "=" * 104]
    L.append("%-15s %7s | %7s %9s %9s | %8s %7s | KGA作答" %
             ("Method", "全卷正", "KG外拒", "KG外作答", "数值误答",
              "禁药安全", "危险"))
    for name in METHODS:
        s = summary.get(name)
        if not s:
            continue
        L.append("%-15s %7.1f | %7.1f %9d %9.1f | %8.1f %7d | %d/%d" % (
            name, 100 * (s["exam_correct_all"] or 0),
            100 * s["beyond_abstain_rate"], s["beyond_answer_n"],
            100 * s["beyond_numeric_wrong_rate"],
            100 * s["ban_safe_rate"], s["ban_unsafe_n"],
            s["kg_answerable_answered"], s["kg_answerable_n"]))
    L.append("=" * 104)
    for k, v in tests.items():
        L.append("  %-42s %s" % (k, v))
    rep = "\n".join(L)
    open(os.path.join(D, "dtu_external_eval.txt"), "w",
         encoding="utf-8").write(rep)
    print(rep)


if __name__ == "__main__":
    main()
