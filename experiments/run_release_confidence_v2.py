# -*- coding: utf-8 -*-
"""
TeaPest-QA v2 置信/选择性预测（158 题，含 30 OOD）。

与 v1 的本质区别：v1 的 106 题 92 题全对、错误仅 14 个且 verbalized 退化为常数 85，
软预警无信号；v2 在诊断层有真实方差、并有 30 道期望拒答的 OOD 题，选择性可被真正评测。

正确性定义（逐题 0/1）：
  answer 题：实体 recall≥0.5；
  OOD 题：系统恰当拒答（与 eval_release_v2 同一套拒答判定）→1。
三路置信（越高越可信）：
  verbalized —— 生成后让模型自报 0-100；
  structural —— 仅问题+KG：链接到病虫→档案完整度；否则命中 KG 症状短语比例；OOD≈0；
  gate       —— 溯源门控（开=1/关=0），即 gated_router 的硬路由信号。
指标：ECE(原始/cross-fit保序)、Brier、AURC、AUROC、固定覆盖率准确率；
软预警阈值表按【校准结构置信的分位数】取阈值，保证非空且反映真实可操作区间。
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "trustkg_rag", "experiments"))

from run_release_experiments import call_llm, score_answer  # noqa: E402
import run_release_v2 as RV2  # noqa: E402
from eval_release_v2 import is_abstain  # noqa: E402
from metrics import (  # noqa: E402
    expected_calibration_error, brier_score, crossfit_calibrate,
    area_under_risk_coverage)

ROOT = os.path.dirname(os.path.abspath(__file__))
_env_d = os.environ.get("TEAPEST_RUN_DIR")
RUN_DIR = _env_d if os.path.isabs(_env_d or "") else (
    os.path.join(ROOT, _env_d) if _env_d
    else os.path.join(ROOT, "rag_results", "release_v2"))
OUT = os.path.join(RUN_DIR, "confidence")
os.makedirs(OUT, exist_ok=True)
CORRECT_RECALL_TAU = 0.5
RELS5 = ["表现症状", "生物防治", "化学防治", "农业防治", "物理防治"]


def verbalized_conf(question, answer):
    prompt = (
        "你之前回答了以下问题：\n\n问题：%s\n\n你的回答：\n%s\n\n"
        "请评估你对该回答正确性的把握。仅输出一个 0-100 的整数，"
        "0=完全不确定/可能答错或超出已知范围，100=完全确定。只输出数字。"
        % (question, answer[:400]))
    txt = call_llm([{"role": "user", "content": prompt}],
                   max_tokens=16, temperature=0)
    import re as _re
    nums = _re.findall(r"\d{1,3}", txt or "")
    if nums:
        return max(0.0, min(1.0, int(nums[0]) / 100.0)), txt.strip()
    return 0.5, (txt or "").strip()


def structural_conf(q):
    link, prof, pests, phrases = RV2.gate_signal(q)
    if pests:
        rels = tuple(RELS5[1:])  # 四类防治
        have = sum(1 for r in rels
                   if any(t["head"] == pests[0] and t["relation"] == r
                          for t in RV2.retriever.triples))
        return 0.6 + 0.1 * have          # 0.6..1.0，档案越全越高
    # 未链接病虫：看题干逐字命中的 KG 症状短语覆盖了多少档案
    if phrases:
        return min(0.6, 0.15 * len(phrases))
    return 0.0


def gate_conf(q):
    _, _, pests, phrases = RV2.gate_signal(q)
    return 1.0 if (pests or phrases) else 0.0


def collect(method):
    preds = {r["qa_id"]: r["generated_answer"]
             for r in json.load(open(os.path.join(
                 RUN_DIR, "preds_%s.json" % method), encoding="utf-8"))}
    cache_p = os.path.join(OUT, "verbalized_%s.json" % method)
    recs = {}
    if os.path.exists(cache_p):
        recs = {r["qa_id"]: r for r in
                json.load(open(cache_p, encoding="utf-8"))}
    rows, t0 = [], time.time()
    for i, q in enumerate(RV2.qa_all):
        qid = q["id"]; ans = preds.get(qid, "")
        if q["expected"] == "abstain":
            correct = 1 if is_abstain(ans) else 0
        else:
            m = score_answer(ans, q)
            correct = 1 if (m and m["recall"] >= CORRECT_RECALL_TAU) else 0
        if qid in recs:
            verb = recs[qid]["verbalized"]
        else:
            verb, raw = verbalized_conf(q["question"], ans)
            recs[qid] = {"qa_id": qid, "verbalized": verb,
                         "verbalized_raw": raw}
            json.dump(list(recs.values()), open(cache_p, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
        rows.append({"qa_id": qid, "stratum": q["stratum"], "type": q["type"],
                     "correct": correct, "verbalized": verb,
                     "structural": round(structural_conf(q), 3),
                     "gate": gate_conf(q)})
        if i % 25 == 0:
            print("  [%d/%d] %.0fs" % (i + 1, len(RV2.qa_all),
                                        time.time() - t0), flush=True)
    return rows


def auroc(y, s):
    import numpy as np
    y = np.array(y); s = np.array(s)
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    # Mann-Whitney U（含并列取 0.5）
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # 并列平均秩
    vals = s[order]; i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        rk = (i + 1 + j + 1) / 2.0
        ranks[order[i:j + 1]] = rk
        i = j + 1
    U = ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2
    return float(U / (len(pos) * len(neg)))


def eval_signal(rows, key):
    y = [r["correct"] for r in rows]
    c = [r[key] for r in rows]
    cal = crossfit_calibrate(y, c, method="isotonic")
    sorted_c = sorted(c)
    taus = sorted(set(round(sorted_c[int(len(sorted_c) * f)], 3)
                      for f in (0.1, 0.2, 0.3, 0.4, 0.5)))
    table = []
    n = len(y); nerr = n - sum(y)
    for tau in taus:
        flag = [i for i in range(n) if cal[i] < tau]
        keep = [i for i in range(n) if cal[i] >= tau]
        caught = sum(1 for i in flag if y[i] == 0)
        table.append({
            "tau": tau, "flag_rate": round(len(flag) / n, 3),
            "keep_coverage": round(len(keep) / n, 3),
            "kept_accuracy": round(sum(y[i] for i in keep) / len(keep), 4)
            if keep else 1.0,
            "error_caught_rate": round(caught / nerr, 3) if nerr else 0.0})
    # 排名版操作曲线：部署上按「底部 f 比例转人工」执行（离散结构分有大量并列、
    # 校准后逐分位阈值会整块切不动；排名口径不受尺度/并列影响，与论文表述一致）。
    order = sorted(range(n), key=lambda i: (c[i], i))
    frac_table = []
    for f in (0.1, 0.2, 0.3, 0.4, 0.5):
        k = int(round(f * n))
        flag = set(order[:k])
        keep = [i for i in range(n) if i not in flag]
        caught = sum(1 for i in flag if y[i] == 0)
        frac_table.append({
            "flag_fraction": f, "flag_n": k,
            "keep_coverage": round(len(keep) / n, 3),
            "kept_accuracy": round(sum(y[i] for i in keep) / len(keep), 4)
            if keep else 1.0,
            "error_caught_rate": round(caught / nerr, 3) if nerr else 0.0})
    return {
        "mean_conf": round(sum(c) / n, 4), "accuracy": round(sum(y) / n, 4),
        "ece": expected_calibration_error(y, c),
        "ece_isotonic": expected_calibration_error(y, cal),
        "brier": brier_score(y, c),
        "aurc": area_under_risk_coverage(y, c),
        "auroc": round(auroc(y, c), 4) if auroc(y, c) is not None else None,
        "warning_table": table,
        "warning_by_fraction": frac_table}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="fused_parallel")
    args = ap.parse_args()
    rows = collect(args.method)

    out = {"method": args.method, "n": len(rows),
           "correct_def": "answer:recall>=.5; OOD:abstained",
           "overall_accuracy": round(sum(r["correct"] for r in rows) / len(rows), 4)}
    for key in ("verbalized", "structural", "gate"):
        out[key] = eval_signal(rows, key)
        # 分层 AUROC：OOD 检出（correct=恰当拒答）
    # OOD 子集上「是否拒答」与信号的关系（高置信却误答=危险）
    ood = [r for r in rows if r["stratum"] == "ood"]
    out["ood_n"] = len(ood)
    out["ood_abstain_rate"] = round(sum(r["correct"] for r in ood) / len(ood), 4)
    json.dump(out, open(os.path.join(OUT, "confidence_%s.json" % args.method),
                        "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump(rows, open(os.path.join(OUT, "per_question_%s.json" % args.method),
                         "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print("\n" + "=" * 78)
    print("置信信号对比 on %s (n=%d, acc=%.3f, OOD拒答=%.3f)" %
          (args.method, out["n"], out["overall_accuracy"],
           out["ood_abstain_rate"]))
    print("=" * 78)
    print("%-11s %8s %7s %8s %7s %7s %7s" %
          ("Signal", "meanC", "ECE", "ECEcal", "Brier", "AURC", "AUROC"))
    for k in ("verbalized", "structural", "gate"):
        s = out[k]
        print("%-11s %8.3f %7.3f %8.3f %7.3f %7.3f %7s" %
              (k, s["mean_conf"], s["ece"], s["ece_isotonic"], s["brier"],
               s["aurc"], s["auroc"]))
    for k in ("structural",):
        print("\n软预警阈值表 (%s, p=校准后):" % k)
        print("%-6s %9s %10s %10s %12s" %
              ("tau", "flag", "keepCov", "keptAcc", "errCaught"))
        for w in out[k]["warning_table"]:
            print("%-6.3f %9.2f %10.2f %10.2f %12.2f" %
                  (w["tau"], 100 * w["flag_rate"],
                   100 * w["keep_coverage"], 100 * w["kept_accuracy"],
                   100 * w["error_caught_rate"]))
        print("底部比例转人工（按原始结构分排名）:")
        print("%-6s %9s %10s %10s %12s" %
              ("frac", "flagN", "keepCov", "keptAcc", "errCaught"))
        for w in out[k]["warning_by_fraction"]:
            print("%-6.2f %9d %10.2f %10.2f %12.2f" %
                  (w["flag_fraction"], w["flag_n"],
                   100 * w["keep_coverage"], 100 * w["kept_accuracy"],
                   100 * w["error_caught_rate"]))


if __name__ == "__main__":
    main()
