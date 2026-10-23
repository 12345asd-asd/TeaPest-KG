# -*- coding: utf-8 -*-
"""
TeaPest-QA v2 评估（158 题，三层）。

答案题(stratum != ood, 128)：否定感知的实体 Hit/Recall/Prec/F1（macro+micro），
  并对诊断题拆分「诊断正确率（病虫名命中）」与「防治覆盖率（控制项 recall）」。
OOD 题(30)：expected=abstain。报「恰当拒答率」与「高置信误答率」，
  按 unknown_pest / crossdomain / contradictory 细分；不并入答案准确率。
门控方法：报 2x2 路由混淆矩阵（answerable/OOD × 作答/拒答）及开闸子集准确率。
显著性：答案题逐题指标的配对非参 bootstrap(10000)。
"""
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "trustkg_rag", "experiments"))

from run_release_experiments import score_answer  # noqa: E402
from metrics import bootstrap_significance_test  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
_env_d = os.environ.get("TEAPEST_RUN_DIR")
D = _env_d if os.path.isabs(_env_d or "") else (
    os.path.join(ROOT, _env_d) if _env_d
    else os.path.join(ROOT, "rag_results", "release_v2"))
MODEL_TAG = os.environ.get("TEAPEST_RUN_TAG", "qwen2.5-14b")
qa_all = json.load(open(os.path.join(ROOT, "release_qa", "qa_v2.json"),
                        encoding="utf-8"))
qa_map = {q["id"]: q for q in qa_all}

ABSTAIN_PAT = re.compile(
    r"无法确定|不能确定|难以确定|无法判断|无法可靠|信息不足|信息不充分|资料不足|"
    r"没有足够|未收录|没有收录|暂不(给出|提供|建议)|建议咨询|请教当地|植保站|农业技术|"
    r"无法给出|不能给出|无法提供|无法准确|难以判断|超出(了)?(现有)?(知识|范围|能力)|"
    r"知识库(中)?(没有|未)|没有找到相关|不具备相关|建议核实|结合田间实测")

METHODS = ["zeroshot", "fewshot", "linear_bm25", "graphrag", "kgrag1hop",
           "kgrag2hop", "kg_only", "fused_parallel", "gated_router"]
LLM_METHODS = ["zeroshot", "fewshot", "linear_bm25", "graphrag", "kgrag1hop",
               "kgrag2hop", "fused_parallel"]


def is_abstain(text):
    return bool(ABSTAIN_PAT.search(text or ""))


def load(name):
    p = os.path.join(D, "preds_%s.json" % name)
    if not os.path.exists(p):
        return None
    return {r["qa_id"]: r["generated_answer"] for r in
            json.load(open(p, encoding="utf-8"))}


def diag_split(q, pred):
    """诊断题：返回 (病虫名命中, 防治项命中数, 防治项总数)。"""
    if q["type"] == "open_diagnosis":
        pest = q["answer_list"][0]
        controls = q["answer_list"][1:]
    elif q["type"] == "symptom_identify":
        pest, controls = q["answer_list"][0], []
    else:  # differential：多个病虫名，无控制项
        pest, controls = None, q["answer_list"]
    m = score_answer(pred, q)
    if m is None:
        return 0.0, 0, 0
    # 病虫名是否被提及（用 find_mentions 同款口径：直接子串 + 否定已在 score 内处理）
    from run_release_experiments import find_mentions, negated
    ments = [e for e, pos in find_mentions(pred) if not negated(pred, pos)]
    if q["type"] == "differential":
        hit = sum(1 for g in controls if g in ments) / len(controls)
        return hit, 0, 0
    d_hit = 1.0 if pest in ments else 0.0
    if not controls:
        return d_hit, 0, 0
    c_hit = sum(1 for g in controls if g in ments)
    return d_hit, c_hit, len(controls)


def eval_method(name, preds):
    ans_rows, ood_rows = [], []
    for q in qa_all:
        pred = preds.get(q["id"], "")
        if q["expected"] == "abstain":
            ood_rows.append({"id": q["id"], "type": q["type"],
                             "abstain": 1 if is_abstain(pred) else 0})
        else:
            m = score_answer(pred, q)
            if m is None:
                continue
            m.update(id=q["id"], stratum=q["stratum"], type=q["type"],
                     difficulty=q["difficulty"], abstain=is_abstain(pred))
            if q["stratum"] == "diagnosis":
                dh, ch, cn = diag_split(q, pred)
                m["diag_hit"] = dh
                if cn:
                    m["control_hit"] = ch / cn
                    m["n_control"] = cn
            ans_rows.append(m)
    return ans_rows, ood_rows


def agg_answer(rows):
    n = len(rows)
    if not n:
        return {}
    micro_c = sum(r["n_correct"] for r in rows)
    micro_g = sum(r["n_gold"] for r in rows)
    micro_p = sum(r["n_pred"] for r in rows)
    return {
        "n": n,
        "hit": round(sum(r["hit"] for r in rows) / n, 4),
        "recall": round(sum(r["recall"] for r in rows) / n, 4),
        "precision": round(sum(r["precision"] for r in rows) / n, 4),
        "f1": round(sum(r["f1"] for r in rows) / n, 4),
        "recall_micro": round(micro_c / max(1, micro_g), 4),
        "precision_micro": round(micro_c / max(1, micro_p), 4),
        "abstain_rate": round(sum(r["abstain"] for r in rows) / n, 4),
    }


def main():
    data = {}
    per_q = {}
    for name in METHODS:
        preds = load(name)
        if preds is None:
            print("missing preds: %s" % name)
            continue
        ans, ood = eval_method(name, preds)
        data[name] = {"all": agg_answer(ans), "ood": ood, "ans_rows": ans}
        per_q[name] = ans

    # 分层聚合
    summary = {}
    for name, d in data.items():
        rows = d["ans_rows"]
        s = {"all_answer": agg_answer(rows)}
        for strat in ("lookup", "diagnosis"):
            sub = [r for r in rows if r["stratum"] == strat]
            s[strat] = agg_answer(sub)
            if strat == "diagnosis":
                dh = [r["diag_hit"] for r in sub if "diag_hit" in r]
                cr = [r["control_hit"] for r in sub if "control_hit" in r]
                s["diagnosis_acc"] = round(sum(dh) / len(dh), 4) if dh else None
                s["control_recall"] = round(sum(cr) / len(cr), 4) if cr else None
                for t in ("symptom_identify", "open_diagnosis", "differential"):
                    ts = [r for r in sub if r["type"] == t]
                    if ts:
                        s["by_type_" + t] = agg_answer(ts)
                        if t == "open_diagnosis":
                            ch = [r["control_hit"] for r in ts if "control_hit" in r]
                            dh2 = [r["diag_hit"] for r in ts if "diag_hit" in r]
                            s["by_type_" + t + "_diagAcc"] = round(
                                sum(dh2) / len(dh2), 4)
                            s["by_type_" + t + "_ctrlRec"] = round(
                                sum(ch) / len(ch), 4)
        for diff in ("easy", "medium", "hard"):
            sub = [r for r in rows if r["difficulty"] == diff]
            if sub:
                s["diff_" + diff] = agg_answer(sub)
        # OOD 拒答
        ood = d["ood"]
        s["ood"] = {
            "n": len(ood),
            "abstain_rate": round(sum(r["abstain"] for r in ood) / len(ood), 4),
            "false_assert_rate": round(
                sum(1 - r["abstain"] for r in ood) / len(ood), 4),
        }
        for t in ("ood_unknown_pest", "ood_crossdomain", "ood_contradictory"):
            ts = [r for r in ood if r["type"] == t]
            if ts:
                s["ood"][t] = round(sum(r["abstain"] for r in ts) / len(ts), 3)
        summary[name] = s

    # 门控 2x2 混淆矩阵
    gate_meta = {}
    mp = os.path.join(D, "gate_meta.json")
    if os.path.exists(mp):
        gate_meta = json.load(open(mp, encoding="utf-8"))
    if gate_meta:
        cm = {"ans_answer": 0, "ans_abstain": 0, "ood_answer": 0,
              "ood_abstain": 0}
        for q in qa_all:
            opened = gate_meta[q["id"]]["gate"] == "answer"
            if q["expected"] == "abstain":
                cm["ood_answer" if opened else "ood_abstain"] += 1
            else:
                cm["ans_answer" if opened else "ans_abstain"] += 1
        summary["__gate_confusion__"] = cm
        # 开闸子集答案准确率（gated 作答部分 = graphrag 答案）
        gr = data.get("graphrag")
        if gr:
            opened_rows = [r for r in gr["ans_rows"]
                           if gate_meta[r["id"]]["gate"] == "answer"]
            summary["__gate_opened_answer_acc__"] = agg_answer(opened_rows)

    # 配对 bootstrap（graphrag vs 其余 LLM 方法，答案题 recall/hit）
    tests = {}
    if "graphrag" in per_q:
        gmap = {r["id"]: r for r in per_q["graphrag"]}
        for name in LLM_METHODS:
            if name == "graphrag" or name not in per_q:
                continue
            omap = {r["id"]: r for r in per_q[name]}
            ids = [i for i in gmap if i in omap]
            for metric in ("recall", "hit"):
                a = [omap[i][metric] for i in ids]
                b = [gmap[i][metric] for i in ids]
                tests["graphrag_vs_%s_%s" % (name, metric)] = \
                    bootstrap_significance_test(a, b, n_bootstrap=10000)

    json.dump({"summary": summary, "bootstrap": tests},
              open(os.path.join(D, "evals_v2.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # ---------- 文本表 ----------
    n_ans = sum(1 for q in qa_all if q["expected"] != "abstain")
    n_ood = len(qa_all) - n_ans
    L = ["TeaPest-QA v2 (n=%d; answer=%d, OOD=%d) %s"
         % (len(qa_all), n_ans, n_ood, MODEL_TAG), "=" * 92]
    L.append("%-15s %5s %7s %7s %7s %7s %8s | %8s %8s" %
             ("Method", "n", "Hit", "Recall", "Prec", "F1", "RecMi",
              "OOD拒答", "OOD误答"))
    for name in METHODS:
        s = summary.get(name)
        if not s:
            continue
        a = s["all_answer"]; o = s["ood"]
        L.append("%-15s %5d %7.1f %7.1f %7.1f %7.1f %8.1f | %8.1f %8.1f" %
                 (name, a["n"], 100 * a["hit"], 100 * a["recall"],
                  100 * a["precision"], 100 * a["f1"],
                  100 * a["recall_micro"], 100 * o["abstain_rate"],
                  100 * o["false_assert_rate"]))
    L.append("=" * 92)
    for name in METHODS:
        s = summary.get(name)
        if not s:
            continue
        L.append("\n[%s]" % name)
        for strat, zh in (("lookup", "事实查询"), ("diagnosis", "诊断")):
            if s.get(strat):
                t = s[strat]
                L.append("  %-6s n=%3d Hit=%.1f Recall=%.1f Prec=%.1f F1=%.1f"
                         % (zh, t["n"], 100 * t["hit"], 100 * t["recall"],
                            100 * t["precision"], 100 * t["f1"]))
        if s.get("diagnosis_acc") is not None:
            L.append("  诊断正确率=%.1f%%  防治覆盖率=%.1f%%" %
                     (100 * s["diagnosis_acc"],
                      100 * (s["control_recall"] or 0)))
        ot = s["open_diagnosis"] if "open_diagnosis" in s else s.get(
            "by_type_open_diagnosis")
        if s.get("by_type_open_diagnosis_diagAcc") is not None:
            L.append("    open_diag n=%d 诊断Acc=%.1f 防治Rec=%.1f" %
                     (s["by_type_open_diagnosis"]["n"],
                      100 * s["by_type_open_diagnosis_diagAcc"],
                      100 * s["by_type_open_diagnosis_ctrlRec"]))
        o = s["ood"]
        L.append("  OOD n=%d 拒答=%.1f%% 误答=%.1f%%  [未知病虫%.0f 跨域%.0f 矛盾%.0f]"
                 % (o["n"], 100 * o["abstain_rate"],
                    100 * o["false_assert_rate"],
                    100 * o.get("ood_unknown_pest", 0),
                    100 * o.get("ood_crossdomain", 0),
                    100 * o.get("ood_contradictory", 0)))
    if "__gate_confusion__" in summary:
        cm = summary["__gate_confusion__"]
        L.append("\n[门控路由混淆矩阵]")
        L.append("  可答题: 作答=%d 误拒=%d | OOD: 正确拒答=%d 泄漏误答=%d" %
                 (cm["ans_answer"], cm["ans_abstain"],
                  cm["ood_abstain"], cm["ood_answer"]))
        oa = summary.get("__gate_opened_answer_acc__")
        if oa:
            L.append("  门控开闸子集 n=%d 准确率 Hit=%.1f Recall=%.1f" %
                     (oa["n"], 100 * oa["hit"], 100 * oa["recall"]))
    L.append("\n[配对bootstrap: graphrag 优于对照的 95%%CI(recall/hit)]")
    for k, v in tests.items():
        L.append("  %-34s diff=%+.3f CI=[%+.3f,%+.3f] p=%.3f" %
                 (k, v["diff"], v["ci_95"][0], v["ci_95"][1], v["p_value"]))
    rep = "\n".join(L)
    open(os.path.join(D, "comparison_v2.txt"), "w",
         encoding="utf-8").write(rep)
    print(rep)


if __name__ == "__main__":
    main()
