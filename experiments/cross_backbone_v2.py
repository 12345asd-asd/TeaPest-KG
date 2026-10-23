# -*- coding: utf-8 -*-
"""
跨骨干对比（qwen2.5-14b-custom vs qwen2.5:7b），TeaPest-QA v2（158 题）。

输入：两个结果目录下由 eval_release_v2.py 生成的 evals_v2.json
（rag_results/release_v2 与 rag_results/release_v2_7b）。
关注点不是“7B 低几分”，而是核心结论是否在第二个骨干上复现：
  R1 结构化证据相对纯 LLM 的巨大增益仍在；
  R2 组织×题型交叉方向一致：profile 诊断强/事实弱，kgrag1hop 反之；
  R3 fused 两层都强、总体最优或并列；
  R4 无门控 OOD 高误答、门控 0 泄漏（门控是确定性的，两骨干规则相同，
     但其开闸子集答案分随骨干变化）。
输出 cross_backbone_v2.json/.txt 到 14B 目录。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))
D14 = os.path.join(ROOT, "rag_results", "release_v2")
D7 = os.path.join(ROOT, "rag_results", "release_v2_7b")
TAG14, TAG7 = "qwen2.5-14b", "qwen2.5-7b"

METHODS = ["zeroshot", "fewshot", "linear_bm25", "kg_only", "kgrag2hop",
           "kgrag1hop", "graphrag", "fused_parallel", "gated_router"]
PRETTY = {
    "zeroshot": "Zero-Shot", "fewshot": "Few-Shot",
    "linear_bm25": "Flat-BM25", "kg_only": "KG-Only",
    "kgrag2hop": "KG-RAG 2-hop", "kgrag1hop": "KG-RAG 1-hop",
    "graphrag": "Profile-RAG", "fused_parallel": "Parallel-Fusion",
    "gated_router": "Gated-RAG",
}


def load(d):
    return json.load(open(os.path.join(d, "evals_v2.json"), encoding="utf-8"))


def cell(s, key, sub="all_answer"):
    return s[sub][key] if s and sub in s else None


def main():
    e14, e7 = load(D14)["summary"], load(D7)["summary"]
    rows, out = [], {}
    for m in METHODS:
        a, b = e14.get(m), e7.get(m)
        if not a or not b:
            continue
        r = {
            "method": m,
            "recall_14b": cell(a, "recall"), "recall_7b": cell(b, "recall"),
            "hit_14b": cell(a, "hit"), "hit_7b": cell(b, "hit"),
            "lookup_r_14b": cell(a, "recall", "lookup"),
            "lookup_r_7b": cell(b, "recall", "lookup"),
            "diag_r_14b": cell(a, "recall", "diagnosis"),
            "diag_r_7b": cell(b, "recall", "diagnosis"),
            "diagacc_14b": a.get("diagnosis_acc"),
            "diagacc_7b": b.get("diagnosis_acc"),
            "ctrl_14b": a.get("control_recall"),
            "ctrl_7b": b.get("control_recall"),
            "ood_abstain_14b": a["ood"]["abstain_rate"],
            "ood_abstain_7b": b["ood"]["abstain_rate"],
        }
        rows.append(r)
        out[m] = r

    def gap(summary, m1, m2, sub, metric):
        a, b = summary.get(m1), summary.get(m2)
        if not a or not b or sub not in a or sub not in b:
            return None
        return a[sub][metric] - b[sub][metric]

    # 复现判定（方向一致性，点估计；显著性由各目录 significance_v2.json 提供）
    checks = {}
    for tag, s in ((TAG14, e14), (TAG7, e7)):
        checks[tag] = {
            # R1: fused vs zeroshot 总体 recall
            "fused_minus_zeroshot_allR":
                round(100 * gap(s, "fused_parallel", "zeroshot",
                                "all_answer", "recall"), 1),
            # R2: profile - kgrag1hop，诊断应为正、事实应为负
            "profile_minus_kgrag_diagR":
                round(100 * gap(s, "graphrag", "kgrag1hop",
                                "diagnosis", "recall"), 1),
            "profile_minus_kgrag_lookupR":
                round(100 * gap(s, "graphrag", "kgrag1hop",
                                "lookup", "recall"), 1),
            # R3: fused 总体 recall 是否 >= 两个单组织
            "fused_ge_both_overall":
                (s["fused_parallel"]["all_answer"]["recall"]
                 >= s["graphrag"]["all_answer"]["recall"] - 1e-9
                 and s["fused_parallel"]["all_answer"]["recall"]
                 >= s["kgrag1hop"]["all_answer"]["recall"] - 1e-9),
            "gate_ood_abstain": s.get("gated_router", {}).get(
                "ood", {}).get("abstain_rate"),
        }
    out["replication_checks"] = checks

    # 门控混淆矩阵（确定性规则，两骨干应相同；列出供核对）
    for tag, s in ((TAG14, e14), (TAG7, e7)):
        cm = s.get("__gate_confusion__")
        if cm:
            out.setdefault("gate_confusion", {})[tag] = cm

    json.dump(out, open(os.path.join(D14, "cross_backbone_v2.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2)

    # ---------- 文本表 ----------
    L = ["TeaPest-QA v2 跨骨干对比 (14B vs 7B; answer=128, OOD=30)",
         "=" * 104]
    L.append("%-16s | %6s %6s | %6s %6s | %6s %6s | %6s %6s | %7s %7s" % (
        "Method", "R14b", "R7b", "Look14", "Look7", "Diag14", "Diag7",
        "ID14", "ID7", "OOD拒14", "OOD拒7"))
    for r in rows:
        L.append("%-16s | %6.1f %6.1f | %6.1f %6.1f | %6.1f %6.1f | "
                 "%6s %6s | %7.1f %7.1f" % (
                     PRETTY[r["method"]],
                     100 * r["recall_14b"], 100 * r["recall_7b"],
                     100 * r["lookup_r_14b"], 100 * r["lookup_r_7b"],
                     100 * r["diag_r_14b"], 100 * r["diag_r_7b"],
                     ("%.0f" % (100 * r["diagacc_14b"]))
                     if r["diagacc_14b"] is not None else "-",
                     ("%.0f" % (100 * r["diagacc_7b"]))
                     if r["diagacc_7b"] is not None else "-",
                     100 * r["ood_abstain_14b"], 100 * r["ood_abstain_7b"]))
    L.append("=" * 104)
    L.append("核心结论复现判定：")
    for tag in (TAG14, TAG7):
        c = checks[tag]
        L.append("  [%s] fused-zero=+%.1f | profile-kgrag 诊断%+.1f 事实%+.1f | "
                 "fused双不弱=%s | 门控OOD拒答=%s" % (
                     tag, c["fused_minus_zeroshot_allR"],
                     c["profile_minus_kgrag_diagR"],
                     c["profile_minus_kgrag_lookupR"],
                     c["fused_ge_both_overall"], c["gate_ood_abstain"]))
    rep = "\n".join(L)
    open(os.path.join(D14, "cross_backbone_v2.txt"), "w",
         encoding="utf-8").write(rep)
    print(rep)


if __name__ == "__main__":
    main()
