# -*- coding: utf-8 -*-
"""
TeaPest-QA v2 高阶消融（对应用户方法学三维度）。

维度1 公平性：主表 linear_bm25(三元组平铺句) / graphrag(病虫档案) / kgrag(细粒度
        子图) 消费同一 413 条三元组、同一 14B 骨干；这里另算每方法平均上下文字符，
        并对 graphrag 做与 kgrag 等字符预算的封顶对照（matched-budget）。
维度2 结构敏感性：
        - 串联(serial cascade)：先用档案「诊断病虫名」(短调用)，再取该病虫完整三元组
          生成防治(长调用)；对比并行融合 fused_parallel(一次调用同时给两路)。
        - 融合权重 λ 扫描：证据=λ 三元组 +(1-λ) 病虫档案，λ∈{.25,.5,.75}，
          端点 λ=0=graphrag、λ=1=kgrag1hop(直接引用主表)，画诊断层 recall 曲线。
维度3 失败边界：KG 覆盖度 α∈{.1,.25,.5,.75,1}，按「病虫×关系」分层固定种子抽样。
        - 结构 oracle（零 LLM）：各 α 下 gold 三元组仍存在的比例（全部120答案题）；
        - 端到端：43 题固定面板上 graphrag 在 α∈{.25,.5,.75} 的 recall（α=1 复用主表）。
所有 LLM 消融只在 58 道诊断题/43 题面板上跑以控成本；逐结果缓存、断点续跑。
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "trustkg_rag", "experiments"))

import run_release_experiments as REX  # noqa: E402
from run_release_experiments import (  # noqa: E402
    retriever, call_llm, overlap_score, _linked, score_answer)
from run_all_experiments import build_vectorg_prompt, build_kgrag_prompt  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
_env_d = os.environ.get("TEAPEST_RUN_DIR")
RUN_DIR = _env_d if os.path.isabs(_env_d or "") else (
    os.path.join(ROOT, _env_d) if _env_d
    else os.path.join(ROOT, "rag_results", "release_v2"))
D = os.path.join(RUN_DIR, "ablation")
os.makedirs(D, exist_ok=True)
qa_all = json.load(open(os.path.join(ROOT, "release_qa", "qa_v2.json"),
                        encoding="utf-8"))
qa_map = {q["id"]: q for q in qa_all}
DIAG_Q = [q for q in qa_all if q["stratum"] == "diagnosis"]
ANS_Q = [q for q in qa_all if q["expected"] == "answer"]
# 固定端到端面板：每 3 题取 1（确定性，约 40 题）
PANEL_Q = [q for i, q in enumerate(ANS_Q) if i % 3 == 0]
RELS = ["表现症状", "生物防治", "化学防治", "农业防治", "物理防治"]
ALPHAS = [0.1, 0.25, 0.5, 0.75, 1.0]


def cache(name, rows, extra=None):
    json.dump(rows, open(os.path.join(D, name), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    if extra:
        json.dump(extra, open(os.path.join(D, name.replace(".json", "_meta.json"),
                                          "w"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)


def load(name):
    p = os.path.join(D, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


# ---------------- 降级 KG（分层固定种子抽样） ----------------
def degrade(alpha, seed=20261015):
    """返回 {(head,rel,tail): triple} 在覆盖度 alpha 下的保留集合（按病虫×关系分层）。"""
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for t in retriever.triples:
        grouped[(t["head"], t["relation"])].append(t)
    keep = {}
    for (h, r), ts in grouped.items():
        ts = sorted(ts, key=lambda x: x["tail"])
        rng.shuffle(ts)
        import math
        k = max(1 if alpha > 0 else 0, math.ceil(len(ts) * alpha)) if alpha < 1 else len(ts)
        for t in ts[:k]:
            keep[(t["head"], t["relation"], t["tail"])] = t
    return keep


def profiles_from(triples):
    byp = defaultdict(list)
    for t in triples:
        if retriever.entity_to_type.get(t["head"]) == "PEST-ORGANISM":
            byp[t["head"]].append(t)
    texts = []
    for pest, trs in byp.items():
        syms = sorted({x["tail"] for x in trs if x["relation"] == "表现症状"})
        parts = ["[%s] 症状:%s。" % (pest, "、".join(syms[:6]))] if syms else []
        for r in ("生物防治", "化学防治", "农业防治", "物理防治"):
            ms = sorted({x["tail"] for x in trs if x["relation"] == r})
            if ms:
                parts.append("%s:%s。" % (r, "、".join(ms[:8])))
        texts.append("".join(parts))
    return texts


# ---------------- 维度1：上下文预算 ----------------
def context_budget():
    """统计主表三方法每题平均喂给 LLM 的字符数。"""
    res = defaultdict(list)
    for q in ANS_Q:
        ents = _linked(q)
        sg = retriever.get_subgraph(ents, hops=1, max_edges=25)
        kg = "\n".join("(%s,%s,%s)" % (e["head"], e["relation"], e["tail"])
                       for e in sg["edges"][:20])
        prof = sorted(((overlap_score(s, q["question"]), s) for s in
                       REX.comm_texts),
                      key=lambda x: -x[0])[:3]
        pt = "".join(s for sc, s in prof if sc > 0)
        lin = sorted(((overlap_score(s, q["question"]), s) for s in
                      REX.lin_sents),
                     key=lambda x: -x[0])[:8]
        lt = "".join(s for sc, s in lin if sc > 0)
        res["kgrag1hop"].append(len(kg)); res["graphrag"].append(len(pt))
        res["linear_bm25"].append(len(lt))
    out = {k: {"mean_chars": round(sum(v) / len(v), 1),
               "n_questions": len(v)} for k, v in res.items()}
    cache("context_budget.json", out)
    return out


# ---------------- 维度2：融合权重 λ（诊断50题） ----------------
def unified_prompt(question, edges, prof):
    kg = "\n".join("- （%s，%s，%s）" % (e["head"], e["relation"], e["tail"])
                   for e in edges) or "无"
    pp = "\n".join("- " + s for s in prof) or "无"
    return [{"role": "system", "content":
             "你是茶叶病虫害诊断专家。综合【结构化三元组】与【病虫档案】作答；"
             "资料未覆盖处须明确说明。\n## 三元组\n%s\n## 病虫档案\n%s\n"
             "要求：先诊断，再分类列防治（生物/化学/物理/农业）。" % (kg, pp)},
            {"role": "user", "content": question}]


def run_lambda():
    rows = load("lambda_sweep.json") or {}
    lams = [0.25, 0.5, 0.75]
    t0 = time.time()
    for lam in lams:
        key = "%.2f" % lam
        rows.setdefault(key, {})
        n_edge = round(20 * lam)
        n_prof = round(3 * (1 - lam))
        for i, q in enumerate(DIAG_Q):
            if q["id"] in rows[key]:
                continue
            ents = _linked(q)
            sg = retriever.get_subgraph(ents, hops=1, max_edges=25)
            edges = sg["edges"][:n_edge]
            sc = sorted(((overlap_score(s, q["question"]), s)
                         for s in REX.comm_texts),
                        key=lambda x: -x[0])
            prof = [s for c, s in sc[:n_prof] if c > 0]
            ans = call_llm(unified_prompt(q["question"], edges, prof),
                           max_tokens=900)
            rows[key][q["id"]] = ans
            if i % 10 == 0:
                print("  [λ=%s %d/50] %.0fs" % (key, i + 1, time.time() - t0),
                      flush=True)
            cache("lambda_sweep.json", rows)
    return rows


# ---------------- 维度2：串联级联（诊断50题） ----------------
PEST_SET = sorted(
    [e for e in retriever.entity_to_type
     if retriever.entity_to_type[e] == "PEST-ORGANISM"], key=len, reverse=True)


def run_serial():
    rows = {r["qa_id"]: r for r in (load("serial_cascade.json") or [])}
    t0 = time.time()
    for i, q in enumerate(DIAG_Q):
        if q["id"] in rows:
            continue
        sc = sorted(((overlap_score(s, q["question"]), s)
                     for s in REX.comm_texts),
                    key=lambda x: -x[0])
        top = [s for c, s in sc[:3] if c > 0]
        # stage1：只诊断病虫名（短调用）
        dmsg = [{"role": "system", "content":
                 "你是茶叶病虫害诊断专家。根据病虫档案，只输出一个最可能的"
                 "病虫害规范名称，不要任何解释。\n档案：\n" + "\n".join(top)},
                {"role": "user", "content": q["question"]}]
        name = call_llm(dmsg, max_tokens=24)
        pest = next((p for p in PEST_SET if p in (name or "")), None)
        if pest is None:  # 回退：档案重叠最高者
            pest = top[0].split("]")[0].lstrip("[") if top else ""
            pest = pest if pest in retriever.entity_to_type else None
        # stage2：取该病虫完整三元组作答
        if pest:
            trs = [t for t in retriever.head_index.get(pest, [])]
            edges = [{"head": t["head"], "relation": t["relation"],
                      "tail": t["tail"]} for t in trs][:25]
            sg = {"edges": edges}
            ans = call_llm(build_kgrag_prompt(q["question"], sg, [], hop=1),
                           max_tokens=900)
        else:
            ans = ("现有资料无法可靠确定病虫，暂不给防治建议，建议咨询植保部门。")
        rows[q["id"]] = {"qa_id": q["id"], "diagnosed": pest,
                         "generated_answer": ans}
        if i % 10 == 0:
            print("  [serial %d/50] %.0fs" % (i + 1, time.time() - t0),
                  flush=True)
        cache("serial_cascade.json", list(rows.values()))
    return list(rows.values())


# ---------------- 维度3：KG 覆盖度临界曲线 ----------------
def oracle_curve():
    """结构层（无 LLM）：各 α 下答案题 gold 仍可由降级 KG 推出的平均比例。

    健全性：α=1 时 gold 全部来自完整 KG，应≈1。按题型判定可用性：
      lookup（context_entity=病虫 p）：gold 防治/症状项需保留为 p 的尾；
      symptom_identify（gold=病虫 p）：p 至少保留 1 条症状边（诊断证据还在）；
      open_diagnosis：病虫 p 需保留症状边，防治项需保留为 p 的尾；
      differential：候选病虫各需保留症状边。
    """
    rows = []
    for a in ALPHAS:
        keep = degrade(a)
        adj = defaultdict(lambda: defaultdict(set))   # head -> rel -> {tails}
        for (h, r, t) in keep:
            adj[h][r].add(t)
        per = []
        for q in ANS_Q:
            typ = q["type"]
            gold = q["answer_list"]
            ok = 0
            if typ in ("pest_control", "pest_all_controls", "pest_symptoms"):
                p = q["context_entity"]
                all_tails = set().union(*adj[p].values()) if adj[p] else set()
                ok = sum(1 for g in gold if g in all_tails)
            elif typ == "symptom_identify":
                p = gold[0]
                ok = 1 if adj[p]["表现症状"] else 0
            elif typ == "open_diagnosis":
                p = gold[0]
                ctrl = gold[1:]
                diag = 1 if adj[p]["表现症状"] else 0
                all_tails = set().union(*adj[p].values()) if adj[p] else set()
                chit = sum(1 for g in ctrl if g in all_tails)
                # 诊断证据与防治等权计入 gold 覆盖
                ok = diag + chit
                gold = gold  # gold 长度=1+len(ctrl)
            elif typ == "differential":
                ok = sum(1 for p in gold if adj[p]["表现症状"])
            per.append(ok / len(gold))
        rows.append({"alpha": a, "oracle_gold_recall":
                     round(sum(per) / len(per), 4)})
    cache("coverage_oracle.json", rows)
    return rows


def run_coverage():
    """端到端：固定面板上 graphrag 在降级 KG 上作答。"""
    rows = load("coverage_e2e.json") or {}
    t0 = time.time()
    for a in [0.25, 0.5, 0.75]:
        key = "%.2f" % a
        rows.setdefault(key, {})
        profs = profiles_from(list(degrade(a).values()))
        for i, q in enumerate(PANEL_Q):
            if q["id"] in rows[key]:
                continue
            top = sorted(((overlap_score(s, q["question"]), s) for s in profs),
                         key=lambda x: -x[0])[:3]
            texts = [{"text": s} for c, s in top if c > 0]
            ans = call_llm(build_vectorg_prompt(q["question"], texts),
                           max_tokens=700)
            rows[key][q["id"]] = ans
            if i % 10 == 0:
                print("  [α=%s %d/%d] %.0fs" %
                      (key, i + 1, len(PANEL_Q), time.time() - t0), flush=True)
            cache("coverage_e2e.json", rows)
    return rows


# ---------------- 汇总评分 ----------------
def _rec(ans, q):
    m = score_answer(ans, q)
    return None if m is None else (m["hit"], m["recall"])


def summarize():
    out = {}
    out["context_budget"] = (load("context_budget.json")
                             if os.path.exists(os.path.join(D, "context_budget.json"))
                             else context_budget())
    # λ 曲线（端点取主表 graphrag/kgrag1hop）
    main = {}
    for m in ("graphrag", "kgrag1hop", "fused_parallel"):
        p = os.path.join(RUN_DIR, "preds_%s.json" % m)
        if os.path.exists(p):
            main[m] = {r["qa_id"]: r["generated_answer"]
                       for r in json.load(open(p, encoding="utf-8"))}
    lam_data = load("lambda_sweep.json")
    curve = []
    for lamv, label in [(0.0, "0(graphrag)"), (0.25, ".25"), (0.5, ".5"),
                        (0.75, ".75"), (1.0, "1(kgrag1hop)")]:
        rs = []
        for q in DIAG_Q:
            if lamv == 0.0 and "graphrag" in main:
                a = main["graphrag"].get(q["id"], "")
            elif lamv == 1.0 and "kgrag1hop" in main:
                a = main["kgrag1hop"].get(q["id"], "")
            elif lamv:
                a = (lam_data or {}).get("%.2f" % lamv, {}).get(q["id"], "")
            else:
                a = ""
            m = _rec(a, q)
            if m:
                rs.append(m)
        if rs:
            curve.append({"lambda": label,
                          "hit": round(sum(x[0] for x in rs) / len(rs), 4),
                          "recall": round(sum(x[1] for x in rs) / len(rs), 4),
                          "n": len(rs)})
    out["lambda_curve_diagnosis"] = curve

    # 串联 vs 并行（诊断50）
    ser = load("serial_cascade.json")
    if ser:
        srows = []
        for r in ser:
            q = qa_map[r["qa_id"]]
            m = _rec(r["generated_answer"], q)
            if m:
                diag = 1.0 if r.get("diagnosed") else 0.0
                # 诊断病虫名是否正确
                if q["type"] == "open_diagnosis":
                    tgt = q["answer_list"][0]
                elif q["type"] == "symptom_identify":
                    tgt = q["answer_list"][0]
                else:
                    tgt = None
                diag_correct = (1.0 if tgt and r.get("diagnosed") == tgt else
                                (diag if not tgt else 0.0))
                srows.append(m + (diag_correct,))
        if "fused_parallel" in main:
            prow = [m for q in DIAG_Q if (m := _rec(
                main["fused_parallel"].get(q["id"], ""), q))]
        else:
            prow = []
        out["series_vs_parallel"] = {
            "serial_cascade": {"n": len(srows),
                               "hit": round(sum(x[0] for x in srows) / len(srows), 4),
                               "recall": round(sum(x[1] for x in srows) / len(srows), 4),
                               "stage1_diag_acc": round(
                                   sum(x[2] for x in srows) / len(srows), 4)},
            "parallel_fused": ({"n": len(prow),
                                "hit": round(sum(x[0] for x in prow) / len(prow), 4),
                                "recall": round(sum(x[1] for x in prow) / len(prow), 4)}
                               if prow else None)}

    # 覆盖度曲线（oracle + e2e 面板）
    out["coverage_oracle"] = load("coverage_oracle.json")
    cov = load("coverage_e2e.json")
    graphrag_preds = main.get("graphrag", {})
    if cov:
        e2e = []
        for a in [0.25, 0.5, 0.75, 1.0]:
            rs = []
            for q in PANEL_Q:
                if a == 1.0:
                    ans = graphrag_preds.get(q["id"], "")
                else:
                    ans = cov.get("%.2f" % a, {}).get(q["id"], "")
                m = _rec(ans, q)
                if m:
                    rs.append(m)
            if rs:
                e2e.append({"alpha": a, "n": len(rs),
                            "hit": round(sum(x[0] for x in rs) / len(rs), 4),
                            "recall": round(sum(x[1] for x in rs) / len(rs), 4)})
        out["coverage_e2e_panel"] = e2e

    cache("ablation_summary.json", out)
    print(json.dumps(out, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    choices=["budget", "lambda", "serial", "oracle",
                             "coverage", "summarize", "all"])
    args = ap.parse_args()
    if args.task in ("budget", "all"):
        print(json.dumps(context_budget(), ensure_ascii=False, indent=2))
    if args.task in ("lambda", "all"):
        run_lambda()
    if args.task in ("serial", "all"):
        run_serial()
    if args.task in ("oracle", "all"):
        print(json.dumps(oracle_curve(), ensure_ascii=False, indent=2))
    if args.task in ("coverage", "all"):
        run_coverage()
    if args.task in ("summarize", "all"):
        summarize()


if __name__ == "__main__":
    main()
