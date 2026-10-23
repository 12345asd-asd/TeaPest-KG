# -*- coding: utf-8 -*-
"""
TeaPest-KGQA 发布版实验（release KG + release QA），本地 Qwen2.5-14B。

与旧 run_local_full_199 的区别：
  - 数据指向 release_kg / release_qa（干净、自洽、带来源）；
  - 评估修正：否定/安全感知匹配、题干复述不计分、macro+micro、逐题指标落盘供校准；
  - 无独立外部语料：文本对照用「三元组平铺句 BM25」与「病虫档案 GraphRAG」，
    与 KG-RAG 同信息源，只比较检索/组织形态（论文如实说明）。
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "trustkg_rag", "experiments"))

from run_all_experiments import (  # noqa: E402
    KGRetriever, build_zeroshot_prompt, build_fewshot_prompt,
    build_vectorg_prompt, build_kgrag_prompt,
)
from local_model import call_llm as _local_call  # noqa: E402

KG_PATH = os.path.join(ROOT, "release_kg", "kg_triples.json")
ENT_PATH = os.path.join(ROOT, "release_kg", "entities.csv")
QA_PATH = os.path.join(ROOT, "release_qa", "qa.json")
OUT = os.path.join(ROOT, "rag_results", "release")
os.makedirs(OUT, exist_ok=True)

NEG_WORDS = ("不建议", "不推荐", "不宜", "不可", "不能", "不要", "不需",
             "禁止", "避免", "忌用", "慎用", "勿用", "不是", "并非",
             "不能用", "不得", "严禁", "勿")
NEG_WINDOW = 10


def call_llm(messages, max_tokens=1024, temperature=0):
    sys_msg = user_msg = None
    user_msg = ""
    for m in messages:
        if m.get("role") == "system":
            sys_msg = m.get("content")
        elif m.get("role") == "user":
            user_msg = m.get("content", "")
    return _local_call(user_msg, max_tokens=max_tokens,
                       temperature=temperature, system_prefix=sys_msg)


retriever = KGRetriever(KG_PATH, ENT_PATH)
qa_all = json.load(open(QA_PATH, encoding="utf-8"))
qa_map = {q["id"]: q for q in qa_all}
print("release KG triples=%d  QA=%d" % (len(retriever.triples), len(qa_all)),
      flush=True)

# ---------- 文本基线语料（均由 release KG 派生，同信息源） ----------
def triple_sentence(t):
    h, r, tl = t["head"], t["relation"], t["tail"]
    if r == "表现症状":
        return "%s的表现症状包括%s。" % (h, tl)
    return "%s可采用%s进行%s。" % (h, tl, r)

lin_sents = [triple_sentence(t) for t in retriever.triples]

pest_tr = defaultdict(list)
for t in retriever.triples:
    if retriever.entity_to_type.get(t["head"]) == "PEST-ORGANISM":
        pest_tr[t["head"]].append(t)
communities = {}
for pest, trs in pest_tr.items():
    syms = sorted({x["tail"] for x in trs if x["relation"] == "表现症状"})
    parts = ["[%s] 症状:%s。" % (pest, "、".join(syms[:6]))] if syms else []
    for r in ("生物防治", "化学防治", "农业防治", "物理防治"):
        ms = sorted({x["tail"] for x in trs if x["relation"] == r})
        if ms:
            parts.append("%s:%s。" % (r, "、".join(ms[:8])))
    communities[pest] = "".join(parts)
comm_texts = list(communities.values())


def bigrams(s):
    return set(s[i:i + 2] for i in range(len(s) - 1))


def overlap_score(text, q):
    qb = bigrams(q)
    if not qb:
        return 0
    return len(bigrams(text) & qb) / len(qb)


# ---------- 链接与方法 ----------
def _linked(q):
    ents = retriever.find_entities(q["question"], min_len=2)
    ctx = q.get("context_entity", "")
    if ctx and ctx not in ents:
        ents.append(ctx)
    return ents if ents else ([ctx] if ctx else [])


def a_kgrag(q, hop):
    ents = _linked(q)
    sg = retriever.get_subgraph(ents, hops=hop,
                                max_edges=25 if hop == 1 else 40)
    return call_llm(build_kgrag_prompt(q["question"], sg, [], hop=hop),
                    max_tokens=900)


def a_zeroshot(q):
    return call_llm(build_zeroshot_prompt(q["question"]), max_tokens=600)


def a_fewshot(q):
    return call_llm(build_fewshot_prompt(q["question"]), max_tokens=600)


def a_linear_bm25(q):
    scored = sorted(((overlap_score(s, q["question"]), s) for s in lin_sents),
                    key=lambda x: -x[0])
    top = [{"text": s} for sc, s in scored[:8] if sc > 0]
    return call_llm(build_vectorg_prompt(q["question"], top), max_tokens=600)


def a_graphrag(q):
    scored = sorted(((overlap_score(s, q["question"]), s)
                     for s in comm_texts), key=lambda x: -x[0])
    top = [{"text": s} for sc, s in scored[:3] if sc > 0]
    return call_llm(build_vectorg_prompt(q["question"], top), max_tokens=700)


def a_kg_only(q):
    ents = _linked(q)
    sg = retriever.get_subgraph(ents, hops=1, max_edges=25)
    lines = ["- %s（%s）：%s" % (e["head"], e["relation"], e["tail"])
             for e in sg["edges"]]
    return "\n".join(lines) if lines else "（图谱中未检索到相关信息）"


METHODS = [
    ("zeroshot", a_zeroshot),
    ("fewshot", a_fewshot),
    ("linear_bm25", a_linear_bm25),
    ("graphrag", a_graphrag),
    ("kg_only", a_kg_only),
    ("kgrag1hop", lambda q: a_kgrag(q, 1)),
    ("kgrag2hop", lambda q: a_kgrag(q, 2)),
]


def run_method(name, fn, limit):
    path = os.path.join(OUT, "preds_%s.json" % name)
    if os.path.exists(path):
        print("skip(exists): %s" % name, flush=True)
        return json.load(open(path, encoding="utf-8"))
    qs = qa_all if limit is None else qa_all[:limit]
    preds, t0 = [], time.time()
    for i, q in enumerate(qs):
        preds.append({"qa_id": q["id"], "method": name,
                      "generated_answer": fn(q)})
        if i % 20 == 0:
            print("  [%s %d/%d] %.0fs" % (name, i + 1, len(qs),
                                            time.time() - t0), flush=True)
    json.dump(preds, open(path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    return preds


# ---------- 修正版实体评估 ----------
ent_list = [r[0] for r in csv.reader(open(ENT_PATH, encoding="utf-8-sig"))
            if len(r) >= 2 and r[1] != "type"]
ent_list.sort(key=len, reverse=True)


def find_mentions(text):
    """返回 [(entity, pos)]，最长优先、命中后挖空防重复计数。"""
    hits, work = [], list(text)
    for ent in ent_list:
        if len(ent) < 2:
            continue
        for m in re.finditer(re.escape(ent), text):
            s, e = m.span()
            if all(work[k] != " " for k in range(s, e)):
                hits.append((ent, s))
                for k in range(s, e):
                    work[k] = " "
    return hits


def negated(text, pos):
    pre = text[max(0, pos - NEG_WINDOW):pos]
    return any(w in pre for w in NEG_WORDS)


def score_answer(pred, q):
    gold = [g for g in q["answer_list"] if g not in q["question"]]
    if not gold:  # 理论不发生（release QA 保证非空）
        return None
    mentions = find_mentions(pred)
    valid = {e for e, pos in mentions if not negated(pred, pos)}
    goldset = set(gold)
    correct = valid & goldset
    rec = len(correct) / len(goldset)
    prec = len(correct) / len(valid) if valid else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"hit": float(bool(correct)), "recall": rec,
            "precision": prec, "f1": f1,
            "n_gold": len(goldset), "n_pred": len(valid),
            "n_correct": len(correct)}


def aggregate(preds, name):
    rows, acc = [], defaultdict(float)
    micro = {"gold": 0, "pred": 0, "correct": 0}
    byd = defaultdict(lambda: [0, 0.0, 0.0])
    byt = defaultdict(lambda: [0, 0.0, 0.0])
    for p in preds:
        q = qa_map.get(p["qa_id"])
        if q is None:        # 该题在更干净的 QA 版本中已被剔除
            continue
        m = score_answer(p["generated_answer"], q)
        if m is None:
            continue
        m.update(qa_id=p["qa_id"], type=q["type"],
                 difficulty=q["difficulty"])
        rows.append(m)
        acc["n"] += 1
        for k in ("hit", "recall", "precision", "f1"):
            acc[k] += m[k]
        micro["gold"] += m["n_gold"]; micro["pred"] += m["n_pred"]
        micro["correct"] += m["n_correct"]
        byd[q["difficulty"]][0] += 1
        byd[q["difficulty"]][1] += m["hit"]; byd[q["difficulty"]][2] += m["recall"]
        byt[q["type"]][0] += 1
        byt[q["type"]][1] += m["hit"]; byt[q["type"]][2] += m["recall"]
    n = int(acc["n"])
    summary = {
        "method": name, "n": n,
        "hit": round(acc["hit"] / n, 4),
        "recall": round(acc["recall"] / n, 4),
        "precision_macro": round(acc["precision"] / n, 4),
        "f1_macro": round(acc["f1"] / n, 4),
        "recall_micro": round(micro["correct"] / max(1, micro["gold"]), 4),
        "precision_micro": round(micro["correct"] / max(1, micro["pred"]), 4),
        "by_difficulty": {d: {"n": v[0], "hit": round(v[1] / v[0], 4),
                              "recall": round(v[2] / v[0], 4)}
                          for d, v in byd.items()},
        "by_type": {t: {"n": v[0], "hit": round(v[1] / v[0], 4),
                        "recall": round(v[2] / v[0], 4)}
                    for t, v in byt.items()},
    }
    return summary, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    all_preds = {}
    for name, fn in METHODS:
        if args.aggregate_only:
            pth = os.path.join(OUT, "preds_%s.json" % name)
            if os.path.exists(pth):
                all_preds[name] = json.load(open(pth, encoding="utf-8"))
        else:
            all_preds[name] = run_method(name, fn, args.limit)

    summaries, per_q = [], {}
    for name, preds in all_preds.items():
        s, rows = aggregate(preds, name)
        summaries.append(s)
        per_q[name] = rows
    summaries.sort(key=lambda x: -x["recall"])

    json.dump(summaries, open(os.path.join(OUT, "evals.json"), "w",
                              encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(per_q, open(os.path.join(OUT, "per_question.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=2)

    lines = ["TeaPest-KGQA release (n=%d, qwen2.5-14b, corrected eval)"
             % (args.limit or len(qa_all)), "=" * 84,
             "%-14s %6s %7s %8s %8s %8s %8s" %
             ("Method", "Hit", "Recall", "PrecMa", "F1Ma",
              "RecMi", "PrecMi"), "-" * 84]
    for s in summaries:
        lines.append("%-14s %6.1f %7.1f %8.1f %8.1f %8.1f %8.1f" % (
            s["method"], 100 * s["hit"], 100 * s["recall"],
            100 * s["precision_macro"], 100 * s["f1_macro"],
            100 * s["recall_micro"], 100 * s["precision_micro"]))
    lines.append("=" * 84)
    k = next((s for s in summaries if s["method"] == "kgrag1hop"),
             summaries[0])
    lines.append("kgrag1hop by difficulty:")
    for d, v in k["by_difficulty"].items():
        lines.append("  %-6s n=%3d Hit=%.1f Recall=%.1f"
                     % (d, v["n"], 100 * v["hit"], 100 * v["recall"]))
    rep = "\n".join(lines)
    open(os.path.join(OUT, "comparison.txt"), "w",
         encoding="utf-8").write(rep)
    print("\n" + rep, flush=True)


if __name__ == "__main__":
    main()
