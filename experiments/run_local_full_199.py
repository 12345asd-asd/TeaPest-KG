# -*- coding: utf-8 -*-
"""
TeaPest-KGQA · 本地 14B 全量（199 题）实验 Runner
====================================================

用本地 GPU 模型 qwen2.5-14b-custom（经 trustkg_rag/experiments/local_model.call_llm）
在【全部 199 题】上跑一套口径一致、可复现、可离线的完整应用实验，产出论文主表。

设计：
  · 检索器 / Prompt 构建器复用 run_all_experiments（有 __main__ 保护，可安全 import）。
  · GraphRAG / TransE / KG-Only / Chunked-BM25 与实体指标口径移植自
    strong_baselines_v2.py / fix_evaluation.py（这两处 import 即执行，无法直接复用）。
  · 断点续跑：每个方法单独存 preds_<method>.json，已存在则跳过；最后统一聚合。

用法：
    python run_local_full_199.py            # 全 199 题
    python run_local_full_199.py --limit 2  # 冒烟（每方法只跑 2 题）
"""

import json, csv, os, sys, time, argparse
from collections import defaultdict
import numpy as np

# Windows 控制台默认 GBK，无法编码 ✓/✅ 等字符 → 强制 UTF-8 输出，避免打印时崩溃
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ---- 路径（项目根）----
ROOT = os.path.dirname(os.path.abspath(__file__))
KG_PATH = os.path.join(ROOT, "augmented_kg_triples.json")
ENTITIES_PATH = os.path.join(ROOT, "augmented_entities.csv")
QA_PATH = os.path.join(ROOT, "qa_pairs.json")
OUT = os.path.join(ROOT, "rag_results", "local_full_199")
os.makedirs(OUT, exist_ok=True)

# ---- 复用检索器 / Prompt ----
sys.path.insert(0, ROOT)
from run_all_experiments import (
    KGRetriever,
    build_zeroshot_prompt, build_fewshot_prompt,
    build_vectorg_prompt, build_kgrag_prompt,
)

# ---- 本地模型入口 + messages 适配器 ----
sys.path.insert(0, os.path.join(ROOT, "trustkg_rag", "experiments"))
from local_model import call_llm as _local_call


def call_llm(messages, max_tokens=1024, temperature=0):
    """把 messages(list[dict]) 接口适配到 local_model.call_llm（user prompt + system 前缀）。"""
    sys_msg = None
    user_msg = ""
    for m in messages:
        role = m.get("role", "")
        if role == "system":
            sys_msg = m.get("content")
        elif role == "user":
            user_msg = m.get("content", "")
    return _local_call(user_msg, max_tokens=max_tokens,
                       temperature=temperature, system_prefix=sys_msg)


# ============================================================
# 加载数据
# ============================================================
retriever = KGRetriever(KG_PATH, ENTITIES_PATH)
qa_all = json.load(open(QA_PATH, encoding="utf-8"))
qa_map = {q["id"]: q for q in qa_all}
test_set = qa_all  # 全部 199 题
print("KG triples=%d  QA=%d" % (len(retriever.triples), len(test_set)), flush=True)


# ============================================================
# 预计算：证据池 / 文本块 / GraphRAG 社区摘要
# ============================================================

# 唯一证据文本池
ev_pool, _seen = [], set()
for t in retriever.triples:
    for e in t.get("evidences", []):
        tx = e.get("text", "")
        if len(tx) > 30 and tx not in _seen:
            _seen.add(tx)
            ev_pool.append({"text": tx, "support": t.get("support", 1)})

# 文本块（Chunked-BM25）
chunks, _seen2 = [], set()
for t in retriever.triples:
    for e in t.get("evidences", []):
        tx = e.get("text", "")
        if len(tx) > 50 and tx not in _seen2:
            _seen2.add(tx)
            if len(tx) > 300:
                for s in range(0, len(tx), 200):
                    chunks.append(tx[s:s + 300])
            else:
                chunks.append(tx)

# GraphRAG 社区摘要（按病虫聚合 症状 + 四类防治）
pest_communities = defaultdict(list)
for t in retriever.triples:
    if retriever.entity_to_type.get(t["head"]) == "PEST-ORGANISM":
        pest_communities[t["head"]].append(t)
    if retriever.entity_to_type.get(t["tail"]) == "PEST-ORGANISM":
        pest_communities[t["tail"]].append(t)

comm_summaries = {}
for pest, trs in pest_communities.items():
    syms = [x["tail"] for x in trs
            if x["relation"] == "表现症状" and x["head"] == pest]
    ctrls = defaultdict(list)
    for x in trs:
        if x["relation"] in ("生物防治", "化学防治", "物理防治", "农业防治") \
                and x["head"] == pest:
            ctrls[x["relation"]].append(x["tail"])
    s = "[%s] " % pest
    if syms:
        s += "症状: %s. " % "; ".join(list(set(syms))[:5])
    for rt, ms in ctrls.items():
        u = list(set(ms))[:5]
        if u:
            s += "%s: %s. " % (rt, "; ".join(u))
    if len(s) > 30:
        comm_summaries[pest] = s

print("ev_pool=%d chunks=%d communities=%d"
      % (len(ev_pool), len(chunks), len(comm_summaries)), flush=True)


# ============================================================
# 各方法的答案构造（questions dict → answer text）
# ============================================================

def _linked(q):
    ents = retriever.find_entities(q["question"], min_len=3)
    ctx = q.get("context_entity", "")
    if ctx and ctx not in ents:
        ents.append(ctx)
    return ents if ents else ([ctx] if ctx else [])


def ans_kgrag(q, hop):
    ents = _linked(q)
    if hop == 1:
        sg = retriever.get_subgraph(ents, hops=1, max_edges=25)
    else:
        sg = retriever.get_subgraph(ents, hops=2, max_edges=40)
    ev = retriever.get_evidence_texts(sg, top_k=5)
    msgs = build_kgrag_prompt(q["question"], sg, ev, hop=hop)
    return call_llm(msgs, max_tokens=1024)


def ans_zeroshot(q):
    return call_llm(build_zeroshot_prompt(q["question"]), max_tokens=512)


def ans_fewshot(q):
    return call_llm(build_fewshot_prompt(q["question"]), max_tokens=512)


def ans_vectorg(q):
    qchars = set(q["question"])
    scored = sorted(
        ((sum(1 for c in qchars if c in e["text"]), e) for e in ev_pool),
        key=lambda x: -x[0])
    top = [s[1] for s in scored[:5] if s[0] > 0]
    return call_llm(build_vectorg_prompt(q["question"], top), max_tokens=512)


def ans_chunked(q):
    qchars = set(q["question"])
    scored = sorted(
        ((sum(1 for c in qchars if c in ch), ch) for ch in chunks),
        key=lambda x: -x[0])
    top = [s[1] for s in scored[:5] if s[0] > 0]
    ctx = "\n\n".join("[%d]%s" % (i + 1, c) for i, c in enumerate(top)) \
        if top else "无参考资料"
    msgs = [{"role": "user",
             "content": "参考资料:\n%s\n\n问题:%s\n\n诊断并给出防治建议:"
                        % (ctx, q["question"])}]
    return call_llm(msgs, max_tokens=1024)


def ans_graphrag(q):
    found = retriever.find_entities(q["question"], min_len=2)
    relevant = []
    for e in found:
        for pest in pest_communities:
            if (e in pest) or (len(e) >= 3 and len(pest) >= 3 and pest in e):
                if pest in comm_summaries and pest not in relevant:
                    relevant.append(pest)
    if not relevant:
        for e in found[:3]:
            if e in comm_summaries:
                relevant.append(e)
    ctx = "\n".join(comm_summaries[p] for p in relevant[:5])
    if not ctx:
        return call_llm([{"role": "user", "content": q["question"]}],
                       max_tokens=512)
    return call_llm([{"role": "user",
                      "content": "##社区知识摘要\n%s\n\n##问题\n%s\n\n"
                                 "参考上述知识给出专业诊断和防治建议:"
                                 % (ctx, q["question"])}],
                   max_tokens=1024)


def ans_kg_only(q):
    ents = retriever.find_entities(q["question"], min_len=3)
    parts = []
    for e in ents[:5]:
        et = retriever.entity_to_type.get(e, "")
        nbrs = []
        for t in retriever.entity_index.get(e, [])[:8]:
            if t["head"] == e:
                nbrs.append("%s→%s" % (t["relation"], t["tail"]))
            if t["tail"] == e:
                nbrs.append("%s←%s" % (t["head"], t["relation"]))
        if nbrs:
            parts.append("[%s](%s): %s" % (e, et, "; ".join(nbrs)))
    if not parts:
        return "知识图谱中未找到相关信息。"
    return "知识图谱查询结果:\n" + "\n".join(parts)


# ---- TransE（懒训练，仅在该方法执行时）----
_transe_cache = {}

def _train_transe():
    if _transe_cache:
        return _transe_cache
    ent_list = sorted(retriever.entity_index.keys())
    rels = sorted(set(t["relation"] for t in retriever.triples))
    e2id = {e: i for i, e in enumerate(ent_list)}
    r2id = {r: i for i, r in enumerate(rels)}
    tl = [(e2id[t["head"]], r2id[t["relation"]], e2id[t["tail"]])
          for t in retriever.triples
          if t["head"] in e2id and t["tail"] in e2id
          and t["relation"] in r2id]
    dim, lr, epochs = 128, 0.01, 300
    np.random.seed(42)
    E = np.random.randn(len(ent_list), dim).astype(np.float64) * 0.1
    R = np.random.randn(len(rels), dim).astype(np.float64) * 0.1
    for i in range(len(ent_list)):
        E[i] /= (np.linalg.norm(E[i]) + 1e-8)
    print("  TransE training %d epochs (%d triples)..." % (epochs, len(tl)),
          flush=True)
    for ep in range(epochs):
        np.random.shuffle(tl)
        for start in range(0, len(tl), 256):
            batch = tl[start:start + 256]
            bsize = len(batch)
            hi = [b[0] for b in batch]
            ri = [b[1] for b in batch]
            ti = [b[2] for b in batch]
            h, r, tpos = E[hi], R[ri], E[ti]
            negt = np.random.randint(0, len(ent_list), size=bsize)
            margin = 1.0
            m_idx = np.where(
                (np.sum((h + r - tpos) ** 2, axis=1)
                 - np.sum((h + r - E[negt]) ** 2, axis=1) + margin) > 0)[0]
            for j in m_idx:
                hg = 2 * (h[j] + r[j] - tpos[j]) \
                    - 2 * (h[j] + r[j] - E[negt[j]])
                rg = hg
                tg = -2 * (h[j] + r[j] - tpos[j])
                ng = 2 * (h[j] + r[j] - E[negt[j]])
                E[hi[j]] -= lr * hg
                R[ri[j]] -= lr * rg
                E[ti[j]] -= lr * tg
                E[negt[j]] -= lr * ng
        for i in range(len(ent_list)):
            E[i] /= (np.linalg.norm(E[i]) + 1e-8)
        if ep % 100 == 0:
            print("    epoch %d" % ep, flush=True)
    np.save(os.path.join(OUT, "transe_entity_embs.npy"), E)
    _transe_cache.update(E=E, e2id=e2id, ent_list=ent_list)
    return _transe_cache


def ans_transe(q):
    c = _train_transe()
    E, e2id, ent_list = c["E"], c["e2id"], c["ent_list"]
    found = retriever.find_entities(q["question"], min_len=2)
    qidx = [e2id[e] for e in found if e in e2id]
    if qidx:
        qemb = E[qidx].mean(axis=0)
        sims = np.dot(E, qemb)
        topi = np.argsort(sims)[-16:][::-1]
        top_ents = [ent_list[i] for i in topi
                    if retriever.entity_to_type.get(ent_list[i], "") in
                    ("PEST-ORGANISM", "CONTROL-BIOLOGICAL", "CONTROL-CHEMICAL",
                     "CONTROL-PHYSICAL", "CONTROL-AGRICULTURAL",
                     "PEST-SYMPTOM")][:8]
        all_ents = list(set(found[:3] + top_ents))
    else:
        all_ents = found[:3]
    sg = retriever.get_subgraph(all_ents, hops=1)
    ev = retriever.get_evidence_texts(sg, top_k=5)
    kl = "\n".join("(%s,%s,%s)" % (e["head"], e["relation"], e["tail"])
                   for e in sg["edges"][:20])
    el = "\n".join("[%d]%s" % (j + 1, t["text"][:300])
                   for j, t in enumerate(ev[:5]))
    if sg["n_edges"] > 0:
        content = "##知识图谱\n%s\n\n##证据\n%s\n\n##问题\n%s\n\n诊断并给出防治建议:" \
                  % (kl, el, q["question"])
    else:
        content = q["question"]
    return call_llm([{"role": "user", "content": content}], max_tokens=1024)


# ============================================================
# 断点续跑执行器
# ============================================================

# 按重要性排序：KG-RAG 最先，保证提前停止时关键行已就绪
METHODS = [
    ("kgrag1hop", lambda q: ans_kgrag(q, 1)),
    ("kgrag2hop", lambda q: ans_kgrag(q, 2)),
    ("zeroshot", ans_zeroshot),
    ("kg_only", ans_kg_only),
    ("chunked_bm25", ans_chunked),
    ("graphrag", ans_graphrag),
    ("abl_no_kg", lambda q: _ans_ablation(q, "kg")),
    ("abl_no_ev", lambda q: _ans_ablation(q, "ev")),
    ("vectorg_bm25", ans_vectorg),
    ("fewshot", ans_fewshot),
    ("transe", ans_transe),
]


def _ans_ablation(q, which):
    ents = _linked(q)
    sg = retriever.get_subgraph(ents, hops=2, max_edges=40)
    ev = retriever.get_evidence_texts(sg, top_k=5)
    if which == "kg":  # 只用证据文本，无结构化 KG
        if ev:
            msgs = build_vectorg_prompt(q["question"], ev)
        else:
            msgs = build_zeroshot_prompt(q["question"])
        return call_llm(msgs, max_tokens=512)
    # 只用 KG 三元组，无证据文本
    kg_lines = ["  (%s, %s, %s) [置信度=%s]"
                % (e["head"], e["relation"], e["tail"], e.get("support", 1))
                for e in sg["edges"][:20]]
    msgs = [{"role": "system",
             "content": "你是茶叶病虫害诊断专家。请仅根据以下知识图谱事实回答。\n\n"
                        "## 知识图谱事实\n%s\n\n请给出诊断和防治建议："
                        % ("\n".join(kg_lines) if kg_lines else "无")},
            {"role": "user", "content": q["question"]}]
    return call_llm(msgs, max_tokens=512)


def run_method(name, builder, limit):
    path = os.path.join(OUT, "preds_%s.json" % name)
    if os.path.exists(path):
        print("skip(exists): %s" % name, flush=True)
        return json.load(open(path, encoding="utf-8"))
    qs = test_set if limit is None else test_set[:limit]
    preds = []
    t0 = time.time()
    for i, q in enumerate(qs):
        ans = builder(q)
        preds.append({"qa_id": q["id"], "method": name,
                      "generated_answer": ans})
        if i % 20 == 0:
            print("  [%s %d/%d] (%.0fs)" % (name, i + 1, len(qs),
                                             time.time() - t0), flush=True)
    json.dump(preds, open(path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("  saved %s (%d, %.0fs)" % (name, len(preds), time.time() - t0),
          flush=True)
    return preds


# ============================================================
# 聚合：实体 Hit/Recall/Precision/F1（口径同 fix_evaluation.py）
# ============================================================

kg_entities = [r[0] for r in csv.reader(
    open(ENTITIES_PATH, encoding="utf-8-sig")) if len(r) >= 2]
kg_entities.sort(key=len, reverse=True)


def extract_mentioned(text):
    mentioned, remaining = [], text
    for ent in kg_entities:
        if len(ent) < 2:
            continue
        if ent in remaining:
            mentioned.append(ent)
            remaining = remaining.replace(ent, " ", 1)
    return mentioned


def proper_metrics(pred, ref_ents):
    if not ref_ents:
        return {"hit": 1.0, "recall": 1.0, "precision": 1.0, "f1": 1.0,
                "ref_count": 0, "pred_count": 0, "correct": 0}
    pe = extract_mentioned(pred)
    correct = set(e for e in pe if e in set(ref_ents))
    rec = len(correct) / len(ref_ents)
    prec = len(correct) / len(pe) if pe else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return {"hit": 1.0 if len(correct) > 0 else 0.0,
            "recall": rec, "precision": prec, "f1": f1,
            "ref_count": len(ref_ents), "pred_count": len(pe),
            "correct": len(correct)}


def aggregate(preds, method):
    rows = []
    agg = {"total": 0, "hit": 0.0, "recall": 0.0, "precision": 0.0,
           "f1": 0.0, "ref": 0, "pred": 0, "correct": 0}
    byt = defaultdict(lambda: {"c": 0, "h": 0.0, "r": 0.0,
                                "p": 0.0, "f": 0.0})
    byd = defaultdict(lambda: {"c": 0, "h": 0.0, "r": 0.0,
                                "p": 0.0, "f": 0.0})
    for p in preds:
        q = qa_map[p["qa_id"]]
        ref = q.get("answer_list", q.get("entities", []))
        m = proper_metrics(p.get("generated_answer", ""), ref)
        qt, df = q.get("type", "unknown"), q.get("difficulty", "unknown")
        m.update(qa_id=p["qa_id"], type=qt, difficulty=df)
        rows.append(m)
        agg["total"] += 1
        for k in ["hit", "recall", "precision", "f1"]:
            agg[k] += m[k]
        agg["ref"] += m["ref_count"]
        agg["pred"] += m["pred_count"]
        agg["correct"] += m["correct"]
        for bucket, key in ((byt, qt), (byd, df)):
            b = bucket[key]
            b["c"] += 1
            b["h"] += m["hit"]
            b["r"] += m["recall"]
            b["p"] += m["precision"]
            b["f"] += m["f1"]

    n = agg["total"]

    def simplify(group):
        out = {}
        for k, b in group.items():
            c = b["c"]
            out[k] = {"n": c, "hit": round(b["h"] / c, 4),
                      "recall": round(b["r"] / c, 4),
                      "precision": round(b["p"] / c, 4),
                      "f1": round(b["f"] / c, 4)}
        return out

    summary = {"method": method, "n": n,
               "hit": round(agg["hit"] / n, 4),
               "recall": round(agg["recall"] / n, 4),
               "precision": round(agg["precision"] / n, 4),
               "f1": round(agg["f1"] / n, 4),
               "avg_ref": round(agg["ref"] / n, 1),
               "avg_pred": round(agg["pred"] / n, 1),
               "avg_correct": round(agg["correct"] / n, 1),
               "by_type": simplify(byt),
               "by_difficulty": simplify(byd)}
    return summary, rows


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="每方法只跑前 N 题（冒烟用）")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="不生成，仅从已有 preds 聚合")
    args = ap.parse_args()

    all_preds = {}
    for name, builder in METHODS:
        if args.aggregate_only:
            path = os.path.join(OUT, "preds_%s.json" % name)
            if os.path.exists(path):
                all_preds[name] = json.load(open(path, encoding="utf-8"))
            continue
        all_preds[name] = run_method(name, builder, args.limit)

    # 聚合
    summaries, per_q = [], {}
    for name, preds in all_preds.items():
        s, rows = aggregate(preds, name)
        summaries.append(s)
        per_q[name] = rows

    summaries.sort(key=lambda x: -x["f1"])

    json.dump(summaries,
              open(os.path.join(OUT, "corrected_evals_full199.json"), "w",
                   encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump(per_q,
              open(os.path.join(OUT, "per_question_full199.json"), "w",
                   encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # 文本对比表
    lines = ["TeaPest-KGQA 本地14B 全量结果 (n=%d, qwen2.5-14b-custom)"
             % (args.limit or 199),
             "=" * 78,
             "%-22s %6s %7s %7s %7s" % ("Method", "Hit", "Recall", "Prec", "F1"),
             "-" * 58]
    for s in summaries:
        lines.append("%-22s %6.1f %7.1f %7.1f %7.1f"
                     % (s["method"], 100 * s["hit"], 100 * s["recall"],
                        100 * s["precision"], 100 * s["f1"]))
    lines.append("=" * 78)
    lines.append("\n-- By difficulty (Hit / Recall) --")
    any_s = next((s for s in summaries if s["method"] == "kgrag1hop"),
                 summaries[0])
    for d in ["easy", "medium", "hard"]:
        if d in any_s["by_difficulty"]:
            b = any_s["by_difficulty"][d]
            lines.append("  %-7s n=%3d  Hit=%.1f Recall=%.1f"
                         % (d, b["n"], 100 * b["hit"], 100 * b["recall"]))
    report = "\n".join(lines)
    open(os.path.join(OUT, "comparison_table_full199.txt"), "w",
         encoding="utf-8").write(report)
    print("\n" + report, flush=True)
    print("\n✅ 聚合完成 → %s" % OUT, flush=True)


if __name__ == "__main__":
    main()
