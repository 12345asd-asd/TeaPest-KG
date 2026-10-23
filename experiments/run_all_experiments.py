# -*- coding: utf-8 -*-
"""
TeaPest-KGQA 全部实验 Runner
===============================
使用 DeepSeek API (Anthropic 兼容端点) 跑:
  E8:  LLM-ZeroShot   — 纯 LLM 不检索
  E9:  LLM-FewShot    — 3-shot prompt
  E10: Vector-RAG     — BM25 + LLM
  E11: KG-RAG         — 子图检索 + 证据排序 + LLM
  E12: Hop Depth 对比 — 1-hop vs 2-hop
  E13: 消融实验      — w/o KG, w/o Evidence, w/o Reasoning

输出: rag_results/ 下所有实验结果 JSON
"""

import json, csv, os, sys, time, re
from collections import defaultdict, Counter

os.makedirs("rag_results", exist_ok=True)

# ============================================================
# Config
# ============================================================
API_BASE = "https://api.deepseek.com/anthropic"
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")  # set this env var; never hardcode keys
MODEL = "deepseek-v4-pro"

KG_PATH = "augmented_kg_triples.json"
ENTITIES_PATH = "augmented_entities.csv"
QA_PATH = "qa_pairs.json"

# ============================================================
# 1. Anthropic-compatible API caller
# ============================================================
def call_llm(messages, max_tokens=1024, temperature=0):
    """Call DeepSeek via Anthropic Messages API"""
    import urllib.request
    import urllib.error

    body = json.dumps({
        "model": MODEL,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{API_BASE}/v1/messages",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01",
        },
    )

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
                # DeepSeek Anthropic format: content has thinking+text blocks
                if "content" in result:
                    for c in result["content"]:
                        if c.get("type") == "text":
                            return c["text"]
                    # fallback: return last content's text/thinking
                    if result["content"]:
                        last = result["content"][-1]
                        return last.get("text", last.get("thinking", str(result)))
                return str(result)
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")
            print(f"  API error (attempt {attempt+1}): {e.code} {err[:200]}")
            if attempt < 2:
                time.sleep(2 ** attempt)
        except Exception as e:
            print(f"  Network error (attempt {attempt+1}): {e}")
            if attempt < 2:
                time.sleep(3)
    return "[API_ERROR]"


# ============================================================
# 2. KG Retriever (local JSON)
# ============================================================
class KGRetriever:
    def __init__(self, kg_path, entities_path):
        with open(kg_path, "r", encoding="utf-8") as f:
            self.triples = json.load(f)
        self.entity_to_type = {}
        with open(entities_path, "r", encoding="utf-8-sig") as f:
            for row in csv.reader(f):
                if len(row) >= 2:
                    self.entity_to_type[row[0].strip()] = row[1].strip()
        # Indexes
        self.head_index = defaultdict(list)
        self.tail_index = defaultdict(list)
        self.entity_index = defaultdict(list)
        for t in self.triples:
            self.head_index[t["head"]].append(t)
            self.tail_index[t["tail"]].append(t)
            self.entity_index[t["head"]].append(t)
            self.entity_index[t["tail"]].append(t)

    def find_entities(self, text, min_len=3):
        found = []
        for ent in self.entity_index:
            if len(ent) >= min_len and ent in text:
                found.append(ent)
        found.sort(key=len, reverse=True)
        return found

    def get_subgraph(self, entities, hops=2, max_edges=40):
        visited_nodes = set()
        visited_edges = set()
        frontier = set(entities)
        for _ in range(hops):
            next_frontier = set()
            for node in frontier:
                visited_nodes.add(node)
                for t in self.entity_index.get(node, []):
                    ek = (t["head"], t["relation"], t["tail"])
                    if ek not in visited_edges:
                        visited_edges.add(ek)
                        visited_nodes.add(t["head"])
                        visited_nodes.add(t["tail"])
                        next_frontier.add(t["tail"] if t["head"] == node else t["head"])
            frontier = next_frontier - visited_nodes
            if len(visited_edges) >= max_edges:
                break

        edges = []
        for h, r, tl in visited_edges:
            full = None
            for t in self.entity_index.get(h, []):
                if t["head"] == h and t["relation"] == r and t["tail"] == tl:
                    full = t; break
            edges.append({
                "head": h, "relation": r, "tail": tl,
                "head_type": full["head_type"] if full else "",
                "tail_type": full["tail_type"] if full else "",
                "support": full.get("support", 1) if full else 1,
                "evidences": full.get("evidences", []) if full else [],
            })
        return {"edges": edges, "n_edges": len(edges), "n_nodes": len(visited_nodes)}

    def direct_lookup(self, entity, relation=None):
        results = []
        for t in self.head_index.get(entity, []):
            if relation is None or t["relation"] == relation:
                results.append(t)
        for t in self.tail_index.get(entity, []):
            if relation is None or t["relation"] == relation:
                results.append(t)
        return results

    def get_evidence_texts(self, subgraph, top_k=5):
        texts = []
        seen = set()
        for e in subgraph["edges"]:
            for ev in e.get("evidences", []):
                txt = ev.get("text", "")
                if len(txt) > 20 and txt not in seen:
                    seen.add(txt)
                    texts.append({"text": txt, "support": e.get("support", 1),
                                  "triple": f"({e['head']}, {e['relation']}, {e['tail']})"})
        texts.sort(key=lambda x: -x["support"])
        return texts[:top_k]


# ============================================================
# 3. Prompt Builders
# ============================================================
def build_zeroshot_prompt(question):
    return [
        {"role": "system", "content": "你是一位茶叶病虫害诊断专家。请根据你的知识回答用户问题。如果信息不充分，请诚实说明。"},
        {"role": "user", "content": question},
    ]

def build_fewshot_prompt(question):
    examples = """示例1:
问: 茶尺蠖有哪些生物防治方法？
答: 茶尺蠖的生物防治方法包括：使用苏云金杆菌（Bt）制剂、释放赤眼蜂等寄生性天敌、施用茶尺蠖核型多角体病毒、保护利用绒茧蜂等自然天敌。

示例2:
问: 茶树出现褐色圆形病斑，边缘有黄色晕圈，可能是什么病害？
答: 可能为茶炭疽病。其典型症状为叶片出现褐色圆形病斑，边缘有黄色晕圈，后期病斑上产生黑色小点（分生孢子器）。

示例3:
问: 茶树叶片出现黑色煤烟状霉层，如何防治？
答: 这是煤烟病，通常由介壳虫或黑刺粉虱的分泌物诱发。防治方法：(1)化学防治可喷施矿物油乳油或石硫合剂；(2)农业防治加强修剪通风；(3)生物防治保护利用寄生蜂等天敌；(4)物理防治可悬挂黄板诱杀虫媒。"""
    return [
        {"role": "system", "content": f"你是一位茶叶病虫害诊断专家。以下是一些诊断问答示例：\n\n{examples}\n\n请参考以上示例的方式回答用户问题，提供具体、专业的诊断和防治建议。"},
        {"role": "user", "content": question},
    ]

def build_vectorg_prompt(question, evidence_texts):
    ev_str = "\n".join(f"[{i+1}] {t['text'][:300]}" for i, t in enumerate(evidence_texts[:5]))
    return [
        {"role": "system", "content": "你是一位茶叶病虫害诊断专家。请根据以下参考资料回答用户问题。"},
        {"role": "user", "content": f"参考资料：\n{ev_str}\n\n用户问题：{question}\n\n请根据参考资料给出诊断和防治建议："},
    ]

def build_kgrag_prompt(question, subgraph, evidence_texts, hop=2):
    # KG facts
    kg_lines = []
    for e in subgraph["edges"][:20]:
        kg_lines.append(f"  ({e['head']}, {e['relation']}, {e['tail']}) [置信度={e.get('support',1)}]")

    # Evidence
    ev_lines = []
    for i, t in enumerate(evidence_texts[:5]):
        ev_lines.append(f"  [{i+1}] {t['text'][:300]}")

    reasoning_hint = ""
    if hop == 2:
        reasoning_hint = "\n请展示推理链: 症状 → 病因(病虫害) → 各类防治方法。"

    return [
        {"role": "system", "content": f"""你是一位茶叶病虫害诊断专家。请根据以下知识图谱事实和原文证据进行诊断推理。

## 知识图谱事实（结构化知识）
{chr(10).join(kg_lines) if kg_lines else '无'}

## 原文证据
{chr(10).join(ev_lines) if ev_lines else '无'}

## 回答要求
1. 先给出诊断结论（是什么病虫害）
2. 分类给出防治建议：生物防治、化学防治、物理防治、农业防治
3. 引用知识图谱中的事实支撑你的判断{reasoning_hint}"""},
        {"role": "user", "content": question},
    ]


# ============================================================
# 4. Evaluator
# ============================================================
class Evaluator:
    def entity_recall(self, pred, ref_entities):
        if not ref_entities: return 1.0
        return sum(1 for e in ref_entities if e in pred) / len(ref_entities)

    def has_any_entity(self, pred, ref_entities):
        if not ref_entities: return 1.0
        return 1.0 if any(e in pred for e in ref_entities) else 0.0

    def evaluate_all(self, predictions, qa_map):
        by_type = defaultdict(lambda: {"count": 0, "has_entity": 0, "entity_recall": 0.0})
        per_q = []
        for p in predictions:
            ref = qa_map.get(p["qa_id"], {})
            ref_ents = ref.get("answer_list", ref.get("entities", []))
            qtype = ref.get("type", "unknown")
            has = self.has_any_entity(p.get("generated_answer", ""), ref_ents)
            rec = self.entity_recall(p.get("generated_answer", ""), ref_ents)
            m = by_type[qtype]; m["count"] += 1; m["has_entity"] += has; m["entity_recall"] += rec
            per_q.append({"qa_id": p["qa_id"], "type": qtype, "entity_recall": round(rec, 3),
                          "has_entity": has, "method": p.get("method", "unknown"),
                          "question": ref.get("question", "")[:100],
                          "answer": p.get("generated_answer", "")[:300]})
        summary = {}
        for qt, m in by_type.items():
            n = m["count"]
            summary[qt] = {"count": n, "hit": round(m["has_entity"] / n, 3) if n else 0,
                           "recall": round(m["entity_recall"] / n, 3) if n else 0}
        total = sum(m["count"] for m in by_type.values())
        t_has = sum(m["has_entity"] for m in by_type.values())
        t_rec = sum(m["entity_recall"] for m in by_type.values())
        return {
            "overall": {"total": total, "hit": round(t_has / total, 3) if total else 0,
                        "recall": round(t_rec / total, 3) if total else 0},
            "by_type": summary, "per_question": per_q,
        }


# ============================================================
# 5. Experiment Runner
# ============================================================
def run():
    print("=" * 60)
    print("TeaPest-KGQA 全实验 Runner")
    print(f"API: {API_BASE}  Model: {MODEL}")
    print("=" * 60)

    # Load data
    retriever = KGRetriever(KG_PATH, ENTITIES_PATH)
    with open(QA_PATH, "r", encoding="utf-8") as f:
        qa_items = json.load(f)
    qa_map = {q["id"]: q for q in qa_items}

    # Select subset for LLM experiments (to save API cost)
    # Easy: 30, Medium: 30, Hard: 14 = 74 questions
    easy_qs = [q for q in qa_items if q["difficulty"] == "easy"]
    med_qs = [q for q in qa_items if q["difficulty"] == "medium"]
    hard_qs = [q for q in qa_items if q["difficulty"] == "hard"]
    import random
    random.seed(42)
    test_set = (
        random.sample(easy_qs, min(30, len(easy_qs))) +
        random.sample(med_qs, min(30, len(med_qs))) +
        hard_qs
    )
    print(f"Test set: {len(test_set)} questions ({len(easy_qs[:30])}E/{min(30,len(med_qs))}M/{len(hard_qs)}H)")
    test_map = {q["id"]: q for q in test_set}

    evaluator = Evaluator()
    all_method_results = {}

    # ═══════════════════════════════════════════════
    # E8: LLM-ZeroShot
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 40)
    print("E8: LLM-ZeroShot")
    print("=" * 40)
    zs_preds = []
    for i, q in enumerate(test_set):
        print(f"  [{i+1}/{len(test_set)}] {q['id']}: {q['question'][:80]}...")
        msgs = build_zeroshot_prompt(q["question"])
        ans = call_llm(msgs, max_tokens=512)
        zs_preds.append({"qa_id": q["id"], "method": "llm_zeroshot",
                          "generated_answer": ans, "reference_answer": q.get("answer_text", "")})
        if i % 10 == 9:
            print(f"    → {ans[:100]}...")
        time.sleep(0.3)  # rate limit

    zs_result = evaluator.evaluate_all(zs_preds, test_map)
    all_method_results["LLM-ZeroShot"] = zs_result
    print(f"  Overall: Hit={zs_result['overall']['hit']:.1%} Recall={zs_result['overall']['recall']:.1%}")

    # ═══════════════════════════════════════════════
    # E9: LLM-FewShot
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 40)
    print("E9: LLM-FewShot (3-shot)")
    print("=" * 40)
    fs_preds = []
    for i, q in enumerate(test_set):
        print(f"  [{i+1}/{len(test_set)}] {q['id']}")
        msgs = build_fewshot_prompt(q["question"])
        ans = call_llm(msgs, max_tokens=512)
        fs_preds.append({"qa_id": q["id"], "method": "llm_fewshot",
                          "generated_answer": ans, "reference_answer": q.get("answer_text", "")})
        time.sleep(0.3)

    fs_result = evaluator.evaluate_all(fs_preds, test_map)
    all_method_results["LLM-FewShot"] = fs_result
    print(f"  Overall: Hit={fs_result['overall']['hit']:.1%} Recall={fs_result['overall']['recall']:.1%}")

    # ═══════════════════════════════════════════════
    # E10: Vector-RAG (BM25 evidence → LLM)
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 40)
    print("E10: Vector-RAG (BM25 retrieval)")
    print("=" * 40)
    # Pre-build all evidence text pool
    all_evidences = []
    for t in retriever.triples:
        for ev in t.get("evidences", []):
            txt = ev.get("text", "")
            if len(txt) > 30:
                all_evidences.append({"text": txt, "support": t.get("support", 1)})
    # Dedup
    seen_txt = set()
    ev_pool = []
    for e in all_evidences:
        if e["text"] not in seen_txt:
            seen_txt.add(e["text"])
            ev_pool.append(e)
    print(f"  Evidence pool: {len(ev_pool)} unique texts")

    vr_preds = []
    for i, q in enumerate(test_set):
        print(f"  [{i+1}/{len(test_set)}] {q['id']}")
        # BM25-like scoring
        q_chars = set(q["question"])
        scored = []
        for ev in ev_pool:
            score = sum(1 for c in q_chars if c in ev["text"])
            if score > 0:
                scored.append((score, ev))
        scored.sort(key=lambda x: -x[0])
        top_ev = [s[1] for s in scored[:5]]
        msgs = build_vectorg_prompt(q["question"], top_ev)
        ans = call_llm(msgs, max_tokens=512)
        vr_preds.append({"qa_id": q["id"], "method": "vector_rag",
                          "generated_answer": ans, "reference_answer": q.get("answer_text", "")})
        time.sleep(0.3)

    vr_result = evaluator.evaluate_all(vr_preds, test_map)
    all_method_results["Vector-RAG"] = vr_result
    print(f"  Overall: Hit={vr_result['overall']['hit']:.1%} Recall={vr_result['overall']['recall']:.1%}")

    # ═══════════════════════════════════════════════
    # E11: KG-RAG (Ours)
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 40)
    print("E11: KG-RAG (Ours)")
    print("=" * 40)
    kgrag_preds = []
    kgrag_times = []
    for i, q in enumerate(test_set):
        print(f"  [{i+1}/{len(test_set)}] {q['id']}: {q['question'][:80]}...")
        t0 = time.time()
        entities = retriever.find_entities(q["question"], min_len=3)
        ctx = q.get("context_entity", "")
        if ctx and ctx not in entities:
            entities.append(ctx)
        subgraph = retriever.get_subgraph(entities if entities else [ctx], hops=2)
        evidence = retriever.get_evidence_texts(subgraph, top_k=5)
        msgs = build_kgrag_prompt(q["question"], subgraph, evidence, hop=2)
        ans = call_llm(msgs, max_tokens=1024)
        elapsed = time.time() - t0
        kgrag_times.append(elapsed)
        kgrag_preds.append({
            "qa_id": q["id"], "method": "kg_rag", "generated_answer": ans,
            "reference_answer": q.get("answer_text", ""),
            "subgraph_nodes": subgraph["n_nodes"], "subgraph_edges": subgraph["n_edges"],
            "evidence_count": len(evidence), "entity_count": len(entities),
            "time_seconds": round(elapsed, 2),
        })
        if i % 5 == 0:
            print(f"    subgraph={subgraph['n_edges']}edges, {len(evidence)}ev, {elapsed:.1f}s")
            print(f"    → {ans[:150]}...")
        time.sleep(0.3)

    kgrag_result = evaluator.evaluate_all(kgrag_preds, test_map)
    all_method_results["KG-RAG"] = kgrag_result
    print(f"  Overall: Hit={kgrag_result['overall']['hit']:.1%} Recall={kgrag_result['overall']['recall']:.1%}")
    print(f"  Avg time: {sum(kgrag_times)/len(kgrag_times):.1f}s")

    # ═══════════════════════════════════════════════
    # E12: Hop Depth (1-hop vs 2-hop)
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 40)
    print("E12: Hop Depth Comparison (1-hop)")
    print("=" * 40)
    hop1_preds = []
    for i, q in enumerate(test_set):
        entities = retriever.find_entities(q["question"], min_len=3)
        ctx = q.get("context_entity", "")
        if ctx and ctx not in entities: entities.append(ctx)
        subgraph_1h = retriever.get_subgraph(entities if entities else [ctx], hops=1, max_edges=25)
        evidence_1h = retriever.get_evidence_texts(subgraph_1h, top_k=5)
        msgs = build_kgrag_prompt(q["question"], subgraph_1h, evidence_1h, hop=1)
        ans = call_llm(msgs, max_tokens=1024)
        hop1_preds.append({"qa_id": q["id"], "method": "kg_rag_1hop",
                            "generated_answer": ans, "reference_answer": q.get("answer_text", "")})
        if i % 15 == 0: print(f"  [{i+1}/{len(test_set)}]")
        time.sleep(0.3)

    hop1_result = evaluator.evaluate_all(hop1_preds, test_map)
    all_method_results["KG-RAG-1hop"] = hop1_result
    print(f"  1-hop: Hit={hop1_result['overall']['hit']:.1%} Recall={hop1_result['overall']['recall']:.1%}")
    print(f"  2-hop: Hit={kgrag_result['overall']['hit']:.1%} Recall={kgrag_result['overall']['recall']:.1%}")

    # ═══════════════════════════════════════════════
    # E13: Ablation Study
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 40)
    print("E13: Ablation Study")
    print("=" * 40)

    # E13a: w/o KG (only evidence texts, no structured KG)
    print("  [Ablation A] w/o KG (evidence only)")
    wo_kg_preds = []
    for i, q in enumerate(test_set):
        entities = retriever.find_entities(q["question"], min_len=3)
        ctx = q.get("context_entity", "")
        if ctx and ctx not in entities: entities.append(ctx)
        subgraph = retriever.get_subgraph(entities if entities else [ctx], hops=2)
        evidence = retriever.get_evidence_texts(subgraph, top_k=5)
        # Use vector-rag style prompt (no KG facts)
        msgs = build_vectorg_prompt(q["question"], evidence) if evidence else build_zeroshot_prompt(q["question"])
        ans = call_llm(msgs, max_tokens=512)
        wo_kg_preds.append({"qa_id": q["id"], "method": "ablation_no_kg",
                             "generated_answer": ans, "reference_answer": q.get("answer_text", "")})
        if i % 20 == 0: print(f"    [{i+1}/{len(test_set)}]")
        time.sleep(0.3)
    wo_kg_result = evaluator.evaluate_all(wo_kg_preds, test_map)
    all_method_results["Ablation-w/o-KG"] = wo_kg_result
    print(f"    w/o KG: Hit={wo_kg_result['overall']['hit']:.1%} Recall={wo_kg_result['overall']['recall']:.1%}")

    # E13b: w/o Evidence (only KG triples, no text)
    print("  [Ablation B] w/o Evidence (KG only)")
    wo_ev_preds = []
    for i, q in enumerate(test_set):
        entities = retriever.find_entities(q["question"], min_len=3)
        ctx = q.get("context_entity", "")
        if ctx and ctx not in entities: entities.append(ctx)
        subgraph = retriever.get_subgraph(entities if entities else [ctx], hops=2)
        # KG-only prompt
        kg_lines = [f"  ({e['head']}, {e['relation']}, {e['tail']}) [置信度={e.get('support',1)}]" for e in subgraph["edges"][:20]]
        msgs = [
            {"role": "system", "content": f"你是茶叶病虫害诊断专家。请仅根据以下知识图谱事实回答。\n\n## 知识图谱事实\n{chr(10).join(kg_lines) if kg_lines else '无'}\n\n请给出诊断和防治建议："},
            {"role": "user", "content": q["question"]},
        ]
        ans = call_llm(msgs, max_tokens=512)
        wo_ev_preds.append({"qa_id": q["id"], "method": "ablation_no_evidence",
                             "generated_answer": ans, "reference_answer": q.get("answer_text", "")})
        if i % 20 == 0: print(f"    [{i+1}/{len(test_set)}]")
        time.sleep(0.3)
    wo_ev_result = evaluator.evaluate_all(wo_ev_preds, test_map)
    all_method_results["Ablation-w/o-Evidence"] = wo_ev_result
    print(f"    w/o Evidence: Hit={wo_ev_result['overall']['hit']:.1%} Recall={wo_ev_result['overall']['recall']:.1%}")

    # ═══════════════════════════════════════════════
    # Save ALL results
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Saving Results...")
    print("=" * 60)

    # Convert defaultdict to dict for JSON
    def to_dict(d):
        if isinstance(d, defaultdict):
            return {k: to_dict(v) for k, v in d.items()}
        if isinstance(d, dict):
            return {k: to_dict(v) for k, v in d.items()}
        return d

    serializable = {}
    for k, v in all_method_results.items():
        serializable[k] = to_dict(v)

    with open("rag_results/full_experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    # Save all predictions
    all_preds = {
        "llm_zeroshot": zs_preds,
        "llm_fewshot": fs_preds,
        "vector_rag": vr_preds,
        "kg_rag": kgrag_preds,
        "kg_rag_1hop": hop1_preds,
        "ablation_no_kg": wo_kg_preds,
        "ablation_no_evidence": wo_ev_preds,
    }
    with open("rag_results/all_predictions.json", "w", encoding="utf-8") as f:
        json.dump(all_preds, f, ensure_ascii=False, indent=2)

    # Summary table
    print("\n" + "=" * 70)
    print("FINAL RESULTS TABLE")
    print("=" * 70)
    print(f"{'Method':<30s} {'Hit Rate':>10s} {'Entity Recall':>14s}")
    print("-" * 56)
    for method in ["LLM-ZeroShot", "LLM-FewShot", "Vector-RAG", "KG-RAG",
                   "KG-RAG-1hop", "Ablation-w/o-KG", "Ablation-w/o-Evidence"]:
        if method in all_method_results:
            o = all_method_results[method]["overall"]
            print(f"{method:<30s} {o['hit']:>9.1%} {o['recall']:>14.1%}")

    # By-type for KG-RAG
    print("\n--- KG-RAG by Question Type ---")
    for qt in ["pest_control", "symptom_diagnosis", "two_hop_diagnosis",
               "contrastive", "pest_all_controls", "open_diagnosis"]:
        if qt in kgrag_result["by_type"]:
            m = kgrag_result["by_type"][qt]
            print(f"  {qt:<25s}: n={m['count']:2d}  hit={m['hit']:.1%}  recall={m['recall']:.1%}")

    # By difficulty for KG-RAG
    print("\n--- KG-RAG by Difficulty ---")
    diff = defaultdict(lambda: {"count": 0, "recall": 0.0})
    for pq in kgrag_result["per_question"]:
        ref = test_map.get(pq["qa_id"], {})
        d = ref.get("difficulty", "unknown")
        diff[d]["count"] += 1
        diff[d]["recall"] += pq["entity_recall"]
    for d in ["easy", "medium", "hard"]:
        if diff[d]["count"] > 0:
            print(f"  {d:10s}: n={diff[d]['count']:2d}  recall={diff[d]['recall']/diff[d]['count']:.1%}")

    # Save summary
    with open("rag_results/final_summary.txt", "w", encoding="utf-8") as f:
        f.write("TeaPest-KGQA 全实验最终结果\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"API: {API_BASE}  Model: {MODEL}\n")
        f.write(f"Test set: {len(test_set)} questions\n\n")
        f.write(f"{'Method':<30s} {'Hit Rate':>10s} {'Entity Recall':>14s}\n")
        f.write("-" * 56 + "\n")
        for method in ["LLM-ZeroShot", "LLM-FewShot", "Vector-RAG", "KG-RAG",
                       "KG-RAG-1hop", "Ablation-w/o-KG", "Ablation-w/o-Evidence"]:
            if method in all_method_results:
                o = all_method_results[method]["overall"]
                f.write(f"{method:<30s} {o['hit']:>9.1%} {o['recall']:>14.1%}\n")
        f.write(f"\n--- KG-RAG by Difficulty ---\n")
        for d in ["easy", "medium", "hard"]:
            if diff[d]["count"] > 0:
                f.write(f"  {d:10s}: n={diff[d]['count']}  recall={diff[d]['recall']/diff[d]['count']:.1%}\n")

    print(f"\n✅ All results saved to rag_results/")
    print(f"   - full_experiment_results.json   (evaluation summaries)")
    print(f"   - all_predictions.json           (all generated answers)")
    print(f"   - final_summary.txt              (human-readable table)")


if __name__ == "__main__":
    run()
