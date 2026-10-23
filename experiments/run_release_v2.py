# -*- coding: utf-8 -*-
"""
TeaPest-QA v2 主运行器（release KG + qa_v2.json，158 题三层）。

在 run_release_experiments 之上：
  - 题目指向 release_qa/qa_v2.json（lookup/diagnosis/ood 三层）；
  - 复用 v1 已生成的 14 道 open_diagnosis（QA0186-199，题干未变）答案，其余重算；
  - 新增方法：
      fused_parallel —— 并行融合（同一次调用同时给「细粒度三元组」与「病虫档案」，
                        信息源与两路单独方法完全相同，只改组织/组合方式）；
      gated_router   —— 结构覆盖门控：链接到病虫或档案症状命中则作答(复用 graphrag)，
                        否则不调 LLM、直接输出拒答模板（OOD/矛盾题应在此被挡下）。
所有方法同一本地骨干 Qwen2.5-14B、temp=0；逐方法逐题缓存，断点续跑。
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

from run_release_experiments import (  # noqa: E402
    retriever, call_llm, overlap_score, lin_sents, comm_texts,
    a_zeroshot, a_fewshot, a_linear_bm25, a_graphrag, a_kg_only,
    _linked, build_vectorg_prompt,
)
from run_all_experiments import build_kgrag_prompt  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
# 独立手册测试集：TEAPEST_QA=release_qa/qa_external.json 时换题但复用全部方法
QA_V2 = os.path.join(ROOT, *os.environ.get(
    "TEAPEST_QA", os.path.join("release_qa", "qa_v2.json")).split("/"))
V1_DIR = os.path.join(ROOT, "rag_results", "release")
# 跨骨干实验：TEAPEST_RUN_DIR 指向独立结果目录（如 rag_results/release_v2_7b）；
# TEAPEST_NO_V1_SEED=1 时不复用 14B 的 v1 开放题答案（跨骨干必须重算）。
_env_dir = os.environ.get("TEAPEST_RUN_DIR")
OUT = _env_dir if os.path.isabs(_env_dir or "") else (
    os.path.join(ROOT, _env_dir) if _env_dir
    else os.path.join(ROOT, "rag_results", "release_v2"))
os.makedirs(OUT, exist_ok=True)
NO_V1_SEED = bool(os.environ.get("TEAPEST_NO_V1_SEED"))

qa_all = json.load(open(QA_V2, encoding="utf-8"))
qa_map = {q["id"]: q for q in qa_all}

ABSTAIN_TEXT = ("根据现有植保知识库无法可靠判断该问题（相关病虫可能未被收录，"
                "或描述信息不足/相互矛盾）。为避免误导，暂不给出具体诊断与用药，"
                "建议咨询当地植保站或农业技术推广部门、结合田间实测后再处理。")

# 门控规则（先验固定，不在测试集上调）——溯源式，而非词面相似度：
#   开闸作答 当且仅当 链接到 PEST-ORGANISM，或题干逐字命中至少一条
#   长度≥4 的 KG 症状实体短语。词面 bigram 相似度仅记录、不用于门控
#   （它对「近域未收录病虫」无法分离，实验中作为对照阈值曲线报告）。
SYMPTOM_PHRASES = sorted(
    (e for e, t in retriever.entity_to_type.items()
     if t == "PEST-SYMPTOM" and len(e) >= 4),
    key=len, reverse=True)
GATE_PROFILE_TAU = 0.12  # 仅用于宽松对照，不参与主门控


# ---------- 检索 ----------
def a_kgrag(q, hop):
    ents = _linked(q)
    sg = retriever.get_subgraph(ents, hops=hop,
                                max_edges=25 if hop == 1 else 40)
    return call_llm(build_kgrag_prompt(q["question"], sg, [], hop=hop),
                    max_tokens=900)


def top_profiles(q, k=3):
    scored = sorted(((overlap_score(s, q["question"]), s) for s in comm_texts),
                    key=lambda x: -x[0])
    return [(sc, s) for sc, s in scored[:k] if sc > 0], \
        max([sc for sc, _ in scored[:1]], default=0.0)


def a_fused_parallel(q):
    """并行融合：1跳细粒度三元组 + top-1 病虫档案，同一次调用。"""
    ents = _linked(q)
    sg = retriever.get_subgraph(ents, hops=1, max_edges=25)
    prof, _ = top_profiles(q, k=1)
    kg_lines = ["- （%s，%s，%s）" % (e["head"], e["relation"], e["tail"])
                for e in sg["edges"][:20]]
    prof_lines = ["- " + s for _, s in prof]
    sysmsg = (
        "你是一位茶叶病虫害诊断专家。下面同时给出【结构化三元组】与【病虫档案】"
        "两类相互补充的资料，请综合二者作答。\n\n"
        "## 结构化三元组（精确关系）\n%s\n\n"
        "## 病虫档案（症状+四类防治汇总）\n%s\n\n"
        "## 要求\n1) 先给诊断结论；2) 分类列出生物/化学/物理/农业防治；"
        "3) 只使用上述资料，资料没有则明确说明无法确定。"
        % ("\n".join(kg_lines) if kg_lines else "无",
           "\n".join(prof_lines) if prof_lines else "无"))
    return call_llm([{"role": "system", "content": sysmsg},
                     {"role": "user", "content": q["question"]}],
                    max_tokens=900)


def symptom_phrase_hits(q):
    txt = q["question"]
    return [s for s in SYMPTOM_PHRASES if s in txt]


def gate_signal(q):
    """返回 (pest链接分, 档案bigram峰值, 病虫列表, 命中的KG症状短语)。"""
    ents = _linked(q)
    pests = [e for e in ents
             if retriever.entity_to_type.get(e) == "PEST-ORGANISM"]
    best = 0.0
    for s in comm_texts:
        sc = overlap_score(s, q["question"])
        if sc > best:
            best = sc
    return (1.0 if pests else 0.0), best, pests, symptom_phrase_hits(q)


def a_gated_router(q, graphrag_answer):
    """溯源门控：链接到病虫或逐字命中 KG 症状短语则作答，否则确定性拒答。"""
    link, prof, pests, phrases = gate_signal(q)
    opened = bool(pests) or len(phrases) >= 1
    if opened:
        return graphrag_answer, {"gate": "answer", "link": link,
                                 "profile_ov": round(prof, 3),
                                 "rule": "pest" if pests else "phrase"}
    return ABSTAIN_TEXT, {"gate": "abstain", "link": link,
                          "profile_ov": round(prof, 3),
                          "rule": "none"}


# ---------- 方法注册表 ----------
# (name, needs_llm, v1_cache_file_or_None)
# 顺序：关键方法优先（断点时先保住主表头条），纯 LLM/文本基线在后
METHODS = [
    ("graphrag", True, "preds_graphrag.json"),
    ("kgrag1hop", True, "preds_kgrag1hop.json"),
    ("kgrag2hop", True, "preds_kgrag2hop.json"),
    ("fused_parallel", True, None),
    ("kg_only", False, "preds_kg_only.json"),
    ("zeroshot", True, "preds_zeroshot.json"),
    ("fewshot", True, "preds_fewshot.json"),
    ("linear_bm25", True, "preds_linear_bm25.json"),
]

GEN_FN = {
    "zeroshot": a_zeroshot, "fewshot": a_fewshot,
    "linear_bm25": a_linear_bm25, "graphrag": a_graphrag,
    "kgrag1hop": lambda q: a_kgrag(q, 1),
    "kgrag2hop": lambda q: a_kgrag(q, 2),
    "kg_only": a_kg_only, "fused_parallel": a_fused_parallel,
}


def seed_from_v1(v1_name):
    """ {qa_id: answer} 仅当 v1 中该题题干与 v2 完全一致时复用（14 道 open）。 """
    if not v1_name:
        return {}
    p = os.path.join(V1_DIR, v1_name)
    if not os.path.exists(p):
        return {}
    seed = {}
    for r in json.load(open(p, encoding="utf-8")):
        q = qa_map.get(r["qa_id"])
        if q is not None:
            seed[r["qa_id"]] = r["generated_answer"]
    return seed


def run_method(name, limit):
    path = os.path.join(OUT, "preds_%s.json" % name)
    recs = {}
    if os.path.exists(path):
        recs = {r["qa_id"]: r for r in json.load(open(path, encoding="utf-8"))}
    seed = {} if NO_V1_SEED else \
        seed_from_v1(dict((n, v) for n, _, v in METHODS)[name])
    qs = qa_all if limit is None else qa_all[:limit]
    t0 = time.time()
    n_fail = [0]
    for i, q in enumerate(qs):
        if q["id"] in recs:
            continue
        if q["id"] in seed:
            ans, src = seed[q["id"]], "v1cache"
        else:
            ans, src = GEN_FN[name](q), "gen"
            if not (ans or "").strip():
                # 模型调用失败（如 Ollama 不可达）：绝不缓存空答案，下轮重试
                n_fail[0] += 1
                print("  [warn] %s 空答案(调用失败)，不缓存" % q["id"], flush=True)
                if n_fail[0] >= 5:
                    raise RuntimeError("连续空答案，疑似模型服务不可用，中止")
                continue
            n_fail[0] = 0
        recs[q["id"]] = {"qa_id": q["id"], "method": name,
                         "generated_answer": ans, "source": src}
        if i % 15 == 0:
            print("  [%s %d/%d] %.0fs (%s)" %
                  (name, i + 1, len(qs), time.time() - t0, src), flush=True)
        json.dump(list(recs.values()), open(path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
    json.dump(list(recs.values()), open(path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    return list(recs.values())


def run_gated(graphrag_preds):
    path = os.path.join(OUT, "preds_gated_router.json")
    gmap = {r["qa_id"]: r["generated_answer"] for r in graphrag_preds}
    out, meta = [], {}
    for q in qa_all:
        ans, m = a_gated_router(q, gmap.get(q["id"], ""))
        meta[q["id"]] = m
        out.append({"qa_id": q["id"], "method": "gated_router",
                    "generated_answer": ans, "source": m["gate"]})
    json.dump(out, open(path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump(meta, open(os.path.join(OUT, "gate_meta.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=2)
    n_ab = sum(1 for v in meta.values() if v["gate"] == "abstain")
    print("gated_router: abstain %d/%d" % (n_ab, len(meta)), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None, help="逗号分隔，只跑这些方法")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None

    will_llm = any(needs for n, needs, _ in METHODS
                   if (not only or n in only))
    if will_llm:
        from local_model import health_check
        if not health_check():
            print("Ollama 不可达，请先启动模型服务再重试（已缓存结果不会丢）。",
                  flush=True)
            sys.exit(2)

    graphrag_preds = None
    for name, _, _ in METHODS:
        if only and name not in only:
            continue
        print("=== %s ===" % name, flush=True)
        preds = run_method(name, args.limit)
        if name == "graphrag":
            graphrag_preds = preds
    if graphrag_preds is None and (not only or "gated_router" in only):
        graphrag_preds = json.load(
            open(os.path.join(OUT, "preds_graphrag.json"), encoding="utf-8"))
    if not only or "gated_router" in only:
        run_gated(graphrag_preds)


if __name__ == "__main__":
    main()
