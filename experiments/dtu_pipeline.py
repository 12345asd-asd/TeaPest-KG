# -*- coding: utf-8 -*-
"""
DTU 检索-门控管线（trace 落盘与 LLM 方法共用）：
  - 载入 KG 视图（干净 release KG 或污染注入视图）；
  - 问题链接：实体子串 + context_entity + 科级泛化扩展（螨类/蚜虫/粉虱/天敌…）；
  - 候选证据单元：1-hop 子图 ∪ top1 病虫档案全部三元组，去重、enrich；
  - gate_units 离线重放（任意 cfg）；
  - 门控后病虫档案重建（档案不得绕开单元门控）；
  - 无信任标签的 prompt builder（不向 LLM 泄漏分数）。
"""
import json
import math
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

from run_all_experiments import KGRetriever  # noqa: E402
from run_release_experiments import overlap_score, comm_texts  # noqa: E402
import dtu_trust as D  # noqa: E402

KG_PATH = os.path.join(ROOT, "release_kg", "kg_triples.json")
ENT_PATH = os.path.join(ROOT, "release_kg", "entities.csv")

# 强词法证据回退（preregistered 2026-10-17）：无病虫/症状直链时，top1 病虫档案
# bigram 覆盖率 ≥0.22（既有宽松对照 GATE_PROFILE_TAU=.12 的近两倍严格度）才作为
# 结构性链接证据；低于此仍判 no_link。结构性常量，不在答案结果上调参。
PROFILE_FALLBACK_TAU = 0.22

# 症状子图投票阈值（离线分离分析后固定 2026-10-18；未在答案结果上调参）：
# 题干词命中症状节点 → IDF 加权投票给病虫节点，top1 份额 ≥.35 才作为图链接证据。
# .35 档离线表现：58 诊断题救回 48（含 3 道 prereg 误拒），放入 4/30 近域 OOD。
SYMPTOM_VOTE_TAU = 0.35


def build_symptom_voter(ret):
    """症状节点空间投票器：词（症状短语 2-5 字窗口）→ 宿主病虫倒排 + IDF。"""
    pest_syms = {}
    for t in ret.triples:
        if t["relation"] == D.REL_SYMPTOM:
            pest_syms.setdefault(t["head"], set()).add(t["tail"])
    pests = list(pest_syms)
    n_pest = max(1, len(pests))
    inv = {}
    for p, ss in pest_syms.items():
        terms = set()
        for s in ss:
            for n in (2, 3, 4, 5):
                for i in range(len(s) - n + 1):
                    w = s[i:i + n]
                    if re.search(r"[一-鿿]", w):
                        terms.add(w)
        for w in terms:
            inv.setdefault(w, set()).add(p)

    def vote(question):
        votes = {}
        n_fired = 0
        for w, hosts in inv.items():
            if w in question:
                n_fired += 1
                weight = math.log(n_pest / len(hosts)) + 0.5
                for p in hosts:
                    votes[p] = votes.get(p, 0.0) + weight
        if not votes:
            return None, 0.0, 0
        total = sum(votes.values())
        top = max(votes.items(), key=lambda x: x[1])
        return top[0], round(top[1] / total, 4), n_fired

    return vote

# 科级泛化：题干无具体病虫链接时，按类别词扩展到该类全部病虫（外部手册题常见）
FAMILY_CUES = [
    ("天敌", None, [D.REL_BIO]),                    # 聚合所有病虫的生物防治
    ("螨", "螨", None), ("蚜", "蚜", None),
    ("粉虱", "粉虱", None), ("叶蝉", "叶蝉", None),
    ("尺蠖", "尺蠖", None), ("刺蛾", "刺蛾", None),
    ("象甲", "象甲", None), ("蓟马", "蓟马", None),
    ("蝽", "蝽", None),
]


def load_view(kg_view=None):
    path = kg_view or KG_PATH
    ret = KGRetriever(path, ENT_PATH)
    raw = {(t["head"], t["relation"], t["tail"]): t for t in ret.triples}
    return ret, raw


def family_expand(question, ret):
    """返回 (扩展病虫列表, 命中类别词, 强制槽位 or None)。"""
    pests_all = [e for e, t in ret.entity_to_type.items()
                 if t == "PEST-ORGANISM"]
    for cue, sub, force_slots in FAMILY_CUES:
        if cue in question:
            if sub is None:
                return pests_all, cue, force_slots
            hit = [p for p in pests_all if sub in p]
            if hit:
                return hit, cue, None
    return [], None, None


def link_question(q, ret, vote_result=None):
    """返回 dict: ents/pests/phrases/family/forced_slots/link_ok/link_rule。

    vote_result=(pest, score, n_fired) 时，在直链/科级泛化/强档案回退全失败后，
    用症状子图投票结果作为图链接证据（rule=symptom_vote）。
    """
    ents = ret.find_entities(q["question"], min_len=2)
    ctx = q.get("context_entity", "")
    if ctx and ctx not in ents:
        ents.append(ctx)
    if not ents and ctx:
        ents = [ctx]
    pests = [e for e in ents if ret.entity_to_type.get(e) == "PEST-ORGANISM"]
    phrases = [e for e, t in ret.entity_to_type.items()
               if t == "PEST-SYMPTOM" and len(e) >= 4 and e in q["question"]]
    fam, cue, forced = (None, None, None)
    prof_pest, prof_sc = (None, 0.0)
    if not pests:
        fam_pests, cue, forced = family_expand(q["question"], ret)
        if cue:
            fam = fam_pests
            for p in fam_pests:
                if p not in pests:
                    pests.append(p)
                if p not in ents:
                    ents.append(p)
    if not pests and not phrases:
        prof_pest, prof_sc = top_profile_pest(q["question"], ret)
        if prof_pest and prof_sc >= PROFILE_FALLBACK_TAU:
            pests.append(prof_pest)
            ents.append(prof_pest)
    vote_sc = 0.0
    if not pests and not phrases and vote_result:
        vp, vsc, nf = vote_result
        if vp and vsc >= SYMPTOM_VOTE_TAU:
            pests.append(vp)
            ents.append(vp)
            vote_sc = vsc
    link_ok = bool(pests) or bool(phrases)
    if pests:
        if vote_sc:
            rule = "symptom_vote"
        else:
            rule = "pest" + ("+family" if fam else "")
            if prof_pest and not fam:
                rule = "profile_strong"
    elif phrases:
        rule = "phrase"
    else:
        rule = "none"
    return {"ents": ents, "pests": pests, "phrases": phrases,
            "family_cue": cue, "forced_slots": forced,
            "profile_fallback": prof_pest if rule == "profile_strong" else None,
            "profile_fallback_sc": round(prof_sc, 3),
            "symptom_vote_sc": vote_sc,
            "link_ok": link_ok, "link_rule": rule}


def top_profile_pest(question, ret):
    """bigram 重叠 top1 档案的病虫名（与 run_release_v2 同口径）。"""
    best_sc, best_pest, best_text = 0.0, None, None
    for s in comm_texts:
        sc = overlap_score(s, question)
        if sc > best_sc:
            best_sc, best_text = sc, s
    if best_sc > 0 and best_text:
        best_pest = best_text.split("]")[0].strip("[")
    return best_pest, best_sc


_VOTER_CACHE = {}


def get_voter(ret):
    key = id(ret)
    if key not in _VOTER_CACHE:
        _VOTER_CACHE[key] = build_symptom_voter(ret)
    return _VOTER_CACHE[key]


def candidate_units(q, ret, raw, tier_reg, safety_reg, hop=1, max_edges=25,
                    cfg=None):
    """1-hop 子图 ∪ top1 档案病虫的全部三元组；返回 (units, meta)。"""
    raw_vote = get_voter(ret)(q["question"])  # 总是计算，trace 留底、离线可重放
    vote_result = raw_vote if cfg and cfg.get("symptom_vote_link") else None
    link = link_question(q, ret, vote_result=vote_result)
    sg = ret.get_subgraph(link["ents"], hops=hop, max_edges=max_edges)
    units, seen = [], set()

    def add(triple, order):
        key = (triple["head"], triple["relation"], triple["tail"])
        if key in seen:
            return
        seen.add(key)
        units.append(D.enrich_unit(triple, len(units), tier_reg, safety_reg))

    for e in sg["edges"]:
        triple = raw.get((e["head"], e["relation"], e["tail"]))
        if triple:
            add(triple, len(units))
    # 链接病虫（含科级泛化扩展）的全部三元组显式并入，兜住子图 25 条截断
    for p in link["pests"]:
        for t in ret.triples:
            if t["head"] == p:
                add(t, len(units))
    prof_pest, prof_sc = top_profile_pest(q["question"], ret)
    if prof_pest:
        for t in ret.triples:
            if t["head"] == prof_pest:
                add(t, len(units))
    meta = {"link": link, "profile_pest": prof_pest, "profile_score": round(prof_sc, 3),
            "symptom_vote": {"pest": raw_vote[0], "score": raw_vote[1],
                             "n_fired": raw_vote[2]},
            "n_raw": len(units)}
    return units, meta


def run_gate(q, units, meta, cfg=None, timed=True):
    """对一题执行门控（科级泛化的强制槽位经 extra_relations 注入）。"""
    return D.gate_units(
        units, q["question"],
        link_ok=meta["link"]["link_ok"],
        q_type=q.get("type"), q_stratum=q.get("stratum"),
        linked_pests=meta["link"]["pests"],
        cfg=cfg, timed=timed,
        extra_relations=meta["link"].get("forced_slots"))


# ---------- 门控后档案重建（与 run_release_experiments.communities 同模板） ----------
def gated_profile_text(pest, kept):
    trs = [u for u in kept if u["head"] == pest]
    if not trs:
        return ""
    syms = sorted({u["tail"] for u in trs if u["relation"] == D.REL_SYMPTOM})
    parts = ["[%s] 症状:%s。" % (pest, "、".join(syms[:6]))] if syms else []
    for r in D.CONTROL_RELATIONS:
        ms = sorted({u["tail"] for u in trs if u["relation"] == r})
        if ms:
            parts.append("%s:%s。" % (r, "、".join(ms[:8])))
    return "".join(parts)


def build_dtu_kgrag_prompt(question, kept, hop=1):
    kg_lines = ["- （%s，%s，%s）" % (u["head"], u["relation"], u["tail"])
                for u in kept[:20]]
    hint = "\n请展示推理链: 症状 → 病因(病虫害) → 各类防治方法。" if hop == 2 else ""
    sysmsg = (
        "你是一位茶叶病虫害诊断专家。请根据以下知识图谱事实进行诊断推理。\n\n"
        "## 知识图谱事实（结构化知识）\n%s\n\n"
        "## 回答要求\n1. 先给出诊断结论（是什么病虫害）；"
        "2. 分类给出防治建议：生物防治、化学防治、物理防治、农业防治；"
        "3. 引用知识图谱中的事实支撑判断；4. 只使用上述事实，事实没有的内容不要编造，"
        "无法确定时明确说明。%s" % ("\n".join(kg_lines) if kg_lines else "无", hint))
    return [{"role": "system", "content": sysmsg},
            {"role": "user", "content": question}]


def build_dtu_fused_prompt(question, kept, prof_pest, profile_text):
    kg_lines = ["- （%s，%s，%s）" % (u["head"], u["relation"], u["tail"])
                for u in kept[:20]]
    prof_lines = ["- " + profile_text] if profile_text else []
    sysmsg = (
        "你是一位茶叶病虫害诊断专家。下面同时给出【结构化三元组】与【病虫档案】"
        "两类相互补充的资料，请综合二者作答。\n\n"
        "## 结构化三元组（精确关系）\n%s\n\n"
        "## 病虫档案（症状+四类防治汇总）\n%s\n\n"
        "## 要求\n1) 先给诊断结论；2) 分类列出生物/化学/物理/农业防治；"
        "3) 只使用上述资料，资料没有则明确说明无法确定，"
        "不编造阈值、剂量、安全间隔期等资料中不存在的数值；"
        "4) 涉及农药使用须符合茶园安全要求。"
        % ("\n".join(kg_lines) if kg_lines else "无",
           "\n".join(prof_lines) if prof_lines else "无"))
    return [{"role": "system", "content": sysmsg},
            {"role": "user", "content": question}]


def trace_record(q, units, meta, kg_tag):
    """构造一行 trace（只存预算信号与候选单元，不存决策——决策离线重放）。"""
    return {
        "schema": "dtu-trace-v1",
        "kg_tag": kg_tag,
        "qa_id": q["id"],
        "stratum": q.get("stratum"),
        "q_type": q.get("type"),
        "kg_answerable": q.get("kg_answerable"),
        "expected": q.get("expected"),
        "question": q["question"],
        "context_entity": q.get("context_entity", ""),
        "link": {"ents": meta["link"]["ents"], "pests": meta["link"]["pests"],
                 "phrases": meta["link"]["phrases"],
                 "family_cue": meta["link"]["family_cue"],
                 "link_rule": meta["link"]["link_rule"]},
        "profile_pest": meta["profile_pest"],
        "profile_score": meta["profile_score"],
        "symptom_vote": meta.get("symptom_vote"),
        "units": [{"tid": u["tid"], "h": u["head"], "r": u["relation"],
                   "t": u["tail"], "domains": u["domains"],
                   "top_tier": u["top_tier"], "n_domains": u["n_domains"],
                   "R": u["R"], "safety_block": u["safety_block"],
                   "banned": u["banned"], "injected": u["injected"],
                   "schema_known": u["schema_known"]} for u in units],
    }
