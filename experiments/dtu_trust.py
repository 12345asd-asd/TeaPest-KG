# -*- coding: utf-8 -*-
"""
DTU（Data Trust Unit）证据单元信任门控 —— 核心纯函数模块（零 LLM、确定性）。

概念来源：Yu et al., Smart Agricultural Technology（DOI 10.1016/j.atech.2026.102520）
把 IoT 链路节点抽象为带 R(reliability)/S(security-sensitivity)/Re(recoverability)/
I(integrity) 四维信任的 Data Trust Unit，转发前 T<T_threshold 即剪枝。
本模块把「候选检索证据单元（KG 三元组）」作为 DTU，在其进入 LLM 上下文之前门控。

关键设计约束（见项目计划 C:\\Users\\86130\\.claude\\plans\\idempotent-scribbling-cascade.md）：
  - release KG 为已人工核实的干净图谱，R 低分单元在干净集上**不主动硬剪**
    （τ 默认档下近似零剪枝）；R 硬剪只作用于注入污染单元，R 在干净集只决定软排序。
  - I 维度承担主力门控：安全注册表硬否决、schema 白名单、NUMERIC 槽位
    （阈值/间隔期/剂量等 KG schema 不覆盖的属性）→ 确定性拒答。
  - 剪枝必须可解释（每条带 reason）；任何 prompt 文本中不出现信任分数。
"""
import json
import math
import os
import re
import time
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.abspath(__file__))
DTU_DIR = os.path.join(ROOT, "release_kg", "dtu")

REL_SYMPTOM = "表现症状"
REL_BIO = "生物防治"
REL_CHEM = "化学防治"
REL_PHY = "物理防治"
REL_AGR = "农业防治"
CONTROL_RELATIONS = [REL_BIO, REL_CHEM, REL_PHY, REL_AGR]
VALID_RELATIONS = {REL_SYMPTOM} | set(CONTROL_RELATIONS)

# 预注册阈值（preregistered 2026-10-17）：
# 干净 KG 单源 tier4 单元 R≈.35，τ=.30 → 干净集零硬剪；
# τ_s=.55 只作用于污染注入单元（injected=True）；网格扫描仅用于敏感性图，不参与选点。
TAU_NORMAL = 0.30
TAU_SENSITIVE = 0.55

FORM_SUFFIX = ("可分散油悬浮剂", "可湿性粉剂", "水分散粒剂", "悬浮乳剂", "可溶液剂",
               "可溶粉剂", "微乳剂", "水乳剂", "悬浮剂", "浓乳剂", "种衣剂", "拌种剂",
               "胶悬剂", "颗粒剂", "乳油", "水剂", "烟剂", "粉剂")
SPLIT_RE = re.compile(r"[·・、+]")

# ---------- 槽位关键词（子串/正则；最长最具体优先由调用方处理） ----------
RELATION_CUES = {
    REL_SYMPTOM: ["症状", "表现", "什么样", "特征", "怎么识别", "危害状", "为害状",
                  "病斑", "什么病", "什么虫", "什么害", "哪些害"],
    REL_BIO: ["生物防治", "天敌", "寄生蜂", "瓢虫", "捕食螨", "草蛉", "蜘蛛",
              "Bt", "苏云金", "白僵菌", "绿僵菌", "核型多角体", "病毒制剂",
              "信息素", "性诱剂", "性诱", "释放"],
    REL_CHEM: ["化学防治", "药剂", "农药", "喷施", "喷药", "杀菌剂", "杀虫剂",
               "杀螨剂", "乳油", "可湿性粉剂", "用什么药", "什么药", "哪种药",
               "防治用", "喷布", "喷雾防治"],
    REL_PHY: ["物理防治", "黄板", "蓝板", "粘虫板", "诱虫板", "杀虫灯", "灯光诱杀",
              "人工捕捉", "人工捕杀", "摘除"],
    REL_AGR: ["农业防治", "耕作", "修剪", "清园", "封园", "施肥", "中耕",
              "合理采摘", "及时采摘", "通风", "排水", "除草", "耕翻", "台刈"],
}
# NUMERIC 槽位：KG 五种关系 schema 完全不覆盖 → schema 级 unsupported
NUMERIC_CUES = [
    ("安全间隔期", [r"安全间隔期", r"间隔期", r"采收前\s*\d*\s*天", r"采摘前\s*\d*\s*天",
                  r"施药后.{0,6}天", r"多少天可以", r"多少天能"]),
    ("防治阈值", [r"防治指标", r"虫口密度", r"百叶(若虫|螨|卵|虫)", r"百丛",
                r"芽梢发病率", r"有?蚜?芽梢率", r"百叶.{0,4}数",
                r"每(叶|丛|平方|株|亩|公顷|厘米)",
                r"达到.{0,8}(防治|用药)", r"防治阈值",
                r"施药适期", r"什么生长期", r"龄期", r"几龄",
                r"防效.{0,6}(百分之|多少)", r"百分之多少"]),
    ("剂量用量", [r"用量多少", r"剂量", r"稀释多少", r"\d+\s*倍液", r"每亩.{0,6}(用量|毫升|克|用水)",
                r"每公顷", r"多少倍液", r"浓度多少", r"多少毫升", r"多少克",
                r"多少张", r"多少个", r"多少厘米", r"不低于多少度"]),
]
BAN_ASK_RES = [r"可以(用|使用|喷施)", r"能不能(用|使用)", r"能否(用|使用|喷施)",
               r"可不可以", r"允许(用|使用)", r"是否允许",
               r"能(用|使用|喷施).{0,20}吗", r"喷施.{0,10}吗"]

# qa_v2 题型先验（避免诊断题关键词漏检导致误杀防治证据）
QTYPE_SLOTS = {
    "pest_symptoms": [REL_SYMPTOM],
    "pest_all_controls": CONTROL_RELATIONS,
    "symptom_identify": [REL_SYMPTOM],
    "differential": [REL_SYMPTOM],
    "open_diagnosis": [REL_SYMPTOM] + CONTROL_RELATIONS,
}


def load_json(path):
    return json.load(open(path, encoding="utf-8"))


def load_source_registry(path=None):
    reg = load_json(path or os.path.join(DTU_DIR, "source_tiers.json"))
    return {d: v["tier"] for d, v in reg["domains"].items()}


def load_safety_registry(path=None):
    reg = load_json(path or os.path.join(DTU_DIR, "banned_agrochemicals.json"))
    alias2name = {}
    for c in reg["chemicals"]:
        for a in c["aliases"]:
            alias2name[a] = c["name"]
    return {"records": reg["chemicals"], "alias2name": alias2name}


# ---------- 单元（三元组）静态打分 ----------
_TIER_SCORE = {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.4}


def triple_domains(triple):
    out = []
    seen = set()
    for u in triple.get("sources", []):
        d = urlparse(u).netloc.lower()
        if d and d not in seen:
            seen.add(d)
            out.append(d)
    return out


def r_reliability(domains, tier_reg):
    """R：最高来源层级为主（.75），独立域名佐证为辅（饱和函数）。"""
    if not domains:
        return 0.0, 0, 0
    tiers = [tier_reg[d] for d in domains if d in tier_reg]
    if not tiers:
        return 0.0, 0, 0
    top = max(tiers)
    base = _TIER_SCORE[top]
    d = len(set(domains))
    r_corr = 1 - math.exp(-(d - 1) / 1.5)
    r = 0.75 * base + 0.25 * (0.5 * base + 0.5 * r_corr)
    return round(r, 4), d, top


def split_chemical_components(tail):
    """剥离剂型后缀、按混配分隔符拆成分（全等匹配禁药别名，禁裸子串）。"""
    t = tail.strip()
    for f in sorted(FORM_SUFFIX, key=len, reverse=True):
        if t.endswith(f):
            t = t[: -len(f)]
            break
    return [c.strip() for c in SPLIT_RE.split(t) if c.strip()]


def safety_verdict(triple, safety_reg):
    """仅对化学防治单元：tail 成分全等命中禁用别名 → 返回命中详情。"""
    if triple.get("relation") != REL_CHEM:
        return None
    alias2name = safety_reg["alias2name"]
    for comp in split_chemical_components(triple["tail"]):
        if comp in alias2name:
            return {"banned_name": alias2name[comp], "matched": comp}
    # 注入型越界 tail（如「六六六粉」）：去尾缀后仍以别名开头也判否决
    for comp in split_chemical_components(triple["tail"]):
        for alias, name in alias2name.items():
            if len(alias) >= 3 and comp.startswith(alias):
                return {"banned_name": name, "matched": alias + "*"}
    return None


def enrich_unit(triple, idx, tier_reg, safety_reg, injected=False):
    """补全一个候选单元的全部静态信任字段。"""
    domains = triple_domains(triple)
    R, n_dom, top_tier = r_reliability(domains, tier_reg)
    sv = safety_verdict(triple, safety_reg)
    return {
        "tid": idx,
        "head": triple["head"], "relation": triple["relation"],
        "tail": triple["tail"],
        "head_type": triple.get("head_type", ""),
        "tail_type": triple.get("tail_type", ""),
        "domains": domains, "top_tier": top_tier,
        "n_domains": n_dom, "R": R,
        "safety_block": bool(sv), "banned": sv,
        "injected": bool(injected or triple.get("_dtu_injected")),
        "schema_known": triple["relation"] in VALID_RELATIONS,
    }


# ---------- 查询槽位检测 ----------
def _hit(patterns, text):
    return [p for p in patterns if p in text]


def detect_ban_question(question, safety_reg):
    if not any(re.search(p, question) for p in BAN_ASK_RES):
        return None
    for alias, name in safety_reg["alias2name"].items():
        if len(alias) >= 2 and alias in question:
            return {"banned_name": name, "matched": alias}
    return None


def detect_slots(question, q_type=None, q_stratum=None, safety_reg=None):
    """返回 {ask_relations, ask_numeric, ban, safety_sensitive, hits}。"""
    hits = {}
    # 1) 题型先验（qa_v2 / external 已知类型）
    priors = None
    if q_stratum == "external":
        if q_type == "chemical":
            priors = [REL_CHEM]
        elif q_type in ("threshold", "interval"):
            priors = []
        elif q_type == "ban":
            priors = []
    else:
        priors = QTYPE_SLOTS.get(q_type)

    ask_relations = list(priors) if priors is not None else []
    # 2) 关键词（仅在无先验时驱动；有先验时只做记录）
    for rel, cues in RELATION_CUES.items():
        h = _hit(cues, question)
        if h:
            hits[rel] = h
            if priors is None and rel not in ask_relations:
                ask_relations.append(rel)

    # 3) NUMERIC 槽位（schema 不覆盖）——任何题型都要检出
    ask_numeric = []
    numeric_hits = {}
    for slot, pats in NUMERIC_CUES:
        hh = [p for p in pats if re.search(p, question)]
        if hh:
            ask_numeric.append(slot)
            numeric_hits[slot] = hh

    ban = detect_ban_question(question, safety_reg) if safety_reg else None
    safety_sensitive = bool(ban) or bool(ask_numeric) or (REL_CHEM in ask_relations)
    return {
        "ask_relations": ask_relations,
        "ask_numeric": ask_numeric,
        "numeric_hits": numeric_hits,
        "ban": ban,
        "safety_sensitive": safety_sensitive,
        "hits": hits,
        "priors": priors,
    }


DEFAULT_CFG = {
    "tau": TAU_NORMAL,
    "tau_s": TAU_SENSITIVE,
    "use_R_prune": True,        # 对注入单元按 R 硬剪
    "r_prune_clean": False,     # 干净单元也按 R 硬剪（消融；默认 False=单源保护）
    "use_safety_block": True,   # 禁药注册表硬否决
    "asymmetric_S": True,       # 安全敏感槽用 tau_s；False=对称阈值
    "numeric_block": True,      # NUMERIC schema 不覆盖 → 拒答
    "soft_mode": False,         # True=非安全单元不剪只降序（软降权对照）
    "protect_single_source": True,
    "link_bypass": False,       # 绕过链接门控（ungated 对照）
    "schema_strict": True,      # False=未知 relation 也允许进入上下文（ungated 对照）
    "symptom_vote_link": False, # 症状子图投票作为图链接证据（pipeline 侧使用）
}


def gate_units(units, question, link_ok=True, q_type=None, q_stratum=None,
               linked_pests=None, tier_reg=None, safety_reg=None, cfg=None,
               timed=False, extra_relations=None):
    """
    units: enrich_unit() 列表（候选证据单元，含 schema 未知单元）。
    linked_pests: 问题链接到的 PEST-ORGANISM 列表（含 context_entity）。
    返回 kept/pruned/units/slot_coverage/decision/reasons/τ/耗时。
    """
    t0 = time.perf_counter()
    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    tier_reg = tier_reg or load_source_registry()
    safety_reg = safety_reg or load_safety_registry()
    linked_pests = set(linked_pests or [])
    slot = detect_slots(question, q_type, q_stratum, safety_reg)
    for rel in (extra_relations or []):
        if rel not in slot["ask_relations"]:
            slot["ask_relations"].append(rel)
    if extra_relations and REL_CHEM in extra_relations:
        slot["safety_sensitive"] = True
    tau_s = cfg["tau_s"] if cfg["asymmetric_S"] else cfg["tau"]

    reasons = []

    # 0) ban 问句优先（即使无 pest 链接，如 E-BA-01）→ 确定性安全回答
    if slot["ban"]:
        return _result("ban_veto", ["ban_question:%s" % slot["ban"]["banned_name"]],
                       units, [], slot, cfg, linked_pests, t0, timed)

    # 1) 链接门控（沿用既有 gate_signal 口径，由调用方计算）
    if not link_ok and not cfg.get("link_bypass"):
        return _result("abstain", ["no_link"], units, [], slot, cfg,
                       linked_pests, t0, timed)

    # 2) NUMERIC 槽位 schema 级不覆盖 → 确定性拒答（零 LLM）
    if cfg["numeric_block"] and slot["ask_numeric"]:
        return _result("abstain",
                       ["numeric_unsupported:%s" % ",".join(slot["ask_numeric"])],
                       units, [], slot, cfg, linked_pests, t0, timed)

    # 3) schema 白名单 + 安全硬否决 + R 门控
    kept, pruned = [], []
    for u in units:
        if not u["schema_known"] and cfg.get("schema_strict", True):
            pruned.append((u, "schema_unknown"))
            continue
        if cfg["use_safety_block"] and u["safety_block"]:
            pruned.append((u, "safety_block:%s" % u["banned"]["banned_name"]))
            continue
        if cfg["use_R_prune"] and (u["injected"] or cfg["r_prune_clean"]):
            tau = tau_s if (slot["safety_sensitive"] or
                            u["relation"] == REL_CHEM) else cfg["tau"]
            n_same_slot = sum(1 for o in units
                              if o["head"] == u["head"]
                              and o["relation"] == u["relation"])
            protected = (cfg["protect_single_source"]
                         and not u["injected"] and n_same_slot <= 1)
            if not cfg["soft_mode"] and u["R"] < tau and not protected:
                pruned.append((u, "low_R:%.2f<%.2f" % (u["R"], tau)))
                continue
        kept.append(u)

    # 4) 实例槽位覆盖：明确所求关系槽在「链接病虫」上无幸存单元
    coverage = {}
    slot_empty = []
    for rel in slot["ask_relations"]:
        suv = [u for u in kept if u["relation"] == rel
               and (not linked_pests or u["head"] in linked_pests)]
        scov = [u for u in units if u["relation"] == rel
                and (not linked_pests or u["head"] in linked_pests)]
        coverage[rel] = {"n_units": len(scov), "n_surviving": len(suv),
                         "covered": bool(suv)}
        if not suv and scov:
            # 门控前有、门控后空 → 门控导致（主要发生在污染视图）
            slot_empty.append((rel, "emptied_by_gate"))
        elif not suv:
            slot_empty.append((rel, "absent_in_kg"))

    # 仅在（安全敏感 或 单一明确槽位）且槽位本就不在 KG 时升级拒答；
    # 多槽开放题（all_controls/open_diagnosis）不因单槽缺失而拒答。
    hard_empty = [r for r, why in slot_empty if why == "emptied_by_gate"]
    if hard_empty:
        return _result("abstain", ["slot_emptied:%s" % ",".join(hard_empty)],
                       units, pruned, slot, cfg, linked_pests, t0, timed,
                       coverage=coverage)
    # 单一明确槽位（有关键词证据；external 题型本身即明确）且 KG 无覆盖：
    #   内部题仅安全敏感槽升级拒答（KG 派生题上理论不触发）；
    #   external 题任何明确槽缺失都拒答（手册事实在 KG schema/实例之外）。
    single_specific = len(slot["ask_relations"]) == 1 and (
        bool(slot["hits"]) or q_stratum == "external")
    if slot_empty and single_specific and (
            slot["safety_sensitive"] or q_stratum == "external"):
        rels = ",".join(r for r, _ in slot_empty)
        return _result("abstain", ["slot_not_covered:%s" % rels],
                       units, pruned, slot, cfg, linked_pests, t0, timed,
                       coverage=coverage)

    return _result("answer", [], units, pruned, slot, cfg, linked_pests,
                   t0, timed, coverage=coverage, kept=kept)


def _result(decision, reasons, units, pruned, slot, cfg, linked_pests,
            t0, timed, coverage=None, kept=None):
    if kept is None:
        kept = [u for u in units if u["schema_known"]
                and not (cfg["use_safety_block"] and u["safety_block"])]
    out = {
        "decision": decision,
        "reasons": reasons,
        "kept": kept,
        "pruned": [{"tid": u["tid"], "head": u["head"], "relation": u["relation"],
                    "tail": u["tail"], "injected": u["injected"], "R": u["R"],
                    "reason": why} for u, why in pruned],
        "n_kept": len(kept), "n_pruned": len(pruned),
        "slot_detect": slot,
        "slot_coverage": coverage or {},
        "tau": cfg["tau"], "tau_s": (cfg["tau_s"] if cfg["asymmetric_S"]
                                     else cfg["tau"]),
        "config": {k: cfg[k] for k in ("use_R_prune", "r_prune_clean",
                                       "use_safety_block", "asymmetric_S",
                                       "numeric_block", "soft_mode",
                                       "protect_single_source")},
    }
    if timed:
        out["gate_ms"] = round((time.perf_counter() - t0) * 1000, 3)
    return out


def ban_response_text(banned_name):
    return ("不可以。%s属于茶园禁（限）用农药，茶树病虫害防治中不允许使用；"
            "为保障茶叶质量安全，请改用绿色防控或当前登记允许的药剂，"
            "并咨询当地植保站。" % banned_name)
