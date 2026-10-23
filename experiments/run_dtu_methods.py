# -*- coding: utf-8 -*-
"""
DTU 方法运行器：dtu_fused（头条）/ dtu_kgrag1hop / dtu_gated。

  dtu_fused     —— 候选证据单元门控后：幸存三元组 + 门控后重建的 top1 病虫档案，
                   一次 LLM 调用；门控拒答/禁药否决零调用（确定性文本）。
  dtu_kgrag1hop —— 同门控，prompt 仅含幸存三元组（无置信度标签）。
  dtu_gated     —— 开闸时复用 dtu_fused 答案，非answer决策输出确定性模板
                   （incremental LLM 调用=0，成本对照方法；与现有 gated_router 同构）。

环境变量（沿用项目约定）：
  TEAPEST_QA         题目文件（相对项目根，默认 release_qa/qa_v2.json）
  TEAPEST_RUN_DIR    结果目录（相对项目根，默认 rag_results/release_v2_dtu）
  TEAPEST_MODEL      Ollama 模型（local_model.py 读取，默认 14b）
  TEAPEST_TIMING_LOG 逐调用计时 jsonl（local_model.py 埋点）
命令行：
  --only a,b,c   --limit N   --kg-view 文件   --cfg prereg|soft|no_safety|symmetric
"""
import argparse
import json
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "trustkg_rag", "experiments"))

import dtu_pipeline as P  # noqa: E402
import dtu_trust as D  # noqa: E402

QA_PATH = os.path.join(ROOT, *os.environ.get(
    "TEAPEST_QA", os.path.join("release_qa", "qa_v2.json")).split("/"))
_env_dir = os.environ.get("TEAPEST_RUN_DIR")
OUT = _env_dir if os.path.isabs(_env_dir or "") else (
    os.path.join(ROOT, _env_dir) if _env_dir
    else os.path.join(ROOT, "rag_results", "release_v2_dtu"))
os.makedirs(OUT, exist_ok=True)

ABSTAIN_TEXT = ("根据现有植保知识库无法可靠判断该问题（相关病虫可能未被收录，"
                "或描述信息不足/相互矛盾）。为避免误导，暂不给出具体诊断与用药，"
                "建议咨询当地植保站或农业技术推广部门、结合田间实测后再处理。")

CFG_VARIANTS = {
    "prereg": {},
    "soft": {"soft_mode": True},
    "no_safety": {"use_safety_block": False},
    "symmetric": {"asymmetric_S": False},
    "no_numeric": {"numeric_block": False},
    "r_prune": {"r_prune_clean": True, "tau": 0.45},
    "symvote": {"symptom_vote_link": True},
    "ungated": {"use_R_prune": False, "use_safety_block": False,
                "numeric_block": False, "link_bypass": True,
                "schema_strict": False},
}


def load_cache(path):
    if os.path.exists(path):
        return {r["qa_id"]: r for r in
                json.load(open(path, encoding="utf-8"))}
    return {}


def load_meta(path):
    """gate meta 以 {qa_id: {...}} dict 落盘。"""
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    return {}


def gate_one(q, ret, raw, tier_reg, safety_reg, cfg):
    units, meta = P.candidate_units(q, ret, raw, tier_reg, safety_reg, cfg=cfg)
    g = P.run_gate(q, units, meta, cfg=cfg, timed=True)
    if meta["link"].get("symptom_vote_sc"):
        g["symptom_vote_sc"] = meta["link"]["symptom_vote_sc"]
    return units, meta, g


def deterministic_output(g):
    if g["decision"] == "ban_veto":
        name = g["slot_detect"]["ban"]["banned_name"]
        return D.ban_response_text(name), "dtu_ban"
    return ABSTAIN_TEXT, "dtu_abstain"


def suffix(kg_tag):
    return ("__" + kg_tag) if kg_tag != "clean" else ""


def run_one_method(name, qa, ret, raw, tier_reg, safety_reg, cfg,
                   kg_tag, deps, cfg_tag="prereg"):
    """name ∈ dtu_fused / dtu_kgrag1hop。"""
    sfx = suffix(kg_tag) + ("" if cfg_tag == "prereg" else "__" + cfg_tag)
    cache_path = os.path.join(OUT, "preds_%s%s.json" % (name, sfx))
    meta_path = os.path.join(OUT, "gate_%s%s.json" % (name, sfx))
    recs = load_cache(cache_path)
    metas = load_meta(meta_path)
    call_llm = deps.get("call_llm")
    t0 = time.time()
    n_fail = 0
    for i, q in enumerate(qa):
        if q["id"] in recs:
            continue
        units, meta, g = gate_one(q, ret, raw, tier_reg, safety_reg, cfg)
        kept = g["kept"]
        prof_pest = meta["profile_pest"]
        mrec = {"decision": g["decision"], "reasons": g["reasons"],
                "n_kept": g["n_kept"], "n_pruned": g["n_pruned"],
                "link_rule": meta["link"]["link_rule"],
                "family_cue": meta["link"].get("family_cue"),
                "profile_pest": prof_pest, "gate_ms": g.get("gate_ms"),
                "slots": g["slot_detect"]["ask_relations"],
                "ask_numeric": g["slot_detect"]["ask_numeric"]}
        if g["decision"] != "answer":
            ans, src = deterministic_output(g)
        else:
            profile_text = P.gated_profile_text(prof_pest, kept) if prof_pest else ""
            if name == "dtu_fused":
                msgs = P.build_dtu_fused_prompt(
                    q["question"], kept, prof_pest, profile_text)
            else:
                msgs = P.build_dtu_kgrag_prompt(q["question"], kept, hop=1)
            ans = call_llm(msgs, max_tokens=900)
            if not (ans or "").strip():
                n_fail += 1
                print("  [warn] %s 空答案，不缓存" % q["id"], flush=True)
                if n_fail >= 5:
                    raise RuntimeError("连续空答案，疑似模型服务不可用")
                continue
            src = "gen"
            n_fail = 0
        recs[q["id"]] = {"qa_id": q["id"], "method": name,
                         "generated_answer": ans, "source": src}
        metas[q["id"]] = mrec
        if i % 15 == 0:
            print("  [%s %d/%d] %.0fs" % (name, i + 1, len(qa),
                                          time.time() - t0), flush=True)
        json.dump(list(recs.values()), open(cache_path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        json.dump(metas, open(meta_path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
    json.dump(list(recs.values()), open(cache_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump(metas, open(meta_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    n_det = sum(1 for r in recs.values() if r["source"] != "gen")
    print("%s: %d 题，确定性非作答 %d（ban/abstain），缓存 %s"
          % (name, len(recs), n_det, os.path.basename(cache_path)), flush=True)
    return recs


def run_gated(fused_recs, kg_tag, cfg_tag="prereg"):
    """dtu_gated 与 dtu_fused 答案一致（开闸复用），0 增量调用成本对照。"""
    sfx = suffix(kg_tag) + ("" if cfg_tag == "prereg" else "__" + cfg_tag)
    path = os.path.join(OUT, "preds_dtu_gated%s.json" % sfx)
    out = [{"qa_id": r["qa_id"], "method": "dtu_gated",
            "generated_answer": r["generated_answer"], "source": r["source"]}
           for r in fused_recs.values()]
    json.dump(out, open(path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("dtu_gated: %d 题（复用 dtu_fused，0 增量调用）" % len(out), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="dtu_kgrag1hop,dtu_fused,dtu_gated")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--kg-view", default=None)
    ap.add_argument("--cfg", default="prereg")
    ap.add_argument("--ids", default=None, help="逗号分隔的 qa_id 列表/文件")
    args = ap.parse_args()
    only = set(args.only.split(","))

    qa_all = json.load(open(QA_PATH, encoding="utf-8"))
    if args.ids:
        if os.path.exists(os.path.join(ROOT, args.ids)):
            ids = {x.strip() for x in open(
                os.path.join(ROOT, args.ids), encoding="utf-8")
                if x.strip() and not x.startswith("#")}
        else:
            ids = {x.strip() for x in args.ids.split(",") if x.strip()}
        qa = [q for q in qa_all if q["id"] in ids]
        print("ids 过滤: %d 题" % len(qa), flush=True)
    else:
        qa = qa_all[:args.limit] if args.limit else qa_all
    kg_tag = "clean"
    view_path = None
    if args.kg_view:
        view_path = os.path.join(ROOT, *args.kg_view.split("/"))
        kg_tag = os.path.splitext(os.path.basename(args.kg_view))[0]
    ret, raw = P.load_view(view_path)
    tier_reg = D.load_source_registry()
    safety_reg = D.load_safety_registry()
    cfg = dict(D.DEFAULT_CFG, **CFG_VARIANTS[args.cfg])

    needs_llm = bool(only & {"dtu_fused", "dtu_kgrag1hop"})
    deps = {}
    if needs_llm:
        from local_model import call_llm as _call, health_check
        if not health_check():
            print("Ollama 不可达，请先启动模型服务（已缓存结果不丢）。", flush=True)
            sys.exit(2)

        def call_llm(messages, max_tokens=900):
            sys_msg = next((m["content"] for m in messages
                            if m["role"] == "system"), "")
            user_msg = next((m["content"] for m in messages
                             if m["role"] == "user"), "")
            return _call(user_msg, max_tokens=max_tokens, temperature=0,
                         system_prefix=sys_msg)

        deps["call_llm"] = call_llm

    cfg_tag = args.cfg
    fused = None
    if "dtu_fused" in only:
        fused = run_one_method("dtu_fused", qa, ret, raw, tier_reg,
                               safety_reg, cfg, kg_tag, deps, cfg_tag)
    if "dtu_kgrag1hop" in only:
        run_one_method("dtu_kgrag1hop", qa, ret, raw, tier_reg, safety_reg,
                       cfg, kg_tag, deps, cfg_tag)
    if "dtu_gated" in only:
        if fused is None:
            sfx = suffix(kg_tag) + ("" if cfg_tag == "prereg"
                                    else "__" + cfg_tag)
            p = os.path.join(OUT, "preds_dtu_fused%s.json" % sfx)
            fused = {r["qa_id"]: r for r in
                     json.load(open(p, encoding="utf-8"))}
        run_gated(fused, kg_tag, cfg_tag)


if __name__ == "__main__":
    main()
