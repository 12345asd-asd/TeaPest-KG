# -*- coding: utf-8 -*-
"""
污染注入离线评测（零 LLM）：污染单元进入 LLM 上下文的携带率。

对每个污染视图（k×seed）统计：被问题检索到（presented）的注入单元中，
经各门控配置后幸存（kept）的比例，按污染类型拆分；另报干净单元误伤率。
结果 -> rag_results/release_v2_dtu/pollution/pollution_eval.json/.txt
"""
import glob
import json
import os
import sys
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import dtu_trust as D  # noqa: E402
from eval_dtu_retrieval import to_units  # noqa: E402

POLL = os.path.join(ROOT, "rag_results", "release_v2_dtu", "pollution")
TR = os.path.join(ROOT, "rag_results", "dtu_traces")

CONFIGS = {
    # 无门控：检索到即携带（query 门控只在整题无链接时挡，等价对照）
    "ungated": None,
    "prereg": {},
    "no_safety": {"use_safety_block": False},
    "no_safety_no_R": {"use_safety_block": False, "use_R_prune": False},
    "no_numeric": {"numeric_block": False},
    "soft": {"soft_mode": True},
    "rprune45": {"r_prune_clean": True, "tau": 0.45, "tau_s": 0.55},
}
KINDS = ("wrong_control", "banned_chem", "banned_high_tier",
         "numeric_oos", "numeric_inline")


def replay_gate(rec, cfg):
    """返回 (kept_ids, decision)；只有 decision=answer 单元才真正进入 LLM。"""
    if cfg is None:
        # ungated：schema 白名单也不启用（模拟现状 kgrag/fused 无门控）
        return {u["tid"] for u in rec["units"]}, "answer"
    g = D.gate_units(
        to_units(rec), rec["question"],
        link_ok=rec["link"]["link_rule"] != "none",
        q_type=rec["q_type"], q_stratum=rec["stratum"],
        linked_pests=rec["link"]["pests"],
        cfg=dict(D.DEFAULT_CFG, **cfg),
        extra_relations=[D.REL_BIO]
        if rec["link"].get("family_cue") == "天敌" else None)
    return {u["tid"] for u in g["kept"]}, g["decision"]


def main():
    traces = sorted(glob.glob(os.path.join(
        TR, "qa_v2__kg_polluted_k*.jsonl")))
    out = {}
    for path in traces:
        tag = os.path.basename(path)[len("qa_v2__"):-len(".jsonl")]
        man_p = os.path.join(POLL, "manifest_" + tag + ".json")
        if not os.path.exists(man_p):
            continue
        manifest = json.load(open(man_p, encoding="utf-8"))
        kind_map = {(m["head"], m["relation"], m["tail"]): m["kind"]
                    for m in manifest["items"]}
        recs = [json.loads(l) for l in open(path, encoding="utf-8")]
        kseed = tag.replace("kg_polluted_", "")
        res = {"n_injected": manifest["n_actual"], "configs": {}}
        for cname, cfg in CONFIGS.items():
            presented = defaultdict(int)
            carried = defaultdict(int)
            clean_presented = clean_pruned = 0
            for rec in recs:
                kept_ids, decision = replay_gate(rec, cfg)
                sent = decision == "answer"   # 非 answer 决策不向 LLM 发送任何单元
                for u in rec["units"]:
                    key = (u["h"], u["r"], u["t"])
                    if u["injected"]:
                        kind = kind_map.get(key, "unknown")
                        presented[kind] += 1
                        presented["__all__"] += 1
                        if sent and u["tid"] in kept_ids:
                            carried[kind] += 1
                            carried["__all__"] += 1
                    elif cfg is not None and sent:
                        clean_presented += 1
                        if u["tid"] not in kept_ids and u["schema_known"]:
                            clean_pruned += 1
            rates = {}
            for kind in KINDS + ("__all__",):
                if presented[kind]:
                    rates[kind] = {
                        "presented": presented[kind],
                        "carried": carried[kind],
                        "carry_rate": round(carried[kind]
                                           / presented[kind], 4)}
            res["configs"][cname] = {
                "by_kind": rates,
                "clean_collateral": round(clean_pruned / clean_presented, 5)
                if clean_presented else 0.0}
        out[kseed] = res

    json.dump(out, open(os.path.join(POLL, "pollution_eval.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2)

    # 汇总：按 k 平均 3 seed
    L = ["DTU pollution carry-rate (3 seeds 平均)", "=" * 96]
    ks = sorted({k.split("_seed")[0] for k in out})
    for k in ks:
        seeds = [kk for kk in out if kk.startswith(k + "_seed")]
        L.append("\n### %s" % k)
        L.append("%-15s %9s %9s %9s %9s %9s %9s %9s" %
                 ("config", "wrongCtl", "banT4", "banHT1",
                  "numOOS", "numInline", "all", "误伤clean"))
        for cname in CONFIGS:
            cells = []
            for kind in ("wrong_control", "banned_chem", "banned_high_tier",
                         "numeric_oos", "numeric_inline", "__all__"):
                vals = [out[s]["configs"][cname]["by_kind"]
                        .get(kind, {}).get("carry_rate") for s in seeds]
                vals = [v for v in vals if v is not None]
                cells.append(100 * sum(vals) / len(vals) if vals else None)
            col = [out[s]["configs"][cname]["clean_collateral"]
                   for s in seeds]
            L.append("%-15s %8s%% %8s%% %8s%% %8s%% %8s%% %8s%% %8.2f%%" % (
                cname,
                *[("%.1f" % v) if v is not None else "-" for v in cells],
                100 * sum(col) / len(col)))
    rep = "\n".join(L)
    open(os.path.join(POLL, "pollution_eval.txt"), "w",
         encoding="utf-8").write(rep)
    print(rep)


if __name__ == "__main__":
    main()
