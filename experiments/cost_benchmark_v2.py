# -*- coding: utf-8 -*-
"""
延迟/调用成本基准（论文成本表）。

在固定分层小面板（6 lookup + 6 diagnosis = 12 道答案题）上实测每个 LLM 方法
的每次调用 wall-clock 与 token 计数（Ollama 原生 prompt_eval_count /
eval_count），通过 TEAPEST_TIMING_LOG 逐方法写独立 jsonl 实现归因
（_log_timing 在每次调用时读环境变量，可按方法切换）。

门控不产生新调用（开闸时复用 Profile-RAG 答案，关闸时 0 调用），其每问调用数
由 gate_meta.json 的开闸比例给出；KG-Only 为 0；串行级联为 2 次调用。
7B 全量运行（158 问/方法）的全工作日志由 aggregate_cost 分段归因。

用法（先确保 TEAPEST_MODEL 指向目标骨干，结果目录用默认）：
    python cost_benchmark_v2.py run      # 跑面板（约 84+24 次调用）
    python cost_benchmark_v2.py agg      # 汇总两骨干 -> cost_table.json/.txt
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "trustkg_rag", "experiments"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import run_release_v2 as RV2  # noqa: E402
from run_release_experiments import call_llm  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
CDIR = os.path.join(RV2.OUT, "cost")
os.makedirs(CDIR, exist_ok=True)

ONE_CALL = ["zeroshot", "fewshot", "linear_bm25", "graphrag",
            "kgrag1hop", "kgrag2hop", "fused_parallel"]


def panel_questions():
    """固定 12 题面板：lookup/diagnosis 各取均匀 6 题（确定性，不随顺序变）。"""
    ans = [q for q in RV2.qa_all if q["expected"] == "answer"]
    look = [q for q in ans if q["stratum"] == "lookup"]
    diag = [q for q in ans if q["stratum"] == "diagnosis"]
    pick = lambda xs, k: [xs[round(i * (len(xs) - 1) / (k - 1))] for i in range(k)]
    panel = pick(look, 6) + pick(diag, 6)
    ids = [q["id"] for q in panel]
    assert len(ids) == 12 and len(set(ids)) == 12
    return panel


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def run():
    panel = panel_questions()
    json.dump([q["id"] for q in panel],
              open(os.path.join(CDIR, "panel_ids.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("panel:", [q["id"] for q in panel])
    for name in ONE_CALL:
        log = os.path.join(CDIR, "cost_%s.jsonl" % name)
        os.environ["TEAPEST_TIMING_LOG"] = log
        if os.path.exists(log):
            os.remove(log)
        for i, q in enumerate(panel):
            ans = RV2.GEN_FN[name](q)
            if not ans.strip():
                raise RuntimeError("%s %s 空答案，服务异常，中止" % (name, q["id"]))
            print("  [%s %d/12]" % (name, i + 1), flush=True)

    # 串行级联（2 次/问）：内联 ablate 的两步，逐次计时
    from run_release_experiments import overlap_score, retriever
    from run_all_experiments import build_kgrag_prompt
    log = os.path.join(CDIR, "cost_serial.jsonl")
    os.environ["TEAPEST_TIMING_LOG"] = log
    if os.path.exists(log):
        os.remove(log)
    PEST_SET = sorted(
        [e for e in retriever.entity_to_type
         if retriever.entity_to_type[e] == "PEST-ORGANISM"],
        key=len, reverse=True)
    import run_release_experiments as REX
    for i, q in enumerate(panel):
        sc = sorted(((overlap_score(s, q["question"]), s)
                     for s in REX.comm_texts), key=lambda x: -x[0])
        top = [s for c, s in sc[:3] if c > 0]
        dmsg = [{"role": "system", "content":
                 "你是茶叶病虫害诊断专家。根据病虫档案，只输出一个最可能的"
                 "病虫害规范名称，不要任何解释。\n档案：\n" + "\n".join(top)},
                {"role": "user", "content": q["question"]}]
        name = call_llm(dmsg, max_tokens=24)
        pest = next((p for p in PEST_SET if p in (name or "")), None)
        if pest is None:
            pest = top[0].split("]")[0].lstrip("[") if top else ""
            pest = pest if pest in retriever.entity_to_type else None
        if pest:
            trs = retriever.head_index.get(pest, [])
            edges = [{"head": t["head"], "relation": t["relation"],
                      "tail": t["tail"]} for t in trs][:25]
            call_llm(build_kgrag_prompt(q["question"], {"edges": edges}, [],
                                        hop=1), max_tokens=900)
        print("  [serial %d/12]" % (i + 1), flush=True)

    # verbalized 置信附加调用（1 次短调用/问）
    from run_release_confidence_v2 import verbalized_conf
    fused = {r["qa_id"]: r["generated_answer"] for r in
             json.load(open(os.path.join(RV2.OUT, "preds_fused_parallel.json"),
                            encoding="utf-8"))}
    log = os.path.join(CDIR, "cost_verbalized.jsonl")
    os.environ["TEAPEST_TIMING_LOG"] = log
    if os.path.exists(log):
        os.remove(log)
    for i, q in enumerate(panel):
        verbalized_conf(q["question"], fused.get(q["id"], ""))
        print("  [verbalized %d/12]" % (i + 1), flush=True)
    os.environ.pop("TEAPEST_TIMING_LOG", None)
    print("done ->", CDIR)


def stats(recs):
    if not recs:
        return None
    pt = [r["prompt_tokens"] for r in recs if r.get("prompt_tokens")]
    gt = [r["gen_tokens"] for r in recs if r.get("gen_tokens")]
    ts = [r["total_s"] for r in recs]
    return {
        "n_calls": len(recs),
        "prompt_tokens": round(statistics.mean(pt)) if pt else None,
        "gen_tokens": round(statistics.mean(gt)) if gt else None,
        "latency_s": round(statistics.mean(ts), 2),
        "latency_median_s": round(statistics.median(ts), 2),
    }


def agg_panel(tag_dir, label):
    cdir = os.path.join(tag_dir, "cost")
    out = {"backbone": label, "panel": 12, "methods": {}}
    for name in ONE_CALL + ["serial", "verbalized"]:
        recs = read_jsonl(os.path.join(cdir, "cost_%s.jsonl" % name))
        s = stats(recs)
        if s:
            s["calls_per_question"] = 2 if name == "serial" else 1
            out["methods"][name] = s
    return out


def agg_7b_full():
    """7B 全量日志：方法按注册表顺序各跑 158 次成功调用，顺序分段归因。"""
    p = os.path.join(ROOT, "rag_results", "release_v2_7b", "timing_7b.jsonl")
    recs = read_jsonl(p)
    order = [n for n, needs, _ in RV2.METHODS if needs]  # 7 个 LLM 方法
    per = 158
    out = {"backbone": "qwen2.5-7b", "full_run_questions": per, "methods": {}}
    if len(recs) < len(order) * per:
        out["warning"] = "日志不全: %d/%d 行" % (len(recs), len(order) * per)
    for i, name in enumerate(order):
        seg = recs[i * per:(i + 1) * per]
        s = stats(seg)
        if s:
            s["calls_per_question"] = 1
            out["methods"][name] = s
    return out


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "agg"
    if cmd == "run":
        run()
        return
    p14 = agg_panel(os.path.join(ROOT, "rag_results", "release_v2"),
                    "qwen2.5-14b")
    p7 = agg_7b_full()

    # 门控开闸比例（14B 目录的 gate_meta；规则确定性，两骨干一致）
    gm = json.load(open(os.path.join(ROOT, "rag_results", "release_v2",
                                     "gate_meta.json"), encoding="utf-8"))
    n_ans = sum(1 for q in RV2.qa_all if q["expected"] == "answer")
    open_rate = sum(1 for v in gm.values() if v["gate"] == "answer") / len(gm)
    ans_open = sum(1 for q in RV2.qa_all
                   if q["expected"] == "answer"
                   and gm[q["id"]]["gate"] == "answer") / n_ans
    gate_info = {"calls_per_question_overall": round(open_rate, 3),
                 "calls_on_answerable": round(ans_open, 3),
                 "calls_on_ood": 0, "note": "关闸0调用；开闸复用Profile答案"}
    out = {"panel_14b": p14, "full_7b": p7, "gated_router": gate_info,
           "kg_only": {"calls_per_question": 0},
           "hardware": "RTX 3060 12GB, Ollama, Q4_K_M, temperature=0"}
    json.dump(out, open(os.path.join(ROOT, "rag_results", "release_v2",
                                     "cost_table.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    def row(tag, m):
        s = out[tag]["methods"].get(m)
        if not s:
            return "%-16s %6s %6s %7s" % (m, "-", "-", "-")
        return "%-16s %6s %6s %7.2f" % (m, s["prompt_tokens"],
                                        s["gen_tokens"], s["latency_s"])

    L = ["每问调用成本（面板=12答案题；7B全量=158题）", "=" * 70]
    for tag, title in (("panel_14b", "14B 面板"), ("full_7b", "7B 全量")):
        L.append("\n[%s]" % title)
        L.append("%-16s %6s %6s %7s" % ("method", "pTok", "gTok", "lat/s"))
        for m in ONE_CALL + ["serial", "verbalized"]:
            if m in out[tag]["methods"]:
                L.append(row(tag, m))
    L.append("\n[gated_router] " + json.dumps(gate_info, ensure_ascii=False))
    rep = "\n".join(L)
    open(os.path.join(ROOT, "rag_results", "release_v2", "cost_table.txt"),
         "w", encoding="utf-8").write(rep)
    print(rep)


if __name__ == "__main__":
    main()
