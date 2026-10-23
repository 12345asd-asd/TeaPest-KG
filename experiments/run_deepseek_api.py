# -*- coding: utf-8 -*-
"""
DeepSeek API 基线：与 14B/7B 完全相同的题目、检索证据与 prompt，
仅把唯一生成入口 call_llm 替换为 DeepSeek Chat API（OpenAI 兼容）。

用法（PowerShell）:
  $env:DEEPSEEK_API_KEY="sk-..."
  $env:TEAPEST_RUN_DIR="rag_results/release_v2_deepseek"
  $env:TEAPEST_NO_V1_SEED="1"
  python run_deepseek_api.py --methods graphrag,kgrag1hop --limit 4   # 冒烟
  python run_deepseek_api.py                                         # 全量

设计：
  - 复用 run_release_v2 的 GEN_FN / gate（检索、证据组织、prompt 一字不改）；
  - call_llm(messages, max_tokens, temperature) 同签名替换，T=0；
  - 断点续跑：已存在的 qa_id 直接跳过；空/失败答案不写盘；
  - 并发 4（只读检索线程安全；写盘按方法加锁，每题原子落盘）。
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

os.environ.setdefault("TEAPEST_RUN_DIR", "rag_results/release_v2_deepseek")
os.environ.setdefault("TEAPEST_NO_V1_SEED", "1")

import run_release_v2 as R                      # noqa: E402
import run_release_experiments as RE           # noqa: E402

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
BASE_URL = os.environ.get("DEEPSEEK_BASE", "https://api.deepseek.com").rstrip("/")
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

LLM_METHODS = ["graphrag", "kgrag1hop", "kgrag2hop", "fused_parallel",
               "zeroshot", "fewshot", "linear_bm25"]


def ds_call_llm(messages, max_tokens=1024, temperature=0):
    """OpenAI-compatible chat call; same signature as local call_llm."""
    body = json.dumps({
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        BASE_URL + "/chat/completions",
        data=body,
        headers={"Authorization": "Bearer " + API_KEY,
                 "Content-Type": "application/json"},
        method="POST")
    last_err = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                d = json.loads(resp.read().decode("utf-8"))
            return d["choices"][0]["message"]["content"] or ""
        except urllib.error.HTTPError as e:
            last_err = "HTTP %s: %s" % (e.code, e.read()[:300])
            # 429 / 5xx 退避重试；4xx 直接抛
            if e.code in (400, 401, 403, 404):
                raise
        except Exception as e:  # noqa: BLE001
            last_err = repr(e)
        time.sleep(min(2 ** attempt, 30))
    raise RuntimeError("DeepSeek call failed after retries: %s" % last_err)


# 在两个模块的命名空间里同时替换（部分 a_* 在 RE 内引用其模块全局）
R.call_llm = ds_call_llm
RE.call_llm = ds_call_llm

_lock = threading.Lock()


def _load(path):
    if os.path.exists(path):
        return {r["qa_id"]: r for r in json.load(open(path, encoding="utf-8"))}
    return {}


def _save(path, recs):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(list(recs.values()), f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def run_method(name, limit=None, workers=4):
    if not API_KEY:
        sys.exit("ERROR: set DEEPSEEK_API_KEY first")
    path = os.path.join(R.OUT, "preds_%s.json" % name)
    recs = _load(path)
    qs = R.qa_all if limit is None else R.qa_all[:limit]
    todo = [q for q in qs if q["id"] not in recs]
    print("[%s] %d/%d cached, generating %d (model=%s, workers=%d)"
          % (name, len(recs), len(qs), len(todo), MODEL, workers), flush=True)
    if not todo:
        return list(recs.values())
    t0 = time.time()
    done = [0]

    def one(q):
        ans = R.GEN_FN[name](q)
        if not (ans or "").strip():
            raise RuntimeError("empty answer for %s" % q["id"])
        return q["id"], ans

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, q): q for q in todo}
        for fut in as_completed(futs):
            q = futs[fut]
            try:
                qid, ans = fut.result()
            except Exception as e:  # noqa: BLE001
                print("  [FAIL] %s: %s" % (q["id"], e), flush=True)
                continue
            with _lock:
                recs[qid] = {"qa_id": qid, "method": name,
                             "generated_answer": ans, "source": "deepseek"}
                done[0] += 1
                if done[0] % 10 == 0 or done[0] == len(todo):
                    _save(path, recs)
                    print("  [%s] %d/%d  %.0fs" %
                          (name, done[0], len(todo), time.time() - t0),
                          flush=True)
    _save(path, recs)
    return list(recs.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=",".join(LLM_METHODS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gated", action="store_true",
                    help="also rebuild deterministic gated_router")
    args = ap.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        run_method(m, args.limit, args.workers)
    if args.gated:
        gp = [r for r in run_method("graphrag", args.limit, args.workers)] \
            if "graphrag" not in methods else None
        gr = gp if gp is not None else json.load(
            open(os.path.join(R.OUT, "preds_graphrag.json"), encoding="utf-8"))
        R.run_gated(gr)
    print("done ->", R.OUT)


if __name__ == "__main__":
    main()
