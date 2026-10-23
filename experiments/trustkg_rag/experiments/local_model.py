# -*- coding: utf-8 -*-
"""
本地模型 · 唯一调用入口（Single Source of Truth）
================================================

项目里【所有】本地大模型推理都从这里调用，不要在别的文件里手写
urllib / requests / ollama / transformers 去直接调模型。

约定：
    1. 换模型 / 换端点 / 换参数 → 只改本文件顶部 CONFIG 块。
    2. 实验脚本里统一：        from local_model import call_llm
    3. 不要在任何其它 .py 里重复实现调用逻辑。

后端：本地 Ollama（http://127.0.0.1:11434），模型 qwen2.5-14b-custom，
      由 RTX 3060 推理（GPU）。
"""

import json
import os
import time
import urllib.request
import urllib.error

# ==================== 改这里即可，全局生效 ====================
CONFIG = {
    "base_url": "http://127.0.0.1:11434",     # Ollama 服务地址
    # 跨骨干实验：设环境变量 TEAPEST_MODEL=qwen2.5:7b 即可切换，无需改代码
    "model": os.environ.get("TEAPEST_MODEL", "qwen2.5-14b-custom"),
    "gguf": r"H:\ollama_models\qwen2.5-14b-instruct-q4_k_m.gguf",
    "http_timeout": 600,                       # 单次请求超时（秒）
    "max_retries": 3,                          # 失败重试次数
    "n_ctx": 8192,                             # 上下文窗口（KG prompt 较长；q4 14B 在 12GB 显存下可容纳）
}
# ==============================================================


def _request(payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{CONFIG['base_url']}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_llm(user_prompt, max_tokens=2048, temperature=0.0, system_prefix=None):
    """
    本地模型推理入口。签名与原 DeepSeek API 版一致，可即插即用替换。

    参数:
        user_prompt    用户提示（正文）
        max_tokens     最大生成 token 数
        temperature    温度：0=确定性，>0=采样（语义熵等需多采样）
        system_prefix  系统提示（可选）
    返回:
        生成的文本（已 strip）；多次失败返回空字符串 ""。
    """
    payload = {
        "model": CONFIG["model"],
        "prompt": user_prompt,
        "stream": False,
        "options": {
            "num_predict": int(max_tokens),
            "temperature": float(temperature),
            "n_ctx": CONFIG["n_ctx"],
        },
    }
    if system_prefix:
        payload["system"] = system_prefix

    last_err = None
    for attempt in range(CONFIG["max_retries"]):
        try:
            result = _request(payload, CONFIG["http_timeout"])
            _log_timing(result, payload)
            return (result.get("response") or "").strip()
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")
            print(f"    [Ollama HTTP {e.code} attempt {attempt+1}] {err[:120]}")
            # 5xx（服务未起/繁忙）才重试；4xx（如坏 prompt）不重试
            if e.code >= 500 and attempt < CONFIG["max_retries"] - 1:
                time.sleep(2 ** attempt)
                continue
            break
        except urllib.error.URLError as e:
            last_err = str(e)
            print(f"    [Ollama 连接失败 attempt {attempt+1}] {str(e)[:80]}")
            if attempt < CONFIG["max_retries"] - 1:
                time.sleep(3)
        except Exception as e:  # 超时、JSON 解析等
            last_err = str(e)
            print(f"    [Ollama 错误 attempt {attempt+1}] {str(e)[:80]}")
            if attempt < CONFIG["max_retries"] - 1:
                time.sleep(3)

    print(f"    ⚠ call_llm 最终失败: {last_err}")
    return ""


def _log_timing(result, payload):
    """成本表埋点：TEAPEST_TIMING_LOG 指向 jsonl 时，逐次调用追加一行。

    记录 Ollama 原生计数字段（耗时单位 ns→s），失败/未设环境变量时静默跳过。
    """
    path = os.environ.get("TEAPEST_TIMING_LOG")
    if not path:
        return
    try:
        rec = {
            "model": CONFIG["model"],
            "tag": os.environ.get("TEAPEST_RUN_TAG", ""),
            "ts": round(time.time(), 1),
            "prompt_chars": len(payload.get("prompt") or ""),
            "prompt_tokens": result.get("prompt_eval_count"),
            "gen_tokens": result.get("eval_count"),
            "total_s": round((result.get("total_duration") or 0) / 1e9, 3),
            "prompt_eval_s": round((result.get("prompt_eval_duration") or 0) / 1e9, 3),
            "gen_s": round((result.get("eval_duration") or 0) / 1e9, 3),
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass  # 埋点绝不能影响实验主流程


def health_check():
    """轻量自检：服务是否可达、模型是否注册。返回 True/False。"""
    import urllib.error
    try:
        req = urllib.request.Request(
            f"{CONFIG['base_url']}/api/tags",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            names = [m["name"].split(":")[0] for m in json.loads(resp.read()).get("models", [])]
        return CONFIG["model"].split(":")[0] in names
    except Exception as e:
        print(f"    [health_check] 不可达: {e}")
        return False
