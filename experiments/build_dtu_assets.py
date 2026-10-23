# -*- coding: utf-8 -*-
"""
DTU 资产构建/校验（来源层级表 + 禁药安全注册表）。

用法：
  python build_dtu_assets.py             扫描 KG 域名，报告分档覆盖（不写文件）
  python build_dtu_assets.py --template  把未分档域名追加为待填模板（stdout）
  python build_dtu_assets.py --check     硬校验，失败退出码 1：
        1) KG 中每个来源域名都已分档；
        2) 每条三元组至少 1 个可解析域名；
        3) 安全注册表对干净 KG 的化学防治 tail 零命中（防别名误伤合法药剂）；
        4) 注册表自身无重复别名。
另打印审计用分布（R 分桶、单源比例、化学槽来源层级构成）。
"""
import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from urllib.parse import urlparse

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))
KG = os.path.join(ROOT, "release_kg", "kg_triples.json")
DTU_DIR = os.path.join(ROOT, "release_kg", "dtu")
TIERS = os.path.join(DTU_DIR, "source_tiers.json")
BANNED = os.path.join(DTU_DIR, "banned_agrochemicals.json")

# 与 dtu_trust.py 保持一致的剂型后缀 / 混配拆分（校验侧独立副本，防循环导入）
FORM_SUFFIX = ("可湿性粉剂", "悬浮剂", "乳油", "水分散粒剂", "水乳剂", "微乳剂",
               "可溶粉剂", "可溶液剂", "水剂", "烟剂", "粉剂", "颗粒剂", "胶悬剂",
               "浓乳剂", "悬浮乳剂", "种衣剂", "拌种剂", "可分散油悬浮剂")
SPLIT_RE = re.compile(r"[·・、+]")


def split_components(tail):
    t = tail.strip()
    for f in sorted(FORM_SUFFIX, key=len, reverse=True):
        if t.endswith(f):
            t = t[: -len(f)]
            break
    return [c.strip() for c in SPLIT_RE.split(t) if c.strip()]


def domain_of(url):
    return urlparse(url).netloc.lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--template", action="store_true")
    args = ap.parse_args()

    triples = json.load(open(KG, encoding="utf-8"))
    tier_reg = json.load(open(TIERS, encoding="utf-8"))
    ban_reg = json.load(open(BANNED, encoding="utf-8"))
    tiers = {d: v["tier"] for d, v in tier_reg["domains"].items()}

    used = Counter()
    unresolved = Counter()
    for t in triples:
        ds = set()
        for u in t.get("sources", []):
            d = domain_of(u)
            if d:
                ds.add(d)
            else:
                unresolved[u] += 1
        for d in ds:
            used[d] += 1

    missing = sorted(set(used) - set(tiers))
    if args.template:
        tpl = {d: {"tier": None, "org": "待填"} for d in missing}
        print(json.dumps(tpl, ensure_ascii=False, indent=2))
        return

    print("KG triples=%d  使用域名=%d  已分档=%d" %
          (len(triples), len(used), len(set(used) & set(tiers))))
    if missing:
        print("[缺失分档] %s" % missing)

    # 注册表自校验：别名唯一
    alias_map = defaultdict(list)
    for c in ban_reg["chemicals"]:
        for a in c["aliases"]:
            alias_map[a].append(c["name"])
    dup = {a: ns for a, ns in alias_map.items() if len(ns) > 1}

    # 零命中断言：注册表成分 vs 干净 KG 化学防治 tail 拆分成分
    banned_aliases = set(alias_map)
    chem_tails = sorted({t["tail"] for t in triples
                         if t["relation"] == "化学防治"})
    hits = []
    for tail in chem_tails:
        for comp in split_components(tail):
            if comp in banned_aliases:
                hits.append((tail, comp))

    # 审计分布
    tscore = {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.4}

    def R(doms):
        top = max(tiers[d] for d in doms if d in tiers)
        d = len(set(doms))
        rc = 1 - math.exp(-(d - 1) / 1.5)
        return 0.75 * tscore[top] + 0.25 * (0.5 * tscore[top] + 0.5 * rc)

    r_hist = Counter()
    n_single = 0
    chem_slot_tier = defaultdict(set)
    for t in triples:
        doms = {domain_of(u) for u in t.get("sources", []) if domain_of(u)}
        if not tiers.keys() & doms:
            continue
        r_hist[round(R(doms), 3)] += 1
        if len(doms) == 1:
            n_single += 1
        if t["relation"] == "化学防治":
            chem_slot_tier[t["head"]].add(
                max(tiers[d] for d in doms if d in tiers))

    print("R 分布: %s" % dict(sorted(r_hist.items())))
    print("单域名三元组: %d/%d (%.1f%%)" %
          (n_single, len(triples), 100 * n_single / len(triples)))
    weak = [p for p, ts in chem_slot_tier.items() if max(ts) >= 3]
    print("化学槽最高来源仅 tier3/4 的病虫 %d/%d: %s" %
          (len(weak), len(chem_slot_tier), "、".join(sorted(weak))))
    print("注册表别名数=%d 化学防治不同 tail=%d 干净KG命中=%s" %
          (len(banned_aliases), len(chem_tails), hits))

    ok = True
    if args.check:
        if missing:
            print("[FAIL] 有未分档域名"); ok = False
        if sum(unresolved.values()):
            print("[FAIL] 存在不可解析 URL: %s" % dict(unresolved)); ok = False
        if dup:
            print("[FAIL] 注册表别名冲突: %s" % dup); ok = False
        if hits:
            print("[FAIL] 安全注册表误伤干净 KG 合法化学药剂: %s" % hits); ok = False
        if ok:
            print("[PASS] DTU 资产校验全部通过")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
