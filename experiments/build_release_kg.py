# -*- coding: utf-8 -*-
"""
构建 TeaPest-KG 发布版（release）。

唯一金核：authoritative_pest_facts.json（联网核实、带来源 URL）。
保证：
  1. 自洽 —— 每条三元组的 head/tail 都出现在实体表中；
  2. 可溯源 —— 每条三元组继承该病虫事实的来源 URL；
  3. 干净 —— 不含旧 KG 的「相关茶类/共现拼接/边界扩展」噪声。
输出到 release_kg/。
"""
import json
import os
import sys
import csv
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))
FACTS = os.path.join(ROOT, "authoritative_pest_facts.json")
OUT = os.path.join(ROOT, "release_kg")
os.makedirs(OUT, exist_ok=True)

REL_MAP = [
    ("symptoms", "表现症状", "PEST-SYMPTOM"),
    ("agricultural", "农业防治", "CONTROL-AGRICULTURAL"),
    ("physical", "物理防治", "CONTROL-PHYSICAL"),
    ("biological", "生物防治", "CONTROL-BIOLOGICAL"),
    ("chemical", "化学防治", "CONTROL-CHEMICAL"),
]


def main():
    F = json.load(open(FACTS, encoding="utf-8"))
    src_url = F["sources"]
    records = F["pests"] + F["diseases"]

    ent_type = {}          # entity -> type
    type_conflicts = []    # 同一实体被赋予不同类型
    triples = {}           # (h,r,t) -> triple dict (dedup)

    def register(name, etype):
        if not name:
            return
        if name in ent_type and ent_type[name] != etype:
            type_conflicts.append((name, ent_type[name], etype))
        ent_type.setdefault(name, etype)

    for rec in records:
        pest = rec["name"]
        register(pest, "PEST-ORGANISM")
        urls = [src_url[s] for s in rec.get("src", []) if s in src_url]
        for field, rel, tail_type in REL_MAP:
            for item in rec.get(field, []):
                register(item, tail_type)
                key = (pest, rel, item)
                if key not in triples:
                    triples[key] = {
                        "head": pest, "relation": rel, "tail": item,
                        "head_type": "PEST-ORGANISM", "tail_type": tail_type,
                        "sources": urls,
                    }

    # 自洽性校验
    nodes = set(ent_type)
    bad = [(h, r, t) for (h, r, t) in triples if h not in nodes or t not in nodes]
    assert not bad, f"不自洽三元组: {bad[:5]}"

    # 落盘实体表
    with open(os.path.join(OUT, "entities.csv"), "w", encoding="utf-8",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["entity", "type"])
        for name in sorted(ent_type):
            w.writerow([name, ent_type[name]])

    # 落盘三元组
    tri_list = sorted(triples.values(),
                      key=lambda x: (x["head"], x["relation"], x["tail"]))
    json.dump(tri_list, open(os.path.join(OUT, "kg_triples.json"), "w",
                             encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # ---- 统计 ----
    type_count = defaultdict(int)
    for t in ent_type.values():
        type_count[t] += 1
    rel_count = defaultdict(int)
    for t in tri_list:
        rel_count[t["relation"]] += 1
    with_src = sum(1 for t in tri_list if t["sources"])
    n_pest = type_count["PEST-ORGANISM"]

    print("=" * 60)
    print("TeaPest-KG release 统计")
    print("=" * 60)
    print(f"实体总数: {len(ent_type)}  (病虫 {n_pest})")
    for t, c in sorted(type_count.items(), key=lambda x: -x[1]):
        print(f"   {t:22} {c}")
    print(f"三元组总数: {len(tri_list)}")
    for r, c in sorted(rel_count.items(), key=lambda x: -x[1]):
        print(f"   {r:8} {c}")
    print(f"带来源三元组: {with_src}/{len(tri_list)} "
          f"({100*with_src/max(1,len(tri_list)):.0f}%)")
    if type_conflicts:
        print("⚠ 实体类型冲突:")
        for c in type_conflicts:
            print("   ", c)
    else:
        print("✓ 无实体类型冲突；✓ 全部三元组自洽")


if __name__ == "__main__":
    main()
