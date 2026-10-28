# -*- coding: utf-8 -*-
"""After the second annotator returns second_annotator_sheet.csv, compute
raw agreement and Cohen's kappa against the first coder on overlapping items.

First-coder compressed codes:
  "MAP->X"      -> force-map
  "RIGHT:name"  -> coincidental
  "GENERIC ..." -> generic
The second coder writes force-map / coincidental / generic.
"""
import csv, io, json, os, sys
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = ["force-map", "coincidental", "generic"]


def norm_first(v):
    v = str(v)
    if v.startswith("MAP->"):
        return "force-map"
    if v.startswith("RIGHT:"):
        return "coincidental"
    if v.upper().startswith("GENERIC"):
        return "generic"
    return None


def cohen_kappa(a, b):
    n = len(a)
    ca, cb = Counter(a), Counter(b)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((ca[k] / n) * (cb[k] / n) for k in LABELS)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0, po


def main():
    first = json.load(open(os.path.join(HERE, "first_coder_labels.json"), encoding="utf-8"))
    path = os.path.join(HERE, "second_annotator_sheet.csv")
    second = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            lab = (row.get("label_force_map_coincidental_generic") or "").strip().lower()
            if lab:
                second[row["item_id"]] = lab
    common = sorted(set(first) & set(second))
    if not common:
        print("No second-annotator labels found yet. Fill the CSV and rerun.")
        return
    a = [norm_first(first[i]) for i in common]
    b = [second[i] for i in common]
    valid = [(x, y) for x, y in zip(a, b) if x in LABELS and y in LABELS]
    va, vb = zip(*valid) if valid else ([], [])
    kappa, po = cohen_kappa(list(va), list(vb))
    print("overlapping items coded by both:", len(valid))
    print("raw agreement: %.3f" % po)
    print("Cohen kappa: %.3f" % kappa)
    # disagreement list for adjudication
    dis = [(i, x, y) for i, x, y in zip(common, a, b) if x in LABELS and y in LABELS and x != y]
    print("disagreements:", len(dis))
    for i, x, y in dis[:30]:
        print("  ", i, "first=", x, "second=", y)
    json.dump({"n": len(valid), "raw_agreement": po, "kappa": kappa,
               "disagreements": dis},
              open(os.path.join(HERE, "agreement_results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
