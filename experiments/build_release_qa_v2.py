# -*- coding: utf-8 -*-
"""
构建 TeaPest-QA v2（release_qa/qa_v2.json）——三层均衡、可复现、诚实口径。

在 v1(106) 基础上修正：
  - 修复别名归一化造成的烂题干（"茶茶小绿茶小绿叶蝉…"）与 pest_symptoms 重复题；
    题干一律从 release KG 干净生成，不再对旧题做脆弱替换。
  - 三层结构（共 158）：
      A 事实查询 easy=70  ：28 综合防治 + 28 症状 + 14 分类别防治（点名病虫，1跳）
      B 诊断(不点名)=58    ：28 特征辨识(gold=病虫) + 20 症状→怎么防治
                            (gold=病虫+防治，含原14自然题+新增6种) + 10 鉴别多答案
      C OOD/拒答 hard=30  ：16 未收录真实茶树病虫 + 7 跨域/离题 + 7 矛盾无解
                            expected=abstain（answer_list 留空，评分走拒答逻辑，
                            不再用 v1 "空 gold 记满分" 的假象）。
诊断题制造组内方差：28 辨识题为高辨识特征（可解），20 防治题为开放式（一跳够不到
防治），10 鉴别题用多种病虫共享的泛症状（真有歧义）。
gold 全部来自 release KG（在词表、可溯源）；OOD 病虫名经核验确不在词表。
"""
import csv
import json
import os
import sys
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))
TRIPLES = os.path.join(ROOT, "release_kg", "kg_triples.json")
ENT = os.path.join(ROOT, "release_kg", "entities.csv")
V1_QA = os.path.join(ROOT, "release_qa", "qa.json")
OUT_JSON = os.path.join(ROOT, "release_qa", "qa_v2.json")
OUT_TXT = os.path.join(ROOT, "release_qa", "qa_v2_manifest.txt")

RELS = ["生物防治", "化学防治", "农业防治", "物理防治"]

# 14 道 v1 自然语言开放诊断题（保留其原 id 以便复用已生成答案）
V1_OPEN_KEEP = ["QA0186", "QA0187", "QA0188", "QA0189", "QA0190", "QA0191",
                "QA0192", "QA0193", "QA0194", "QA0195", "QA0196", "QA0197",
                "QA0198", "QA0199"]

# 新增 6 道 hard 防治诊断题（覆盖原14种之外；题干不得出现病虫规范名/别名）
NEW_HARD = [
    ("茶毛虫", "茶园里幼虫群集在叶片上啃食，虫体长有成丛毒毛、皮肤接触会刺痛，"
               "严重时整株叶片几乎被啃光，这是什么害虫？该如何防治？"),
    ("茶树介壳虫", "茶树枝干和叶片上附着介壳状小虫刺吸汁液，受害枝干枯死、"
                   "树势衰弱并落叶，还伴生一层黑色煤烟状物，这是什么害虫？怎么防治？"),
    ("咖啡小爪螨", "茶树成叶正面呈红褐色、失去光泽，叶面可见刺吸造成的细密网纹，"
                   "并布满白色蜕皮与尘状物，后期叶片干枯脱落，这是什么螨类？如何防治？"),
    ("茶叶丽瘿螨", "茶树叶片呈紫铜色、无光泽，叶质变脆并向上卷曲开裂，"
                   "叶面有白色蜡质纵条，芽梢萎缩硬化，这是什么螨害？怎样防治？"),
    ("茶跗线螨", "茶树嫩芽叶变褐、生长停滞，叶质硬化增厚，芽叶萎缩多皱，"
                 "这是什么螨类为害？如何防控？"),
    ("茶黄蓟马", "茶树嫩叶被锉吸后卷缩焦枯，叶背出现红褐色纵痕，芽叶卷曲，"
                 "这是什么害虫？应如何防治？"),
]

# 28 道 B1 特征辨识题：手工命制的自然症状描述（不点名病虫，gold=[病虫]）。
# 刻意采用各病虫高辨识特征，使该层「可解」，与 B2/B3 的开放/歧义形成难度梯度。
IDENTIFY = {
    "卵形短须螨": "茶树叶片背面出现紫褐色突起斑，主脉变紫褐色，后期叶柄变黑霉烂并引起落叶，这是什么害螨？",
    "咖啡小爪螨": "茶树成叶正面红褐色、失去光泽，叶面有刺吸造成的细网纹，并附着白色蜕皮尘状物，最后叶片干枯脱落，是什么害螨？",
    "茶丽纹象甲": "茶园成虫咬食嫩叶成弧形缺刻，幼虫生活在土中取食茶苗须根，这是哪种害虫？",
    "茶二叉蚜": "茶树嫩梢叶背有成群虫体聚集刺吸，致叶片皱缩卷曲、新梢抽不出，并分泌蜜露诱发黑色霉层，是什么害虫？",
    "茶云纹叶枯病": "茶树叶片出现不规则褐色大斑、呈云纹状，潮湿时病斑上产生黑色霉层，是什么病害？",
    "茶刺蛾": "茶树幼虫在叶背取食叶肉形成缺刻，虫体具毒刺、人体接触后螫痛发红，是什么害虫？",
    "茶叶丽瘿螨": "茶树叶片呈紫铜色、无光泽，质地变脆并向上卷曲开裂，叶面有白色蜡质纵条，芽梢萎缩硬化，是什么害螨？",
    "茶小卷叶蛾": "茶树幼虫吐丝缀叶成苞、藏在苞内取食成孔洞，受害叶出现透明枯斑，是什么害虫？",
    "茶小绿叶蝉": "茶树芽叶萎缩、嫩叶卷曲、叶缘焦枯，生长点受害明显，是什么害虫？",
    "茶尺蠖": "茶树幼虫啃食叶片形成缺刻或孔洞，受惊时吐丝下垂，严重时整株叶片被吃光，是什么害虫？",
    "茶树介壳虫": "茶树枝干和叶片被介壳状虫体刺吸汁液，树势衰弱、枝条枯死并落叶，是什么害虫？",
    "茶根结线虫": "挖出茶树根部可见串珠状根瘤、须根少，地上部长势弱、叶片黄化，是什么病害？",
    "茶橙瘿螨": "茶树嫩叶和芽上出现橙黄色小点，叶片萎缩变形、叶质变脆、芽叶生长受阻，是什么害螨？",
    "茶毛虫": "茶树幼虫群集啃食叶片，虫体长有成丛毒毛，严重时叶片被吃光，是什么害虫？",
    "茶炭疽病": "茶树成叶出现褐色圆形病斑，边缘有黄色晕圈，后期病斑上产生黑色小点，是什么病害？",
    "茶煤烟病": "茶树叶片表面覆盖一层黑色烟煤状霉层和黑色霉状物，是什么病害？",
    "茶网蝽": "茶树成叶叶面密布密集白色小斑，叶背有黑色黏液排泄物并见刺吸小虫，最终叶片灰白脱落，是什么害虫？",
    "茶膏药病": "茶树枝干上长出膏药状菌膜和灰白色菌丝层，可致枝干枯死，是什么病害？",
    "茶苗根腐病": "茶苗根部变褐腐烂，茎基部出现水渍状病斑，最终植株萎蔫枯死，是什么病害？",
    "茶苗白绢病": "茶苗根部可见白色绢丝状菌丝和菜籽状菌核，是什么病害？",
    "茶藻斑病": "茶树叶片正面出现黄斑和放射状铁锈色斑，叶背有锈色粉状物，用手触碰会沾染锈色粉末，是什么病害？",
    "茶角胸叶甲": "茶树成虫啃食叶肉残留叶脉，叶片布满孔洞，后期仅剩叶脉，是什么害虫？",
    "茶赤叶斑病": "茶树叶片先出现水渍状暗绿色斑点，后变褐色枯焦，远看似火烧状，是什么病害？",
    "茶跗线螨": "茶树嫩芽叶变褐、萎缩多皱，叶质硬化增厚，芽叶生长停滞，是什么害螨？",
    "茶轮斑病": "茶树叶片出现灰白色圆形小斑，扩大后呈同心轮纹状病斑，病斑可穿孔并散生黑色小点，是什么病害？",
    "茶饼病": "茶树嫩叶背面出现白粉和馒头状疱斑，叶片变形，是什么病害？",
    "茶黄蓟马": "茶树嫩叶遭锉食后卷缩焦枯，叶背留下红褐色纵痕，芽叶卷曲，是什么害虫？",
    "黑刺粉虱": "茶树叶片正面出现黄斑，叶背附着黑色刺状若虫，被害叶枯萎脱落，并分泌蜜露诱发黑色霉层，是什么害虫？",
}

# 30 道 OOD：正确行为=拒答/低置信（答案均不在 release KG）
OOD_UNKNOWN_PEST = [
    ("茶白星病", "茶树嫩叶上散生许多圆形白色小斑，中央灰白、边缘暗褐，嫩梢也带病斑，这是什么病害？如何防治？"),
    ("茶芽枯病", "春茶期芽梢尖端变褐枯死，幼叶出现黄褐色不规则病斑，嫩芽容易折断，这是什么病？怎么防治？"),
    ("茶圆赤星病", "茶树成叶上出现圆形赤褐色小斑，中央灰白并散生黑色小粒点，潮湿时病斑生灰色霉层，是什么病？"),
    ("茶黑腐病", "茶树树冠上部叶片成片变褐腐烂，像被开水烫过，雨后病叶覆盖一层黑色霉层，这是什么病害？"),
    ("茶紫纹羽病", "茶苗根部缠绕着紫色根状菌索，根皮变黑腐烂，地上部分黄化枯死，这是什么根部病害？"),
    ("茶菟丝子", "茶树上缠绕黄色丝状藤蔓，吸盘刺入枝条吸取养分，受害树势衰弱，这是什么寄生性有害生物？"),
    ("茶蓑蛾", "茶树上挂着许多纺锤形护囊，幼虫藏在囊内伸出头啃食叶片，护囊外粘缀碎叶小枝，是什么害虫？"),
    ("大蓑蛾", "茶园枝条上吊挂大型袋囊，幼虫负囊取食叶片造成孔洞缺刻，这是什么害虫？"),
    ("扁刺蛾", "茶树叶片被扁平椭圆形、背侧具刺突的幼虫啃食成缺刻，人体接触刺突会痛痒，这是什么害虫？"),
    ("黄刺蛾", "幼虫黄绿色、背部有紫色哑铃形大斑，结灰白间褐色纵纹的硬茧，啃食茶树叶片，是什么害虫？"),
    ("绿刺蛾", "成虫通体翠绿色，老熟幼虫入土结茧，幼虫取食茶树叶片形成缺刻，这是什么害虫？"),
    ("茶长白蚧", "茶树枝干上密布长椭圆形白色介壳，虫体藏于壳下刺吸汁液，受害枝条枯萎，这是什么害虫？"),
    ("红蜡蚧", "茶树枝叶上着生暗红色半球形蜡壳，虫体固着刺吸并诱发煤烟，这是什么介壳虫？"),
    ("龟蜡蚧", "茶树枝条覆盖灰白色龟甲状蜡被，虫体固定刺吸汁液并诱发煤污，这是什么蚧虫？"),
    ("茶梢蛾", "茶树嫩梢被蛀食，受害梢节间肿大、芽叶枯萎，剖开虫道可见小型幼虫，这是什么害虫？"),
    ("铜绿金龟子", "夏季夜间茶树上有具铜绿色金属光泽的甲虫群集啃食叶片，土中还有蛴螬取食根部，这是什么害虫？"),
]
OOD_CROSSDOMAIN = [
    "小麦条锈病在拔节期用什么杀菌剂防治效果最好？",
    "番茄早疫病和晚疫病怎么区分，分别用什么药？",
    "大棚黄瓜叶片背面长出紫黑色霉层是什么病，如何防治？",
    "水稻稻瘟病的穗颈瘟应如何用药预防？",
    "龙井茶的手工炒制一般包含哪些主要工序？",
    "茶园土壤pH偏低，想撒施生石灰调节，具体用量该怎么确定？",
    "茶园直接喷施草甘膦除草对茶树安全吗，需要注意什么？",
]
OOD_CONTRADICTORY = [
    "茶树根部有串珠状根瘤，同时叶背布满白色粉状物和馒头状疱斑，这最可能是同一种什么病？",
    "茶树叶片白天正常，一到夜里就发出蓝色荧光，可能是什么害虫造成的？",
    "我家茶树既不抽新梢也不落叶，叶片保持鲜绿已经四年毫无变化，这是什么病虫？",
    "茶叶上出现规则的正方形黑色病斑，每个病斑边长恰好一厘米，是什么病害？",
    "茶园里发现一种长着六条腿的哺乳动物在啃食主根，这是什么害虫？",
    "茶树被一种既非昆虫也非病菌、肉眼看不见却能咬穿钢板的生物为害，该用什么药？",
    "茶树完全不防治、也从不施肥浇水，却仍连年高产，最可能是哪种病虫害的功劳？",
]


# 10 道 B3 鉴别多答案题（共享泛症状，gold=KG 内所有可造成该表现的病虫）。
# 前 2 题沿用初版题干（已有答案缓存，文本不得改动）；后 8 题的每个候选均经
# release KG 症状集核对——措辞虽不同，语义表现同属一簇（如叶斑+黑色小点、
# 煤烟霉层、根部受害萎蔫、毒毛毒刺食叶等），专家面诊时本就需要列鉴别候选。
DIFFERENTIAL = [
    ("茶树嫩梢和枝干上有刺吸式小虫，体表分泌蜜露，叶片诱发黑色煤烟层；"
     "哪些刺吸式害虫都可能造成这一现象？",
     ["茶二叉蚜", "茶树介壳虫", "黑刺粉虱"]),
    ("茶园发现幼虫把茶树叶片啃食到近乎光秃，可能与哪些食叶性害虫有关？",
     ["茶尺蠖", "茶毛虫"]),
    ("茶树嫩梢和嫩叶皱缩卷曲、芽叶萎缩、新梢抽生受阻，哪些刺吸为害嫩梢的"
     "害虫都可能出现这类表现？",
     ["茶二叉蚜", "茶小绿叶蝉", "茶黄蓟马"]),
    ("茶树成叶失去光泽，叶面呈红褐色至紫铜色变色，主脉附近变深，"
     "后期叶片干枯脱落，哪些害螨可能造成这类表现？",
     ["咖啡小爪螨", "茶叶丽瘿螨", "卵形短须螨"]),
    ("茶园里食叶幼虫啃食叶片形成缺刻，虫体具成丛毒毛或毒刺，"
     "人体接触后皮肤痛痒红肿，哪些害虫可能为害？",
     ["茶毛虫", "茶刺蛾"]),
    ("茶树叶片和枝条表面覆盖一层黑色烟煤状霉层，这一表现可能涉及哪些"
     "病害或害虫？",
     ["茶煤烟病", "茶二叉蚜", "黑刺粉虱", "茶树介壳虫"]),
    ("茶树（尤其茶苗）地上部叶片黄化、树势衰弱甚至萎蔫枯死，"
     "根部检查可见异常，哪些根部病虫害可能造成这类表现？",
     ["茶根结线虫", "茶苗根腐病", "茶苗白绢病"]),
    ("茶树成叶出现褐色病斑，后期病斑上散生或呈轮纹状排列黑色小点，"
     "潮湿时还长出霉层，哪些叶部病害可能有这类表现？",
     ["茶云纹叶枯病", "茶炭疽病", "茶轮斑病"]),
    ("茶树叶片正面出现密集的失绿小白斑或黄斑，叶背有刺吸式小虫及其"
     "黑色排泄物，最终叶片灰白干枯脱落，哪些害虫可能为害？",
     ["茶网蝽", "黑刺粉虱"]),
    ("茶园里甲虫类成虫取食茶树嫩叶，把叶片咬成弧形缺刻、孔洞，"
     "严重时仅残留叶脉，哪些害虫可能为害？",
     ["茶丽纹象甲", "茶角胸叶甲"]),
]


def load():
    triples = json.load(open(TRIPLES, encoding="utf-8"))
    vocab = {r[0] for r in csv.reader(open(ENT, encoding="utf-8-sig"))
             if len(r) >= 2 and r[0] != "entity"}
    byp = defaultdict(lambda: defaultdict(list))
    for t in triples:
        byp[t["head"]][t["relation"]].append(t["tail"])
    for p in byp:
        for r in byp[p]:
            byp[p][r] = list(dict.fromkeys(byp[p][r]))
    v1 = {q["id"]: q for q in json.load(open(V1_QA, encoding="utf-8"))}
    return byp, vocab, v1


def controls(d):
    out = []
    for r in RELS:
        out += d[r]
    return list(dict.fromkeys(out))


def build_identify(pest, d, i):
    """B1: 特征辨识（不点名病虫），gold=[病虫]。题干取自手工命制的自然描述。"""
    return IDENTIFY[pest]


def main():
    byp, vocab, v1 = load()
    pests = sorted(byp)
    assert len(pests) == 28, len(pests)
    rows = []
    nid = [0]

    def add(qid, stratum, qtype, diff, hops, question, gold, ctx, source,
            expected="answer"):
        nid[0] += 1
        rows.append({
            "id": qid, "stratum": stratum, "type": qtype,
            "difficulty": diff, "hops": hops, "question": question,
            "context_entity": ctx, "answer_list": list(dict.fromkeys(gold)),
            "gold_source": source, "expected": expected,
        })

    # ---------- A 事实查询（70） ----------
    # A1 综合防治 28
    for i, p in enumerate(pests):
        add("L-AC-%02d" % (i + 1), "lookup", "pest_all_controls", "easy", 1,
            "%s有哪些综合防治方法？" % p, controls(byp[p]), p, "release_kg")
    # A2 症状 28（干净题干，替换 v1 烂题/重复题）
    for i, p in enumerate(pests):
        add("L-SY-%02d" % (i + 1), "lookup", "pest_symptoms", "easy", 1,
            "%s为害茶树时有哪些典型症状？" % p, byp[p]["表现症状"], p,
            "release_kg")
    # A3 分类别防治 14（四类均衡、gold 非空，轮询）
    want = {"生物防治": 4, "化学防治": 4, "农业防治": 3, "物理防治": 3}
    made = defaultdict(int)
    for r in RELS:
        for p in pests:
            if made[r] >= want[r]:
                break
            if byp[p][r]:
                add("L-CT-%s-%02d" % (r[0], made[r] + 1), "lookup",
                    "pest_control", "easy", 1,
                    "%s可以采取哪些%s措施？" % (p, r), byp[p][r], p,
                    "release_kg")
                made[r] += 1

    # ---------- B1 特征辨识 28（medium） ----------
    for i, p in enumerate(pests):
        q = build_identify(p, byp[p], i)
        add("D-ID-%02d" % (i + 1), "diagnosis", "symptom_identify",
            "medium", 1, q, [p], "", "release_kg")

    # ---------- B2 症状→防治 20（hard） ----------
    # 14 道 v1 自然题（保留 id/文本）
    for i, qid in enumerate(V1_OPEN_KEEP):
        q = v1[qid]
        add(qid, "diagnosis", "open_diagnosis", "hard", 2,
            q["question"], q["answer_list"], "", q.get("gold_source",
            "release_kg+authoritative"))
    # 6 道新增
    for j, (p, q) in enumerate(NEW_HARD):
        add("D-CT-%02d" % (j + 1), "diagnosis", "open_diagnosis", "hard", 2,
            q, [p] + controls(byp[p]), "", "release_kg")

    # ---------- B3 鉴别多答案 10（hard，共享泛症状，真有歧义） ----------
    # 题干只给多病虫共享的泛表现，gold=KG内所有可造成该表现的病虫；
    # 考察的是「列出鉴别候选」而非唯一确诊。每个簇均由 release KG 症状集核对。
    for k, (q, gold) in enumerate(DIFFERENTIAL, start=1):
        add("D-DF-%02d" % k, "diagnosis", "differential", "hard", 1,
            q, gold, "", "release_kg")

    # ---------- C OOD 拒答（30） ----------
    for i, (name, q) in enumerate(OOD_UNKNOWN_PEST):
        add("X-PST-%02d" % (i + 1), "ood", "ood_unknown_pest", "hard", 0,
            q, [], "", "ood_expect_abstain", expected="abstain")
    for i, q in enumerate(OOD_CROSSDOMAIN):
        add("X-DOM-%02d" % (i + 1), "ood", "ood_crossdomain", "hard", 0,
            q, [], "", "ood_expect_abstain", expected="abstain")
    for i, q in enumerate(OOD_CONTRADICTORY):
        add("X-CON-%02d" % (i + 1), "ood", "ood_contradictory", "hard", 0,
            q, [], "", "ood_expect_abstain", expected="abstain")

    # ================= 自校验 =================
    err = []
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        err.append("重复 id")
    # answer 题 gold 非空且在词表
    for r in rows:
        if r["expected"] == "answer":
            if not r["answer_list"]:
                err.append("%s gold 空" % r["id"])
            for g in r["answer_list"]:
                if g not in vocab:
                    err.append("%s gold 不在词表: %s" % (r["id"], g))
        else:
            if r["answer_list"]:
                err.append("%s OOD 题不应有 gold" % r["id"])
    # 诊断题不得自泄病虫名（题干含任一病虫规范名即泄露；允许 D-DF 因 gold 为多答案）
    for r in rows:
        if r["stratum"] == "diagnosis" and r["type"] != "differential":
            leak = [p for p in pests if p in r["question"]]
            if leak:
                err.append("%s 题干泄露病虫名: %s" % (r["id"], leak))
    # OOD 病虫名确不在词表（子串双向检查）
    for name, _ in OOD_UNKNOWN_PEST:
        if any(name in v or v in name for v in vocab if len(v) >= 3):
            err.append("OOD 病虫名与词表冲突: %s" % name)

    # 分层统计
    stat = defaultdict(lambda: defaultdict(int))
    for r in rows:
        stat[r["stratum"]][r["type"]] += 1
    diff = defaultdict(int)
    for r in rows:
        diff[r["difficulty"]] += 1

    json.dump(rows, open(OUT_JSON, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # 人读清单
    L = ["TeaPest-QA v2  共 %d 题" % len(rows), "=" * 78]
    for strat, title in (("lookup", "A 事实查询(easy,点名病虫)"),
                         ("diagnosis", "B 诊断(不点名病虫)"),
                         ("ood", "C OOD/拒答(expected=abstain)")):
        L.append("\n#### %s  [%d 题]" % (title, sum(stat[strat].values())))
        for r in rows:
            if r["stratum"] != strat:
                continue
            gold = "、".join(r["answer_list"]) if r["answer_list"] else "<拒答>"
            L.append("[%s|%s|%s] %s" % (r["id"], r["type"], r["difficulty"],
                                        r["question"]))
            L.append("     gold: %s" % gold)
    L.append("\n" + "=" * 78)
    L.append("分层: " + ", ".join("%s=%d" % (k, sum(stat[k].values()))
                                  for k in ("lookup", "diagnosis", "ood")))
    L.append("难度: " + dict.__repr__(dict(diff)))
    open(OUT_TXT, "w", encoding="utf-8").write("\n".join(L))

    print("n=%d" % len(rows))
    for s in ("lookup", "diagnosis", "ood"):
        print("  %s: %d  %s" % (s, sum(stat[s].values()),
                                dict(stat[s])))
    print("difficulty:", dict(diff))
    if err:
        print("\n⚠ 校验未通过：")
        for e in err:
            print("  -", e)
        sys.exit(1)
    print("✓ gold 非空/全在词表；✓ 诊断题无自泄露；✓ OOD 名未收录；✓ id 唯一")
    print("manifest ->", OUT_TXT)


if __name__ == "__main__":
    main()
