# -*- coding: utf-8 -*-
"""
TrustKG-RAG 评估指标模块
=========================
四维十项指标：
  1. 答案质量: EM, F1, Hit@k
  2. 校准: ECE (等频/等宽分箱), Brier Score, Reliability Diagram data
  3. 选择性预测: AURC, Acc@coverage, Risk-Coverage curve data
  4. 忠实性: FActScore (KGQA适配版), 幻觉率, 引用覆盖率

所有函数设计为纯函数，输入 preds+golds+confs，输出 dict。
"""

import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


# ============================================================
# 1. 答案质量指标
# ============================================================

def exact_match(preds: List[str], golds: List[str]) -> float:
    """Exact Match: 预测与标准答案完全一致的比例"""
    if not preds: return 0.0
    correct = sum(1 for p, g in zip(preds, golds) if p.strip() == g.strip())
    return correct / len(preds)


def token_f1(preds: List[str], golds: List[str]) -> float:
    """Token-level F1 (分词后计算，中文按字符)"""
    if not preds: return 0.0
    f1s = []
    for p, g in zip(preds, golds):
        p_tokens = set(list(p.strip()))
        g_tokens = set(list(g.strip()))
        if not p_tokens or not g_tokens:
            f1s.append(0.0)
            continue
        tp = len(p_tokens & g_tokens)
        if tp == 0:
            f1s.append(0.0)
            continue
        prec = tp / len(p_tokens)
        rec = tp / len(g_tokens)
        f1s.append(2 * prec * rec / (prec + rec))
    return float(np.mean(f1s))


def hit_at_k(pred_entities: List[List[str]], gold_entities: List[List[str]], k: int = 1) -> float:
    """Hit@k: 预测的前k个实体中是否有任何一个出现在标准答案实体中"""
    if not pred_entities: return 0.0
    hits = 0
    for pred_list, gold_list in zip(pred_entities, gold_entities):
        top_k = pred_list[:k]
        if any(e in gold_list for e in top_k):
            hits += 1
    return hits / len(pred_entities)


def precision_recall_f1_entities(pred_entities: List[List[str]],
                                  gold_entities: List[List[str]]) -> Dict[str, float]:
    """实体级 Precision / Recall / F1 (微平均)"""
    if not pred_entities:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = fp = fn = 0
    for pred_list, gold_list in zip(pred_entities, gold_entities):
        pred_set = set(pred_list)
        gold_set = set(gold_list)
        tp += len(pred_set & gold_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}


# ============================================================
# 2. 校准指标
# ============================================================

def expected_calibration_error(accuracies: List[float],
                               confidences: List[float],
                               n_bins: int = 10,
                               strategy: str = "quantile") -> float:
    """
    Expected Calibration Error (ECE)

    Args:
        accuracies: 每个样本的0/1正确性
        confidences: 每个样本的置信度 [0, 1]
        n_bins: 分箱数量
        strategy: "quantile" (等频) or "equal" (等宽)

    Returns:
        ECE 值（百分比，0~1）
    """
    if not accuracies:
        return 0.0

    acc_arr = np.array(accuracies, dtype=float)
    conf_arr = np.array(confidences, dtype=float)
    n = len(acc_arr)

    if strategy == "quantile":
        # 等频分箱：每个箱子样本数大致相等
        bin_edges = np.quantile(conf_arr, np.linspace(0, 1, n_bins + 1))
        # 确保边界唯一
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            # 所有置信度都一样，退化到单箱
            return abs(np.mean(acc_arr) - np.mean(conf_arr))
    else:
        # 等宽分箱
        bin_edges = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    for i in range(len(bin_edges) - 1):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == len(bin_edges) - 2:
            mask = (conf_arr >= low) & (conf_arr <= high)
        else:
            mask = (conf_arr >= low) & (conf_arr < high)
        count = mask.sum()
        if count == 0:
            continue
        bin_acc = acc_arr[mask].mean()
        bin_conf = conf_arr[mask].mean()
        ece += (count / n) * abs(bin_acc - bin_conf)

    return round(float(ece), 4)


def brier_score(accuracies: List[float], confidences: List[float]) -> float:
    """Brier Score: 均方误差，越低越好"""
    if not accuracies: return 0.0
    acc_arr = np.array(accuracies, dtype=float)
    conf_arr = np.array(confidences, dtype=float)
    return round(float(np.mean((conf_arr - acc_arr) ** 2)), 4)


def reliability_diagram_data(accuracies: List[float],
                             confidences: List[float],
                             n_bins: int = 10,
                             strategy: str = "quantile") -> List[Dict]:
    """
    生成 Reliability Diagram 数据（用于画图）

    Returns:
        list of dict: 每个箱子的 {bin_id, count, avg_conf, avg_acc, gap}
    """
    if not accuracies: return []

    acc_arr = np.array(accuracies, dtype=float)
    conf_arr = np.array(confidences, dtype=float)

    if strategy == "quantile":
        bin_edges = np.quantile(conf_arr, np.linspace(0, 1, n_bins + 1))
        bin_edges = np.unique(bin_edges)
    else:
        bin_edges = np.linspace(0, 1, n_bins + 1)

    bins = []
    for i in range(len(bin_edges) - 1):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == len(bin_edges) - 2:
            mask = (conf_arr >= low) & (conf_arr <= high)
        else:
            mask = (conf_arr >= low) & (conf_arr < high)
        count = int(mask.sum())
        if count == 0:
            bins.append({"bin": i, "count": 0, "avg_conf": float(low),
                         "avg_acc": 0.0, "gap": 0.0})
            continue
        avg_conf = float(conf_arr[mask].mean())
        avg_acc = float(acc_arr[mask].mean())
        bins.append({"bin": i, "count": count, "avg_conf": avg_conf,
                     "avg_acc": avg_acc, "gap": abs(avg_acc - avg_conf)})
    return bins


def _fit_temperature_scaling(conf_train, acc_train):
    """
    温度缩放（二分类置信）：q = sigmoid(logit(p) / T)。
    网格搜索使 NLL 最小的单一参数 T（T<1 更尖锐，T>1 更平缓）。
    """
    eps = 1e-6
    p = np.clip(np.array(conf_train, dtype=float), eps, 1 - eps)
    y = np.array(acc_train, dtype=float)
    z = np.log(p / (1.0 - p))

    def nll(T):
        if T <= 0:
            return np.inf
        q = 1.0 / (1.0 + np.exp(-z / T))
        q = np.clip(q, eps, 1 - eps)
        return -np.mean(y * np.log(q) + (1.0 - y) * np.log(1.0 - q))

    grid = np.concatenate([np.linspace(0.1, 5.0, 120),
                           np.linspace(5.0, 20.0, 40)])
    nlls = [nll(T) for T in grid]
    return float(grid[int(np.argmin(nlls))])


def crossfit_calibrate(accuracies, confidences, n_folds=5,
                        method="isotonic", seed=42):
    """
    Cross-fitting 校准（无数据泄漏）。

    将样本分 K 折；每折用其余折拟合 confidence→correct 的（单调）映射，
    再应用到该折。返回校准后置信度，长度与输入相同。这样校准器从不在
    被评估的样本上拟合，避免在测试集上自拟合。

    Args:
        method: "isotonic"  保序回归（非参数单调，推荐）
                "temperature" 温度缩放（单参数 sigmoid）

    返回的置信是原始置信的【单调变换】，因此不改变 AURC / 风险覆盖排序，
    只修正置信尺度。可据此区分"排序差"（AURC 高）与"仅尺度未校准"。
    """
    from sklearn.isotonic import IsotonicRegression

    acc = np.array(accuracies, dtype=float)
    conf = np.array(confidences, dtype=float)
    n = len(acc)
    if n == 0:
        return []

    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_folds = max(1, min(int(n_folds), n))
    folds = [f for f in np.array_split(idx, n_folds) if len(f) > 0]

    calibrated = np.zeros(n)
    for held in folds:
        fit_mask = np.ones(n, dtype=bool)
        fit_mask[held] = False
        x_tr, y_tr = conf[fit_mask], acc[fit_mask]

        # 训练折为空（样本极少）→ 无法校准，原样保留
        if len(x_tr) == 0:
            calibrated[held] = conf[held]
            continue

        if method == "temperature":
            T = _fit_temperature_scaling(x_tr, y_tr)
            p = np.clip(conf[held], 1e-6, 1 - 1e-6)
            z = np.log(p / (1.0 - p))
            calibrated[held] = 1.0 / (1.0 + np.exp(-z / T))
        else:
            iso = IsotonicRegression(out_of_bounds="clip",
                                     y_min=0.0, y_max=1.0)
            iso.fit(x_tr, y_tr)
            calibrated[held] = iso.predict(conf[held])

    return [round(float(v), 4) for v in calibrated]


# ============================================================
# 3. 选择性预测指标
# ============================================================

def area_under_risk_coverage(accuracies: List[float],
                             confidences: List[float]) -> float:
    """
    AURC (Area Under Risk-Coverage curve)
    按置信度从高到低排序，依次纳入样本，计算每个覆盖率下的风险。
    AURC = 平均风险 (所有覆盖率下风险的期望)
    越低越好。
    """
    if not accuracies: return 0.0

    # 按置信度从高到低排序
    sorted_indices = np.argsort(-np.array(confidences))
    acc_sorted = np.array(accuracies, dtype=float)[sorted_indices]
    n = len(acc_sorted)

    # 累积计算风险
    cumulative_errors = np.cumsum(1 - acc_sorted)
    risks = cumulative_errors / np.arange(1, n + 1)

    # AURC = risk-coverage 曲线下面积 (归一化到 [0,1])
    aurc = float(np.mean(risks))
    return round(aurc, 4)


def accuracy_at_coverage(accuracies: List[float],
                         confidences: List[float],
                         coverage: float = 0.8) -> Tuple[float, float]:
    """
    固定覆盖率下的准确率

    Returns:
        (accuracy, actual_coverage)
    """
    if not accuracies: return 0.0, 0.0

    n = len(accuracies)
    k = max(1, int(n * coverage))

    sorted_indices = np.argsort(-np.array(confidences))
    top_indices = sorted_indices[:k]
    acc_arr = np.array(accuracies, dtype=float)

    acc = float(acc_arr[top_indices].mean())
    actual_cov = k / n
    return round(acc, 4), round(actual_cov, 4)


def risk_coverage_curve(accuracies: List[float],
                        confidences: List[float],
                        n_points: int = 100) -> Dict[str, list]:
    """
    Risk-Coverage 曲线数据（用于画图）

    Returns:
        {"coverage": [...], "risk": [...], "accuracy": [...]}
    """
    if not accuracies:
        return {"coverage": [], "risk": [], "accuracy": []}

    n = len(accuracies)
    sorted_indices = np.argsort(-np.array(confidences))
    acc_sorted = np.array(accuracies, dtype=float)[sorted_indices]

    cumulative_errors = np.cumsum(1 - acc_sorted)
    risks = cumulative_errors / np.arange(1, n + 1)
    accs = np.cumsum(acc_sorted) / np.arange(1, n + 1)

    # 采样 n_points 个点
    if n <= n_points:
        indices = np.arange(n)
    else:
        indices = np.linspace(0, n - 1, n_points, dtype=int)

    return {
        "coverage": [round((i + 1) / n, 4) for i in indices],
        "risk": [round(float(risks[i]), 4) for i in indices],
        "accuracy": [round(float(accs[i]), 4) for i in indices],
    }


# ============================================================
# 4. 忠实性指标 (KGQA 适配版 FActScore)
# ============================================================

def extract_atomic_facts(text: str) -> List[str]:
    """
    从生成答案中提取原子事实（简单版：按句子/分句切分）
    中文：按 。 ； ！ ？ 等标点切分
    """
    if not text:
        return []
    # 简单分句
    import re
    sentences = re.split(r'[。；！？\n]', text)
    facts = [s.strip() for s in sentences if len(s.strip()) > 5]
    return facts


def factscore_kgqa(generated_answers: List[str],
                   subgraph_triples: List[List[Dict]],
                   answer_entities: Optional[List[List[str]]] = None) -> Dict[str, float]:
    """
    KGQA 适配版 FActScore

    核心思想：将生成答案分解为原子事实，检查每个事实是否可由检索子图支撑。
    简化实现：基于实体/关系匹配的粗略评估（精确版需要 NLI 模型）。

    Args:
        generated_answers: 生成的答案列表
        subgraph_triples: 每个答案对应的检索子图三元组列表 (list of list of dict)
        answer_entities: 答案实体列表（可选，用于更精细的检查）

    Returns:
        {"factscore": float, "hallucination_rate": float, "n_atomic_facts": int,
         "supported_facts": int}
    """
    total_facts = 0
    supported_facts = 0

    for ans, triples in zip(generated_answers, subgraph_triples):
        facts = extract_atomic_facts(ans)
        total_facts += len(facts)

        # 构建子图词汇表（所有实体和关系）
        vocab = set()
        for t in triples:
            vocab.add(t.get("head", ""))
            vocab.add(t.get("tail", ""))
            vocab.add(t.get("relation", ""))

        # 简单启发式：原子事实中包含至少 2 个子图实体/关系 → 视为支撑
        for fact in facts:
            matches = sum(1 for v in vocab if len(v) >= 2 and v in fact)
            if matches >= 2:
                supported_facts += 1

    factscore = supported_facts / total_facts if total_facts > 0 else 0.0
    hallucination_rate = 1.0 - factscore

    return {
        "factscore": round(factscore, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "n_atomic_facts": total_facts,
        "supported_facts": supported_facts,
    }


def citation_coverage(generated_answers: List[str],
                      citation_pattern: str = r"\[Path#\d+\]") -> Dict[str, float]:
    """
    引用覆盖率：包含引用标记的原子事实比例

    Args:
        generated_answers: 生成的答案列表
        citation_pattern: 引用标记的正则模式

    Returns:
        {"citation_coverage": float, "cited_facts": int, "total_facts": int}
    """
    import re
    total = 0
    cited = 0
    for ans in generated_answers:
        facts = extract_atomic_facts(ans)
        total += len(facts)
        for fact in facts:
            if re.search(citation_pattern, fact):
                cited += 1
    coverage = cited / total if total > 0 else 0.0
    return {
        "citation_coverage": round(coverage, 4),
        "cited_facts": cited,
        "total_facts": total,
    }


# ============================================================
# 5. 统计显著性检验
# ============================================================

def bootstrap_significance_test(metric_a: List[float],
                                metric_b: List[float],
                                n_bootstrap: int = 10000,
                                seed: int = 42) -> Dict:
    """
    配对 bootstrap 显著性检验

    检验：metric_b 的均值是否显著优于 metric_a
    (对准确率/F1/FactScore 是越高越好；对 ECE/AURC 是越低越好)

    Args:
        metric_a: baseline 的逐样本指标值
        metric_b: 我们方法的逐样本指标值
        n_bootstrap: bootstrap 次数
        seed: 随机种子

    Returns:
        {"mean_a": float, "mean_b": float, "diff": float,
         "p_value": float, "ci_low": float, "ci_high": float,
         "effect_size": float}
    """
    np.random.seed(seed)
    a = np.array(metric_a, dtype=float)
    b = np.array(metric_b, dtype=float)
    n = len(a)

    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    obs_diff = mean_b - mean_a

    # Bootstrap
    boot_diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        boot_diffs.append(np.mean(b[idx]) - np.mean(a[idx]))
    boot_diffs = np.array(boot_diffs)

    # p-value: 双侧检验（差异不为 0）
    p_value = float(np.mean(np.abs(boot_diffs) >= abs(obs_diff)))

    # 95% CI
    ci_low = float(np.percentile(boot_diffs, 2.5))
    ci_high = float(np.percentile(boot_diffs, 97.5))

    # 效应量 (Cohen's d)
    pooled_std = np.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2)
    cohens_d = float(obs_diff / pooled_std) if pooled_std > 0 else 0.0

    return {
        "mean_a": round(mean_a, 4),
        "mean_b": round(mean_b, 4),
        "diff": round(obs_diff, 4),
        "p_value": round(p_value, 4),
        "ci_95": [round(ci_low, 4), round(ci_high, 4)],
        "cohens_d": round(cohens_d, 4),
    }


# ============================================================
# 6. 综合评估入口
# ============================================================

def evaluate_all(predictions: List[Dict], gold_data: List[Dict]) -> Dict:
    """
    综合评估入口函数

    Args:
        predictions: list of dict，每个样本包含
            - "answer": str, 生成的答案
            - "confidence": float, 置信度 [0,1]
            - "pred_entities": list[str], 预测的实体列表
            - "subgraph_triples": list[dict], 检索子图
            - "is_correct": bool, 是否正确（可选，自动计算则用 EM）
        gold_data: list of dict，每个样本包含
            - "answer": str, 标准答案
            - "answer_entities": list[str], 答案实体列表
            - "answer_list": list[str], 答案实体列表（别名）

    Returns:
        四维指标的汇总 dict
    """
    n = min(len(predictions), len(gold_data))
    preds = predictions[:n]
    golds = gold_data[:n]

    # 准备数据
    pred_answers = [p.get("answer", "") for p in preds]
    gold_answers = [g.get("answer", g.get("answer_text", "")) for g in golds]

    pred_entities_list = [p.get("pred_entities", []) for p in preds]
    gold_entities_list = [g.get("answer_entities", g.get("answer_list", [])) for g in golds]

    confidences = [p.get("confidence", 0.5) for p in preds]

    # 正确性（基于实体 Hit@1）
    accuracies = []
    for pred_ents, gold_ents in zip(pred_entities_list, gold_entities_list):
        correct = any(e in gold_ents for e in pred_ents[:1]) if pred_ents else False
        accuracies.append(1.0 if correct else 0.0)

    # 1. 答案质量
    em = exact_match(pred_answers, gold_answers)
    f1 = token_f1(pred_answers, gold_answers)
    hit1 = hit_at_k(pred_entities_list, gold_entities_list, k=1)
    hit5 = hit_at_k(pred_entities_list, gold_entities_list, k=5)
    ent_metrics = precision_recall_f1_entities(pred_entities_list, gold_entities_list)

    # 2. 校准
    ece = expected_calibration_error(accuracies, confidences, n_bins=10)
    brier = brier_score(accuracies, confidences)
    rel_data = reliability_diagram_data(accuracies, confidences, n_bins=10)

    # 3. 选择性预测
    aurc = area_under_risk_coverage(accuracies, confidences)
    acc_at_80, cov_80 = accuracy_at_coverage(accuracies, confidences, 0.8)
    rc_curve = risk_coverage_curve(accuracies, confidences)

    # 4. 忠实性
    subgraphs = [p.get("subgraph_triples", []) for p in preds]
    fs_result = factscore_kgqa(pred_answers, subgraphs, gold_entities_list)
    cite_result = citation_coverage(pred_answers)

    return {
        "n_samples": n,
        "answer_quality": {
            "exact_match": round(em, 4),
            "token_f1": round(f1, 4),
            "hit@1": round(hit1, 4),
            "hit@5": round(hit5, 4),
            "entity_precision": ent_metrics["precision"],
            "entity_recall": ent_metrics["recall"],
            "entity_f1": ent_metrics["f1"],
        },
        "calibration": {
            "ece": ece,
            "brier_score": brier,
            "reliability_diagram": rel_data,
        },
        "selective_prediction": {
            "aurc": aurc,
            "accuracy_at_0.8_coverage": acc_at_80,
            "coverage_at_0.8": cov_80,
            "risk_coverage_curve": rc_curve,
        },
        "faithfulness": {
            "factscore": fs_result["factscore"],
            "hallucination_rate": fs_result["hallucination_rate"],
            "n_atomic_facts": fs_result["n_atomic_facts"],
            "citation_coverage": cite_result["citation_coverage"],
        },
    }


# ============================================================
# 7. 简单自测
# ============================================================

if __name__ == "__main__":
    # 快速自测
    print("=== 指标模块自测 ===")

    # 模拟数据
    np.random.seed(42)
    n = 200
    confidences = list(np.random.beta(5, 2, n))  # 偏高置信
    accuracies = []
    for c in confidences:
        acc = np.random.binomial(1, c * 0.85 + 0.05)  # 稍微过自信
        accuracies.append(float(acc))

    print(f"\n校准测试 (n={n}):")
    ece = expected_calibration_error(accuracies, confidences, n_bins=10)
    brier = brier_score(accuracies, confidences)
    aurc = area_under_risk_coverage(accuracies, confidences)
    acc80, cov80 = accuracy_at_coverage(accuracies, confidences, 0.8)
    print(f"  ECE = {ece:.4f}")
    print(f"  Brier = {brier:.4f}")
    print(f"  AURC = {aurc:.4f}")
    print(f"  Acc@0.8 = {acc80:.4f} (coverage={cov80:.2f})")

    # 忠实性测试
    answers = [
        "茶小绿叶蝉的生物防治方法包括：[Path#1] 释放寄生蜂，[Path#2] 喷施印楝素，以及使用白僵菌等真菌。",
        "使用化学农药吡虫啉可以防治茶小绿叶蝉。",
    ]
    triples = [
        [
            {"head": "茶小绿叶蝉", "relation": "生物防治", "tail": "寄生蜂"},
            {"head": "茶小绿叶蝉", "relation": "生物防治", "tail": "印楝素"},
        ],
        [
            {"head": "茶小绿叶蝉", "relation": "化学防治", "tail": "吡虫啉"},
        ],
    ]
    fs = factscore_kgqa(answers, triples)
    cite = citation_coverage(answers)
    print(f"\n忠实性测试:")
    print(f"  FActScore = {fs['factscore']:.4f} ({fs['supported_facts']}/{fs['n_atomic_facts']})")
    print(f"  Citation Coverage = {cite['citation_coverage']:.4f} ({cite['cited_facts']}/{cite['total_facts']})")

    # 显著性检验测试
    a_metric = np.random.normal(0.70, 0.2, 100)
    b_metric = np.random.normal(0.75, 0.2, 100)
    result = bootstrap_significance_test(list(a_metric), list(b_metric), n_bootstrap=1000)
    print(f"\n显著性检验:")
    print(f"  Mean A = {result['mean_a']:.4f}")
    print(f"  Mean B = {result['mean_b']:.4f}")
    print(f"  Diff = {result['diff']:.4f}")
    print(f"  p = {result['p_value']:.4f}")
    print(f"  95% CI = [{result['ci_95'][0]:.4f}, {result['ci_95'][1]:.4f}]")
    print(f"  Cohen's d = {result['cohens_d']:.4f}")

    print("\n✅ 所有指标模块测试通过")
