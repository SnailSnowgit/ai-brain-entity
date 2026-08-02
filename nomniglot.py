# -*- coding: utf-8 -*-
"""N-Omniglot 导出样本的零依赖加载器与少样本评估工具。

数据由 export_nomniglot.py 生成（默认 data/nomniglot_latin.json）：
真实 DAVIS346 事件相机记录的手写字符笔画事件，已按时间积分成
F 帧 G×G 网格（ON-OFF 差值归一）。本模块只读 JSON，纯标准库。
"""
import json
import math
import random
from typing import Dict, List, Tuple

DEFAULT_PATH = "data/nomniglot_latin.json"


def load(path: str = DEFAULT_PATH) -> Dict:
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def sample_to_vector(sample: Dict, mode: str = "mean") -> List[float]:
    """样本 -> 定长向量。mean: 各帧平均 (G*G 维); flatten: 全部帧拼接"""
    frames = sample["frames"]
    g = len(frames[0])
    if mode == "flatten":
        return [v for f in frames for row in f for v in row]
    vec = [0.0] * (g * g)
    for f in frames:
        for y in range(g):
            for x in range(g):
                vec[y * g + x] += f[y][x] / len(frames)
    return vec


def classes_of(samples: List[Dict]) -> List[int]:
    return sorted({s["class"] for s in samples})


def few_shot_split(samples: List[Dict], k_shot: int,
                   seed: int = 0) -> Tuple[List[Dict], List[Dict]]:
    """每类取 k 个作 support，其余作 query（确定性）"""
    rng = random.Random(seed)
    support, query = [], []
    for c in classes_of(samples):
        pool = [s for s in samples if s["class"] == c]
        idx = list(range(len(pool)))
        rng.shuffle(idx)
        for rank, i in enumerate(idx):
            (support if rank < k_shot else query).append(pool[i])
    return support, query


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def nearest_prototype_accuracy(support: List[Dict], query: List[Dict],
                               mode: str = "mean") -> float:
    """k-shot 少样本基准：类原型 = support 向量均值，余弦最近邻"""
    protos: Dict[int, List[float]] = {}
    counts: Dict[int, int] = {}
    for s in support:
        v = sample_to_vector(s, mode)
        if s["class"] not in protos:
            protos[s["class"]] = [0.0] * len(v)
            counts[s["class"]] = 0
        for i, x in enumerate(v):
            protos[s["class"]][i] += x
        counts[s["class"]] += 1
    for c in protos:
        protos[c] = [x / counts[c] for x in protos[c]]
    correct = 0
    for s in query:
        v = sample_to_vector(s, mode)
        pred = max(protos, key=lambda c: _cosine(v, protos[c]))
        correct += int(pred == s["class"])
    return correct / max(len(query), 1)


def stats(samples: List[Dict]) -> Dict:
    ev = [s["n_events"] for s in samples]
    du = [s["duration_us"] for s in samples]
    return {
        "n_samples": len(samples),
        "n_classes": len(classes_of(samples)),
        "events_min": min(ev), "events_mean": sum(ev) // len(ev),
        "events_max": max(ev),
        "duration_s_mean": round(sum(du) / len(du) / 1e6, 2),
    }
