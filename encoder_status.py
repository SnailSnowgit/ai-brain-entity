# -*- coding: utf-8 -*-
"""
多模态编码器状态导出：为「大脑观测台」的多模态编码器 Widget
生成 data/encoder_status.json。纯本地，无任何外部 API。

内容：
  - 当前编码器注册表（list_encoders）
  - 编码器选择优先级链说明
  - 实测对照：同一文件经 自定义编码器 / 伪 embedding 两条通路的
    embedding 统计（维度、均值、范围）与感官层 16 维重采样结果
  - 实验 8（多模态相似性保持）关键数据（若 experiment_results.json 存在）

运行：
    python encoder_status.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from ai_brain_entity import (
    AIBrainEntity,
    encode_image,
    list_encoders,
    register_image_encoder,
    unregister_image_encoder,
)


def _hist32_encoder(path: str):
    """示例自定义模型：文件字节直方图 → 32 维特征（与 models/encoders 示例一致）"""
    with open(path, "rb") as f:
        data = f.read()
    hist = [0.0] * 32
    for b in data:
        hist[b % 32] += 1.0
    total = sum(hist) or 1.0
    return [v / total for v in hist]


def _stats(vec):
    return {
        "dim": len(vec),
        "mean": round(sum(vec) / max(len(vec), 1), 4),
        "min": round(min(vec), 4) if vec else 0.0,
        "max": round(max(vec), 4) if vec else 0.0,
        "head": [round(v, 4) for v in vec[:8]],
    }


def main() -> None:
    out_path = Path("datasets/encoder_status.json")
    out_path.parent.mkdir(exist_ok=True)

    # 1. 构造一个确定的样本文件（内容非均匀，使直方图编码器输出有结构）
    sample = Path("datasets/_encoder_sample.bin")
    sample.write_bytes("大脑多模态编码器探针样本·神经元脉冲".encode("utf-8") * 16)

    # 2. 通路 A：注册自定义编码器（ hist32 ）
    register_image_encoder(_hist32_encoder, name="hist32")
    encoders_with_custom = list_encoders()
    vec_custom = encode_image(str(sample))
    unregister_image_encoder("hist32")

    # 3. 通路 B：无自定义编码器 → 内置链（无依赖环境 = 512 维伪 embedding）
    encoders_builtin = list_encoders()
    vec_builtin = encode_image(str(sample))

    # 4. 两条通路的 embedding 经感官层重采样后的 16 维电流
    brain = AIBrainEntity("EncoderProbe", seed=42)
    cur_custom = brain._normalize_vector(vec_custom, 16)
    cur_builtin = brain._normalize_vector(vec_builtin, 16)

    payload = {
        "meta": {
            "title": "多模态编码器状态",
            "version": "v3.1 可插拔自定义多模态模型",
            "source": "encoder_status.py 本地实测",
        },
        "encoders": {
            "with_custom": encoders_with_custom,
            "builtin": encoders_builtin,
        },
        "priority_chain": [
            "1. 调用时显式传入的 encoder（callable 或注册名）",
            "2. register_*_encoder 注册的默认自定义编码器",
            "3. 内置 CLIP / Whisper（模型名可用 set_*_model 换成自定义微调版）",
            "4. 确定性伪 embedding（无外部模型时兜底，可复现）",
        ],
        "contract": "callable(path) -> 数值序列（长度任意，进感官层前重采样到 16 维）",
        "probe": {
            "sample_bytes": sample.stat().st_size,
            "custom": {
                "name": "hist32（自定义字节直方图模型）",
                "embedding": _stats(vec_custom),
                "sensory_current_16": [round(v, 4) for v in cur_custom],
            },
            "builtin": {
                "name": "内置链（CLIP 不可用时 = 伪 embedding）",
                "embedding": _stats(vec_builtin),
                "sensory_current_16": [round(v, 4) for v in cur_builtin],
            },
        },
        "exp8": None,
    }

    # 5. 实验 8 多模态相似性数据（若实验已运行）
    exp_path = Path("data/experiment_results.json")
    if exp_path.exists():
        exp = json.loads(exp_path.read_text(encoding="utf-8")).get("exp8")
        if exp:
            payload["exp8"] = {
                "similarity_ranking": {
                    "相似对·变体1": exp.get("current_cos_cat_vs_cat_variant1"),
                    "相似对·变体2": exp.get("current_cos_cat_vs_cat_variant2"),
                    "不同对象": exp.get("current_cos_cat_vs_dog"),
                    "随机向量": exp.get("current_cos_cat_vs_random"),
                },
                "e2e": exp.get("e2e_vector_perception"),
                "smoke": exp.get("perceive_image_smoke_test"),
                "interpretation": exp.get("interpretation"),
            }

    sample.unlink()
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"编码器状态已导出 -> {out_path} "
          f"(custom dim={len(vec_custom)}, builtin dim={len(vec_builtin)})")


if __name__ == "__main__":
    main()
