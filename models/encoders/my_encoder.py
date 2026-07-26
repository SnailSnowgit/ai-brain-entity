# -*- coding: utf-8 -*-
"""示例自定义多模态编码器（零依赖）。

演示编码器契约：callable(path: str) -> 数值序列。
用文件字节直方图构造 32 维特征向量——不含语义，但结构完整，
替换为真实模型时只需保持同样的输入输出签名。

用法：
    import sys; sys.path.insert(0, "models/encoders")
    from ai_brain_entity import register_image_encoder
    from my_encoder import encode
    register_image_encoder(encode, name="hist32")
"""


def encode(path: str, bins: int = 32):
    """文件字节直方图 → bins 维归一化特征向量"""
    with open(path, "rb") as f:
        data = f.read()
    hist = [0.0] * bins
    for b in data:
        hist[b % bins] += 1.0
    total = sum(hist) or 1.0
    return [v / total for v in hist]
