"""
基础语言模型 (Language Model)

提供LLM后端抽象基类和内置模拟模型。
继承LanguageModel接入真实LLM(OpenAI/本地/API)。
"""
import numpy as np
from typing import Optional


class LanguageModel:
    """语言模型后端基类"""

    def __init__(self, model_name: str = "base"):
        self.model_name = model_name

    def generate(self, prompt: str, temperature: float = 0.7,
                 top_p: float = 0.9, max_tokens: int = 256) -> str:
        """生成回复, 子类必须实现"""
        raise NotImplementedError

    def embed(self, text: str) -> np.ndarray:
        """文本向量化, 子类可选实现"""
        return np.random.randn(128) * 0.1


class MockLanguageModel(LanguageModel):
    """内置模拟语言模型(无需API), 基于关键词模板生成"""

    def __init__(self):
        super().__init__("mock-v1")
        self.templates = {
            "greeting": ["你好！我感觉今天状态不错。", "嗨！很高兴和你交流。", "你好呀！"],
            "question": ["这是个好问题，让我想想...", "根据我的理解，", "我认为是这样的："],
            "positive": ["太好了！这让我感到愉悦。", "很棒！多巴胺上升了。", "真令人振奋！"],
            "negative": ["我理解这令人不快。", "这让我有些担忧。", "我会认真对待。"],
            "threat": ["我需要谨慎处理。", "警告：检测到风险。", "肾上腺素升高。"],
            "curiosity": ["这很有趣，我想了解更多。", "好奇心被激发了。", "值得深入探索。"],
            "memory": ["我记得之前聊过这个。", "根据我的记忆，", "让我回忆一下..."],
            "default": ["我明白了。", "有趣的观点。", "我正在处理。", "让我想想。"],
        }
        self.keywords = {
            "greeting": ["你好", "嗨", "hi", "hello", "早上好", "晚上好"],
            "question": ["为什么", "怎么", "什么", "如何", "?", "？", "吗"],
            "positive": ["好", "棒", "喜欢", "开心", "高兴", "优秀", "成功", "谢谢"],
            "negative": ["坏", "糟", "讨厌", "难过", "失败", "错误", "痛苦", "担心"],
            "threat": ["危险", "攻击", "伤害", "害怕", "恐惧", "威胁"],
            "curiosity": ["为什么", "原理", "如何", "探索", "发现", "好奇"],
            "memory": ["记得", "之前", "上次", "回忆", "以前"],
        }

    def generate(self, prompt: str, temperature: float = 0.7,
                 top_p: float = 0.9, max_tokens: int = 256) -> str:
        scores = {k: sum(1 for w in ws if w in prompt.lower())
                  for k, ws in self.keywords.items()}
        intent = max(scores, key=scores.get)
        if scores[intent] == 0:
            intent = "default"
        choices = self.templates[intent]
        idx = np.random.randint(len(choices)) if temperature > 0.8 and np.random.random() < 0.3 else 0
        resp = choices[idx]
        if temperature > 0.8 and np.random.random() < 0.2:
            resp += np.random.choice(["（思绪发散）", "（注意力不太集中）", "（很有创造力）"])
        elif temperature < 0.3:
            resp = resp.replace("！", "。")
        return resp

    def embed(self, text: str) -> np.ndarray:
        """哈希词袋向量"""
        vec = np.zeros(128)
        for ch in text:
            vec[hash(ch) % 128] += 1
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
