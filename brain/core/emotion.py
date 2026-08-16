"""
情绪核心模块 (Emotional Core)

包含:
  - EmotionalState: 六维情绪状态(joy/sadness/anger/fear/disgust/surprise)
  - EmotionalCore: 情绪评估、衰减、调制
  - DopamineSystem: 多巴胺奖励预测误差(RPE)

情绪对认知的调制:
  - 情绪效价(valence)影响记忆编码强度
  - 唤醒度(arousal)影响注意力
  - 多巴胺影响探索/利用和创造力
  - 压力(皮质醇)使生成更保守
"""
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class EmotionalState:
    """情绪状态"""
    joy: float = 0.1
    sadness: float = 0.05
    anger: float = 0.0
    fear: float = 0.05
    disgust: float = 0.0
    surprise: float = 0.1

    @property
    def valence(self) -> float:
        """效价: 正(愉悦)~负(不快)"""
        pos = self.joy + self.surprise * 0.5
        neg = self.sadness + self.anger + self.fear + self.disgust
        return float(np.clip(pos - neg, -1.0, 1.0))

    @property
    def arousal(self) -> float:
        """唤醒度: 平静~激动"""
        return float(np.clip(
            self.joy + self.anger + self.fear + self.surprise * 0.7,
            0.0, 1.0
        ))

    def dominant(self) -> str:
        emotions = {'喜悦': self.joy, '悲伤': self.sadness, '愤怒': self.anger,
                    '恐惧': self.fear, '厌恶': self.disgust, '惊讶': self.surprise}
        dom = max(emotions, key=emotions.get)
        if emotions[dom] < 0.15:
            if self.valence > 0.2: return '满足'
            if self.valence < -0.2: return '低落'
            return '平静'
        return dom


class DopamineSystem:
    """多巴胺奖励系统"""

    def __init__(self, baseline: float = 0.3, learning_rate: float = 0.1,
                 discount: float = 0.9):
        self.baseline = baseline
        self.learning_rate = learning_rate
        self.discount = discount
        self.current_dopamine = baseline
        self.value_estimates: Dict[str, float] = {}

    def predict_reward(self, state_key: str) -> float:
        return self.value_estimates.get(state_key, self.baseline)

    def compute_rpe(self, actual_reward: float, state_key: str = "default") -> float:
        """奖励预测误差 = 实际奖励 - 预期奖励"""
        predicted = self.value_estimates.get(state_key, self.baseline)
        rpe = actual_reward - predicted
        # 更新价值估计
        self.value_estimates[state_key] = predicted + self.learning_rate * rpe
        # 多巴胺释放
        self.current_dopamine = float(np.clip(
            self.baseline + rpe * 2.0, 0.0, 1.0
        ))
        return rpe

    def step(self, dt: float = 1.0):
        """多巴胺回归基线"""
        self.current_dopamine += (self.baseline - self.current_dopamine) * 0.05 * dt


class EmotionalCore:
    """情绪核心"""

    def __init__(self, decay_rate: float = 0.02):
        self.state = EmotionalState()
        self.dopamine = DopamineSystem()
        self.decay_rate = decay_rate
        self.emotional_memories = []

    def evaluate_stimulus(self, text: str) -> EmotionalState:
        """评估文本刺激的情绪效价"""
        positive = ["好", "棒", "喜欢", "开心", "高兴", "优秀", "谢谢", "爱",
                    "赞", "成功", "美好", "快乐"]
        negative = ["坏", "糟", "讨厌", "难过", "失败", "错误", "痛苦", "担心",
                    "焦虑", "沮丧", "失望"]
        threat = ["危险", "攻击", "杀", "伤害", "恐惧", "威胁"]

        valence = 0.0
        arousal = 0.0

        for w in positive:
            if w in text:
                valence += 0.15
                arousal += 0.05
        for w in negative:
            if w in text:
                valence -= 0.15
                arousal += 0.1
        for w in threat:
            if w in text:
                valence -= 0.3
                arousal += 0.4

        self.update(valence=valence, arousal=arousal, text=text)
        return self.state

    def update(self, valence: float = 0.0, arousal: float = 0.0,
               reward: float = 0.0, text: str = ""):
        """更新情绪状态"""
        s = self.state
        if valence > 0.2:
            s.joy = min(1.0, s.joy + valence * 0.3)
        elif valence < -0.2:
            if any(w in text for w in ["危险", "攻击", "伤害", "恐惧", "威胁"]):
                s.fear = min(1.0, s.fear + abs(valence) * 0.4)
            else:
                s.sadness = min(1.0, s.sadness + abs(valence) * 0.3)

        if arousal > 0.5:
            s.surprise = min(1.0, s.surprise + 0.2)

        if reward > 0.3:
            s.joy = min(1.0, s.joy + reward * 0.1)

    def step(self, dt: float = 1.0):
        """情绪衰减"""
        s = self.state
        s.joy *= (1 - self.decay_rate * dt)
        s.sadness *= (1 - self.decay_rate * dt)
        s.anger *= (1 - self.decay_rate * 1.5 * dt)
        s.fear *= (1 - self.decay_rate * 1.2 * dt)
        s.disgust *= (1 - self.decay_rate * dt)
        s.surprise *= (1 - self.decay_rate * 3.0 * dt)
        self.dopamine.step(dt)

    def get_memory_modulation(self) -> float:
        """情绪对记忆的调制权重"""
        return 0.5 + abs(self.state.arousal) * 0.5

    def get_attention_bias(self) -> float:
        """情绪对注意力的偏置"""
        return float(np.clip(self.state.arousal * 0.5 + self.state.valence * 0.2, -1, 1))

    def get_generation_params(self, cortisol: float = 0.15,
                              oxytocin: float = 0.2,
                              endorphin: float = 0.1,
                              cognitive_load: float = 0.3,
                              energy: float = 1.0) -> Tuple[float, float]:
        """
        情绪/激素调制LLM采样参数

        Returns:
            (temperature, top_p)
        """
        temperature = 0.7
        top_p = 0.9

        # 多巴胺: 高→创造力
        temperature += 0.2 * (self.dopamine.current_dopamine - 0.3)
        # 皮质醇(压力): 高→保守
        temperature -= 0.3 * max(0, cortisol - 0.2)
        top_p -= 0.2 * max(0, cortisol - 0.3)
        # 催产素: 社交→开放
        temperature += 0.1 * oxytocin
        # 内啡肽: 愉悦→灵活
        temperature += 0.1 * endorphin
        # 认知负荷高→谨慎
        if cognitive_load > 0.7:
            temperature -= 0.15
        # 低能量→保守
        if energy < 0.3:
            temperature -= 0.2

        return (float(np.clip(temperature, 0.1, 1.5)),
                float(np.clip(top_p, 0.5, 1.0)))
