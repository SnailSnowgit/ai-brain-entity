"""
意识模块 (Consciousness)

基于全局工作空间理论(GWT - Global Workspace Theory):
  - 多个并行的无意识模块竞争进入全局工作空间
  - 显著性最高的内容"广播"到全系统, 成为意识内容
  - 意识是"全脑信息整合"的涌现属性

显著性 = w1*情绪强度 + w2*新奇度 + w3*目标相关 + w4*意外度 + noise
"""
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class ConsciousnessLevel(Enum):
    """意识水平"""
    COMA = "昏迷"
    DEEP_SLEEP = "深睡"
    DREAMING = "做梦"
    DROWSY = "嗜睡"
    AWAKE = "清醒"
    FLOW = "心流"
    HYPER = "高度警觉"


@dataclass
class ConsciousContent:
    """进入意识的内容"""
    text: str = ""
    source: str = ""           # 来源模块
    salience: float = 0.5      # 显著性 0-1
    emotional_valence: float = 0.0
    emotional_arousal: float = 0.5
    confidence: float = 0.5
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsciousnessMetrics:
    """意识度量指标"""
    phi: float = 0.0           # 整合信息Φ
    level: ConsciousnessLevel = ConsciousnessLevel.AWAKE
    broadcast_count: int = 0
    mean_salience: float = 0.0
    content_diversity: float = 0.0


class GlobalWorkspace:
    """
    全局工作空间

    并行模块竞争进入意识, 赢家广播到全系统。
    """

    def __init__(self, capacity: int = 7, decay: float = 0.1):
        self.capacity = capacity  # 工作记忆容量7±2
        self.decay = decay
        self.current_content: Optional[ConsciousContent] = None
        self.broadcast_history: List[ConsciousContent] = []
        self.attention_focus: Optional[np.ndarray] = None
        self.total_broadcasts = 0

    def compete(self, candidates: List[ConsciousContent]) -> ConsciousContent:
        """
        全局工作空间竞争

        Args:
            candidates: 各模块提交的候选内容

        Returns:
            获胜的意识内容
        """
        if not candidates:
            return ConsciousContent(text="(静息)", source="none", salience=0.1)

        # 显著性竞争 + 噪声
        noise = np.random.randn(len(candidates)) * 0.05
        scores = [c.salience + noise[i] for i, c in enumerate(candidates)]
        winner_idx = int(np.argmax(scores))
        winner = candidates[winner_idx]
        winner.timestamp = self.total_broadcasts

        # 广播
        self.current_content = winner
        self.broadcast_history.append(winner)
        if len(self.broadcast_history) > 100:
            self.broadcast_history.pop(0)
        self.total_broadcasts += 1

        return winner

    def broadcast(self, candidates: List[ConsciousContent]) -> ConsciousContent:
        """竞争并广播(同compete)"""
        return self.compete(candidates)

    def set_attention(self, focus: np.ndarray):
        """自上而下的注意力调控"""
        self.attention_focus = focus

    def step(self, dt: float = 1.0):
        """意识内容衰减"""
        if self.current_content:
            self.current_content.salience *= (1 - self.decay * dt)


class ConsciousnessSystem:
    """
    意识系统

    整合GWT + Φ信息整合度量
    """

    def __init__(self):
        self.workspace = GlobalWorkspace()
        self.metrics = ConsciousnessMetrics()
        self.awake = True
        self.time = 0.0

    def compute_phi(self, module_activities: Dict[str, float]) -> float:
        """
        计算整合信息Φ(简化版)

        Φ ≈ 全系统互信息 - 各模块独立信息之和
        用模块间活动相关性近似
        """
        if len(module_activities) < 2:
            return 0.0
        values = list(module_activities.values())
        n = len(values)
        # 模块间平均相关性作为整合度
        correlations = []
        for i in range(n):
            for j in range(i + 1, n):
                # 简化: 用活动强度的乘积作为协同度
                correlations.append(values[i] * values[j])
        integration = np.mean(correlations) if correlations else 0.0
        # 分化度: 活动分布的熵
        arr = np.array(values) + 1e-8
        arr = arr / arr.sum()
        differentiation = -np.sum(arr * np.log(arr)) / np.log(n)
        # Φ = 整合 × 分化
        self.metrics.phi = float(integration * differentiation)
        return self.metrics.phi

    def determine_level(self, arousal: float, cognitive_load: float = 0.3,
                        energy: float = 1.0) -> ConsciousnessLevel:
        """根据唤醒度/负荷/能量判断意识水平"""
        if energy < 0.15:
            level = ConsciousnessLevel.DEEP_SLEEP
        elif arousal < 0.15:
            level = ConsciousnessLevel.DROWSY
        elif arousal > 0.85 and cognitive_load > 0.7:
            level = ConsciousnessLevel.HYPER
        elif 0.4 < arousal < 0.7 and cognitive_load > 0.5:
            level = ConsciousnessLevel.FLOW
        elif arousal > 0.3:
            level = ConsciousnessLevel.AWAKE
        else:
            level = ConsciousnessLevel.DEEP_SLEEP
        self.metrics.level = level
        return level

    def build_candidates(self, user_input: str = None,
                         memories: list = None,
                         emotion_state=None,
                         prediction_error: float = 0.0,
                         interoception: dict = None,
                         curiosity: float = 0.5) -> List[ConsciousContent]:
        """从各模块收集意识候选"""
        candidates = []

        # 外部输入(通常最显著)
        if user_input:
            candidates.append(ConsciousContent(
                text=user_input[:100], source="sensory_input",
                salience=0.6 + curiosity * 0.2, confidence=0.9
            ))

        # 记忆候选
        for mem in (memories or [])[:3]:
            candidates.append(ConsciousContent(
                text=str(mem.get('content', ''))[:100],
                source="memory",
                salience=0.3 + mem.get('strength', 0.3) * 0.3,
                emotional_valence=mem.get('emotional_valence', 0.0),
                confidence=0.6
            ))

        # 情绪候选
        if emotion_state and abs(emotion_state.valence) > 0.3:
            candidates.append(ConsciousContent(
                text=f"情绪: {emotion_state.dominant()}",
                source="emotion",
                salience=0.3 + abs(emotion_state.valence) * 0.3,
                emotional_valence=emotion_state.valence,
                emotional_arousal=emotion_state.arousal,
                confidence=0.8
            ))

        # 预测误差(意外)
        if prediction_error > 0.3:
            candidates.append(ConsciousContent(
                text="意外: 预测与实际不符",
                source="predictive_coding",
                salience=0.4 + prediction_error * 0.4,
                confidence=0.5
            ))

        # 内感受
        if interoception and interoception.get('dominant_signal'):
            candidates.append(ConsciousContent(
                text=f"身体: {interoception['dominant_signal']}",
                source="interoception",
                salience=0.2 + interoception.get('discomfort', 0) * 0.5,
                confidence=0.7
            ))

        return candidates

    def step(self, dt: float = 1.0):
        self.time += dt
        self.workspace.step(dt)
        # 更新统计
        if self.workspace.broadcast_history:
            recent = self.workspace.broadcast_history[-10:]
            self.metrics.mean_salience = float(
                np.mean([c.salience for c in recent]))
            self.metrics.broadcast_count = self.workspace.total_broadcasts
            sources = [c.source for c in recent]
            self.metrics.content_diversity = len(set(sources)) / len(sources)
