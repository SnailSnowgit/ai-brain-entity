"""
注意力系统 (Attention System)

双路注意力:
  - 自下而上(Bottom-up): 刺激驱动, 显著性自动捕获(新颖/威胁/强烈)
  - 自上而下(Top-down): 目标/动机驱动, 主动聚焦相关信息

机制:
  - 显著性图(Saliency Map): 对输入各维度计算显著性
  - 注意力门控(Attention Gate): 决定哪些信息进入GWT竞争
  - 聚光灯(Spotlight): 注意力焦点, 有惯性和转移成本
  - 精度分配: 注意力增强预测编码对应层的精度
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto


class AttentionMode(Enum):
    """注意力模式"""
    FOCUSED = auto()     # 聚焦(高浓度, 窄范围)
    DIFFUSE = auto()     # 弥散(低浓度, 宽范围)
    SELECTIVE = auto()   # 选择性
    DIVIDED = auto()     # 分配(多目标)
    ORIENTING = auto()   # 定向(突发刺激)


@dataclass
class AttentionTarget:
    """注意力目标"""
    source: str               # 来源模块
    salience: float           # 显著性 0-1
    vector: np.ndarray        # 内容向量
    bottom_up: float = 0.0    # 自下而上强度
    top_down: float = 0.0     # 自上而下强度
    priority: float = 0.0     # 综合优先级
    sustained: int = 0        # 持续步数


class AttentionSystem:
    """
    注意力系统

    模拟背侧(自上而下)和腹侧(自下而上)注意网络。
    与预测编码精度联动: 被注意的信息获得更高精度权重。
    """

    def __init__(self,
                 vector_dim: int = 128,
                 capacity: float = 1.0,
                 focus_decay: float = 0.05,
                 orienting_threshold: float = 0.6,
                 inertia: float = 0.3,
                 top_down_gain: float = 0.6,
                 bottom_up_gain: float = 0.8):
        self.vector_dim = vector_dim
        self.capacity = capacity
        self.focus_decay = focus_decay
        self.orienting_threshold = orienting_threshold
        self.inertia = inertia
        self.td_gain = top_down_gain
        self.bu_gain = bottom_up_gain

        # 当前注意力焦点
        self.focus_vector: Optional[np.ndarray] = None
        self.focus_strength: float = 0.0
        self.focus_source: str = "none"
        self.focus_duration: int = 0

        # 注意力模式
        self.mode = AttentionMode.DIFFUSE

        # 自上而下目标(来自目标管理/动机)
        self.top_down_bias: np.ndarray = np.zeros(vector_dim)
        self.top_down_strength: float = 0.0

        # 显著性历史(用于习惯化)
        self.recent_inputs: List[np.ndarray] = []
        self.max_history = 20

        # 精度分配(给预测编码的反馈)
        self.precision_gain: np.ndarray = np.ones(vector_dim)

        # 统计
        self.total_orienting = 0
        self.attention_switches = 0
        self.current_targets: List[AttentionTarget] = []

    def set_top_down_bias(self, target_vector: np.ndarray, strength: float = 0.7):
        """设置自上而下注意力偏置(来自目标/动机)"""
        if target_vector.shape[0] < self.vector_dim:
            padded = np.zeros(self.vector_dim)
            padded[:target_vector.shape[0]] = target_vector
            target_vector = padded
        norm = np.linalg.norm(target_vector)
        if norm > 0:
            self.top_down_bias = target_vector / norm
        self.top_down_strength = float(np.clip(strength, 0, 1))

    def clear_top_down_bias(self):
        """清除自上而下偏置"""
        self.top_down_bias = np.zeros(self.vector_dim)
        self.top_down_strength = 0.0

    def _bottom_up_salience(self, vector: np.ndarray,
                            threat: float = 0.0,
                            novelty: float = 0.0) -> float:
        """
        自下而上显著性

        因素: 强度 + 新颖性 + 威胁 + 变化
        """
        # 输入强度
        intensity = float(np.mean(np.abs(vector)))

        # 新颖性(与近期输入的差异)
        if self.recent_inputs:
            similarities = [float(np.abs(np.dot(vector, r)))
                           for r in self.recent_inputs[-5:]]
            novelty = 1.0 - np.mean(similarities)
        else:
            novelty = 1.0

        # 变化检测(与上一输入差异)
        change = 0.0
        if self.recent_inputs:
            change = float(np.mean((vector - self.recent_inputs[-1]) ** 2))

        # 威胁加权(威胁自动捕获注意力)
        threat_capture = threat * 2.0

        salience = (0.3 * intensity + 0.3 * novelty +
                    0.2 * change + threat_capture)
        return float(np.clip(salience, 0, 1))

    def _top_down_match(self, vector: np.ndarray) -> float:
        """自上而下匹配度(与目标偏置的相似度)"""
        if self.top_down_strength < 0.01:
            return 0.0
        norm = np.linalg.norm(vector)
        if norm < 1e-8:
            return 0.0
        similarity = float(np.dot(vector / norm, self.top_down_bias))
        return max(0, similarity) * self.top_down_strength

    def process(self, inputs: Dict[str, np.ndarray],
                threat_level: float = 0.0,
                cognitive_load: float = 0.3) -> List[AttentionTarget]:
        """
        处理多路输入, 分配注意力

        Args:
            inputs: {source_name: vector}
            threat_level: 威胁等级 0-1
            cognitive_load: 认知负荷 0-1

        Returns:
            按优先级排序的注意力目标
        """
        targets = []

        for source, vector in inputs.items():
            if vector.shape[0] < self.vector_dim:
                padded = np.zeros(self.vector_dim)
                padded[:vector.shape[0]] = vector
                vector = padded

            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm

            bu = self._bottom_up_salience(vector, threat_level)
            td = self._top_down_match(vector)

            # 综合优先级
            priority = (self.bu_gain * bu + self.td_gain * td)

            # 认知负荷高时降低整体注意力
            priority *= (1.0 - 0.3 * cognitive_load)

            # 威胁超控
            if threat_level > 0.7 and "threat" in source.lower():
                priority = max(priority, 0.95)

            targets.append(AttentionTarget(
                source=source,
                salience=max(bu, td),
                vector=vector,
                bottom_up=bu,
                top_down=td,
                priority=float(np.clip(priority, 0, 1)),
            ))

        # 按优先级排序
        targets.sort(key=lambda t: t.priority, reverse=True)
        self.current_targets = targets

        # 注意力焦点选择
        if targets:
            top = targets[0]

            # 定向反应: 高显著性突发刺激
            if (top.priority > self.orienting_threshold and
                    top.source != self.focus_source):
                self.total_orienting += 1
                self.attention_switches += 1
                self.focus_duration = 0
                self.mode = AttentionMode.ORIENTING
            elif top.source == self.focus_source:
                self.focus_duration += 1
                self.mode = AttentionMode.FOCUSED if self.focus_duration > 2 else AttentionMode.SELECTIVE
            else:
                self.attention_switches += 1
                self.focus_duration = 0

            # 惯性: 切换有成本
            if (self.focus_vector is not None and
                    top.source != self.focus_source and
                    self.focus_strength > 0.5):
                # 只有新目标显著强于当前焦点才切换
                if top.priority < self.focus_strength + self.inertia * 0.2:
                    top = next((t for t in targets
                               if t.source == self.focus_source), top)

            self.focus_vector = top.vector.copy()
            self.focus_strength = float(np.clip(
                top.priority * (1 - self.focus_decay * self.focus_duration),
                0, 1))
            self.focus_source = top.source

            # 更新显著性历史
            self.recent_inputs.append(top.vector.copy())
            if len(self.recent_inputs) > self.max_history:
                self.recent_inputs.pop(0)

        # 更新精度分配(注意力增强对应维度精度)
        self._update_precision()

        # 衰减
        if self.focus_vector is not None:
            self.focus_strength *= (1 - self.focus_decay)
            if self.focus_strength < 0.05:
                self.focus_strength = 0.0
                self.focus_source = "none"
                self.mode = AttentionMode.DIFFUSE

        return targets

    def _update_precision(self):
        """更新预测编码精度分配(注意力=精度)"""
        # 基础精度
        self.precision_gain = np.ones(self.vector_dim) * 0.5

        if self.focus_vector is not None and self.focus_strength > 0.1:
            # 注意力焦点维度获得高精度
            focus_mask = np.abs(self.focus_vector) > 0.1
            self.precision_gain[focus_mask] += self.focus_strength * 1.5
            # 非焦点维度降低
            self.precision_gain[~focus_mask] *= 0.3

        self.precision_gain = np.clip(self.precision_gain, 0.1, 2.0)

    def get_precision_for_layer(self, layer_idx: int,
                                layer_size: int) -> np.ndarray:
        """获取预测编码某层的精度向量"""
        if layer_size == self.vector_dim:
            return self.precision_gain
        # 不同维度: 插值
        indices = np.linspace(0, self.vector_dim - 1, layer_size).astype(int)
        return self.precision_gain[indices]

    def gate_for_consciousness(self, candidates: list) -> list:
        """
        注意力门控: 筛选进入GWT竞争的候选

        只有通过注意力门控的候选才能进入意识竞争
        """
        if not candidates:
            return []

        gated = []
        for cand in candidates:
            # 候选需要有salience属性
            sal = getattr(cand, 'salience', getattr(cand, 'priority', 0.5))
            source = getattr(cand, 'source', 'unknown')

            # 注意力增强: 与焦点相关的候选增强
            boost = 1.0
            if source == self.focus_source:
                boost = 1.0 + self.focus_strength
            elif self.focus_strength > 0.7:
                boost = 0.5  # 高度聚焦时抑制无关信息

            adjusted_sal = sal * boost
            if adjusted_sal > 0.15:  # 门控阈值
                if hasattr(cand, 'salience'):
                    cand.salience = min(1.0, adjusted_sal)
                gated.append(cand)

        return gated if gated else candidates[:1]  # 至少保留一个

    def get_summary(self) -> Dict:
        return {
            "mode": self.mode.name,
            "focus_source": self.focus_source,
            "focus_strength": round(self.focus_strength, 3),
            "focus_duration": self.focus_duration,
            "top_down_active": self.top_down_strength > 0.1,
            "orienting_count": self.total_orienting,
            "switches": self.attention_switches,
            "tracked_targets": len(self.current_targets),
            "mean_precision": round(float(np.mean(self.precision_gain)), 3),
        }
