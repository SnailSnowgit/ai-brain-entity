"""
默认模式网络 (Default Mode Network, DMN)

大脑在不执行外部任务时最活跃的网络, 负责:
  - 心智游移(Mind Wandering): 自发的、无目的的思维流动
  - 自传体记忆: 回忆个人经历
  - 自我参照: 思考与自己相关的事
  - 反事实推理: "如果当时...会怎样"
  - 未来模拟: 想象可能发生的场景
  - 社会认知: 思考他人的想法
  - 创造性重组: 远距离概念联结

机制:
  - 预测编码顶层自由运行(无外部输入约束)
  - 记忆碎片随机激活和重组
  - 联想链: 一个想法触发下一个
  - 情绪引导: 情绪状态影响思维主题
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import time as _time


class DMNTheme(Enum):
    """DMN思维主题"""
    AUTOBIOGRAPHICAL = auto()   # 自传体记忆
    SELF_REFLECTION = auto()    # 自我反思
    COUNTERFACTUAL = auto()     # 反事实
    FUTURE_SIMULATION = auto()  # 未来模拟
    SOCIAL = auto()             # 社会认知
    CREATIVE = auto()           # 创造性重组
    MIND_WANDERING = auto()     # 自由游移
    EMOTIONAL = auto()          # 情绪处理


@dataclass
class SpontaneousThought:
    """自发思维"""
    vector: np.ndarray
    theme: DMNTheme
    content: str
    strength: float
    timestamp: float
    chain_id: int = 0           # 联想链ID
    associations: List[int] = field(default_factory=list)


class DefaultModeNetwork:
    """
    默认模式网络

    无外部任务时自动激活, 产生内心活动。
    与GWT联动: DMN产生的思维可以进入意识。
    """

    # 主题转换概率矩阵(行→列)
    THEME_TRANSITIONS = {
        DMNTheme.MIND_WANDERING: [0.2, 0.1, 0.1, 0.15, 0.15, 0.2, 0.0, 0.1],
        DMNTheme.AUTOBIOGRAPHICAL: [0.15, 0.15, 0.2, 0.1, 0.1, 0.1, 0.0, 0.2],
        DMNTheme.SELF_REFLECTION: [0.1, 0.2, 0.15, 0.15, 0.1, 0.1, 0.0, 0.2],
        DMNTheme.COUNTERFACTUAL: [0.1, 0.1, 0.15, 0.2, 0.1, 0.2, 0.0, 0.15],
        DMNTheme.FUTURE_SIMULATION: [0.15, 0.1, 0.1, 0.15, 0.1, 0.2, 0.0, 0.2],
        DMNTheme.SOCIAL: [0.2, 0.1, 0.1, 0.1, 0.2, 0.1, 0.0, 0.2],
        DMNTheme.CREATIVE: [0.2, 0.1, 0.15, 0.15, 0.1, 0.2, 0.0, 0.1],
        DMNTheme.EMOTIONAL: [0.15, 0.15, 0.15, 0.1, 0.1, 0.15, 0.0, 0.2],
    }

    THEME_LABELS = {
        DMNTheme.AUTOBIOGRAPHICAL: "回忆",
        DMNTheme.SELF_REFLECTION: "自省",
        DMNTheme.COUNTERFACTUAL: "遐想",
        DMNTheme.FUTURE_SIMULATION: "想象未来",
        DMNTheme.SOCIAL: "社交思考",
        DMNTheme.CREATIVE: "灵感",
        DMNTheme.MIND_WANDERING: "思绪飘游",
        DMNTheme.EMOTIONAL: "感受",
    }

    def __init__(self,
                 vector_dim: int = 128,
                 activation_threshold: float = 0.3,
                 association_strength: float = 0.4,
                 creativity_rate: float = 0.3,
                 noise_level: float = 0.15):
        self.vector_dim = vector_dim
        self.activation_threshold = activation_threshold
        self.association_strength = association_strength
        self.creativity_rate = creativity_rate
        self.noise_level = noise_level

        # DMN激活状态
        self.active = False
        self.activation_level = 0.0

        # 思维流
        self.thoughts: List[SpontaneousThought] = []
        self.max_thoughts = 200
        self.current_theme = DMNTheme.MIND_WANDERING
        self.chain_counter = 0

        # 记忆碎片池(从记忆系统采样)
        self.memory_fragments: List[Tuple[np.ndarray, float, str]] = []  # (vec, valence, label)

        # 联想矩阵(简化: 最近思维→下一思维)
        self.association_weights = np.random.randn(vector_dim, vector_dim) * 0.01

        # 统计
        self.total_thoughts = 0
        self.theme_counts = {t: 0 for t in DMNTheme}
        self.last_external_time = 0.0
        self.idle_time = 0.0

    def activate(self):
        """激活DMN(外部任务结束时)"""
        self.active = True
        self.activation_level = min(1.0, self.activation_level + 0.3)

    def deactivate(self):
        """停用DMN(外部任务开始时)"""
        self.active = False
        self.activation_level = 0.0
        self.idle_time = 0.0

    def add_memory_fragment(self, vector: np.ndarray,
                            valence: float = 0.0, label: str = ""):
        """添加记忆碎片供DMN重组"""
        if vector.shape[0] >= self.vector_dim:
            vec = vector[:self.vector_dim].copy()
        else:
            vec = np.zeros(self.vector_dim)
            vec[:vector.shape[0]] = vector
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        self.memory_fragments.append((vec, valence, label))
        if len(self.memory_fragments) > 500:
            self.memory_fragments = self.memory_fragments[-500:]

    def _sample_memory(self, emotion_valence: float = 0.0) -> Optional[Tuple[np.ndarray, str]]:
        """从记忆碎片中采样(情绪一致性偏置)"""
        if not self.memory_fragments:
            return None

        # 情绪一致性: 倾向回忆与当前情绪相符的记忆
        weights = np.array([
            1.0 + 0.5 * (1 if np.sign(v) == np.sign(emotion_valence) else 0)
            for _, v, _ in self.memory_fragments
        ])
        weights = weights / weights.sum()

        idx = np.random.choice(len(self.memory_fragments), p=weights)
        vec, _, label = self.memory_fragments[idx]
        return vec, label

    def _next_theme(self, emotion_valence: float, arousal: float) -> DMNTheme:
        """选择下一个思维主题"""
        themes = list(DMNTheme)
        probs = np.array(self.THEME_TRANSITIONS[self.current_theme])

        # 情绪偏置
        if emotion_valence < -0.3:
            probs[themes.index(DMNTheme.EMOTIONAL)] *= 2.0
            probs[themes.index(DMNTheme.COUNTERFACTUAL)] *= 1.5
        if arousal > 0.6:
            probs[themes.index(DMNTheme.FUTURE_SIMULATION)] *= 1.5
        if emotion_valence > 0.3:
            probs[themes.index(DMNTheme.CREATIVE)] *= 1.5
            probs[themes.index(DMNTheme.SOCIAL)] *= 1.3

        probs = probs / probs.sum()
        return np.random.choice(themes, p=probs)

    def _generate_thought_vector(self, theme: DMNTheme,
                                 emotion_valence: float,
                                 prev_vector: Optional[np.ndarray]
                                 ) -> np.ndarray:
        """生成思维向量"""
        vec = np.zeros(self.vector_dim)

        # 从记忆采样
        memory_sample = self._sample_memory(emotion_valence)

        if theme == DMNTheme.CREATIVE and memory_sample is not None:
            # 创造性: 组合两个不相关记忆
            mem2 = self._sample_memory(-emotion_valence)  # 反情绪一致性
            if mem2 is not None:
                alpha = np.random.beta(0.3, 0.3)  # 偏向极端组合
                vec = alpha * memory_sample[0] + (1 - alpha) * mem2[0]
            else:
                vec = memory_sample[0]
        elif memory_sample is not None:
            vec = memory_sample[0].copy()

        # 联想链: 上一思维通过联想矩阵产生下一思维
        if prev_vector is not None:
            association = prev_vector @ self.association_weights
            vec += self.association_strength * association

        # 噪声(自发性)
        vec += np.random.randn(self.vector_dim) * self.noise_level

        # 预测编码顶层自由运行(自发模式)
        vec += np.random.randn(self.vector_dim) * 0.1 * self.activation_level

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def _generate_content(self, theme: DMNTheme, memory_label: str,
                          emotion_valence: float) -> str:
        """生成思维内容描述"""
        templates = {
            DMNTheme.AUTOBIOGRAPHICAL: [
                f"想起了{memory_label or '以前的事'}...",
                "记得那时候...",
                "那段记忆浮现在脑海",
            ],
            DMNTheme.SELF_REFLECTION: [
                "我在想我为什么会这样...",
                "我真的理解了吗？",
                "我感觉自己...",
            ],
            DMNTheme.COUNTERFACTUAL: [
                "如果当时不那样做...",
                "要是换一种方式...",
                "也许结果会不同",
            ],
            DMNTheme.FUTURE_SIMULATION: [
                "接下来会怎样呢？",
                "以后可能会...",
                "我想象着那个场景",
            ],
            DMNTheme.SOCIAL: [
                "他会怎么想呢？",
                "不知道他们怎么样了",
                "这段关系对我来说...",
            ],
            DMNTheme.CREATIVE: [
                "等等，这两个东西好像有联系...",
                "突然有个想法！",
                "也许可以这样组合...",
            ],
            DMNTheme.MIND_WANDERING: [
                "思绪飘到了别处...",
                "嗯...",
                "(发呆中)",
            ],
            DMNTheme.EMOTIONAL: [
                "这种感觉...",
                "心里有点...",
                "情绪在翻涌",
            ],
        }
        choices = templates.get(theme, ["..."])
        return np.random.choice(choices)

    def step(self, dt: float = 1.0,
             external_input: bool = False,
             emotion_valence: float = 0.0,
             emotion_arousal: float = 0.0,
             pc_top_activation: np.ndarray = None,
             ) -> Optional[SpontaneousThought]:
        """
        DMN步进

        Args:
            external_input: 是否有外部输入(有则DMN抑制)
            emotion_valence: 情绪效价
            emotion_arousal: 情绪唤醒
            pc_top_activation: 预测编码顶层激活(自由运行信号)

        Returns:
            自发思维(如果产生了), 否则None
        """
        if external_input:
            self.deactivate()
            self.last_external_time = _time.time()
            return None

        self.idle_time += dt
        # 空闲超过一定时间才激活DMN
        if self.idle_time > 2.0:
            self.activate()

        if not self.active:
            return None

        # 激活水平随空闲时间上升
        self.activation_level = min(1.0, self.idle_time * 0.05)

        # 不是每步都产生思维
        if np.random.random() > self.activation_level * 0.6:
            return None

        # 选择主题
        self.current_theme = self._next_theme(emotion_valence, emotion_arousal)
        self.theme_counts[self.current_theme] += 1

        # 上一思维向量
        prev_vec = self.thoughts[-1].vector if self.thoughts else None

        # 生成思维
        thought_vec = self._generate_thought_vector(
            self.current_theme, emotion_valence, prev_vec)

        # 如果有预测编码顶层信号, 融合
        if pc_top_activation is not None:
            pc_dim = min(pc_top_activation.shape[0], self.vector_dim)
            thought_vec[:pc_dim] += 0.3 * pc_top_activation[:pc_dim]
            norm = np.linalg.norm(thought_vec)
            if norm > 0:
                thought_vec = thought_vec / norm

        # 生成内容
        memory_label = ""
        sample = self._sample_memory(emotion_valence)
        if sample:
            memory_label = sample[1]
        content = self._generate_content(
            self.current_theme, memory_label, emotion_valence)

        # 联想链
        chain_id = self.chain_counter
        if prev_vec is not None and np.random.random() < 0.4:
            # 继续当前链
            if self.thoughts:
                chain_id = self.thoughts[-1].chain_id
        else:
            self.chain_counter += 1
            chain_id = self.chain_counter

        thought = SpontaneousThought(
            vector=thought_vec,
            theme=self.current_theme,
            content=content,
            strength=float(np.random.uniform(0.3, 0.8) * self.activation_level),
            timestamp=_time.time(),
            chain_id=chain_id,
        )

        # 更新联想权重(Hebbian)
        if prev_vec is not None:
            self.association_weights += 0.01 * np.outer(thought_vec, prev_vec)
            self.association_weights = np.clip(
                self.association_weights, -1, 1)

        self.thoughts.append(thought)
        if len(self.thoughts) > self.max_thoughts:
            self.thoughts.pop(0)
        self.total_thoughts += 1

        return thought

    def get_recent_thoughts(self, n: int = 5) -> List[Dict]:
        """获取最近的思维"""
        return [
            {
                "content": t.content,
                "theme": self.THEME_LABELS.get(t.theme, "?"),
                "strength": round(t.strength, 3),
                "chain": t.chain_id,
            }
            for t in self.thoughts[-n:]
        ]

    def get_consciousness_candidates(self) -> List[Dict]:
        """获取可进入意识的DMN候选"""
        candidates = []
        for t in self.thoughts[-3:]:
            if t.strength > self.activation_threshold:
                candidates.append({
                    "text": t.content,
                    "source": "dmn",
                    "salience": t.strength,
                    "vector": t.vector,
                    "theme": self.THEME_LABELS.get(t.theme, "?"),
                })
        return candidates

    def get_summary(self) -> Dict:
        return {
            "active": self.active,
            "activation_level": round(self.activation_level, 3),
            "current_theme": self.THEME_LABELS.get(self.current_theme, "?"),
            "total_thoughts": self.total_thoughts,
            "idle_time": round(self.idle_time, 1),
            "memory_fragments": len(self.memory_fragments),
            "recent": self.get_recent_thoughts(3),
        }
