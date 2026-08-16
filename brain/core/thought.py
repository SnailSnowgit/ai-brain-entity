"""
思考模块 (Thought System)

思维流(Thought Stream) + 思考空间(Thought Space):
  - 思维流: 连续的思维序列, 类似内心独白
  - 思考空间: 并行思维竞争, 最强者进入意识
  - 思维类型: 感知/记忆/情绪/抽象/目标导向/自我参照/意象
  - 思维链: 一个思维触发下一个思维(联想)
  - 系统1(快速直觉) / 系统2(缓慢推理)
"""
import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import time


class ThoughtType(Enum):
    """思维类型"""
    PERCEPTUAL = "感知"
    MEMORY = "记忆"
    EMOTIONAL = "情绪"
    ABSTRACT = "抽象"
    GOAL_DIRECTED = "目标导向"
    SELF_REFERENTIAL = "自我参照"
    IMAGERY = "意象"


@dataclass
class Thought:
    """单个思维"""
    content: np.ndarray
    thought_type: ThoughtType = ThoughtType.ABSTRACT
    strength: float = 0.5
    source: str = ""
    timestamp: float = field(default_factory=time.time)
    associations: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def decay(self, rate: float = 0.02):
        self.strength *= (1 - rate)


class ThoughtSpace:
    """
    思考空间: 并行思维竞争

    容量有限(30), 最强思维被选中进入意识。
    """

    def __init__(self, capacity: int = 30, vector_dim: int = 128):
        self.capacity = capacity
        self.vector_dim = vector_dim
        self.thoughts: List[Thought] = []
        self.time = 0.0

    def submit(self, thought: Thought):
        """提交一个思维到思考空间"""
        self.thoughts.append(thought)
        if len(self.thoughts) > self.capacity:
            # 移除最弱的
            self.thoughts.sort(key=lambda t: t.strength)
            self.thoughts.pop(0)

    def select_for_consciousness(self) -> Optional[Thought]:
        """选择最强思维进入意识"""
        if not self.thoughts:
            return None
        # 竞争: 强度 + 噪声
        scores = [t.strength + np.random.randn() * 0.05
                  for t in self.thoughts]
        idx = int(np.argmax(scores))
        winner = self.thoughts[idx]
        return winner

    def get_broadcast_content(self) -> Optional[Thought]:
        """同select_for_consciousness"""
        return self.select_for_consciousness()

    def step(self, dt: float = 1.0):
        """思维衰减和清理"""
        self.time += dt
        for t in self.thoughts:
            t.decay(0.02 * dt)
        # 移除过弱的思维
        self.thoughts = [t for t in self.thoughts if t.strength > 0.05]

    def clear(self):
        self.thoughts.clear()


class ThoughtStream:
    """
    思维流: 连续的思维序列

    记录思维的时间序列, 支持联想链。
    """

    def __init__(self, max_length: int = 100):
        self.stream: List[Thought] = []
        self.max_length = max_length
        self.time = 0.0

    def add(self, thought: Thought):
        """添加思维到流"""
        thought.timestamp = self.time
        # 与上一个思维建立联想
        if self.stream:
            prev_idx = len(self.stream) - 1
            self.stream[prev_idx].associations.append(len(self.stream))
            thought.associations.append(prev_idx)
        self.stream.append(thought)
        if len(self.stream) > self.max_length:
            self.stream.pop(0)

    def get_recent(self, n: int = 5) -> List[Thought]:
        return self.stream[-n:]

    def step(self, dt: float = 1.0):
        self.time += dt


class ThoughtSystem:
    """
    思考系统

    整合思考空间(并行竞争)和思维流(序列记录)。
    支持系统1(快速)和系统2(缓慢)两种思考模式。
    """

    def __init__(self, space_capacity: int = 30, vector_dim: int = 128,
                 stream_length: int = 100, thinking_speed: float = 1.0):
        self.space = ThoughtSpace(capacity=space_capacity, vector_dim=vector_dim)
        self.stream = ThoughtStream(max_length=stream_length)
        self.thinking_speed = thinking_speed
        self.current_thought: Optional[Thought] = None
        self.time = 0.0
        self.system2_active = False
        self.system2_steps = 0

    def input_perceptual(self, content: np.ndarray, strength: float = 0.6):
        """输入感知思维"""
        self.space.submit(Thought(
            content=content, thought_type=ThoughtType.PERCEPTUAL,
            strength=strength, source="perception"
        ))

    def input_memory(self, content: np.ndarray, strength: float = 0.4):
        """输入记忆思维"""
        self.space.submit(Thought(
            content=content, thought_type=ThoughtType.MEMORY,
            strength=strength, source="memory"
        ))

    def input_emotional(self, content: np.ndarray, strength: float = 0.5):
        """输入情绪思维"""
        self.space.submit(Thought(
            content=content, thought_type=ThoughtType.EMOTIONAL,
            strength=strength, source="emotion"
        ))

    def input_goal(self, content: np.ndarray, strength: float = 0.7):
        """输入目标导向思维"""
        self.space.submit(Thought(
            content=content, thought_type=ThoughtType.GOAL_DIRECTED,
            strength=strength, source="goal"
        ))

    def generate_thought(self, trigger: np.ndarray = None,
                         thought_type: ThoughtType = ThoughtType.ABSTRACT,
                         strength: float = 0.4) -> Thought:
        """生成一个新思维(可由外部触发)"""
        if trigger is not None:
            content = trigger + np.random.randn(*trigger.shape) * 0.1
        else:
            content = np.random.randn(self.space.vector_dim) * 0.1
            norm = np.linalg.norm(content)
            if norm > 0:
                content = content / norm * 0.3

        thought = Thought(
            content=content, thought_type=thought_type,
            strength=strength, source="internal"
        )
        self.space.submit(thought)
        return thought

    def activate_system2(self, steps: int = 3):
        """激活系统2(深度思考)"""
        self.system2_active = True
        self.system2_steps = steps

    def step(self, dt: float = 1.0) -> Dict[str, Any]:
        """
        思考一步

        Returns:
            dict with 'conscious_thought', 'space_size', 'system2'
        """
        self.time += dt

        # 系统2: 生成额外的推理思维
        if self.system2_active and self.system2_steps > 0:
            self.system2_steps -= 1
            # 链式思考: 基于当前最强思维生成下一个
            current = self.space.select_for_consciousness()
            if current is not None:
                next_content = current.content + np.random.randn(
                    *current.content.shape) * 0.05
                self.space.submit(Thought(
                    content=next_content,
                    thought_type=ThoughtType.ABSTRACT,
                    strength=current.strength * 0.8,
                    source="system2"
                ))
            if self.system2_steps == 0:
                self.system2_active = False

        # 选择进入意识的思维
        conscious = self.space.select_for_consciousness()
        if conscious and conscious.strength > 0.3:
            self.current_thought = conscious
            self.stream.add(conscious)

        # 衰减
        self.space.step(dt)
        self.stream.step(dt)

        return {
            'conscious_thought': self.current_thought,
            'space_size': len(self.space.thoughts),
            'stream_length': len(self.stream.stream),
            'system2': self.system2_active,
            'thought_type': self.current_thought.thought_type.value
            if self.current_thought else None,
        }

    def get_current_thought(self) -> Optional[Thought]:
        return self.current_thought

    def get_summary(self) -> Dict[str, Any]:
        return {
            'current_type': self.current_thought.thought_type.value
            if self.current_thought else None,
            'current_strength': round(self.current_thought.strength, 3)
            if self.current_thought else 0,
            'space_size': len(self.space.thoughts),
            'stream_length': len(self.stream.stream),
            'system2_active': self.system2_active,
        }
