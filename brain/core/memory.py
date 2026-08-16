"""
类脑记忆模块 (Memory System)

三级记忆架构:
  感觉缓冲(SensoryBuffer)  → 短期记忆(ShortTermMemory) → 长期记忆(LongTermMemory)
  容量200, 持续数秒          容量100, 持续数十秒           容量5000, 持久

特性:
  - 艾宾浩斯遗忘曲线
  - 情绪增强记忆(高唤醒=更强巩固)
  - 多巴胺门控(STP→LTP)
  - 联想检索
  - 记忆巩固
"""
import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import time


@dataclass
class MemoryItem:
    """单条记忆"""
    content: np.ndarray
    timestamp: float = 0.0
    strength: float = 0.5
    emotional_valence: float = 0.0
    retrieval_count: int = 0
    associations: List[int] = field(default_factory=list)

    def decay(self, dt: float, decay_rate: float = 0.001):
        """遗忘衰减"""
        self.strength *= np.exp(-decay_rate * dt)


class SensoryBuffer:
    """感觉缓冲: 大容量, 极短保持"""

    def __init__(self, capacity: int = 200, duration: float = 3.0):
        self.capacity = capacity
        self.duration = duration
        self.buffer: List[MemoryItem] = []
        self.current_time = 0.0

    def add(self, item: MemoryItem):
        self.buffer.append(item)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def step(self, dt: float) -> List[MemoryItem]:
        """返回转移到短期记忆的项"""
        self.current_time += dt
        transferred = []
        remaining = []
        for item in self.buffer:
            age = self.current_time - item.timestamp
            if age > self.duration:
                if item.strength > 0.3:
                    transferred.append(item)
            else:
                remaining.append(item)
        self.buffer = remaining
        return transferred


class ShortTermMemory:
    """短期记忆: 有限容量, 工作记忆"""

    def __init__(self, capacity: int = 100, decay_rate: float = 0.005,
                 consolidation_threshold: float = 0.6):
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.consolidation_threshold = consolidation_threshold
        self.memory: List[MemoryItem] = []
        self.current_time = 0.0

    def add(self, item: MemoryItem):
        self.memory.append(item)
        if len(self.memory) > self.capacity:
            # 移除最弱的
            self.memory.sort(key=lambda x: x.strength)
            self.memory.pop(0)

    def step(self, dt: float, dopamine: float = 0.3) -> List[MemoryItem]:
        """返回巩固到长期记忆的项"""
        self.current_time += dt
        consolidated = []
        remaining = []
        for item in self.memory:
            item.decay(dt, self.decay_rate)
            # 多巴胺增强巩固
            threshold = self.consolidation_threshold * (1.0 - dopamine * 0.5)
            if item.strength >= threshold and item.retrieval_count >= 1:
                consolidated.append(item)
            elif item.strength > 0.05:
                remaining.append(item)
        self.memory = remaining
        return consolidated

    def retrieve(self, cue: np.ndarray, top_k: int = 3) -> List[MemoryItem]:
        """检索最相关的记忆"""
        if not self.memory:
            return []
        scores = []
        for item in self.memory:
            if item.content.shape == cue.shape:
                sim = float(np.dot(cue, item.content) /
                            (np.linalg.norm(cue) * np.linalg.norm(item.content) + 1e-8))
            else:
                sim = 0.0
            scores.append((sim, item))
        scores.sort(key=lambda x: x[0], reverse=True)
        results = []
        for sim, item in scores[:top_k]:
            item.retrieval_count += 1
            item.strength = min(1.0, item.strength + 0.1)  # 检索增强
            results.append(item)
        return results


class LongTermMemory:
    """长期记忆: 大容量, 持久存储"""

    def __init__(self, capacity: int = 5000, forgetting_rate: float = 0.0001,
                 consolidation_rate: float = 0.01):
        self.capacity = capacity
        self.forgetting_rate = forgetting_rate
        self.consolidation_rate = consolidation_rate
        self.memory: List[MemoryItem] = []
        self.semantic_network: Dict[int, List[int]] = {}
        self.current_time = 0.0

    def store(self, item: MemoryItem):
        item.strength = min(1.0, item.strength)
        self.memory.append(item)
        idx = len(self.memory) - 1
        # 情绪增强
        if abs(item.emotional_valence) > 0.5:
            item.strength = min(1.0, item.strength + 0.2)
        if len(self.memory) > self.capacity:
            self.memory.sort(key=lambda x: x.strength)
            self.memory.pop(0)

    def retrieve(self, cue: np.ndarray, top_k: int = 5) -> List[MemoryItem]:
        if not self.memory:
            return []
        scored = []
        for item in self.memory:
            if item.content.shape == cue.shape:
                sim = float(np.dot(cue, item.content) /
                            (np.linalg.norm(cue) * np.linalg.norm(item.content) + 1e-8))
            else:
                sim = 0.0
            # 强度加权
            score = sim * (0.5 + 0.5 * item.strength)
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, item in scored[:top_k]:
            item.retrieval_count += 1
            item.strength = min(1.0, item.strength + 0.05)
            results.append(item)
        return results

    def consolidate(self):
        """记忆巩固: 突触归一化, 强化重要记忆"""
        for item in self.memory:
            if item.retrieval_count > 2:
                item.strength = min(1.0, item.strength + self.consolidation_rate)
            item.decay(1.0, self.forgetting_rate)

    def step(self, dt: float):
        self.current_time += dt
        for item in self.memory:
            item.decay(dt, self.forgetting_rate * 0.1)


class MemorySystem:
    """三级记忆系统"""

    def __init__(self, sensory_buffer_size: int = 200,
                 stm_size: int = 100, ltm_size: int = 5000):
        self.sensory_buffer = SensoryBuffer(capacity=sensory_buffer_size)
        self.short_term = ShortTermMemory(capacity=stm_size)
        self.long_term = LongTermMemory(capacity=ltm_size)
        self.current_time = 0.0
        self.sensory_to_stm_threshold = 0.3
        self.stm_to_ltm_threshold = 0.6

    def input_sensory(self, pattern: np.ndarray, emotional_valence: float = 0.0):
        """输入感觉记忆"""
        item = MemoryItem(
            content=pattern.copy(),
            timestamp=self.current_time,
            strength=0.4 + abs(emotional_valence) * 0.3,
            emotional_valence=emotional_valence
        )
        self.sensory_buffer.add(item)

    def retrieve(self, cue: np.ndarray, top_k: int = 3) -> Optional[MemoryItem]:
        """检索记忆(先短期后长期)"""
        stm_results = self.short_term.retrieve(cue, top_k=1)
        if stm_results:
            return stm_results[0]
        ltm_results = self.long_term.retrieve(cue, top_k=1)
        if ltm_results:
            return ltm_results[0]
        return None

    def step(self, dt: float = 1.0, dopamine_level: float = 0.3):
        """记忆流转: 感觉→短期→长期"""
        self.current_time += dt

        # 感觉 → 短期
        transferred = self.sensory_buffer.step(dt)
        for item in transferred:
            if item.strength >= self.sensory_to_stm_threshold:
                self.short_term.add(item)

        # 短期 → 长期
        consolidated = self.short_term.step(dt, dopamine=dopamine_level)
        for item in consolidated:
            self.long_term.store(item)

        # 长期巩固
        self.long_term.step(dt)

    def get_stats(self) -> Dict[str, int]:
        return {
            'sensory_buffer_count': len(self.sensory_buffer.buffer),
            'stm_count': len(self.short_term.memory),
            'ltm_count': len(self.long_term.memory),
        }
