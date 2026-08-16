"""
睡眠与记忆重放 (Sleep & Memory Replay)

睡眠阶段:
  - 清醒(Wake): 正常认知
  - N1(浅睡): 入睡过渡, 片段化思维
  - N2(中睡): 睡眠纺锤波, 运动记忆巩固
  - N3(深睡): 慢波, 陈述性记忆巩固, 突触缩放
  - REM(快速眼动): 做梦, 情绪记忆处理, 创造性重组

功能:
  - 经验重放: 重放短期记忆, 巩固到长期
  - Hebbian重训练: 重放时更新预测编码权重
  - 突触缩放: 全局权重归一化(防止爆炸, 保留相对强度)
  - 记忆整合: 关联记忆连接, 图式构建
  - 情绪处理: REM中降低情绪记忆的情绪负荷
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import time as _time


class SleepStage(Enum):
    WAKE = "wake"
    N1 = "n1"
    N2 = "n2"
    N3 = "n3"
    REM = "rem"


@dataclass
class ReplayItem:
    """待重放的经验"""
    vector: np.ndarray
    emotional_valence: float
    emotional_arousal: float
    timestamp: float
    replay_count: int = 0
    importance: float = 0.5


class SleepSystem:
    """
    睡眠系统

    管理睡眠-觉醒周期, 在离线阶段重放记忆、巩固学习、归一化突触。
    """

    # 睡眠周期时长(模拟分钟)
    CYCLE_DURATION = 90.0
    # 各阶段比例
    STAGE_PROPORTIONS = {
        SleepStage.N1: 0.05,
        SleepStage.N2: 0.45,
        SleepStage.N3: 0.25,
        SleepStage.REM: 0.25,
    }

    def __init__(self,
                 replay_lr: float = 0.02,
                 consolidation_rate: float = 0.1,
                 synaptic_scale_target: float = 1.0,
                 synaptic_scale_rate: float = 0.05,
                 emotional_decay_in_rem: float = 0.3,
                 replay_batch_size: int = 10):
        self.replay_lr = replay_lr
        self.consolidation_rate = consolidation_rate
        self.scale_target = synaptic_scale_target
        self.scale_rate = synaptic_scale_rate
        self.emotional_rem_decay = emotional_decay_in_rem
        self.replay_batch = replay_batch_size

        # 睡眠状态
        self.stage = SleepStage.WAKE
        self.is_asleep = False
        self.sleep_time = 0.0
        self.cycle_position = 0.0
        self.cycles_completed = 0

        # 待重放的经验缓冲区
        self.replay_buffer: List[ReplayItem] = []
        self.max_buffer = 1000

        # 统计
        self.total_replays = 0
        self.total_consolidations = 0
        self.total_synaptic_scaling = 0
        self.last_sleep_duration = 0.0
        self.sleep_history: List[Dict] = []

        # 重放生成的内容(梦/想象)
        self.current_dream: Optional[np.ndarray] = None
        self.dream_content: List[str] = []

    def add_experience(self, vector: np.ndarray,
                       emotional_valence: float = 0.0,
                       emotional_arousal: float = 0.0,
                       importance: float = None):
        """添加待重放的经验"""
        if importance is None:
            importance = min(1.0, abs(emotional_valence) * 0.5 +
                           emotional_arousal * 0.3 + 0.2)
        self.replay_buffer.append(ReplayItem(
            vector=vector.copy(),
            emotional_valence=emotional_valence,
            emotional_arousal=emotional_arousal,
            timestamp=_time.time(),
            importance=importance,
        ))
        if len(self.replay_buffer) > self.max_buffer:
            # 保留重要的, 丢弃旧的不重要的
            self.replay_buffer.sort(
                key=lambda x: x.importance * (1 + x.replay_count * 0.1))
            self.replay_buffer = self.replay_buffer[-self.max_buffer:]

    def can_sleep(self, fatigue: float = 0.0, energy: float = 1.0) -> bool:
        """判断是否可以入睡"""
        return (not self.is_asleep and
                (fatigue > 0.5 or energy < 0.3))

    def fall_asleep(self):
        """入睡"""
        if self.is_asleep:
            return
        self.is_asleep = True
        self.stage = SleepStage.N1
        self.sleep_time = 0.0
        self.cycle_position = 0.0
        self.dream_content = []

    def wake_up(self):
        """醒来"""
        if not self.is_asleep:
            return
        self.last_sleep_duration = self.sleep_time
        self.sleep_history.append({
            "duration": self.sleep_time,
            "cycles": self.cycles_completed,
            "replays": self.total_replays,
        })
        self.is_asleep = False
        self.stage = SleepStage.WAKE
        self.current_dream = None

    def _update_stage(self):
        """更新睡眠阶段(90分钟周期)"""
        cycle_pos = (self.sleep_time % self.CYCLE_DURATION) / self.CYCLE_DURATION

        # 前半夜N3多, 后半夜REM多
        night_progress = min(1.0, self.sleep_time / (self.CYCLE_DURATION * 4))
        rem_boost = night_progress * 0.15

        if cycle_pos < 0.05:
            self.stage = SleepStage.N1
        elif cycle_pos < 0.50:
            self.stage = SleepStage.N2
        elif cycle_pos < 0.75 - rem_boost:
            self.stage = SleepStage.N3
        else:
            self.stage = SleepStage.REM

        # 周期计数
        new_cycles = int(self.sleep_time / self.CYCLE_DURATION)
        if new_cycles > self.cycles_completed:
            self.cycles_completed = new_cycles

    def _select_replay_batch(self) -> List[ReplayItem]:
        """选择重放批次(优先重要和近期)"""
        if not self.replay_buffer:
            return []

        # 重要性加权采样
        weights = np.array([
            item.importance * (1.0 / (1.0 + item.replay_count))
            for item in self.replay_buffer
        ])
        weights = weights / weights.sum()

        n = min(self.replay_batch, len(self.replay_buffer))
        indices = np.random.choice(
            len(self.replay_buffer), n, replace=False, p=weights)

        return [self.replay_buffer[i] for i in indices]

    def _replay_n3(self, pc_network, memory_system) -> int:
        """
        N3深睡: 陈述性记忆重放 + Hebbian重训练 + 突触缩放

        Returns:
            重放数量
        """
        batch = self._select_replay_batch()
        if not batch:
            return 0

        for item in batch:
            # 重放到预测编码网络
            target_dim = pc_network.layers[0].size
            vec = item.vector
            if vec.shape[0] < target_dim:
                padded = np.zeros(target_dim)
                padded[:vec.shape[0]] = vec
                vec = padded
            elif vec.shape[0] > target_dim:
                vec = vec[:target_dim]

            # 多次重放(重要的多放)
            n_replays = max(1, int(item.importance * 3))
            for _ in range(n_replays):
                pc_network.step(vec, dt=1.0)

            # 巩固到长期记忆
            if memory_system is not None:
                mem_vec = item.vector[:128] if item.vector.shape[0] >= 128 else item.vector
                memory_system.input_sensory(
                    mem_vec, emotional_valence=item.emotional_valence)
                memory_system.step(
                    dt=1.0, dopamine_level=0.3 + item.importance * 0.3)

            item.replay_count += 1
            self.total_replays += 1
            self.total_consolidations += 1

        # 突触缩放(全局归一化)
        self._synaptic_scaling(pc_network)

        return len(batch)

    def _replay_rem(self, pc_network, memory_system) -> int:
        """
        REM睡眠: 情绪记忆处理 + 创造性重组 + 做梦

        Returns:
            重放数量
        """
        batch = self._select_replay_batch()
        if not batch:
            return 0

        # REM: 重组记忆碎片(创造性)
        vectors = [item.vector for item in batch]
        if len(vectors) >= 2:
            # 随机组合两个记忆(梦的奇特性)
            idx1, idx2 = np.random.choice(len(vectors), 2, replace=False)
            mix_ratio = np.random.beta(0.5, 0.5)  # 偏向极端组合
            dream_vec = mix_ratio * vectors[idx1] + (1 - mix_ratio) * vectors[idx2]
            norm = np.linalg.norm(dream_vec)
            if norm > 0:
                dream_vec = dream_vec / norm
            self.current_dream = dream_vec

        for item in batch:
            # REM中降低情绪负荷
            if abs(item.emotional_valence) > 0.3 or item.emotional_arousal > 0.5:
                item.emotional_valence *= (1 - self.emotional_rem_decay)
                item.emotional_arousal *= (1 - self.emotional_rem_decay)
                item.importance *= 0.9  # 处理后重要性降低

            # 重放(学习率更低, 不做突触缩放)
            target_dim = pc_network.layers[0].size
            vec = item.vector
            if vec.shape[0] < target_dim:
                padded = np.zeros(target_dim)
                padded[:vec.shape[0]] = vec
                vec = padded
            pc_network.step(vec, dt=1.0)

            item.replay_count += 1
            self.total_replays += 1

        return len(batch)

    def _replay_n2(self, pc_network) -> int:
        """N2中睡: 程序性/运动记忆重放"""
        batch = self._select_replay_batch()
        for item in batch:
            target_dim = pc_network.layers[0].size
            vec = item.vector
            if vec.shape[0] < target_dim:
                padded = np.zeros(target_dim)
                padded[:vec.shape[0]] = vec
                vec = padded
            # N2重放更慢更精确
            pc_network.step(vec, dt=0.5)
            item.replay_count += 1
            self.total_replays += 1
        return len(batch)

    def _synaptic_scaling(self, pc_network):
        """
        突触缩放: 全局权重归一化

        保持权重相对模式, 但将总强度拉回目标值。
        这是防止Hebbian学习导致权重爆炸的关键稳态机制。
        """
        total_weight_sq = 0.0
        n_weights = 0
        for layer in pc_network.layers:
            total_weight_sq += float(np.sum(layer.top_down_weights ** 2))
            total_weight_sq += float(np.sum(layer.bottom_up_weights ** 2))
            n_weights += layer.top_down_weights.size
            n_weights += layer.bottom_up_weights.size

        if n_weights == 0:
            return

        current_scale = np.sqrt(total_weight_sq / n_weights)
        if current_scale < 1e-8:
            return

        # 向目标缩放
        scale_factor = 1.0 + self.scale_rate * (
            self.scale_target / current_scale - 1.0)
        scale_factor = float(np.clip(scale_factor, 0.8, 1.2))

        for layer in pc_network.layers:
            layer.top_down_weights *= scale_factor
            layer.bottom_up_weights *= scale_factor

        self.total_synaptic_scaling += 1

    def step(self, dt: float = 1.0, pc_network=None,
             memory_system=None) -> Dict:
        """
        睡眠步进

        Args:
            dt: 时间步长(模拟分钟)
            pc_network: 预测编码网络(用于重放)
            memory_system: 记忆系统(用于巩固)

        Returns:
            睡眠状态dict
        """
        if not self.is_asleep:
            return {"asleep": False, "stage": "wake"}

        self.sleep_time += dt
        self._update_stage()

        replays = 0
        # 不同阶段执行不同重放
        if self.stage == SleepStage.N3 and pc_network is not None:
            replays = self._replay_n3(pc_network, memory_system)
        elif self.stage == SleepStage.REM and pc_network is not None:
            replays = self._replay_rem(pc_network, memory_system)
        elif self.stage == SleepStage.N2 and pc_network is not None:
            replays = self._replay_n2(pc_network)

        return {
            "asleep": True,
            "stage": self.stage.value,
            "sleep_time": self.sleep_time,
            "cycles": self.cycles_completed,
            "replays_this_step": replays,
            "dreaming": self.stage == SleepStage.REM and self.current_dream is not None,
            "buffer_size": len(self.replay_buffer),
        }

    def sleep_cycle(self, pc_network=None, memory_system=None,
                    n_cycles: int = 4) -> Dict:
        """
        执行完整睡眠周期(阻塞式)

        Args:
            n_cycles: 睡眠周期数(默认4个周期=6小时)
        """
        self.fall_asleep()
        total_replays = 0
        stages_log = []

        total_minutes = self.CYCLE_DURATION * n_cycles
        steps = int(total_minutes / 5.0)  # 每步5分钟

        for _ in range(steps):
            result = self.step(dt=5.0, pc_network=pc_network,
                             memory_system=memory_system)
            total_replays += result.get("replays_this_step", 0)
            stages_log.append(result["stage"])

        self.wake_up()

        return {
            "duration": self.last_sleep_duration,
            "cycles": self.cycles_completed,
            "total_replays": total_replays,
            "consolidations": self.total_consolidations,
            "synaptic_scalings": self.total_synaptic_scaling,
            "stage_distribution": {
                s.value: stages_log.count(s.value) / len(stages_log)
                for s in SleepStage
            },
        }

    def get_summary(self) -> Dict:
        return {
            "is_asleep": self.is_asleep,
            "stage": self.stage.value,
            "sleep_time": round(self.sleep_time, 1),
            "cycles_completed": self.cycles_completed,
            "replay_buffer": len(self.replay_buffer),
            "total_replays": self.total_replays,
            "total_consolidations": self.total_consolidations,
            "total_synaptic_scaling": self.total_synaptic_scaling,
        }
