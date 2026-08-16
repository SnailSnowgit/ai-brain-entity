"""
稳态调节系统 (Homeostasis)

维持内部平衡:
  - 能量: 随认知活动消耗, 休息/睡眠恢复
  - 疲劳: 随清醒时间累积, 睡眠重置
  - 昼夜节律: 24h周期, 调制激素基线和警觉度
  - 内感受: 身体状态信号 → 影响情绪和动机
  - 需求系统: 饥饿/口渴/社交/认知需求

与动机系统联动: 稳态失衡产生驱动力(饥饿→觅食, 疲劳→休息)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto


class NeedType(Enum):
    """生理/心理需求"""
    ENERGY = "energy"           # 能量
    REST = "rest"               # 休息
    SOCIAL = "social"           # 社交
    COGNITIVE = "cognitive"     # 认知刺激
    SAFETY = "safety"           # 安全
    AUTONOMY = "autonomy"       # 自主


@dataclass
class Need:
    """单个需求"""
    type: NeedType
    level: float = 0.5      # 当前满足度 0-1 (1=完全满足)
    optimal: float = 0.7    # 最优水平
    decay_rate: float = 0.005  # 自然衰减率
    weight: float = 1.0     # 重要性权重

    @property
    def deficit(self) -> float:
        """缺失程度(驱动强度)"""
        return max(0, self.optimal - self.level) * self.weight

    @property
    def is_satisfied(self) -> bool:
        return self.level >= self.optimal * 0.9


class CircadianRhythm:
    """
    昼夜节律(24小时周期)

    基于简化的双过程模型:
    - 过程S(睡眠压力): 清醒时累积, 睡眠时消散
    - 过程C(昼夜节律): 24h正弦波, 调制警觉度
    """

    def __init__(self, start_hour: float = 8.0):
        self.hour = start_hour  # 当前时刻(小时)
        self.cycle_length = 24.0

    def advance(self, hours: float = 0.1):
        """推进时间"""
        self.hour = (self.hour + hours) % self.cycle_length

    @property
    def alertness(self) -> float:
        """警觉度 0-1 (双过程模型)"""
        # 过程C: 昼夜节律(9点最高, 3点最低)
        phase = (self.hour - 9.0) / 24.0 * 2 * np.pi
        circadian = 0.5 + 0.5 * np.cos(phase)
        return float(np.clip(circadian, 0, 1))

    @property
    def melatonin(self) -> float:
        """褪黑素水平(夜间高)"""
        phase = (self.hour - 3.0) / 24.0 * 2 * np.pi
        return float(np.clip(0.5 + 0.5 * np.cos(phase), 0, 1))

    @property
    def cortisol_level(self) -> float:
        """皮质醇(早晨高)"""
        phase = (self.hour - 9.0) / 24.0 * 2 * np.pi
        return float(np.clip(0.5 + 0.5 * np.cos(phase), 0, 1))

    @property
    def is_night(self) -> bool:
        return self.hour < 6.0 or self.hour >= 22.0


class HomeostaticSystem:
    """
    稳态调节系统

    管理能量、疲劳、昼夜节律和需求满足。
    稳态失衡产生内感受信号, 驱动动机和情绪。
    """

    def __init__(self, start_hour: float = 8.0):
        # 能量与疲劳
        self.energy = 1.0          # 0-1
        self.fatigue = 0.0         # 0-1
        self.sleep_pressure = 0.0  # 过程S
        self.arousal_level = 0.5   # 生理唤醒

        # 昼夜节律
        self.circadian = CircadianRhythm(start_hour)

        # 需求系统
        self.needs: Dict[NeedType, Need] = {
            NeedType.ENERGY: Need(NeedType.ENERGY, level=0.8, decay_rate=0.008, weight=1.5),
            NeedType.REST: Need(NeedType.REST, level=0.7, decay_rate=0.005, weight=1.3),
            NeedType.SOCIAL: Need(NeedType.SOCIAL, level=0.5, decay_rate=0.003, weight=0.8),
            NeedType.COGNITIVE: Need(NeedType.COGNITIVE, level=0.5, decay_rate=0.004, weight=1.0),
            NeedType.SAFETY: Need(NeedType.SAFETY, level=0.8, decay_rate=0.001, weight=2.0),
            NeedType.AUTONOMY: Need(NeedType.AUTONOMY, level=0.6, decay_rate=0.002, weight=0.7),
        }

        # 代谢参数
        self.base_metabolic_rate = 0.002   # 基础消耗/步
        self.cognitive_cost = 0.004        # 深度思考额外消耗
        self.energy_recovery_rate = 0.02   # 休息恢复
        self.sleep_recovery_rate = 0.05    # 睡眠恢复
        self.fatigue_rate = 0.003          # 疲劳累积/步
        self.fatigue_recovery = 0.04       # 睡眠时疲劳恢复

        # 内感受信号
        self.interoception: Dict[str, float] = {
            "hunger": 0.0,
            "tiredness": 0.0,
            "stress": 0.0,
            "comfort": 0.5,
        }

        # 统计
        self.time_awake = 0.0
        self.activity_level = 0.5

    def expend_energy(self, amount: float):
        """消耗能量(认知活动/运动)"""
        self.energy = max(0.0, self.energy - amount)
        self.activity_level = float(np.clip(self.activity_level + amount * 2, 0, 1))

    def rest(self, amount: float = 0.1):
        """休息恢复能量"""
        self.energy = min(1.0, self.energy + self.energy_recovery_rate * amount)
        self.activity_level *= 0.9

    def satisfy_need(self, need_type: NeedType, amount: float = 0.2):
        """满足需求"""
        if need_type in self.needs:
            need = self.needs[need_type]
            need.level = float(np.clip(need.level + amount, 0, 1))

    def _update_interoception(self):
        """更新内感受信号"""
        self.interoception["hunger"] = 1.0 - self.needs[NeedType.ENERGY].level
        self.interoception["tiredness"] = self.fatigue
        self.interoception["stress"] = float(np.clip(
            self.fatigue * 0.3 + (1 - self.needs[NeedType.SAFETY].level) * 0.7, 0, 1))
        self.interoception["comfort"] = float(np.clip(
            self.energy * 0.3 +
            self.needs[NeedType.SAFETY].level * 0.3 +
            self.needs[NeedType.SOCIAL].level * 0.2 +
            self.circadian.alertness * 0.2, 0, 1))

    def get_drive_signals(self) -> Dict[str, float]:
        """
        获取稳态驱动信号(传给动机系统)

        Returns:
            驱动信号dict, 可直接加到动机评估中
        """
        signals = {}
        for need_type, need in self.needs.items():
            signals[need_type.value] = need.deficit

        # 疲劳→休息驱动
        signals["rest"] = self.fatigue * 1.5
        # 昼夜节律影响
        signals["alertness"] = self.circadian.alertness
        # 能量低→觅食/节能
        if self.energy < 0.3:
            signals["conserve_energy"] = (0.3 - self.energy) * 3

        return signals

    def get_modulation(self) -> Dict[str, float]:
        """
        获取对其他模块的调制参数

        Returns:
            调制参数(影响认知精度/情绪基线/学习率等)
        """
        # 疲劳降低认知精度
        cognitive_efficiency = float(np.clip(
            1.0 - self.fatigue * 0.5 - (1 - self.energy) * 0.3, 0.2, 1.0))

        # 昼夜节律调制
        circadian_factor = self.circadian.alertness

        # 综合认知能力
        cognitive_capacity = cognitive_efficiency * (0.5 + 0.5 * circadian_factor)

        return {
            "cognitive_efficiency": cognitive_efficiency,
            "cognitive_capacity": float(np.clip(cognitive_capacity, 0.1, 1.0)),
            "arousal_modulation": float(np.clip(
                self.circadian.alertness * 0.5 + self.activity_level * 0.5, 0, 1)),
            "learning_rate_mod": float(np.clip(
                cognitive_efficiency * (1.0 - self.fatigue * 0.3), 0.2, 1.0)),
            "emotional_baseline_valence": float(np.clip(
                self.interoception["comfort"] * 0.4 -
                self.interoception["stress"] * 0.3, -0.5, 0.5)),
            "melatonin": self.circadian.melatonin,
            "cortisol_circadian": self.circadian.cortisol_level,
        }

    def step(self, dt: float = 1.0, is_sleeping: bool = False,
             cognitive_demand: float = 0.3, social_interaction: bool = False,
             threat: bool = False) -> Dict:
        """
        稳态步进

        Args:
            dt: 时间步长(小时)
            is_sleeping: 是否在睡眠
            cognitive_demand: 认知负荷 0-1
            social_interaction: 是否有社交
            threat: 是否有威胁
        """
        # 推进昼夜节律
        self.circadian.advance(dt * 0.1)  # 每步约6分钟

        if is_sleeping:
            # 睡眠: 恢复能量, 消除疲劳
            self.energy = min(1.0, self.energy + self.sleep_recovery_rate * dt)
            self.fatigue = max(0.0, self.fatigue - self.fatigue_recovery * dt)
            self.sleep_pressure = max(0.0, self.sleep_pressure - 0.1 * dt)
            self.activity_level *= 0.95
            self.time_awake = 0.0
        else:
            # 清醒: 消耗能量, 累积疲劳
            energy_cost = (self.base_metabolic_rate +
                          cognitive_demand * self.cognitive_cost) * dt
            self.energy = max(0.0, self.energy - energy_cost)
            self.fatigue = min(1.0, self.fatigue + self.fatigue_rate * dt *
                              (1 + cognitive_demand * 0.5))
            self.sleep_pressure = min(1.0, self.sleep_pressure + 0.005 * dt)
            self.time_awake += dt
            self.activity_level = float(np.clip(
                self.activity_level * 0.95 + cognitive_demand * 0.05, 0, 1))

        # 需求自然衰减
        for need in self.needs.values():
            need.level = max(0.0, need.level - need.decay_rate * dt)

        # 社交满足
        if social_interaction:
            self.satisfy_need(NeedType.SOCIAL, 0.05 * dt)
            self.satisfy_need(NeedType.AUTONOMY, 0.01 * dt)

        # 安全需求
        if threat:
            self.needs[NeedType.SAFETY].level = max(
                0.0, self.needs[NeedType.SAFETY].level - 0.1 * dt)
        else:
            self.satisfy_need(NeedType.SAFETY, 0.02 * dt)

        # 认知需求(思考时满足)
        if cognitive_demand > 0.3:
            self.satisfy_need(NeedType.COGNITIVE, 0.03 * dt)

        # 能量低时认知需求也降
        if self.energy < 0.3:
            self.needs[NeedType.COGNITIVE].level *= 0.95

        # 更新内感受
        self._update_interoception()

        return {
            "energy": round(self.energy, 3),
            "fatigue": round(self.fatigue, 3),
            "sleep_pressure": round(self.sleep_pressure, 3),
            "circadian_alertness": round(self.circadian.alertness, 3),
            "hour": round(self.circadian.hour, 1),
            "needs": {n.type.value: round(n.level, 3) for n in self.needs.values()},
            "interoception": {k: round(v, 3) for k, v in self.interoception.items()},
            "modulation": self.get_modulation(),
        }

    def get_summary(self) -> Dict:
        return {
            "energy": round(self.energy, 3),
            "fatigue": round(self.fatigue, 3),
            "hour": round(self.circadian.hour, 1),
            "alertness": round(self.circadian.alertness, 3),
            "is_night": self.circadian.is_night,
            "needs": {n.type.value: round(n.level, 3) for n in self.needs.values()},
            "dominant_need": max(self.needs.values(),
                                key=lambda n: n.deficit).type.value,
        }
