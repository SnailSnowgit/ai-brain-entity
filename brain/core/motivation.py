"""
内在动机模块 (Intrinsic Motivation)

基于自我决定理论(SDT)和好奇心驱动:
  - 好奇心(Curiosity): 对新颖/未知信息的探索欲望
  - 胜任感(Competence): 完成任务/解决问题的满足感
  - 自主性(Autonomy): 自我主导的需求
  - 社交连接(Social): 与他人互动的需求
  - 确定性(Certainty): 降低不确定性的需求

内在动机产生内在奖励, 通过多巴胺系统驱动学习和探索。
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


class DriveType(Enum):
    """动机类型"""
    CURIOSITY = "好奇心"
    COMPETENCE = "胜任感"
    AUTONOMY = "自主性"
    SOCIAL = "社交"
    CERTAINTY = "确定性"
    AVOIDANCE = "回避"


@dataclass
class Drive:
    """单个动机"""
    type: DriveType
    level: float = 0.5       # 当前强度 0-1
    baseline: float = 0.5    # 基线水平
    decay: float = 0.01      # 衰减率
    weight: float = 1.0      # 权重

    def satisfy(self, amount: float):
        """满足动机"""
        self.level = float(np.clip(self.level - amount * 0.3, 0.0, 1.0))

    def deprivate(self, amount: float):
        """剥夺(增强动机)"""
        self.level = float(np.clip(self.level + amount * 0.2, 0.0, 1.0))

    def step(self, dt: float = 1.0):
        """回归基线"""
        self.level += (self.baseline - self.level) * self.decay * dt


class CuriosityEngine:
    """
    好奇心引擎

    基于预测误差: 适度的意外激发好奇心, 过大意外导致恐惧。
    好奇心 = f(新颖度, 复杂度, 可学习性)
    """

    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.familiarity: Dict[str, float] = {}  # 熟悉度记忆
        self.novelty_threshold = 0.3
        self.complexity_optimal = 0.5  # 最优复杂度(金发姑娘区)

    def assess(self, stimulus_key: str, complexity: float = 0.5,
               prediction_error: float = 0.0) -> float:
        """
        评估好奇心水平

        Args:
            stimulus_key: 刺激标识
            complexity: 复杂度 0-1
            prediction_error: 预测误差 0-1

        Returns:
            好奇心水平 0-1
        """
        # 新颖度: 不熟悉的更新奇
        familiarity = self.familiarity.get(stimulus_key, 0.0)
        novelty = 1.0 - familiarity

        # 金发姑娘效应: 太简单无聊, 太难挫败
        complexity_fit = 1.0 - abs(complexity - self.complexity_optimal) * 2

        # 预测误差贡献: 适度意外最有趣
        pe_contribution = prediction_error * (1.0 - prediction_error) * 4

        curiosity = (0.4 * novelty + 0.3 * complexity_fit +
                     0.3 * pe_contribution)
        curiosity = float(np.clip(curiosity, 0.0, 1.0))

        # 更新熟悉度
        self.familiarity[stimulus_key] = min(
            1.0, familiarity + self.learning_rate)

        return curiosity

    def get_intrinsic_reward(self, curiosity: float) -> float:
        """好奇心转化为内在奖励"""
        return curiosity * 0.5


class MotivationSystem:
    """
    内在动机系统

    管理多个动机, 产生内在奖励信号, 驱动行为选择。
    """

    def __init__(self):
        self.drives: Dict[DriveType, Drive] = {
            DriveType.CURIOSITY: Drive(DriveType.CURIOSITY, baseline=0.6, weight=1.2),
            DriveType.COMPETENCE: Drive(DriveType.COMPETENCE, baseline=0.4, weight=1.0),
            DriveType.AUTONOMY: Drive(DriveType.AUTONOMY, baseline=0.3, weight=0.8),
            DriveType.SOCIAL: Drive(DriveType.SOCIAL, baseline=0.4, weight=0.9),
            DriveType.CERTAINTY: Drive(DriveType.CERTAINTY, baseline=0.5, weight=1.0),
            DriveType.AVOIDANCE: Drive(DriveType.AVOIDANCE, baseline=0.2, weight=1.5),
        }
        self.curiosity_engine = CuriosityEngine()
        self.total_reward = 0.0
        self.time = 0.0

    def get_dominant(self) -> Tuple[DriveType, float]:
        """获取当前主导动机"""
        strongest = max(self.drives.values(),
                        key=lambda d: d.level * d.weight)
        return strongest.type, strongest.level * strongest.weight

    def evaluate(self, user_input: str = None,
                 prediction_error: float = 0.0,
                 task_completed: bool = False,
                 social_interaction: bool = False,
                 threat_detected: bool = False) -> Dict[str, float]:
        """
        评估当前情境, 更新动机, 返回内在奖励

        Returns:
            dict with 'reward', 'curiosity', 'dominant_drive', 'explore_prob'
        """
        rewards = {}

        # 好奇心
        if user_input:
            complexity = min(1.0, len(user_input) / 200)
            key = str(hash(user_input[:50]))
            curiosity = self.curiosity_engine.assess(
                key, complexity=complexity,
                prediction_error=prediction_error)
            self.drives[DriveType.CURIOSITY].satisfy(curiosity * 0.3)
            rewards['curiosity'] = self.curiosity_engine.get_intrinsic_reward(curiosity)
        else:
            curiosity = self.drives[DriveType.CURIOSITY].level
            rewards['curiosity'] = 0.0

        # 胜任感: 完成任务
        if task_completed:
            self.drives[DriveType.COMPETENCE].satisfy(0.5)
            rewards['competence'] = 0.4
        else:
            rewards['competence'] = 0.0

        # 社交
        if social_interaction:
            self.drives[DriveType.SOCIAL].satisfy(0.3)
            rewards['social'] = 0.2
        else:
            rewards['social'] = 0.0

        # 确定性: 预测误差低=确定性高
        certainty_reward = (1.0 - prediction_error) * 0.1
        self.drives[DriveType.CERTAINTY].satisfy(prediction_error * 0.1)
        rewards['certainty'] = certainty_reward

        # 回避: 威胁
        if threat_detected:
            self.drives[DriveType.AVOIDANCE].deprivate(0.5)
            rewards['avoidance'] = -0.3
        else:
            rewards['avoidance'] = 0.0

        # 自主性: 每次交互略微满足
        rewards['autonomy'] = 0.05 if user_input else 0.0

        total = sum(rewards.values())
        self.total_reward += total

        # 探索概率: 好奇心高时更倾向探索
        dom_type, dom_level = self.get_dominant()
        explore_prob = float(np.clip(
            self.drives[DriveType.CURIOSITY].level * 0.5 +
            self.drives[DriveType.AUTONOMY].level * 0.2,
            0.05, 0.95
        ))

        return {
            'reward': float(np.clip(total, -1.0, 1.0)),
            'curiosity': curiosity,
            'dominant_drive': dom_type.value,
            'dominant_level': round(dom_level, 3),
            'explore_prob': explore_prob,
        }

    def step(self, dt: float = 1.0):
        """动机衰减/回归基线"""
        self.time += dt
        for drive in self.drives.values():
            drive.step(dt)

    def get_state(self) -> Dict[str, float]:
        return {dt.value.lower(): round(d.level, 3)
                for dt, d in self.drives.items()}
