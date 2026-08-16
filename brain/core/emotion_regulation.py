"""
情绪调节系统 (Emotion Regulation)

不只是产生情绪, 还能主动调节情绪。

调节策略(Gross过程模型):
  1. 情境选择(Situation Selection): 接近/回避引发情绪的情境
  2. 情境修正(Situation Modification): 改变情境
  3. 注意力部署(Attention Deployment): 转移注意力/反刍
  4. 认知重评(Cognitive Reappraisal): 重新解释事件意义
  5. 表达抑制(Response Suppression): 抑制情绪表达(消耗认知资源)

额外策略:
  - 正念/接纳(Mindfulness): 观察情绪不评判
  - 社会支持寻求(Social Support)
  - 自我安抚(Self-soothing)
  - 反刍(Rumination): 消极重复思考(适应不良)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import time as _time


class RegulationStrategy(Enum):
    """情绪调节策略"""
    COGNITIVE_REAPPRAISAL = ("认知重评", 0.7, 0.3)    # (名称, 效果, 认知消耗)
    ATTENTION_SHIFT = ("注意力转移", 0.5, 0.2)
    EXPRESSIVE_SUPPRESSION = ("表达抑制", 0.4, 0.5)
    MINDFULNESS = ("正念接纳", 0.6, 0.2)
    SOCIAL_SUPPORT = ("社会支持", 0.6, 0.1)
    SITUATION_SELECTION = ("情境选择", 0.8, 0.1)
    SELF_SOOTHING = ("自我安抚", 0.4, 0.1)
    RUMINATION = ("反刍", -0.3, 0.3)       # 适应不良
    ACCEPTANCE = ("接纳", 0.5, 0.15)
    DISTRACTION = ("分心", 0.4, 0.15)

    def __init__(self, label, effectiveness, cost):
        self.label = label
        self.effectiveness = effectiveness
        self.cost = cost


@dataclass
class EmotionEvent:
    """情绪事件"""
    trigger: str
    emotion_type: str
    intensity: float
    valence: float
    timestamp: float
    regulated: bool = False
    strategy_used: Optional[str] = None
    regulation_success: float = 0.0


class EmotionRegulation:
    """
    情绪调节系统

    根据情绪类型、强度和认知资源选择调节策略,
    并评估调节效果。
    """

    # 情绪类型→推荐策略映射
    STRATEGY_AFFINITY = {
        "恐惧": [RegulationStrategy.COGNITIVE_REAPPRAISAL,
                RegulationStrategy.ATTENTION_SHIFT,
                RegulationStrategy.SOCIAL_SUPPORT],
        "愤怒": [RegulationStrategy.COGNITIVE_REAPPRAISAL,
                RegulationStrategy.MINDFULNESS,
                RegulationStrategy.EXPRESSIVE_SUPPRESSION],
        "悲伤": [RegulationStrategy.SOCIAL_SUPPORT,
                RegulationStrategy.SELF_SOOTHING,
                RegulationStrategy.ACCEPTANCE],
        "厌恶": [RegulationStrategy.ATTENTION_SHIFT,
                RegulationStrategy.SITUATION_SELECTION],
        "焦虑": [RegulationStrategy.MINDFULNESS,
                RegulationStrategy.COGNITIVE_REAPPRAISAL,
                RegulationStrategy.DISTRACTION],
        "喜悦": [RegulationStrategy.ACCEPTANCE,
                RegulationStrategy.SITUATION_SELECTION],
    }

    def __init__(self):
        # 策略使用经验(策略→成功率)
        self.strategy_experience: Dict[RegulationStrategy, List[bool]] = {
            s: [] for s in RegulationStrategy
        }

        # 情绪事件历史
        self.emotion_history: List[EmotionEvent] = []
        self.max_history = 200

        # 当前调节状态
        self.current_strategy: Optional[RegulationStrategy] = None
        self.regulation_active: bool = False
        self.regulation_duration: int = 0
        self.cognitive_cost: float = 0.0

        # 长期指标
        self.emotional_stability: float = 0.5  # 情绪稳定性
        self.regulation_skill: float = 0.3     # 调节技能(随练习提升)
        self.rumination_tendency: float = 0.2  # 反刍倾向
        self.wellbeing: float = 0.5            # 心理健康

        # 反刍状态
        self.ruminating: bool = False
        self.rumination_target: Optional[str] = None
        self.rumination_count: int = 0

    def assess_emotion(self, emotion_type: str, intensity: float,
                       valence: float, cognitive_resources: float = 1.0,
                       social_available: bool = False) -> Dict:
        """
        评估情绪并决定是否需要调节

        Args:
            emotion_type: 情绪类型(恐惧/愤怒/悲伤等)
            intensity: 强度 0-1
            valence: 效价 -1~1
            cognitive_resources: 可用认知资源 0-1
            social_available: 是否有社交支持

        Returns:
            评估结果dict
        """
        # 记录事件
        event = EmotionEvent(
            trigger="",
            emotion_type=emotion_type,
            intensity=intensity,
            valence=valence,
            timestamp=_time.time(),
        )
        self.emotion_history.append(event)
        if len(self.emotion_history) > self.max_history:
            self.emotion_history.pop(0)

        # 判断是否需要调节
        needs_regulation = False
        reasons = []

        if intensity > 0.6:
            needs_regulation = True
            reasons.append("高强度")
        if valence < -0.4:
            needs_regulation = True
            reasons.append("负效价")
        if emotion_type in ("恐惧", "愤怒") and intensity > 0.4:
            needs_regulation = True
            reasons.append("高唤醒负面情绪")

        # 选择策略
        strategy = None
        if needs_regulation:
            strategy = self._select_strategy(
                emotion_type, intensity, cognitive_resources, social_available)

        return {
            "needs_regulation": needs_regulation,
            "reasons": reasons,
            "recommended_strategy": strategy.label if strategy else None,
            "intensity": intensity,
            "valence": valence,
        }

    def _select_strategy(self, emotion_type: str, intensity: float,
                         cognitive_resources: float,
                         social_available: bool) -> RegulationStrategy:
        """选择最优调节策略"""
        candidates = self.STRATEGY_AFFINITY.get(
            emotion_type,
            [RegulationStrategy.COGNITIVE_REAPPRAISAL,
             RegulationStrategy.MINDFULNESS]
        )

        # 评分每个候选策略
        best = None
        best_score = -float("inf")

        for strategy in candidates:
            # 基础效果
            score = strategy.effectiveness

            # 认知资源不足时, 选择低消耗策略
            if cognitive_resources < strategy.cost:
                score -= 0.5

            # 社交支持不可用时排除
            if strategy == RegulationStrategy.SOCIAL_SUPPORT and not social_available:
                score -= 1.0

            # 经验加成(过去成功的策略更可能被选)
            experience = self.strategy_experience[strategy]
            if experience:
                success_rate = sum(experience) / len(experience)
                score += 0.2 * success_rate

            # 调节技能加成
            score += 0.1 * self.regulation_skill

            # 高强度情绪倾向用快速策略
            if intensity > 0.7 and strategy.effectiveness > 0.6:
                score += 0.1

            # 反刍倾向
            if strategy == RegulationStrategy.RUMINATION:
                score += self.rumination_tendency * 0.3

            if score > best_score:
                best_score = score
                best = strategy

        return best or RegulationStrategy.MINDFULNESS

    def regulate(self, emotion_type: str, intensity: float, valence: float,
                 cognitive_resources: float = 1.0,
                 social_available: bool = False,
                 reappraisal_content: str = None) -> Dict:
        """
        执行情绪调节

        Args:
            emotion_type: 情绪类型
            intensity: 当前强度
            valence: 当前效价
            cognitive_resources: 认知资源
            social_available: 社交支持可用
            reappraisal_content: 重评内容(可选)

        Returns:
            调节结果: {new_intensity, new_valence, success, cost}
        """
        assessment = self.assess_emotion(
            emotion_type, intensity, valence,
            cognitive_resources, social_available)

        if not assessment["needs_regulation"]:
            return {
                "regulated": False,
                "new_intensity": intensity,
                "new_valence": valence,
                "success": 0.0,
            }

        strategy = self._select_strategy(
            emotion_type, intensity, cognitive_resources, social_available)

        self.current_strategy = strategy
        self.regulation_active = True
        self.regulation_duration += 1
        self.cognitive_cost = strategy.cost

        # 计算调节效果
        base_effect = strategy.effectiveness
        skill_mod = 1.0 + self.regulation_skill * 0.5
        resource_mod = min(1.0, cognitive_resources / max(0.1, strategy.cost))

        success = float(np.clip(
            base_effect * skill_mod * resource_mod *
            (0.7 + 0.3 * np.random.random()), 0, 1))

        # 应用调节
        new_intensity = intensity
        new_valence = valence

        if strategy == RegulationStrategy.COGNITIVE_REAPPRAISAL:
            # 重评: 降低强度, 改善效价
            new_intensity *= (1 - 0.5 * success)
            new_valence += 0.3 * success

        elif strategy == RegulationStrategy.ATTENTION_SHIFT:
            # 注意力转移: 降低强度
            new_intensity *= (1 - 0.4 * success)

        elif strategy == RegulationStrategy.EXPRESSIVE_SUPPRESSION:
            # 抑制: 外在表达降低但内在强度可能反弹
            new_intensity *= (1 - 0.3 * success)
            # 抑制消耗认知资源, 长期可能反弹
            if np.random.random() < 0.3:
                new_intensity = min(1.0, new_intensity * 1.2)

        elif strategy == RegulationStrategy.MINDFULNESS:
            # 正念: 降低强度, 不改变效价但减少次生情绪
            new_intensity *= (1 - 0.4 * success)

        elif strategy == RegulationStrategy.SOCIAL_SUPPORT:
            # 社会支持: 改善效价
            new_intensity *= (1 - 0.3 * success)
            new_valence += 0.4 * success

        elif strategy == RegulationStrategy.SELF_SOOTHING:
            new_intensity *= (1 - 0.3 * success)
            new_valence += 0.2 * success

        elif strategy == RegulationStrategy.RUMINATION:
            # 反刍: 适应不良, 增强负面情绪
            new_intensity = min(1.0, intensity * (1 + 0.3 * success))
            new_valence -= 0.2 * success
            self.ruminating = True
            self.rumination_count += 1

        elif strategy == RegulationStrategy.ACCEPTANCE:
            new_intensity *= (1 - 0.3 * success)

        elif strategy == RegulationStrategy.DISTRACTION:
            new_intensity *= (1 - 0.35 * success)

        new_intensity = float(np.clip(new_intensity, 0, 1))
        new_valence = float(np.clip(new_valence, -1, 1))

        # 更新经验
        self.strategy_experience[strategy].append(success > 0.5)

        # 更新长期指标
        self.regulation_skill = min(1.0, self.regulation_skill + 0.005 * success)
        if strategy != RegulationStrategy.RUMINATION:
            self.emotional_stability = min(
                1.0, self.emotional_stability + 0.003 * success)
            self.wellbeing = min(
                1.0, self.wellbeing + 0.002 * success)
        else:
            self.emotional_stability = max(0, self.emotional_stability - 0.005)
            self.wellbeing = max(0, self.wellbeing - 0.005)

        # 记录调节结果
        if self.emotion_history:
            self.emotion_history[-1].regulated = True
            self.emotion_history[-1].strategy_used = strategy.label
            self.emotion_history[-1].regulation_success = success

        return {
            "regulated": True,
            "strategy": strategy.label,
            "new_intensity": round(new_intensity, 3),
            "new_valence": round(new_valence, 3),
            "success": round(success, 3),
            "cognitive_cost": strategy.cost,
            "intensity_reduction": round(intensity - new_intensity, 3),
        }

    def stop_regulation(self):
        """停止当前调节"""
        self.regulation_active = False
        self.current_strategy = None
        self.regulation_duration = 0
        self.cognitive_cost = 0.0

    def step(self) -> Dict:
        """调节系统步进"""
        # 反刍自然消退
        if self.ruminating and np.random.random() < 0.1:
            self.ruminating = False

        # 情绪稳定性自然回归
        self.emotional_stability = float(np.clip(
            self.emotional_stability * 0.999 + 0.5 * 0.001, 0, 1))

        return {
            "active": self.regulation_active,
            "strategy": self.current_strategy.label if self.current_strategy else None,
            "ruminating": self.ruminating,
            "stability": round(self.emotional_stability, 3),
            "skill": round(self.regulation_skill, 3),
            "wellbeing": round(self.wellbeing, 3),
        }

    def get_summary(self) -> Dict:
        # 策略使用统计
        strategy_stats = {}
        for strategy, history in self.strategy_experience.items():
            if history:
                strategy_stats[strategy.label] = {
                    "uses": len(history),
                    "success_rate": round(sum(history) / len(history), 2),
                }

        return {
            "active": self.regulation_active,
            "current_strategy": (self.current_strategy.label
                                if self.current_strategy else None),
            "stability": round(self.emotional_stability, 3),
            "regulation_skill": round(self.regulation_skill, 3),
            "wellbeing": round(self.wellbeing, 3),
            "rumination_count": self.rumination_count,
            "strategy_stats": strategy_stats,
        }
