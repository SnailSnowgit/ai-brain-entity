"""
元认知与自我模型 (Metacognition & Self Model)

元认知: 对自己认知过程的认知
  - 元记忆(Metamemory): 知道自己知不知道(feeling of knowing)
  - 置信度校准: 对自己回答的确信程度
  - 错误检测: 发现自己犯错
  - 认知监控: 实时评估自己的认知状态

自我模型:
  - 自我概念: 对"我是谁"的表征
  - 自我叙事: 连贯的人生故事线
  - 能力自我评估: 擅长什么/不擅长什么
  - 自我效能感: 对能否完成任务的信念
  - 自尊: 对自我价值的整体评价
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import time as _time


class ConfidenceLevel(Enum):
    """置信度等级"""
    GUESSING = (0.0, 0.3, "猜测")
    UNCERTAIN = (0.3, 0.5, "不确定")
    FAIRLY_SURE = (0.5, 0.75, "比较确定")
    CONFIDENT = (0.75, 0.9, "有信心")
    CERTAIN = (0.9, 1.01, "非常确定")

    def __init__(self, lo, hi, label):
        self.lo = lo
        self.hi = hi
        self.label = label

    @classmethod
    def classify(cls, confidence: float) -> "ConfidenceLevel":
        for level in cls:
            if level.lo <= confidence < level.hi:
                return level
        return cls.UNCERTAIN


@dataclass
class SelfBelief:
    """自我信念"""
    domain: str               # 领域
    ability: float = 0.5      # 能力估计 0-1
    experience: int = 0       # 经验次数
    success: int = 0          # 成功次数
    failure: int = 0          # 失败次数
    last_updated: float = 0.0

    @property
    def success_rate(self) -> float:
        total = self.success + self.failure
        return self.success / total if total > 0 else 0.5

    def update(self, success: bool, learning_rate: float = 0.1):
        self.experience += 1
        if success:
            self.success += 1
        else:
            self.failure += 1
        # 能力估计向成功率靠拢
        target = self.success_rate
        self.ability += learning_rate * (target - self.ability)
        self.last_updated = _time.time()


@dataclass
class NarrativeEpisode:
    """叙事片段(自我故事中的一章)"""
    content: str
    valence: float
    importance: float
    timestamp: float
    tags: List[str] = field(default_factory=list)


class MetaMemory:
    """
    元记忆: 知道自己记不记得

    模拟:
      - FOK(Feeling of Knowing): 觉得自己知道但暂时想不起来
      - TOT(Tip of the Tongue): 话到嘴边现象
      - JOL(Judgment of Learning): 学习判断
    """

    def __init__(self):
        self.fok_history: List[Tuple[float, bool]] = []  # (FOK判断, 实际是否记得)
        self.jol_accuracy: List[Tuple[float, float]] = []  # (学习判断, 实际保持)

    def feeling_of_knowing(self, query_vector: np.ndarray,
                           memory_system, threshold: float = 0.3) -> Dict:
        """
        判断自己是否知道某个信息(不实际检索)

        Returns:
            {fok: 0-1, likely_known: bool, tot_state: bool}
        """
        # 基于线索熟悉度的FOK
        # 简化: 用查询向量与记忆系统中项目的相似度
        familiarity = 0.3  # 基础熟悉感

        # 检查短期记忆
        if hasattr(memory_system, 'short_term') and hasattr(memory_system.short_term, 'memory'):
            for item in memory_system.short_term.memory[-10:]:
                sim = float(np.dot(query_vector[:item.content.shape[0]],
                                   item.content[:query_vector.shape[0]]))
                familiarity = max(familiarity, abs(sim))

        # 检查长期记忆
        if hasattr(memory_system, 'long_term') and hasattr(memory_system.long_term, 'memory'):
            for item in memory_system.long_term.memory[-20:]:
                dim = min(query_vector.shape[0], item.content.shape[0])
                sim = float(np.dot(query_vector[:dim], item.content[:dim]))
                familiarity = max(familiarity, abs(sim))

        fok = float(np.clip(familiarity, 0, 1))
        likely_known = fok > threshold

        # TOT状态: 高FOK但检索困难(0.4-0.7区间)
        tot_state = 0.4 < fok < 0.7

        return {
            "fok": fok,
            "likely_known": likely_known,
            "tot_state": tot_state,
        }

    def judgment_of_learning(self, item_strength: float,
                             study_time: float) -> float:
        """学习判断: 觉得自己以后能记住多少"""
        # 基于编码强度和学习时间
        jol = item_strength * (1 - np.exp(-study_time * 0.5))
        return float(np.clip(jol, 0, 1))

    def update_calibration(self, predicted: float, actual: float):
        """更新置信度校准(预测vs实际)"""
        self.fok_history.append((predicted, actual))
        if len(self.fok_history) > 100:
            self.fok_history.pop(0)

    def get_calibration_error(self) -> float:
        """校准误差(越低越准确)"""
        if not self.fok_history:
            return 0.5
        preds = np.array([p for p, _ in self.fok_history])
        acts = np.array([a for _, a in self.fok_history])
        return float(np.mean(np.abs(preds - acts)))


class SelfModel:
    """
    自我模型

    维护自我概念、能力信念、自我叙事。
    """

    def __init__(self):
        # 自我概念(多维向量)
        self.self_concept = np.random.randn(32) * 0.1
        self.self_concept_norm = np.zeros(32)  # 规范自我(理想)

        # 能力信念
        self.beliefs: Dict[str, SelfBelief] = {
            "语言理解": SelfBelief("语言理解", ability=0.5),
            "问题解决": SelfBelief("问题解决", ability=0.5),
            "记忆": SelfBelief("记忆", ability=0.5),
            "情绪识别": SelfBelief("情绪识别", ability=0.5),
            "创造力": SelfBelief("创造力", ability=0.5),
            "社交": SelfBelief("社交", ability=0.5),
            "学习": SelfBelief("学习", ability=0.5),
            "推理": SelfBelief("推理", ability=0.5),
        }

        # 自我叙事
        self.narrative: List[NarrativeEpisode] = []
        self.narrative_theme = "我是一个正在学习和成长的认知系统。"

        # 自尊
        self.self_esteem = 0.5
        self.self_efficacy = 0.5

        # 自我连续感
        self.continuity = 0.5  # 过去自我与现在自我的连续感

        # 统计
        self.total_reflections = 0
        self.error_detections = 0
        self.self_corrections = 0

    def update_belief(self, domain: str, success: bool):
        """更新能力信念"""
        if domain in self.beliefs:
            self.beliefs[domain].update(success)
            # 自我效能感: 所有能力信念的加权平均
            abilities = [b.ability for b in self.beliefs.values()]
            self.self_efficacy = float(np.mean(abilities))
            # 自尊受成功/失败影响
            self.self_esteem += 0.05 * (0.2 if success else -0.15)
            self.self_esteem = float(np.clip(self.self_esteem, 0, 1))

    def add_narrative(self, content: str, valence: float = 0.0,
                      importance: float = 0.3, tags: List[str] = None):
        """添加叙事片段"""
        self.narrative.append(NarrativeEpisode(
            content=content, valence=valence, importance=importance,
            timestamp=_time.time(), tags=tags or [],
        ))
        if len(self.narrative) > 500:
            self.narrative.pop(0)

        # 重要事件影响自我概念
        if importance > 0.6:
            self.self_concept += valence * importance * 0.05
            self.self_concept = np.tanh(self.self_concept)

    def get_self_description(self) -> str:
        """生成自我描述"""
        top_abilities = sorted(
            self.beliefs.items(),
            key=lambda x: x[1].ability, reverse=True)[:3]
        weak_abilities = sorted(
            self.beliefs.items(),
            key=lambda x: x[1].ability)[:2]

        strengths = "、".join(f"{name}({b.ability:.0%})"
                             for name, b in top_abilities)
        weaknesses = "、".join(f"{name}({b.ability:.0%})"
                              for name, b in weak_abilities)

        return (f"我擅长{strengths}; "
                f"在{weaknesses}方面还需要提升。"
                f"自我效能感{self.self_efficacy:.0%}。")


class Metacognition:
    """
    元认知系统

    监控和调节认知过程。
    """

    def __init__(self, vector_dim: int = 128):
        self.vector_dim = vector_dim
        self.meta_memory = MetaMemory()
        self.self_model = SelfModel()

        # 认知监控
        self.current_cognitive_load = 0.0
        self.processing_fluency = 0.5  # 加工流畅性
        self.error_likelihood = 0.0    # 错误可能性
        self.confidence = 0.5          # 当前置信度

        # 认知策略
        self.strategies = {
            "deep_analysis": 0.5,     # 深度分析
            "quick_heuristic": 0.5,   # 快速启发
            "memory_search": 0.5,     # 记忆搜索
            "analogical": 0.5,        # 类比推理
            "decompose": 0.5,         # 问题分解
        }

        # 错误检测
        self.prediction_errors_history: List[float] = []
        self.confidence_history: List[float] = []
        self.performance_history: List[bool] = []

        # 元认知决策
        self.need_more_time = False
        self.need_more_info = False
        self.should_switch_strategy = False

    def monitor(self, prediction_error: float, processing_time: float = 0.1,
                task_difficulty: float = 0.5, memory_accessible: bool = True,
                pc_error: float = 0.0) -> Dict:
        """
        监控当前认知过程

        Args:
            prediction_error: 预测误差
            processing_time: 加工时间
            task_difficulty: 任务难度
            memory_accessible: 记忆是否可访问
            pc_error: 预测编码误差

        Returns:
            监控状态dict
        """
        # 加工流畅性(误差小+时间短=流畅)
        self.processing_fluency = float(np.clip(
            1.0 - prediction_error * 2 - processing_time * 0.1, 0, 1))

        # 认知负荷
        self.current_cognitive_load = float(np.clip(
            task_difficulty * 0.5 + prediction_error * 0.5, 0, 1))

        # 错误可能性(高误差+低流畅性=可能出错)
        self.error_likelihood = float(np.clip(
            prediction_error * 0.6 + (1 - self.processing_fluency) * 0.4, 0, 1))

        # 置信度(基于流畅性和误差一致性)
        # 流畅性高→自信, 但也要校准
        raw_confidence = self.processing_fluency * (1 - prediction_error)
        # 校准调整
        cal_error = self.meta_memory.get_calibration_error()
        self.confidence = float(np.clip(
            raw_confidence * (1 - cal_error * 0.3), 0, 1))

        # 元认知决策
        self.need_more_time = (self.confidence < 0.4 and
                               self.error_likelihood > 0.5)
        self.need_more_info = (not memory_accessible and
                               self.confidence < 0.5)
        self.should_switch_strategy = (self.error_likelihood > 0.7 and
                                       processing_time > 0.5)

        # 记录历史
        self.prediction_errors_history.append(prediction_error)
        self.confidence_history.append(self.confidence)
        if len(self.prediction_errors_history) > 50:
            self.prediction_errors_history.pop(0)
            self.confidence_history.pop(0)

        return {
            "fluency": round(self.processing_fluency, 3),
            "cognitive_load": round(self.current_cognitive_load, 3),
            "error_likelihood": round(self.error_likelihood, 3),
            "confidence": round(self.confidence, 3),
            "confidence_level": ConfidenceLevel.classify(self.confidence).label,
            "need_more_time": self.need_more_time,
            "need_more_info": self.need_more_info,
            "switch_strategy": self.should_switch_strategy,
        }

    def evaluate_response(self, predicted_confidence: float,
                          actual_correct: bool, domain: str = "学习"):
        """
        评估回答结果, 更新校准和自我模型

        Args:
            predicted_confidence: 预测的置信度
            actual_correct: 实际是否正确
            domain: 能力领域
        """
        self.meta_memory.update_calibration(predicted_confidence,
                                            float(actual_correct))
        self.self_model.update_belief(domain, actual_correct)
        self.performance_history.append(actual_correct)
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)

        if not actual_correct and predicted_confidence > 0.7:
            # 高置信但错误 = 错误检测
            self.self_model.error_detections += 1

    def detect_error(self, expected: np.ndarray, actual: np.ndarray,
                     threshold: float = 0.3) -> bool:
        """错误检测: 比较预期和实际结果"""
        error = float(np.mean((expected - actual) ** 2))
        is_error = error > threshold
        if is_error:
            self.self_model.error_detections += 1
        return is_error

    def select_strategy(self, task_difficulty: float) -> str:
        """选择认知策略"""
        strategies = list(self.strategies.keys())
        weights = np.array([self.strategies[s] for s in strategies])

        # 困难任务倾向深度分析, 简单任务倾向快速启发
        if task_difficulty > 0.6:
            weights[strategies.index("deep_analysis")] *= 2
            weights[strategies.index("decompose")] *= 1.5
        else:
            weights[strategies.index("quick_heuristic")] *= 2

        # 错误率高时尝试其他策略
        if self.error_likelihood > 0.6:
            weights *= np.random.dirichlet(np.ones(len(strategies)) * 2)

        weights = weights / weights.sum()
        chosen = np.random.choice(strategies, p=weights)
        return chosen

    def reflect(self) -> Dict:
        """
        自我反思

        Returns:
            反思结果dict
        """
        self.self_model.total_reflections += 1

        # 近期表现
        recent = self.performance_history[-20:]
        recent_accuracy = (sum(recent) / len(recent)) if recent else 0.5

        # 置信度校准
        cal_error = self.meta_memory.get_calibration_error()

        # 生成反思内容
        if recent_accuracy > 0.7:
            reflection = "最近表现不错，我对这方面越来越有信心了。"
            valence = 0.3
        elif recent_accuracy < 0.3:
            reflection = "最近出错比较多，我需要调整策略，放慢速度。"
            valence = -0.3
        else:
            reflection = "表现一般，有些地方还需要加强。"
            valence = 0.0

        if cal_error > 0.3:
            reflection += " 我对自己判断的准确性还需要更好地校准。"

        # 添加到叙事
        self.self_model.add_narrative(
            reflection, valence=valence, importance=0.4,
            tags=["reflection"])

        return {
            "reflection": reflection,
            "recent_accuracy": round(recent_accuracy, 3),
            "calibration_error": round(cal_error, 3),
            "self_efficacy": round(self.self_model.self_efficacy, 3),
            "self_esteem": round(self.self_model.self_esteem, 3),
            "total_reflections": self.self_model.total_reflections,
            "error_detections": self.self_model.error_detections,
            "self_description": self.self_model.get_self_description(),
        }

    def get_summary(self) -> Dict:
        return {
            "confidence": round(self.confidence, 3),
            "confidence_level": ConfidenceLevel.classify(self.confidence).label,
            "fluency": round(self.processing_fluency, 3),
            "cognitive_load": round(self.current_cognitive_load, 3),
            "error_likelihood": round(self.error_likelihood, 3),
            "self_efficacy": round(self.self_model.self_efficacy, 3),
            "self_esteem": round(self.self_model.self_esteem, 3),
            "calibration_error": round(self.meta_memory.get_calibration_error(), 3),
            "total_reflections": self.self_model.total_reflections,
            "error_detections": self.self_model.error_detections,
        }
