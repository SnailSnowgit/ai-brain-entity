"""
小脑 (Cerebellum)

功能:
  - 前馈模型: 预测动作的 sensory consequence
  - 时序协调: 精确控制动作序列的时序
  - 误差校正: 比较预期与实际, 输出校正信号
  - 自动化: 反复练习的序列从刻意(皮层控制)转为自动(小脑控制)
  - 学习曲线: 从慢/易错到快/精确

结构(简化):
  - 颗粒细胞: 输入扩展编码
  - 浦肯野细胞: 输出校正信号(抑制性)
  - 下橄榄核: 误差信号输入(爬行纤维)
  - 小脑核: 最终输出
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto


class SequenceStatus(Enum):
    """序列执行状态"""
    NOT_STARTED = auto()
    EXECUTING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    ERROR = auto()


@dataclass
class MotorCommand:
    """运动/动作指令"""
    vector: np.ndarray       # 指令向量
    timestamp: float
    predicted_outcome: Optional[np.ndarray] = None
    actual_outcome: Optional[np.ndarray] = None
    correction: Optional[np.ndarray] = None


@dataclass
class ActionSequence:
    """动作序列"""
    id: str
    name: str
    commands: List[np.ndarray] = field(default_factory=list)
    current_step: int = 0
    status: SequenceStatus = SequenceStatus.NOT_STARTED
    repetitions: int = 0
    success_count: int = 0
    error_history: List[float] = field(default_factory=list)
    timing: List[float] = field(default_factory=list)  # 每步耗时

    @property
    def automaticity(self) -> float:
        """自动化程度 0-1 (随重复增加)"""
        if self.repetitions == 0:
            return 0.0
        # 幂律学习曲线
        return float(1.0 - np.exp(-0.05 * self.repetitions))

    @property
    def avg_error(self) -> float:
        if not self.error_history:
            return 1.0
        return float(np.mean(self.error_history[-20:]))

    @property
    def is_automated(self) -> bool:
        return self.automaticity > 0.7 and self.avg_error < 0.3


class Cerebellum:
    """
    小脑

    负责:
    1. 前馈模型: 给定当前状态+动作, 预测下一状态
    2. 误差校正: 实际vs预测的差异 → 校正信号
    3. 时序学习: 学习动作序列的精确时序
    4. 自动化: 练习使序列变得流畅、无需意识参与
    """

    def __init__(self,
                 state_dim: int = 128,
                 command_dim: int = 64,
                 learning_rate: float = 0.02,
                 granular_expansion: int = 4):
        self.state_dim = state_dim
        self.command_dim = command_dim
        self.lr = learning_rate
        self.granular_dim = state_dim * granular_expansion

        # 前馈模型权重(颗粒细胞→浦肯野细胞)
        # 输入: [当前状态; 动作指令] → 预测下一状态
        input_dim = state_dim + command_dim
        self.granular_weights = np.random.randn(input_dim, self.granular_dim) * 0.1
        self.purkinje_weights = np.random.randn(self.granular_dim, state_dim) * 0.05

        # 时序预测: 学习动作间的时间间隔
        self.timing_weights = np.random.randn(command_dim, command_dim) * 0.1

        # 校正增益(误差→校正的映射)
        self.correction_gain = 0.3

        # 动作序列库
        self.sequences: Dict[str, ActionSequence] = {}

        # 当前执行
        self.current_sequence: Optional[ActionSequence] = None
        self.prediction_error_history: List[float] = []

        # 统计
        self.total_corrections = 0
        self.total_predictions = 0

    def _granular_expand(self, x: np.ndarray) -> np.ndarray:
        """颗粒细胞扩展编码(稀疏化)"""
        activation = x @ self.granular_weights
        # 稀疏ReLU(仅top 20%激活)
        threshold = np.percentile(activation, 80)
        return np.maximum(0, activation - threshold)

    def predict_outcome(self, current_state: np.ndarray,
                        command: np.ndarray) -> np.ndarray:
        """
        前馈模型: 预测动作结果

        Args:
            current_state: 当前状态
            command: 动作指令

        Returns:
            预测的下一状态
        """
        x = np.concatenate([current_state, command])
        granular = self._granular_expand(x)
        prediction = granular @ self.purkinje_weights
        self.total_predictions += 1
        return prediction

    def compute_correction(self, predicted: np.ndarray,
                           actual: np.ndarray) -> np.ndarray:
        """
        计算校正信号(感觉预测误差)

        小脑比较"我预测会发生什么"和"实际发生了什么",
        输出校正信号给运动皮层。
        """
        sensory_error = actual - predicted
        correction = sensory_error * self.correction_gain
        self.total_corrections += 1
        return correction

    def learn(self, current_state: np.ndarray, command: np.ndarray,
              actual_next_state: np.ndarray):
        """
        学习: 更新前馈模型(监督学习, 误差由爬行纤维传递)

        Args:
            current_state: 当前状态
            command: 执行的指令
            actual_next_state: 实际下一状态
        """
        # 前向
        x = np.concatenate([current_state, command])
        granular = self._granular_expand(x)
        predicted = granular @ self.purkinje_weights

        # 误差
        error = actual_next_state - predicted
        self.prediction_error_history.append(float(np.mean(error ** 2)))
        if len(self.prediction_error_history) > 200:
            self.prediction_error_history.pop(0)

        # 浦肯野权重更新(梯度下降)
        grad = granular.reshape(-1, 1) @ error.reshape(1, -1)
        self.purkinje_weights += self.lr * grad
        self.purkinje_weights = np.clip(self.purkinje_weights, -5, 5)

        # 颗粒权重微调
        granular_grad = (error @ self.purkinje_weights.T)
        self.granular_weights += self.lr * 0.1 * x.reshape(-1, 1) @ granular_grad.reshape(1, -1)
        self.granular_weights = np.clip(self.granular_weights, -5, 5)

    def register_sequence(self, seq_id: str, name: str,
                          commands: List[np.ndarray] = None):
        """注册动作序列"""
        self.sequences[seq_id] = ActionSequence(
            id=seq_id, name=name,
            commands=commands or [],
        )

    def start_sequence(self, seq_id: str) -> bool:
        """开始执行序列"""
        if seq_id not in self.sequences:
            return False
        seq = self.sequences[seq_id]
        seq.status = SequenceStatus.EXECUTING
        seq.current_step = 0
        self.current_sequence = seq
        return True

    def execute_step(self, current_state: np.ndarray) -> Optional[Dict]:
        """
        执行序列中的一步

        Returns:
            {command, predicted_outcome, is_automatic, progress}
            序列结束返回None
        """
        if self.current_sequence is None:
            return None
        seq = self.current_sequence
        if seq.status != SequenceStatus.EXECUTING:
            return None
        if seq.current_step >= len(seq.commands):
            seq.status = SequenceStatus.COMPLETED
            seq.repetitions += 1
            self.current_sequence = None
            return None

        command = seq.commands[seq.current_step]
        predicted = self.predict_outcome(current_state, command)

        result = {
            "command": command,
            "predicted_outcome": predicted,
            "step": seq.current_step,
            "total_steps": len(seq.commands),
            "is_automatic": seq.is_automated,
            "automaticity": seq.automaticity,
            "progress": seq.current_step / len(seq.commands),
        }

        seq.current_step += 1
        return result

    def provide_feedback(self, actual_outcome: np.ndarray,
                         success: bool = True):
        """提供实际结果反馈, 用于学习和校正"""
        if self.current_sequence is None:
            return

        seq = self.current_sequence
        if seq.current_step > 0:
            last_command = seq.commands[seq.current_step - 1]
            # 学习
            # 需要当前状态, 简化: 用actual_outcome作为基准
            # 实际应用中应保存执行时的状态
            error = float(np.mean((actual_outcome -
                                   self.predict_outcome(actual_outcome, last_command)) ** 2))
            seq.error_history.append(error)

        if success:
            seq.success_count += 1

    def get_coordination_quality(self) -> float:
        """
        协调质量 0-1

        基于近期预测误差: 误差越小, 协调越好
        """
        if not self.prediction_error_history:
            return 0.5
        recent = self.prediction_error_history[-50:]
        avg_error = np.mean(recent)
        return float(np.clip(1.0 - avg_error * 5, 0, 1))

    def get_automated_sequences(self) -> List[str]:
        """获取已自动化的序列"""
        return [sid for sid, seq in self.sequences.items() if seq.is_automated]

    def get_learning_curve(self, seq_id: str) -> List[float]:
        """获取学习曲线(误差历史)"""
        if seq_id in self.sequences:
            return self.sequences[seq_id].error_history
        return []

    def modulate_timing(self, base_speed: float = 1.0) -> float:
        """
        根据自动化程度调整执行速度

        自动化程度越高, 执行越快越流畅
        """
        if self.current_sequence:
            speed = base_speed * (1.0 + self.current_sequence.automaticity * 0.5)
        else:
            speed = base_speed
        return speed

    def get_summary(self) -> Dict:
        return {
            "sequences": len(self.sequences),
            "automated_sequences": len(self.get_automated_sequences()),
            "coordination_quality": round(self.get_coordination_quality(), 3),
            "total_predictions": self.total_predictions,
            "total_corrections": self.total_corrections,
            "recent_prediction_error": round(
                np.mean(self.prediction_error_history[-10:])
                if self.prediction_error_history else 0, 4),
            "current_sequence": (self.current_sequence.name
                                if self.current_sequence else None),
        }
