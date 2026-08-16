"""
预测编码模块 (Predictive Coding)

预测编码理论: 大脑不断生成对世界的预测, 用预测误差更新内部模型。

层级结构:
  高层(抽象) → 预测 → 低层(感觉)
  低层(感觉) → 误差 → 高层(抽象)

每一层:
  - prediction: 对下层输入的预测
  - error: 实际输入与预测的差异
  - 权重更新: 最小化预测误差
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class PCLayer:
    """预测编码层"""
    size: int
    name: str = "layer"
    activation: np.ndarray = field(default_factory=lambda: np.zeros(1))
    prediction: np.ndarray = field(default_factory=lambda: np.zeros(1))
    error: np.ndarray = field(default_factory=lambda: np.zeros(1))
    # 自上而下的预测权重(来自上一层)
    top_down_weights: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))
    # 自下而上的误差权重
    bottom_up_weights: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))
    lr: float = 0.01
    precision: float = 1.0  # 精度(误差权重)

    def __post_init__(self):
        self.activation = np.zeros(self.size)
        self.prediction = np.zeros(self.size)
        self.error = np.zeros(self.size)


class PredictiveCodingNetwork:
    """
    多层预测编码网络

    信号流:
      输入 → 底层误差 → 更新底层表征 → 向上传播误差
      高层预测 → 向下传播 → 与底层比较 → 误差
    """

    def __init__(self, layer_sizes: List[int] = None,
                 learning_rate: float = 0.01,
                 time_constant: float = 10.0):
        """
        Args:
            layer_sizes: 各层神经元数, 从输入到高层
                         默认[512, 650, 256] ≈ 100万参数
            learning_rate: 学习率
            time_constant: 时间常数(ms)
        """
        if layer_sizes is None:
            layer_sizes = [512, 650, 256]

        self.lr = learning_rate
        self.time_constant = time_constant
        self.layers: List[PCLayer] = []

        for i, size in enumerate(layer_sizes):
            layer = PCLayer(size=size, name=f"layer_{i}", lr=learning_rate)
            # 初始化权重
            if i > 0:
                prev_size = layer_sizes[i - 1]
                # He初始化
                scale = np.sqrt(2.0 / prev_size)
                # top_down: (upper_size, lower_size) — upper @ W → lower
                layer.top_down_weights = np.random.randn(size, prev_size) * scale
                # bottom_up: (lower_size, upper_size) — lower @ W → upper
                layer.bottom_up_weights = np.random.randn(prev_size, size) * scale
            self.layers.append(layer)

        self.prediction_errors: List[float] = []
        self.total_error: float = 0.0
        self.step_count: int = 0

    def predict_down(self) -> List[np.ndarray]:
        """自上而下生成预测"""
        predictions = [None] * len(self.layers)
        for i in range(len(self.layers) - 1, 0, -1):
            upper = self.layers[i]
            lower = self.layers[i - 1]
            # 高层激活 → 预测低层输入
            pred = np.tanh(upper.activation @ upper.top_down_weights)
            lower.prediction = pred
            predictions[i - 1] = pred
        return predictions

    def compute_errors(self, external_input: np.ndarray = None) -> List[np.ndarray]:
        """计算各层预测误差"""
        errors = []
        for i, layer in enumerate(self.layers):
            if i == 0 and external_input is not None:
                # 底层: 外部输入 vs 预测
                layer.error = (external_input - layer.prediction) * layer.precision
            elif i > 0:
                # 高层: 下层误差通过bottom_up权重传播
                lower = self.layers[i - 1]
                layer.error = lower.error @ layer.bottom_up_weights * layer.precision
            errors.append(layer.error.copy())
        return errors

    def update_activations(self, dt: float = 1.0):
        """更新各层激活值(梯度下降最小化误差)"""
        for layer in self.layers:
            # 误差驱动的激活更新(阻尼)
            layer.activation += self.lr * 0.1 * layer.error * dt
            layer.activation = np.tanh(layer.activation)

    def update_weights(self, dt: float = 1.0):
        """更新权重(Hebbian学习)"""
        for i in range(1, len(self.layers)):
            upper = self.layers[i]
            lower = self.layers[i - 1]
            # top_down: (upper_size, lower_size) = outer(upper_act, lower_err)
            grad_td = np.outer(upper.activation, lower.error)
            upper.top_down_weights += self.lr * 0.1 * grad_td * dt
            # bottom_up: (lower_size, upper_size) = outer(lower_act, upper_err)
            grad_bu = np.outer(lower.activation, upper.error)
            upper.bottom_up_weights += self.lr * 0.1 * grad_bu * dt
            # 权重裁剪防止爆炸
            np.clip(upper.top_down_weights, -5, 5, out=upper.top_down_weights)
            np.clip(upper.bottom_up_weights, -5, 5, out=upper.bottom_up_weights)

    def step(self, external_input: np.ndarray = None,
             dt: float = 1.0) -> Dict[str, float]:
        """
        执行一步预测编码

        Args:
            external_input: 外部输入向量(底层)
            dt: 时间步长

        Returns:
            dict with 'mean_error', 'total_error', 'n_layers'
        """
        self.step_count += 1

        if external_input is not None:
            self.layers[0].activation = np.tanh(external_input)

        # 1. 自上而下预测
        self.predict_down()

        # 2. 计算误差
        self.compute_errors(external_input)

        # 3. 更新激活
        self.update_activations(dt)

        # 4. 更新权重
        self.update_weights(dt)

        # 统计
        mean_error = float(np.mean([
            np.mean(np.abs(layer.error)) for layer in self.layers
        ]))
        self.total_error = mean_error
        self.prediction_errors.append(mean_error)
        if len(self.prediction_errors) > 1000:
            self.prediction_errors.pop(0)

        return {
            'mean_error': mean_error,
            'total_error': self.total_error,
            'n_layers': len(self.layers),
            'top_activation_norm': float(
                np.linalg.norm(self.layers[-1].activation)),
        }

    def get_prediction(self) -> np.ndarray:
        """获取高层对底层的预测"""
        if len(self.layers) >= 2:
            return self.layers[0].prediction.copy()
        return self.layers[0].activation.copy()

    def get_last_error(self) -> float:
        return self.prediction_errors[-1] if self.prediction_errors else 0.0

    def reset(self):
        """重置网络状态"""
        for layer in self.layers:
            layer.activation = np.zeros(layer.size)
            layer.prediction = np.zeros(layer.size)
            layer.error = np.zeros(layer.size)
        self.prediction_errors.clear()
        self.total_error = 0.0
        self.step_count = 0
