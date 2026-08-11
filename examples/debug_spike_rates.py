"""检查spike_rates的实际值"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from brain import Brain

brain = Brain(
    sensory_neurons=200,
    association_neurons=500,
    decision_neurons=20
)

# 输入刺激
stim = np.random.rand(200) * 0.5
brain.input_stimulus(stim, modality=0)

# 运行几步
for i in range(10):
    brain.step(dt=1.0)

print("=== spike_rates 实际值 ===")
sensory_rates = brain.network.sensory.spike_rates
association_rates = brain.network.association.spike_rates
decision_rates = brain.network.decision.spike_rates

print(f"sensory: min={np.min(sensory_rates):.2f}, max={np.max(sensory_rates):.2f}, "
      f"mean={np.mean(sensory_rates):.2f}, std={np.std(sensory_rates):.2f}")
print(f"association: min={np.min(association_rates):.2f}, max={np.max(association_rates):.2f}, "
      f"mean={np.mean(association_rates):.2f}, std={np.std(association_rates):.2f}")
print(f"decision: min={np.min(decision_rates):.2f}, max={np.max(decision_rates):.2f}, "
      f"mean={np.mean(decision_rates):.2f}, std={np.std(decision_rates):.2f}")
print()

# 检查归一化后的值
print("=== 除以100后的值 ===")
sensory_norm = sensory_rates / 100.0
association_norm = association_rates / 100.0
decision_norm = decision_rates / 100.0

print(f"sensory: min={np.min(sensory_norm):.4f}, max={np.max(sensory_norm):.4f}, "
      f"mean={np.mean(sensory_norm):.4f}")
print(f"association: min={np.min(association_norm):.4f}, max={np.max(association_norm):.4f}, "
      f"mean={np.mean(association_norm):.4f}")
print(f"decision: min={np.min(decision_norm):.4f}, max={np.max(decision_norm):.4f}, "
      f"mean={np.mean(decision_norm):.4f}")
print()

# 检查分块后的宏通道
print("=== 分块后的宏通道 ===")
def block_mean(x, n_blocks=8):
    edges = np.linspace(0, len(x), n_blocks + 1).astype(int)
    edges[-1] = len(x)
    return np.array([x[edges[k]:edges[k+1]].mean() for k in range(len(edges)-1)])

sensory_blocks = block_mean(sensory_norm)
association_blocks = block_mean(association_norm)
decision_blocks = block_mean(decision_norm)

print(f"sensory blocks: {sensory_blocks}")
print(f"  std: {np.std(sensory_blocks):.6f}")
print(f"association blocks: {association_blocks}")
print(f"  std: {np.std(association_blocks):.6f}")
print(f"decision blocks: {decision_blocks}")
print(f"  std: {np.std(decision_blocks):.6f}")
print()

# 检查互信息
print("=== 互信息计算 ===")
# 模拟滑动窗口
T = 50
sensory_series = np.random.rand(T, 8) * 0.1 + 0.9  # 接近1的随机值
association_series = np.random.rand(T, 8) * 0.1 + 0.9

# z-score标准化
X = (sensory_series - sensory_series.mean(axis=0)) / np.where(sensory_series.std(axis=0) > 1e-12, sensory_series.std(axis=0), 1.0)
Y = (association_series - association_series.mean(axis=0)) / np.where(association_series.std(axis=0) > 1e-12, association_series.std(axis=0), 1.0)

print(f"X std (per dim): {X.std(axis=0)}")
print(f"Y std (per dim): {Y.std(axis=0)}")
print()

# 计算协方差
a = 0.1
T_actual = T
Cx = (1 - a) * (X.T @ X) / T_actual + a * np.eye(X.shape[1])
Cy = (1 - a) * (Y.T @ Y) / T_actual + a * np.eye(Y.shape[1])
Z = np.concatenate([X, Y], axis=1)
Cz = (1 - a) * (Z.T @ Z) / T_actual + a * np.eye(Z.shape[1])

sx, ldx = np.linalg.slogdet(Cx)
sy, ldy = np.linalg.slogdet(Cy)
sz, ldz = np.linalg.slogdet(Cz)

print(f"logdet Cx: {ldx:.4f}")
print(f"logdet Cy: {ldy:.4f}")
print(f"logdet Cz: {ldz:.4f}")
print()

mi = 0.5 * (ldx + ldy - ldz)
bias = X.shape[1] * Y.shape[1] / (2.0 * T_actual)
print(f"MI raw: {mi:.4f} nats")
print(f"bias: {bias:.4f} nats")
print(f"MI corrected: {max(0, mi - bias):.4f} nats")
