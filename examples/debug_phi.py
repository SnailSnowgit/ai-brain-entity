"""调试Φ值计算"""
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
stim = np.random.rand(200) * 0.8
brain.input_stimulus(stim, modality=0)

# 运行几步
for i in range(10):
    brain.step(dt=1.0)

# 检查各层活动模式
print("=== 各层活动模式 ===")
sensory = brain.network.sensory.get_activity_pattern()
association = brain.network.association.get_activity_pattern()
decision = brain.network.decision.get_activity_pattern()

print(f"sensory shape: {sensory.shape}")
print(f"sensory mean: {np.mean(sensory):.4f}")
print(f"sensory std: {np.std(sensory):.4f}")
print(f"sensory min: {np.min(sensory):.4f}")
print(f"sensory max: {np.max(sensory):.4f}")
print()

print(f"association shape: {association.shape}")
print(f"association mean: {np.mean(association):.4f}")
print(f"association std: {np.std(association):.4f}")
print(f"association min: {np.min(association):.4f}")
print(f"association max: {np.max(association):.4f}")
print()

print(f"decision shape: {decision.shape}")
print(f"decision mean: {np.mean(decision):.4f}")
print(f"decision std: {np.std(decision):.4f}")
print(f"decision min: {np.min(decision):.4f}")
print(f"decision max: {np.max(decision):.4f}")
print()

# 检查熵计算
print("=== 熵计算 ===")
def compute_entropy(x):
    binary = (x > 0.5).astype(float)
    if len(binary) == 0:
        return 0.0
    p1 = np.mean(binary)
    p0 = 1.0 - p1
    if p0 <= 0 or p1 <= 0:
        return 0.0
    entropy = -p0 * np.log2(p0) - p1 * np.log2(p1)
    return entropy

sensory_entropy = compute_entropy(sensory)
association_entropy = compute_entropy(association)
decision_entropy = compute_entropy(decision)

print(f"sensory entropy: {sensory_entropy:.4f}")
print(f"association entropy: {association_entropy:.4f}")
print(f"decision entropy: {decision_entropy:.4f}")
print()

# 整体熵
whole = np.concatenate([sensory, association, decision])
whole_entropy = compute_entropy(whole)
print(f"whole entropy: {whole_entropy:.4f}")
print()

# Φ值
avg_parts = (sensory_entropy + association_entropy + decision_entropy) / 3.0
phi = max(0.0, whole_entropy - avg_parts)
print(f"raw phi: {phi:.6f}")
print(f"normalized phi (*5): {min(1.0, phi * 5):.6f}")
print(f"normalized phi (*10): {min(1.0, phi * 10):.6f}")
print(f"normalized phi (*20): {min(1.0, phi * 20):.6f}")
print()

# 检查二值化的比例
print("=== 二值化比例（>0.5） ===")
print(f"sensory: {np.mean(sensory > 0.5):.4f}")
print(f"association: {np.mean(association > 0.5):.4f}")
print(f"decision: {np.mean(decision > 0.5):.4f}")
print(f"whole: {np.mean(whole > 0.5):.4f}")
print()

# 试试用连续值的熵（高斯分布近似）
print("=== 连续熵（高斯近似） ===")
def continuous_entropy(x):
    """计算连续信号的微分熵（高斯近似）"""
    sigma = np.std(x)
    if sigma < 1e-10:
        return 0.0
    return 0.5 * np.log2(2 * np.pi * np.e * sigma ** 2)

sensory_cent = continuous_entropy(sensory)
association_cent = continuous_entropy(association)
decision_cent = continuous_entropy(decision)
whole_cent = continuous_entropy(whole)

print(f"sensory continuous entropy: {sensory_cent:.4f}")
print(f"association continuous entropy: {association_cent:.4f}")
print(f"decision continuous entropy: {decision_cent:.4f}")
print(f"whole continuous entropy: {whole_cent:.4f}")
print()

avg_parts_cent = (sensory_cent + association_cent + decision_cent) / 3.0
phi_cent = max(0.0, whole_cent - avg_parts_cent)
print(f"raw phi (continuous): {phi_cent:.6f}")
print(f"normalized phi_cent (*0.1): {min(1.0, phi_cent * 0.1):.6f}")
print()

# 用互信息的方式计算
print("=== 互信息方式 ===")
# 互信息 I(X;Y) = H(X) + H(Y) - H(X,Y)
# 整合信息可以看作各部分之间的互信息之和
def mutual_information(x, y, bins=10):
    """计算两个向量的互信息"""
    # 分箱
    x_bins = np.digitize(x, np.linspace(x.min(), x.max(), bins))
    y_bins = np.digitize(y, np.linspace(y.min(), y.max(), bins))
    
    # 联合分布
    joint = np.zeros((bins, bins))
    for i in range(len(x)):
        xi = min(x_bins[i] - 1, bins - 1)
        yi = min(y_bins[i] - 1, bins - 1)
        joint[xi, yi] += 1
    joint /= joint.sum()
    
    # 边缘分布
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    
    # 互信息
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if joint[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += joint[i, j] * np.log2(joint[i, j] / (px[i] * py[j]))
    
    return mi

mi_sa = mutual_information(sensory, association)
mi_sd = mutual_information(sensory, decision)
mi_ad = mutual_information(association, decision)

print(f"MI(sensory, association): {mi_sa:.4f}")
print(f"MI(sensory, decision): {mi_sd:.4f}")
print(f"MI(association, decision): {mi_ad:.4f}")
print(f"Total MI: {mi_sa + mi_sd + mi_ad:.4f}")
print(f"Normalized total MI: {min(1.0, (mi_sa + mi_sd + mi_ad) / 2.0):.4f}")
