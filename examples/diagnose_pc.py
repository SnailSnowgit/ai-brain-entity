"""预测编码系统全面诊断"""
import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.predictive_coding import PredictiveCodingNetwork, ActiveInference

print("=" * 70)
print("  预测编码系统诊断报告")
print("  Predictive Coding System Diagnostics")
print("=" * 70)
print()

# ===== 1. 顶层预测误差诊断 =====
print("【1】顶层预测误差诊断")
print("-" * 50)

pc = PredictiveCodingNetwork(
    layer_sizes=[50, 30, 10],
    layer_names=["感官", "特征", "概念"],
    learning_rate=0.02,
    time_constant=5.0,
    seed=42
)

# 输入刺激
stimulus = np.sin(np.linspace(0, 4*np.pi, 50)) * 0.5 + 0.5

# 运行几步
for _ in range(50):
    pc.step(stimulus)

for i, layer in enumerate(pc.layers):
    error_mean = np.mean(np.abs(layer.prediction_error))
    activity_mean = np.mean(np.abs(layer.activity))
    print(f"  层 {i} ({layer.name}):")
    print(f"    活动均值: {activity_mean:.4f}")
    print(f"    预测误差均值: {error_mean:.4f}")
    print(f"    误差/活动比: {error_mean/max(activity_mean, 1e-8):.2%}")
    print()

print("  问题: 顶层预测误差为 0? ", "是" if np.mean(np.abs(pc.layers[-1].prediction_error)) < 1e-10 else "否")
print()

# ===== 2. 精度加权诊断 =====
print("【2】精度加权（注意力）诊断")
print("-" * 50)

pc2 = PredictiveCodingNetwork(
    layer_sizes=[40, 20, 10],
    learning_rate=0.01,
    time_constant=10.0,
    seed=123
)

noise_levels = [0.0, 0.1, 0.3, 0.5, 0.8]
precisions = []
errors = []

base_stim = np.random.rand(40) * 0.5 + 0.25

for noise in noise_levels:
    # 先学习基础模式
    for _ in range(30):
        pc2.step(base_stim)
    
    # 加噪声测试
    noisy_stim = base_stim + np.random.randn(40) * noise
    for _ in range(10):
        pc2.step(noisy_stim)
    
    prec = pc2.layers[0].precision
    err = np.mean(np.abs(pc2.layers[0].prediction_error))
    precisions.append(prec)
    errors.append(err)
    print(f"  噪声 {noise:.1f}: 精度={prec:.4f}, 误差={err:.4f}")

print()
precision_range = max(precisions) - min(precisions)
print(f"  精度变化范围: {precision_range:.4f}")
print(f"  问题: 精度变化太小? ", "是" if precision_range < 0.05 else "否")
print()

# ===== 3. 主动推理诊断 =====
print("【3】主动推理诊断")
print("-" * 50)

pc3 = PredictiveCodingNetwork(
    layer_sizes=[30, 15, 5],
    layer_names=["感官", "联合", "运动"],
    learning_rate=0.02,
    time_constant=5.0,
    seed=456
)

ai = ActiveInference(
    pc_network=pc3,
    num_actions=4,
    action_dim=30,
    horizon=3,
    learning_rate=0.01
)

goal = np.sin(np.linspace(0, 3*np.pi, 30)) * 0.5 + 0.5
ai.set_goal(goal)

current_state = np.random.rand(30) * 0.3
initial_state = current_state.copy()

fe_history = []
error_history = []

for step in range(100):
    fe, action = ai.step(current_state)
    fe_history.append(fe)
    current_state = current_state * 0.9 + action * 0.1
    error = np.mean(np.abs(current_state - goal))
    error_history.append(error)

print(f"  初始自由能: {fe_history[0]:.4f}")
print(f"  最终自由能: {fe_history[-1]:.4f}")
print(f"  自由能下降: {fe_history[0] - fe_history[-1]:.4f} ({(fe_history[0]-fe_history[-1])/fe_history[0]*100:.1f}%)")
print()
print(f"  初始目标误差: {np.mean(np.abs(initial_state - goal)):.4f}")
print(f"  最终目标误差: {error_history[-1]:.4f}")
print(f"  误差下降: {error_history[0] - error_history[-1]:.4f} ({(error_history[0]-error_history[-1])/error_history[0]*100:.1f}%)")
print()
print(f"  问题: 目标误差仍然很高? ", "是" if error_history[-1] > 0.3 else "否")
print()

# ===== 4. 自由能分解诊断 =====
print("【4】自由能分解诊断")
print("-" * 50)

fe_state = pc.compute_free_energy()
print(f"  总自由能: {fe_state.total_free_energy:.4f}")
print(f"  准确率项: {fe_state.accuracy_term:.4f}")
print(f"  复杂度项: {fe_state.complexity_term:.4f}")
print(f"  准确率/复杂度比: {abs(fe_state.accuracy_term)/max(fe_state.complexity_term, 1e-8):.2f}")
print()

# ===== 总结 =====
print("=" * 70)
print("  诊断总结")
print("=" * 70)

issues = []
if np.mean(np.abs(pc.layers[-1].prediction_error)) < 1e-10:
    issues.append("顶层预测误差为 0（缺少先验约束）")
if precision_range < 0.05:
    issues.append("精度加权变化幅度太小（注意力效果不明显）")
if error_history[-1] > 0.3:
    issues.append("主动推理目标误差高（策略学习效率低）")

if issues:
    print("  发现问题:")
    for i, issue in enumerate(issues, 1):
        print(f"    {i}. {issue}")
else:
    print("  未发现明显问题")

print()
