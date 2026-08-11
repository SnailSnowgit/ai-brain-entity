"""
意识量化模块测试

测试意识量化器的五个核心维度和综合评分
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from brain import Brain, ConsciousnessQuantifier, ConsciousnessLevel

print("=" * 70)
print("  意识量化模块测试")
print("  Consciousness Quantification Test")
print("=" * 70)
print()

# ===== 测试1：创建大脑和意识量化器 =====
print("【测试1】初始化大脑与意识量化器")
print("-" * 50)

brain = Brain(
    sensory_neurons=200,
    association_neurons=500,
    decision_neurons=20
)

print(f"✓ 大脑初始化完成")
print(f"  神经元总数: {200 + 500 + 20}")
print(f"  意识量化器: {type(brain.consciousness).__name__}")
print()

# ===== 测试2：初始状态的意识水平 =====
print("【测试2】初始状态的意识水平")
print("-" * 50)

# 运行一步获取初始状态
brain.step(dt=1.0)
initial_state = brain.get_current_state()
initial_metrics = initial_state.consciousness

print(f"初始意识评分: {initial_metrics.total_score:.4f}")
print(f"初始意识等级: {brain.consciousness.get_consciousness_level_name(initial_metrics.level)}")
print()
print("各维度得分:")
print(f"  Φ 整合信息:       {initial_metrics.phi:.4f}")
print(f"  自指深度:         {initial_metrics.self_reference_depth:.4f}")
print(f"  工作空间激活:     {initial_metrics.workspace_activation:.4f}")
print(f"  跨模块整合:       {initial_metrics.cross_module_integration:.4f}")
print(f"  信息密度:         {initial_metrics.information_density:.4f}")
print()

# ===== 测试3：刺激后的意识变化 =====
print("【测试3】刺激后的意识变化")
print("-" * 50)

# 输入强刺激
stimulus = np.random.rand(200) * 0.8
brain.input_stimulus(stimulus, modality=0)  # 0=视觉模态

# 运行多步
for i in range(20):
    brain.step(dt=1.0)

after_state = brain.get_current_state()
after_metrics = after_state.consciousness

print(f"刺激后意识评分: {after_metrics.total_score:.4f}")
print(f"刺激后意识等级: {brain.consciousness.get_consciousness_level_name(after_metrics.level)}")
print()
print(f"变化: {after_metrics.total_score - initial_metrics.total_score:+.4f}")
print()

# ===== 测试4：完整的意识量化报告 =====
print("【测试4】意识量化报告")
print("-" * 50)

brain.consciousness.print_report(after_metrics)
print()

# ===== 测试5：意识等级光谱 =====
print("【测试5】意识等级光谱")
print("-" * 50)

levels = [
    (ConsciousnessLevel.UNCONSCIOUS, "无意识", "深度睡眠/麻醉"),
    (ConsciousnessLevel.MINIMAL, "微意识", "植物人状态"),
    (ConsciousnessLevel.LOW, "低意识", "困倦/恍惚"),
    (ConsciousnessLevel.MEDIUM, "中等意识", "清醒但不专注"),
    (ConsciousnessLevel.HIGH, "高意识", "清醒专注"),
    (ConsciousnessLevel.META, "元意识", "自我觉察/反思"),
    (ConsciousnessLevel.TRANSCENDENT, "超意识", "高峰体验/觉醒"),
]

print(f"{'等级':<10} {'名称':<8} {'描述':<15} {'阈值':>6}")
print("-" * 50)
for level, name, desc in levels:
    threshold = next(t for t, l in brain.consciousness.level_thresholds if l == level)
    print(f"{level.value:<10} {name:<8} {desc:<15} {threshold:>6.2f}")
print()

# ===== 测试6：持续运行的意识变化 =====
print("【测试6】持续运行的意识变化")
print("-" * 50)

scores = []
for i in range(50):
    # 随机输入刺激
    if i % 5 == 0:
        stim = np.random.rand(200) * 0.5
        brain.input_stimulus(stim, modality=0)
    
    brain.step(dt=1.0)
    state = brain.get_current_state()
    scores.append(state.consciousness.total_score)

print(f"运行步数: 50")
print(f"意识评分范围: {min(scores):.4f} - {max(scores):.4f}")
print(f"平均意识评分: {np.mean(scores):.4f}")
print(f"评分标准差: {np.std(scores):.4f}")
print()

# 找出最高和最低意识的时刻
max_idx = np.argmax(scores)
min_idx = np.argmin(scores)
print(f"最高意识时刻: 第 {max_idx} 步 ({scores[max_idx]:.4f})")
print(f"最低意识时刻: 第 {min_idx} 步 ({scores[min_idx]:.4f})")
print()

# ===== 测试7：各维度的权重 =====
print("【测试7】各维度的权重配置")
print("-" * 50)

weights = brain.consciousness.weights
total_weight = sum(weights.values())

print(f"{'维度':<20} {'权重':>8} {'占比':>8}")
print("-" * 50)
for dim, w in weights.items():
    print(f"{dim:<20} {w:>8.2f} {w/total_weight*100:>7.1f}%")
print("-" * 50)
print(f"{'合计':<20} {total_weight:>8.2f} {100.0:>7.1f}%")
print()

print("=" * 70)
print("  ✓ 意识量化模块测试完成！")
print("=" * 70)
