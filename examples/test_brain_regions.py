"""
脑区域模块测试脚本
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.brain_regions import BrainRegions
import numpy as np

print("测试脑区域系统...\n")

brain_regions = BrainRegions(
    sensory_dim=64,
    motor_dim=8
)

print("\n运行100步测试...")

for i in range(100):
    # 随机感觉输入
    stimulus = np.random.rand(64).astype(np.float32) * 0.5
    brain_regions.input_sensory(0, stimulus)
    
    # 位置更新（模拟空间移动）
    x = 0.5 + 0.3 * np.sin(i * 0.1)
    y = 0.5 + 0.3 * np.cos(i * 0.1)
    brain_regions.hippocampus.update_place_cells((x, y))
    
    brain_regions.step(dt=1.0)
    
    if i % 20 == 0:
        stats = brain_regions.get_stats()
        h = stats['hippocampus']
        p = stats['prefrontal']
        print(f"  步{i:3d}: 记忆={h['episodic_memories']:3d}, "
              f"WM={p['active_working_memory']}, "
              f"CA3={h['ca3_activity_mean']:.3f}")

stats = brain_regions.get_stats()

print("\n" + "=" * 50)
print("  海马体 (Hippocampus)")
print("=" * 50)
h = stats['hippocampus']
print(f"  DG活跃神经元: {h['dg_active_neurons']}")
print(f"  CA3活动均值: {h['ca3_activity_mean']:.4f}")
print(f"  CA1活动均值: {h['ca1_activity_mean']:.4f}")
print(f"  情景记忆数: {h['episodic_memories']}")
print(f"  形成记忆次数: {h['memories_formed']}")
print(f"  模式完成次数: {h['pattern_completions']}")
print(f"  巩固次数: {h['consolidated_count']}")
print(f"  活跃位置细胞: {h['active_place_cells']}")
print(f"  θ相位: {h['theta_phase']:.2f} rad")
print(f"  新奇信号: {h['novelty_signal']:.4f}")

print("\n" + "=" * 50)
print("  前额叶皮层 (Prefrontal Cortex)")
print("=" * 50)
p = stats['prefrontal']
print(f"  活跃工作记忆: {p['active_working_memory']}")
print(f"  工作记忆负荷: {p['working_memory_load']:.3f}")
print(f"  当前目标: {p['current_goal']}")
print(f"  目标重要性: {p['goal_importance']:.3f}")
print(f"  注意力控制: {p['attention_control']:.3f}")
print(f"  认知负荷: {p['cognitive_load']:.3f}")

print("\n" + "=" * 50)
print("  丘脑 (Thalamus)")
print("=" * 50)
t = stats['thalamus']
print(f"  觉醒水平: {t['arousal_level']:.3f}")
print(f"  平均门控: {t['mean_gating']:.3f}")
print(f"  门控变化次数: {t['gate_changes']}")

print("\n" + "=" * 50)
print("  基底神经节 (Basal Ganglia)")
print("=" * 50)
bg = stats['basal_ganglia']
print(f"  最佳动作值: {bg['best_action_value']:.3f}")
print(f"  选中动作: {bg['selected_action']}")
print(f"  选择次数: {bg['actions_selected']}")
print(f"  累计奖励: {bg['total_reward']:.3f}")

# 测试记忆提取
print("\n" + "=" * 50)
print("  记忆提取测试")
print("=" * 50)
cue = np.random.rand(64).astype(np.float32) * 0.5
recalled = brain_regions.hippocampus.recall_memory(cue)
if recalled:
    print(f"  ✓ 成功提取记忆 #{recalled.memory_id}")
    print(f"    强度: {recalled.strength:.3f}")
    print(f"    提取次数: {recalled.recall_count}")
else:
    print(f"  ✗ 未找到匹配记忆")

print("\n✓ 脑区域模块测试完成！")
