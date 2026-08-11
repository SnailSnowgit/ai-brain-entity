"""
高级脑区模块测试脚本
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.advanced_regions import AdvancedBrainRegions
import numpy as np

print("=" * 60)
print("  高级脑区模块测试")
print("=" * 60)

# 创建高级脑区系统
brain = AdvancedBrainRegions(
    sensory_dim=64,
    motor_dim=32
)

print("\n运行100步测试...")
print("-" * 60)

for i in range(100):
    # 随机感觉输入
    sensory = np.random.rand(64).astype(np.float32) * 0.5
    
    # 小脑输入
    brain.cerebellum.input_mossy_fibers(sensory)
    
    # 每20步给一个误差信号（学习）
    if i % 20 == 0 and i > 0:
        error = np.random.rand(100).astype(np.float32) * 0.3
        brain.cerebellum.input_climbing_fiber(error)
    
    # 扣带回输入
    brain.cingulate.input_signals(sensory, sensory * 0.8)
    
    # 顶叶输入
    spatial = np.random.rand(32, 32).astype(np.float32)
    brain.parietal.input_visual_spatial(spatial)
    somato = np.random.rand(32).astype(np.float32)
    brain.parietal.input_somatosensory(somato)
    
    # 随机转移注意
    if i % 15 == 0:
        x = np.random.randint(0, 32)
        y = np.random.randint(0, 32)
        brain.parietal.shift_attention(x, y)
    
    # 岛叶输入
    body = np.random.rand(32).astype(np.float32)
    brain.insula.input_interoceptive(body)
    emotion = np.random.rand(32).astype(np.float32)
    brain.insula.input_emotional(emotion)
    
    # 默认模式网络：每30步切换状态
    if i % 30 < 15:
        brain.default_mode.activate_default_mode()
    else:
        brain.default_mode.deactivate_default_mode()
    
    # 执行一步
    brain.step(dt=1.0)
    
    if i % 20 == 0:
        stats = brain.get_stats()
        dmn = stats['default_mode']
        cg = stats['cingulate']
        print(f"  步{i:3d}: DMN={'ON' if dmn['is_active'] else 'OFF'}, "
              f"冲突={cg['conflict_level']:.3f}, "
              f"心智游移={dmn['mind_wandering_level']:.3f}")

stats = brain.get_stats()

# ===== 小脑 =====
print("\n" + "=" * 60)
print("  小脑 (Cerebellum)")
print("=" * 60)
c = stats['cerebellum']
print(f"  颗粒细胞活动均值: {c['granule_activity_mean']:.4f}")
print(f"  浦肯野细胞活动均值: {c['purkinje_activity_mean']:.4f}")
print(f"  运动程序数: {c['num_programs']}")
print(f"  接收误差次数: {c['errors_received']}")
print(f"  内部时钟: {c['internal_clock']:.1f}")

# ===== 扣带回 =====
print("\n" + "=" * 60)
print("  扣带回 (Cingulate Cortex)")
print("=" * 60)
cg = stats['cingulate']
print(f"  ACC活动均值: {cg['acc_activity_mean']:.4f}")
print(f"  PCC活动均值: {cg['pcc_activity_mean']:.4f}")
print(f"  冲突水平: {cg['conflict_level']:.4f}")
print(f"  错误信号: {cg['error_signal']:.4f}")
print(f"  疼痛水平: {cg['pain_level']:.4f}")
print(f"  动机水平: {cg['motivation_level']:.4f}")
print(f"  自我参照: {cg['self_reference']:.4f}")
print(f"  情绪调节: {cg['emotional_regulation']:.4f}")
print(f"  检测到冲突: {cg['conflicts_detected']} 次")
print(f"  检测到错误: {cg['errors_detected']} 次")

# ===== 默认模式网络 =====
print("\n" + "=" * 60)
print("  默认模式网络 (Default Mode Network)")
print("=" * 60)
dmn = stats['default_mode']
print(f"  激活状态: {'激活(静息态)' if dmn['is_active'] else '抑制(任务态)'}")
print(f"  DMN整体活动: {dmn['dmn_activity']:.4f}")
print(f"  心智游移程度: {dmn['mind_wandering_level']:.4f}")
print(f"  自我反思程度: {dmn['self_reflection_level']:.4f}")
print(f"  时间视角: {dmn['time_perspective']:.2f} (0=过去, 0.5=现在, 1=未来)")
print(f"  mPFC活动均值: {dmn['mpfc_activity_mean']:.4f}")
print(f"  PCC活动均值: {dmn['pcc_activity_mean']:.4f}")
print(f"  角回活动均值: {dmn['angular_gyrus_activity_mean']:.4f}")
print(f"  心智游移片段: {dmn['mind_wandering_episodes']} 次")
print(f"  自我反思时间: {dmn['self_reflection_time']:.1f}")

# ===== 顶叶 =====
print("\n" + "=" * 60)
print("  顶叶 (Parietal Lobe)")
print("=" * 60)
p = stats['parietal']
print(f"  注意焦点位置: {p['attention_location']}")
print(f"  注意半径: {p['attention_radius']}")
print(f"  注意转移次数: {p['attention_shifts']} 次")
print(f"  SPL(顶上小叶)活动: {p['spl_activity_mean']:.4f}")
print(f"  IPL(顶下小叶)活动: {p['ipl_activity_mean']:.4f}")
print(f"  体感活动均值: {p['somatosensory_mean']:.4f}")
print(f"  数字线激活: 数字 {p['number_line_active']}")

# ===== 岛叶 =====
print("\n" + "=" * 60)
print("  岛叶 (Insula)")
print("=" * 60)
ins = stats['insula']
print(f"  身体拥有感: {ins['body_ownership']:.4f}")
print(f"  情绪意识: {ins['emotional_awareness']:.4f}")
print(f"  内感受准确性: {ins['interoceptive_accuracy']:.4f}")
print(f"  前岛叶活动均值: {ins['anterior_activity_mean']:.4f}")
print(f"  后岛叶活动均值: {ins['posterior_activity_mean']:.4f}")
print(f"  味觉活动: {ins['taste_activity']}")
print(f"  内感受更新次数: {ins['interoceptive_updates']}")

# 测试小脑学习
print("\n" + "=" * 60)
print("  小脑运动学习测试")
print("=" * 60)

target = np.random.rand(32).astype(np.float32) * 0.5
print(f"  目标模式维度: {len(target)}")

# 学习前误差
output = brain.cerebellum.step()
initial_error = np.mean(np.abs(target - output[:32]))
print(f"  学习前误差: {initial_error:.4f}")

print("\n✓ 高级脑区模块测试完成！")
print("=" * 60)
