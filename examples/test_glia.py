"""
胶质细胞系统测试
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain import Astrocyte, Oligodendrocyte, Microglia, GlialNetwork, Brain

print("=" * 60)
print("胶质细胞系统测试")
print("=" * 60)

print("\n测试1：单个星形胶质细胞")
print("-" * 40)

astro = Astrocyte(0)
print(f"初始钙浓度: {astro.intracellular_calcium:.3f}")
print(f"初始激活水平: {astro.activation_level:.3f}")

# 刺激星形胶质细胞
print("\n施加谷氨酸刺激...")
for i in range(20):
    astro.stimulate(glutamate_amount=0.1, potassium_increase=0.05)
    state = astro.step(dt=1.0)
    if i % 5 == 0:
        print(f"  第{i}步: 钙={state['calcium_level']:.3f}, 激活={state['activation']:.3f}, "
              f"钙波={state['calcium_wave']}, 钾={state['potassium']:.2f}")

print(f"\n突触调制因子: {astro.get_synaptic_modulation():.3f}")

print("\n测试2：少突胶质细胞")
print("-" * 40)

oligo = Oligodendrocyte(0)
print(f"初始髓鞘段数: {len(oligo.myelin_segments)}")
print(f"初始能量储备: {oligo.energy_reserve:.3f}")

# 髓鞘化轴突
oligo.myelinate_axon(0, initial_myelin=0.3)
oligo.myelinate_axon(1, initial_myelin=0.5)
print(f"髓鞘化后段数: {len(oligo.myelin_segments)}")
print(f"轴突0传导速度因子: {oligo.get_conduction_speed_factor(0):.3f}")
print(f"轴突1传导速度因子: {oligo.get_conduction_speed_factor(1):.3f}")

# 活动依赖的髓鞘可塑性
print("\n高活动刺激髓鞘增厚...")
axon_activity = {0: 0.8, 1: 0.2}  # 轴突0高活动，轴突1低活动
for i in range(50):
    oligo.step(dt=1.0, axon_activity=axon_activity)
    if i % 10 == 0:
        m0 = oligo.myelin_segments[0]['myelin_thickness']
        m1 = oligo.myelin_segments[1]['myelin_thickness']
        print(f"  第{i}步: 轴突0髓鞘={m0:.3f}, 轴突1髓鞘={m1:.3f}")

print("\n测试3：小胶质细胞")
print("-" * 40)

micro = Microglia(0)
print(f"初始状态: {micro.state}")
print(f"初始激活水平: {micro.activation_level:.3f}")

# 监测突触
synapses = [
    (0, 1, 0.8),
    (2, 3, 0.1),  # 弱突触
    (4, 5, 0.15),  # 弱突触
    (6, 7, 0.05),  # 很弱的突触
    (8, 9, 0.9),
]

print("\n监测周围突触（含3个弱突触）...")
for i in range(20):
    micro.survey(synapses, damage_signals=0.0)
    state = micro.step(dt=1.0)
    if i % 5 == 0:
        print(f"  第{i}步: 状态={state['state']}, 激活={state['activation']:.3f}, "
              f"突起={state['process_extension']:.2f}")

# 测试损伤响应
print("\n施加损伤信号...")
for i in range(10):
    micro.survey(synapses, damage_signals=0.3)
    state = micro.step(dt=1.0)
    if i % 3 == 0:
        print(f"  第{i}步: 状态={state['state']}, 炎症={state['inflammation']:.3f}")

print("\n测试4：胶质网络")
print("-" * 40)

glia_net = GlialNetwork(
    num_astrocytes=20,
    num_oligodendrocytes=5,
    num_microglia=3,
    network_size=(5, 5)
)

summary = glia_net.get_summary()
print(f"星形胶质细胞: {summary['astrocytes']} 个")
print(f"少突胶质细胞: {summary['oligodendrocytes']} 个")
print(f"小胶质细胞: {summary['microglia']} 个")
print(f"星形胶质连接: {summary['astrocyte_connections']} 条")

# 建立神经元映射
neuron_positions = {i: (np.random.uniform(0, 5), np.random.uniform(0, 5)) for i in range(30)}
glia_net.map_neurons_to_astrocytes(neuron_positions)
print(f"映射神经元数: {len(glia_net.neuron_astrocyte_mapping)}")

# 模拟神经元活动刺激
print("\n神经元活动刺激胶质细胞...")
for i in range(30):
    neuron_activity = {j: np.random.uniform(0, 0.5) for j in range(30)}
    state = glia_net.step(dt=1.0, neuron_activity=neuron_activity)
    if i % 10 == 0:
        print(f"  第{i}步:")
        print(f"    星形胶质: 平均钙={state['astrocytes']['avg_calcium']:.3f}, "
              f"激活={state['astrocytes']['avg_activation']:.3f}, "
              f"钙波数={state['astrocytes']['active_calcium_waves']}")
        print(f"    小胶质: 炎症={state['microglia']['avg_inflammation']:.3f}, "
              f"监测中={state['microglia']['surveying']}个")

print("\n测试5：完整大脑中的胶质细胞")
print("-" * 40)

brain = Brain(
    sensory_neurons=50,
    association_neurons=80,
    decision_neurons=4
)

print("创建大脑成功")
glia_summary = brain.glia.get_summary()
print(f"胶质细胞: 星形={glia_summary['astrocytes']}, "
      f"少突={glia_summary['oligodendrocytes']}, "
      f"小胶质={glia_summary['microglia']}")

# 运行几步
print("\n运行大脑...")
for i in range(20):
    stimulus = np.random.rand(50) * 2.0
    brain.input_stimulus(stimulus, modality=0)
    state = brain.step(dt=1.0)
    if i % 5 == 0:
        print(f"  第{i}步:")
        print(f"    感官活动: {state.sensory_activity:.1f} Hz")
        print(f"    胶质状态: 星形激活={state.glia_stats['astrocytes']['avg_activation']:.3f}, "
              f"钙波={state.glia_stats['astrocytes']['active_calcium_waves']}")

print("\n" + "=" * 60)
print("所有测试完成！")
print("=" * 60)
