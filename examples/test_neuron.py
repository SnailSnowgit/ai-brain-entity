"""
简单测试：验证神经元能否正常激发
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain import SpikingNeuron, NeuralLayer, SensoryLayer

print("测试1：单个神经元激发")
print("-" * 40)

neuron = SpikingNeuron(0, threshold=1.0, input_gain=1.0)
print(f"初始膜电位: {neuron.membrane_potential}")
print(f"阈值: {neuron.threshold}")

# 施加持续输入
spike_count = 0
for i in range(20):
    neuron.receive_input(0.5)  # 每步输入0.5
    spiked = neuron.step(dt=1.0, current_time=i)
    if spiked:
        spike_count += 1
    if i % 5 == 0:
        print(f"  第{i}步: 膜电位={neuron.membrane_potential:.3f}, 激发={spiked}")

print(f"\n20步内激发次数: {spike_count}")

print("\n测试2：神经网络层")
print("-" * 40)

layer = NeuralLayer("test", num_neurons=10, excitatory_ratio=0.8, sparse_connectivity=0.5)
print(f"创建 {layer.num_neurons} 个神经元的层")
print(f"层内连接数: {sum(len(n.output_synapses) for n in layer.neurons)}")

# 施加输入
input_vec = np.ones(10) * 0.5
layer.apply_external_input(input_vec)

total_spikes = 0
for i in range(10):
    spikes = layer.step(dt=1.0, current_time=i)
    total_spikes += np.sum(spikes)
    if i % 3 == 0:
        print(f"  第{i}步: 激发数={np.sum(spikes)}, 平均活动={layer.get_mean_firing_rate():.1f} Hz")

print(f"\n10步总激发数: {total_spikes}")

print("\n测试3：感官层")
print("-" * 40)

sensory = SensoryLayer(num_neurons=20, num_modalities=2)
print(f"感官层: {sensory.num_neurons} 神经元, {sensory.num_modalities} 模态")
print(f"模态大小: {sensory.modality_size}")

# 输入刺激
stimulus = np.ones(10) * 1.0
sensory.receive_stimulus(stimulus, modality=0)

total_spikes = 0
for i in range(10):
    spikes = sensory.step(dt=1.0, current_time=i)
    total_spikes += np.sum(spikes)
    if i % 3 == 0:
        print(f"  第{i}步: 激发数={np.sum(spikes)}, 活动率={sensory.get_mean_firing_rate():.1f} Hz")

print(f"\n10步总激发数: {total_spikes}")
print(f"感官适应度: {np.mean(sensory.adaptation_levels):.3f}")

print("\n测试完成！")
