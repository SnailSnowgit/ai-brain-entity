"""
调试脚本：检查神经元为什么不激发
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain import SpikingNeuron, NeuralLayer, SensoryLayer

print("调试1：单个神经元，强输入")
print("-" * 50)

neuron = SpikingNeuron(0, threshold=0.8, input_gain=1.0, membrane_time_constant=10.0)
print(f"阈值: {neuron.threshold}")
print(f"输入增益: {neuron.input_gain}")
print(f"时间常数: {neuron.membrane_time_constant}")

# 施加强输入
for i in range(10):
    neuron.receive_input(5.0)  # 强输入
    spiked = neuron.step(dt=1.0, current_time=i)
    print(f"  第{i}步: V={neuron.membrane_potential:.4f}, 输入={neuron.synaptic_input:.4f}, 激发={spiked}")

print("\n调试2：NeuralLayer，直接施加输入")
print("-" * 50)

layer = NeuralLayer("test", num_neurons=5, excitatory_ratio=1.0, sparse_connectivity=0.0)
print(f"神经元数: {layer.num_neurons}")
print(f"每个神经元阈值: {[n.threshold for n in layer.neurons]}")

# 直接施加输入
input_vec = np.ones(5) * 5.0
print(f"输入向量: {input_vec}")
layer.apply_external_input(input_vec)

for i in range(10):
    spikes = layer.step(dt=1.0, current_time=i)
    print(f"  第{i}步: 激发数={np.sum(spikes)}, 活动模式={layer.activity_pattern}")
    # 打印第一个神经元的状态
    print(f"    神经元0: V={layer.neurons[0].membrane_potential:.4f}, 输入={layer.neurons[0].synaptic_input:.4f}")

print("\n调试3：SensoryLayer")
print("-" * 50)

sensory = SensoryLayer(num_neurons=10, num_modalities=1)
print(f"感官层神经元数: {sensory.num_neurons}")
print(f"模态大小: {sensory.modality_size}")
print(f"适应度: {sensory.adaptation_levels[:5]}")

# 输入刺激
stimulus = np.ones(5) * 2.0
print(f"输入刺激: {stimulus}")
sensory.receive_stimulus(stimulus, modality=0)

for i in range(10):
    spikes = sensory.step(dt=1.0, current_time=i)
    if i % 2 == 0:
        print(f"  第{i}步: 激发数={np.sum(spikes)}, 平均活动={sensory.get_mean_firing_rate():.1f}")
        print(f"    神经元0: V={sensory.neurons[0].membrane_potential:.4f}")

print("\n调试完成！")
