"""
测试三层网络通路
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain import ThreeLayerNetwork

print("测试三层网络通路")
print("=" * 50)

# 创建网络
net = ThreeLayerNetwork(
    sensory_size=20,
    association_size=30,
    decision_size=4,
    num_modalities=2
)

print(f"感官层: {net.sensory.num_neurons} 神经元")
print(f"联想层: {net.association.num_neurons} 神经元")
print(f"决策层: {net.decision.num_neurons} 神经元")
print(f"感官→联想连接矩阵: {net.sensory_to_association_weights.shape}")
print(f"联想→决策连接矩阵: {net.association_to_decision_weights.shape}")

# 输入刺激
print("\n输入刺激...")
stimulus = np.ones(10) * 1.5  # 10个神经元的强刺激
net.input_stimulus(stimulus, modality=0)

# 运行几步
for i in range(20):
    net.step(dt=1.0)
    
    if i % 5 == 0:
        print(f"\n第 {i} 步:")
        print(f"  感官层活动率: {net.sensory.get_mean_firing_rate():.1f} Hz")
        print(f"  联想层活动率: {net.association.get_mean_firing_rate():.1f} Hz")
        print(f"  决策层活动率: {net.decision.get_mean_firing_rate():.1f} Hz")
        print(f"  激活概念: {len(net.get_active_concepts())} 个")
        decision = net.get_decision()
        print(f"  当前决策: {decision[1]} (置信度: {decision[2]:.2f})")

print("\n测试完成！")
