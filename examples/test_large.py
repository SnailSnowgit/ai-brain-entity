"""
大规模网络测试脚本 - 验证版
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.large_network import LargeThreeLayerNetwork
import numpy as np
import time

print('=' * 60)
print('测试 50,000 神经元网络')
print('=' * 60)

t0 = time.time()
net = LargeThreeLayerNetwork(
    sensory_neurons=5000,
    association_neurons=40000,
    decision_neurons=5000,
    connection_density=0.005,
    feedforward_density=0.01
)
build_time = time.time() - t0
print(f'\n构建时间: {build_time:.2f}s')

print('\n运行 20 步（强刺激）...')
for i in range(20):
    # 更强的刺激
    stim = np.random.rand(1000).astype(np.float32) * 10.0
    net.input_stimulus(stim, modality=0)
    
    t = time.time()
    net.step(dt=1.0)
    dt = time.time() - t
    
    stats = net.get_stats()
    if i % 5 == 0:
        print(f'  步{i:2d}: {dt*1000:5.1f}ms | '
              f'感官={stats["sensory_rate"]:6.1f}Hz ({stats["sensory_active"]:5,}活跃) | '
              f'联想={stats["association_rate"]:6.1f}Hz ({stats["association_active"]:6,}活跃) | '
              f'决策={stats["decision_rate"]:6.1f}Hz')

print('\n' + '=' * 60)
print('测试完成！')
print('=' * 60)
