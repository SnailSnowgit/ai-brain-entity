"""
50万神经元大规模网络测试
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.large_network import LargeThreeLayerNetwork
import numpy as np
import time

print('=' * 70)
print('  500,000 神经元超大规模网络测试')
print('=' * 70)

# 50万神经元分配：感官5万，联想42.5万，决策2.5万
sensory_n = 50000
association_n = 425000
decision_n = 25000
total = sensory_n + association_n + decision_n

print(f'\n总神经元数: {total:,}')
print(f'  感官层: {sensory_n:,}')
print(f'  联想层: {association_n:,}')
print(f'  决策层: {decision_n:,}')

# 使用极低的连接密度（0.01% = 万分之一）
# 这样每个神经元平均有 ~50 个输入连接
connection_density = 0.0001
feedforward_density = 0.0002

print(f'\n连接密度: {connection_density*100:.3f}% (万分之一)')
print(f'层间连接密度: {feedforward_density*100:.3f}%')

print('\n' + '-' * 70)
print('开始构建网络...')
print('-' * 70)

t0 = time.time()
net = LargeThreeLayerNetwork(
    sensory_neurons=sensory_n,
    association_neurons=association_n,
    decision_neurons=decision_n,
    connection_density=connection_density,
    feedforward_density=feedforward_density
)
build_time = time.time() - t0

print(f'\n✓ 网络构建完成！')
print(f'  构建时间: {build_time:.1f}s ({build_time/60:.1f}分钟)')

print('\n' + '-' * 70)
print('运行测试 (10步)')
print('-' * 70)

step_times = []
for i in range(10):
    # 强刺激
    stim = np.random.rand(sensory_n // 5).astype(np.float32) * 20.0
    net.input_stimulus(stim, modality=0)
    
    t = time.time()
    net.step(dt=1.0)
    dt = time.time() - t
    step_times.append(dt)
    
    stats = net.get_stats()
    print(f'  步{i:2d}: {dt*1000:6.1f}ms | '
          f'感官={stats["sensory_rate"]:7.1f}Hz ({stats["sensory_active"]:7,}活跃) | '
          f'联想={stats["association_rate"]:7.1f}Hz ({stats["association_active"]:8,}活跃) | '
          f'决策={stats["decision_rate"]:6.1f}Hz')

avg_time = np.mean(step_times)
print(f'\n平均每步时间: {avg_time*1000:.1f}ms')
print(f'每秒可运行: {1.0/avg_time:.1f} 步')

stats = net.get_stats()
print(f'\n最终状态:')
print(f'  感官层活跃: {stats["sensory_active"]:,} / {sensory_n:,} ({stats["sensory_active"]/sensory_n*100:.1f}%)')
print(f'  联想层活跃: {stats["association_active"]:,} / {association_n:,} ({stats["association_active"]/association_n*100:.1f}%)')
print(f'  决策层活跃: {stats["decision_active"]:,} / {decision_n:,}')

print('\n' + '=' * 70)
print('测试完成！')
print('=' * 70)
