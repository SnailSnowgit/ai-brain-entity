"""
大规模神经网络演示
展示500万神经元级别的稀疏神经网络
"""

import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain import LargeScaleThreeLayerNetwork


def demo_large_scale():
    print("=" * 60)
    print("大规模神经网络演示")
    print("=" * 60)
    print()
    
    # 配置：默认100万神经元（可以修改为500万）
    # 注意：500万神经元约需3.75GB内存
    total_neurons = 1000000  # 100万
    connections_per_neuron = 50
    
    print(f"配置:")
    print(f"  总神经元数: {total_neurons:,}")
    print(f"  每神经元连接数: {connections_per_neuron}")
    print()
    
    # 创建网络
    print("创建大规模神经网络...")
    t0 = time.time()
    
    network = LargeScaleThreeLayerNetwork(
        sensory_neurons=int(total_neurons * 0.1),
        association_neurons=int(total_neurons * 0.9) - 10,
        decision_neurons=10,
        connections_per_neuron=connections_per_neuron
    )
    
    create_time = time.time() - t0
    print(f"网络创建完成: {create_time:.1f}s")
    print()
    
    # 内存统计
    mem = network.sensory.get_memory_usage()
    mem += network.association.get_memory_usage()
    total_mem = mem['total'] + (
        network.sensory_to_assoc_indices.nbytes +
        network.sensory_to_assoc_weights.nbytes +
        network.assoc_to_decision_indices.nbytes +
        network.assoc_to_decision_weights.nbytes
    ) / (1024 * 1024)
    
    print(f"内存使用: {total_mem:.1f} MB ({total_mem/1024:.2f} GB)")
    print()
    
    # 运行演示
    print("运行网络...")
    print()
    
    sensory_n = network.sensory.num_neurons
    
    # 阶段1：持续刺激
    print("--- 阶段1：持续感官刺激 ---")
    for step in range(10):
        # 随机刺激
        stimulus = np.random.rand(sensory_n).astype(np.float32) * 0.3
        # 在特定区域增强刺激
        center = sensory_n // 2
        width = sensory_n // 10
        stimulus[center-width:center+width] *= 3
        
        network.input_stimulus(stimulus)
        state = network.step(dt=1.0)
        
        if step % 2 == 0:
            print(f"  第 {step} 步: "
                  f"感官={state['sensory']['firing_rate']:.1f} Hz, "
                  f"联想={state['association']['firing_rate']:.1f} Hz, "
                  f"决策={state['decision']['firing_rate']:.1f} Hz")
    
    print()
    
    # 阶段2：无刺激（观察衰减）
    print("--- 阶段2：无刺激（观察活动衰减） ---")
    for step in range(10):
        state = network.step(dt=1.0)
        if step % 2 == 0:
            print(f"  第 {step} 步: "
                  f"感官={state['sensory']['firing_rate']:.1f} Hz, "
                  f"联想={state['association']['firing_rate']:.1f} Hz")
    
    print()
    
    # 性能统计
    print("--- 性能统计 ---")
    summary = network.get_summary()
    print(f"  总神经元数: {summary['total_neurons']:,}")
    print(f"  感官层神经元: {summary['sensory_neurons']:,}")
    print(f"  联想层神经元: {summary['association_neurons']:,}")
    print(f"  决策层神经元: {summary['decision_neurons']:,}")
    print()
    
    # 估算不同规模的性能
    print("--- 规模扩展估算 ---")
    print(f"  {'规模':>12} | {'创建时间':>10} | {'内存':>10} | {'每步时间':>10}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    
    for scale in [100000, 500000, 1000000, 2000000, 5000000]:
        ratio = scale / total_neurons
        est_create = create_time * ratio
        est_mem = total_mem * ratio
        est_step = 0.15 * ratio  # 估算每步时间
        
        print(f"  {scale:>12,} | {est_create:>8.1f}s | {est_mem/1024:>7.2f}GB | {est_step:>8.2f}s")
    
    print()
    print("=" * 60)
    print("演示完成！")
    print("=" * 60)
    print()
    print("提示：")
    print("  - 修改 total_neurons 变量可以调整网络规模")
    print("  - 500万神经元约需3.75GB内存，创建约20秒，每步约0.8秒")
    print("  - connections_per_neuron 控制每个神经元的连接数")
    print("  - 稀疏连接大幅减少了内存和计算量")


if __name__ == "__main__":
    demo_large_scale()
