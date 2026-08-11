"""测试大规模网络性能"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from brain.large_network import LargeThreeLayerNetwork

print("=" * 60)
print("  大规模网络性能测试")
print("=" * 60)
print()

# 测试不同规模
sizes = [
    ("1万神经元", 10000),
    ("5万神经元", 50000),
    ("10万神经元", 100000),
]

for name, total in sizes:
    sensory = int(total * 0.1)
    association = int(total * 0.85)
    decision = total - sensory - association
    
    print(f"【{name}】")
    print(f"  感官: {sensory:,} + 联想: {association:,} + 决策: {decision:,}")
    
    start = time.time()
    net = LargeThreeLayerNetwork(
        sensory_neurons=sensory,
        association_neurons=association,
        decision_neurons=decision,
        connection_density=0.01,
        seed=42
    )
    init_time = time.time() - start
    print(f"  构建时间: {init_time:.2f}s")
    
    # 运行10步
    start = time.time()
    for i in range(10):
        stimulus = np.random.rand(sensory) * 0.3
        net.input_stimulus(stimulus)
        net.step(1.0)
    step_time = time.time() - start
    
    print(f"  10步时间: {step_time:.3f}s")
    print(f"  平均每步: {step_time/10*1000:.1f}ms")
    print(f"  每秒步数: {10/step_time:.1f} steps/s")
    print()

print("=" * 60)
print("  测试完成！")
print("=" * 60)
