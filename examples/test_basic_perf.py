"""测试基础网络性能（新默认参数）"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from brain import ThreeLayerNetwork

print("=" * 60)
print("  基础网络性能测试（新默认参数）")
print("=" * 60)
print()

start = time.time()
net = ThreeLayerNetwork()
init_time = time.time() - start

total = net.sensory.num_neurons + net.association.num_neurons + net.decision.num_neurons
print(f"总神经元: {total:,}")
print(f"  感官层: {net.sensory.num_neurons:,}")
print(f"  联想层: {net.association.num_neurons:,}")
print(f"  决策层: {net.decision.num_neurons:,}")
print(f"初始化时间: {init_time:.3f}s")
print()

# 运行50步
start = time.time()
for i in range(50):
    stimulus = np.random.rand(net.sensory.num_neurons) * 0.3
    net.input_stimulus(stimulus)
    net.step(1.0)
step_time = time.time() - start

print(f"50步运行时间: {step_time:.3f}s")
print(f"平均每步: {step_time/50*1000:.1f}ms")
print(f"每秒步数: {50/step_time:.1f} steps/s")
print()
print("=" * 60)
