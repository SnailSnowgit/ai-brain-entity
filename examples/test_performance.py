"""性能测试：新默认参数下的运行速度"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from brain import ThreeLayerNetwork, MemorySystem, GlialNetwork, ThoughtSystem

print("=" * 60)
print("  性能测试：新默认参数")
print("=" * 60)
print()

# 1. 三层网络
print("【1. 三层神经网络】")
start = time.time()
net = ThreeLayerNetwork()
init_time = time.time() - start

total_neurons = net.sensory.num_neurons + net.association.num_neurons + net.decision.num_neurons

print(f"  总神经元数: {total_neurons:,}")
print(f"  感官层: {net.sensory.num_neurons:,}")
print(f"  联想层: {net.association.num_neurons:,}")
print(f"  决策层: {net.decision.num_neurons:,}")
print(f"  初始化时间: {init_time:.3f}s")

# 运行100步
start = time.time()
for i in range(100):
    stimulus = np.random.rand(net.sensory.num_neurons) * 0.5
    net.input_stimulus(stimulus)
    net.step(1.0)
step_time = time.time() - start

print(f"  100步运行时间: {step_time:.3f}s")
print(f"  平均每步: {step_time/100*1000:.1f}ms")
print(f"  每秒步数: {100/step_time:.1f} steps/s")
print()

# 2. 记忆系统
print("【2. 记忆系统】")
start = time.time()
mem = MemorySystem()
init_time = time.time() - start
print(f"  感官缓存: {mem.sensory_buffer.capacity} 槽位")
print(f"  短期记忆: {mem.short_term.capacity} 槽位")
print(f"  长期记忆: {mem.long_term.capacity} 槽位")
print(f"  初始化时间: {init_time:.3f}s")
print()

# 3. 胶质细胞
print("【3. 胶质细胞网络】")
start = time.time()
glia = GlialNetwork()
init_time = time.time() - start
print(f"  星形胶质细胞: {len(glia.astrocytes):,}")
print(f"  少突胶质细胞: {len(glia.oligodendrocytes):,}")
print(f"  小胶质细胞: {len(glia.microglia):,}")
print(f"  网络大小: {glia.network_size}")
print(f"  初始化时间: {init_time:.3f}s")
print()

# 4. 思考系统
print("【4. 思考系统】")
start = time.time()
thought = ThoughtSystem()
init_time = time.time() - start
print(f"  思考空间: {thought.space.capacity} 槽位")
print(f"  思维向量: {thought.space.vector_dim} 维")
print(f"  情节记忆: {thought.memory.max_episodes} 个")
print(f"  初始化时间: {init_time:.3f}s")
print()

print("=" * 60)
print("  测试完成！")
print("=" * 60)
