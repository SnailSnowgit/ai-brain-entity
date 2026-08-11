"""快速测试所有核心模块"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain import *

print("=== 核心模块测试 ===")
print()

# 1. 基础网络
net = ThreeLayerNetwork()
print(f"1. 三层网络: {net.sensory.num_neurons} + {net.association.num_neurons} + {net.decision.num_neurons} 神经元 ✓")

# 2. 记忆系统
mem = MemorySystem()
print(f"2. 记忆系统: 感官{mem.sensory_buffer.capacity} + 短期{mem.short_term_memory.capacity} + 长期{mem.long_term_memory.capacity} ✓")

# 3. 调制系统
mod = ModulationSystem()
print(f"3. 调制系统: 情绪+注意+多巴胺 ✓")

# 4. 胶质细胞
glia = GlialNetwork(100)
print(f"4. 胶质网络: {len(glia.astrocytes)}星形 + {len(glia.oligodendrocytes)}少突 + {len(glia.microglia)}小胶质 ✓")

# 5. 思考系统
thought = ThoughtSystem(150)
print(f"5. 思考系统: 空间{thought.thought_space.capacity}槽位 ✓")

# 6. 感知系统
perc = PerceptualSystem()
print(f"6. 感知系统: 视觉+语言 ✓")

# 7. 脑区域
regions = BrainRegions()
print(f"7. 脑区域: 海马+前额叶+丘脑+基底节 ✓")

# 8. 遗传进化
gene_pool = GenePool(population_size=10)
print(f"8. 遗传系统: 种群{gene_pool.population_size} ✓")

print()
print("所有核心模块正常！")
