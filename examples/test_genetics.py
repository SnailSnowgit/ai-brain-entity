"""
遗传与进化系统测试脚本
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.genetics import GenePool, Genome, Gene, GeneType, GeneExpression, EvolutionaryLearning, create_default_genome
import numpy as np

print("=" * 60)
print("  遗传与进化系统测试")
print("=" * 60)

# 1. 测试基因
print("\n1. 基因测试")
print("-" * 40)

gene = Gene(
    gene_id="test_gene",
    gene_type=GeneType.STRUCTURAL,
    value=0.5,
    dominance=0.7
)
print(f"  基因ID: {gene.gene_id}")
print(f"  类型: {gene.gene_type.value}")
print(f"  值: {gene.value}")
print(f"  显性: {gene.dominance}")

# 突变测试
mutations = 0
for _ in range(1000):
    if gene.mutate(mutation_rate=0.1):
        mutations += 1
print(f"  1000次尝试突变数: {mutations} (预期~100)")

# 2. 测试基因组
print("\n2. 基因组测试")
print("-" * 40)

genome = create_default_genome("test_genome")
stats = genome.get_gene_stats()
print(f"  基因组ID: {genome.genome_id}")
print(f"  基因总数: {stats['total_genes']}")
print(f"  类型分布: {stats['type_counts']}")
print(f"  基因值均值: {stats['mean_value']:.4f}")

# 克隆测试
clone = genome.clone()
print(f"  克隆成功: {clone.genome_id}")

# 交叉测试
genome2 = create_default_genome("test_genome2")
child = genome.crossover(genome2, method="uniform")
print(f"  交叉子代基因数: {len(child.genes)}")
print(f"  子代代数: {child.generation}")

# 3. 测试基因库
print("\n3. 基因库测试")
print("-" * 40)

gene_pool = GenePool(
    population_size=30,
    mutation_rate=0.02,
    selection_pressure=0.3,
    elitism=2
)

template = create_default_genome("template")
gene_pool.initialize_with_template(template)

pool_stats = gene_pool.get_stats()
print(f"  种群大小: {pool_stats['population_size']}")

# 4. 测试进化
print("\n4. 进化测试 (10代)")
print("-" * 40)

# 简单适应度函数：基因值接近0.5的程度
def fitness_func(genome):
    values = [g.value for g in genome.genes.values()]
    fitness = 1.0 - np.mean([abs(v - 0.5) for v in values])
    return fitness

best = gene_pool.evolve(num_generations=10, fitness_func=fitness_func)

best_stats = best.get_gene_stats()
print(f"\n进化结果:")
print(f"  最佳适应度: {best.fitness:.4f}")
print(f"  最佳基因组: {best.genome_id}")
print(f"  进化代数: {gene_pool.generation}")
print(f"  基因数: {best_stats['total_genes']}")

# 5. 测试基因表达
print("\n5. 基因表达测试")
print("-" * 40)

expression = GeneExpression()
params = expression.express(best)

print("  表达的参数:")
for key, value in params.items():
    print(f"    {key}: {value}")

# 6. 测试进化学习
print("\n6. 进化学习测试 (5代)")
print("-" * 40)

evo_learning = EvolutionaryLearning(
    population_size=10,
    mutation_rate=0.02,
    lifetime_learning_rate=0.1,
    baldwin_effect_strength=0.3
)

evo_learning.initialize_population()

# 简单环境函数
def simple_env(params, learning=False, step=0):
    values = [v for v in params.values() if isinstance(v, (int, float))]
    base_fitness = sum(values) / len(values) / 100.0  # 归一化
    
    if learning:
        # 学习过程中逐步改进
        base_fitness += step * 0.001
    
    return base_fitness

best_genome, evo_stats = evo_learning.evolve_with_learning(
    num_generations=5,
    env_function=simple_env,
    learning_steps_per_gen=20
)

print(f"\n进化学习结果:")
print(f"  最佳适应度: {evo_stats['best_fitness']:.4f}")
print(f"  进化代数: {evo_stats['num_generations']}")

# 7. 进化历史
print("\n7. 进化历史")
print("-" * 40)

history = gene_pool.best_fitness_history
print("  最佳适应度变化:")
for i, fit in enumerate(history):
    bar = "█" * int(fit * 30)
    print(f"    第{i+1:2d}代: {fit:.4f} {bar}")

print("\n" + "=" * 60)
print("✓ 遗传与进化系统测试完成！")
print("=" * 60)
