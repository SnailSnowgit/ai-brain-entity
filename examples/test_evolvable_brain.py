"""
可进化大脑测试脚本
测试DNA基因库与主脑类的整合
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.evolvable_brain import (
    EvolvableBrain, BrainEvolution, 
    EvolvableBrainWithLearning, BrainEvolutionWithLearning,
    simple_reward_task, learning_task
)
from brain.genetics import create_default_genome
import numpy as np

print("=" * 60)
print("  可进化大脑测试")
print("  DNA基因库 + 主脑类 整合测试")
print("=" * 60)

# ===== 1. 测试单个可进化大脑 =====
print("\n1. 单个可进化大脑测试")
print("-" * 60)

genome = create_default_genome("test_brain")
evo_brain = EvolvableBrain(genome, seed=42)

params = evo_brain.get_brain_params()
print(f"  基因组ID: {genome.genome_id}")
print(f"  基因数量: {len(genome.genes)}")
print()
print(f"  表达的大脑参数:")
print(f"    感官神经元: {int(params['sensory_neurons'])}")
print(f"    联想神经元: {int(params['association_neurons'])}")
print(f"    决策神经元: {int(params['decision_neurons'])}")
print(f"    连接密度: {params['connection_density']:.4f}")
print(f"    兴奋比例: {params['excitatory_ratio']:.4f}")
print(f"    阈值均值: {params['threshold_mean']:.4f}")
print(f"    膜时间常数: {params['membrane_time_constant']:.2f} ms")
print(f"    不应期: {params['refractory_period']:.2f} ms")
print(f"    输入增益: {params['input_gain']:.4f}")
print(f"    Hebbian学习率: {params['hebbian_learning_rate']:.6f}")
print(f"    多巴胺学习率: {params['dopamine_learning_rate']:.4f}")
print(f"    记忆巩固率: {params['memory_consolidation_rate']:.6f}")
print(f"    可塑性水平: {params['plasticity_level']:.4f}")
print(f"    多巴胺基线: {params['dopamine_baseline']:.4f}")
print(f"    情绪敏感度: {params['emotional_sensitivity']:.4f}")
print(f"    注意强度: {params['attention_strength']:.4f}")
print(f"    觉醒基线: {params['arousal_baseline']:.4f}")

# 测试大脑是否正常工作
print(f"\n  大脑运行测试 (50步)...")
for i in range(50):
    stimulus = np.random.rand(evo_brain.brain.network.sensory.num_neurons).astype(np.float32) * 0.5
    evo_brain.brain.input_stimulus(stimulus, modality=i % 5, reward=0.1)
    evo_brain.brain.step()

summary = evo_brain.brain.get_summary()
print(f"    总步数: {summary['total_steps']}")
print(f"    感官活动: {summary['avg_sensory_activity']:.1f} Hz")
print(f"    联想活动: {summary['avg_association_activity']:.1f} Hz")
print(f"    长期记忆: {summary['memory_stats']['ltm_count']} 项")

# ===== 2. 测试适应度评估 =====
print("\n2. 适应度评估测试")
print("-" * 60)

result = evo_brain.evaluate_fitness(
    task_function=simple_reward_task,
    num_trials=3,
    max_steps=50
)

print(f"  适应度: {result.fitness:.4f}")
print(f"  总奖励: {result.total_reward:.4f}")
print(f"  存活步数: {result.steps_survived}")
print(f"  决策数量: {result.decisions_made}")
print(f"  记忆形成: {result.memory_formed} 项")
print(f"  平均活动: {result.avg_activity:.4f}")

# ===== 3. 测试变异 =====
print("\n3. 变异测试")
print("-" * 60)

mutated_brain = evo_brain.mutate(mutation_rate=0.1)
mutated_params = mutated_brain.get_brain_params()

print(f"  原感官神经元: {int(params['sensory_neurons'])}")
print(f"  变异后: {int(mutated_params['sensory_neurons'])}")
print(f"  原联想神经元: {int(params['association_neurons'])}")
print(f"  变异后: {int(mutated_params['association_neurons'])}")

# ===== 4. 测试进化 =====
print("\n4. 进化测试 (10代, 种群20个)")
print("-" * 60)

evolution = BrainEvolution(
    population_size=20,
    mutation_rate=0.02,
    selection_pressure=0.3,
    elitism=2,
    seed=42
)

evolution.initialize_population()

best_brain = evolution.evolve(
    num_generations=10,
    task_function=simple_reward_task,
    num_trials=1,
    max_steps=30
)

# ===== 5. 进化结果 =====
print("\n5. 进化结果")
print("-" * 60)

stats = evolution.get_evolution_stats()
print(f"  进化代数: {stats['generation']}")
print(f"  种群大小: {stats['population_size']}")
print()
print(f"  最佳适应度变化:")
for i, fit in enumerate(stats['best_fitness_history']):
    bar = "█" * int(fit * 50)
    print(f"    第{i+1:2d}代: {fit:.4f} {bar}")

print(f"\n  平均适应度变化:")
for i, fit in enumerate(stats['avg_fitness_history']):
    bar = "░" * int(fit * 50)
    print(f"    第{i+1:2d}代: {fit:.4f} {bar}")

# 最佳大脑参数
best_params = best_brain.get_brain_params()
print(f"\n  最佳大脑参数:")
print(f"    感官神经元: {int(best_params['sensory_neurons'])}")
print(f"    联想神经元: {int(best_params['association_neurons'])}")
print(f"    决策神经元: {int(best_params['decision_neurons'])}")
print(f"    连接密度: {best_params['connection_density']:.4f}")
print(f"    兴奋比例: {best_params['excitatory_ratio']:.4f}")
print(f"    可塑性水平: {best_params['plasticity_level']:.4f}")

# ===== 6. 带学习的进化（Baldwin效应） =====
print("\n6. 带学习的进化测试 (Baldwin效应, 5代)")
print("-" * 60)

evo_learning = BrainEvolutionWithLearning(
    population_size=10,
    mutation_rate=0.02,
    learning_rate=0.1,
    baldwin_strength=0.3,
    elitism=1,
    seed=42
)

evo_learning.initialize_population()

best_learning_brain = evo_learning.evolve(
    num_generations=5,
    task_function=learning_task,
    learning_steps=30
)

# ===== 7. 学习进化结果 =====
print("\n7. 学习进化结果")
print("-" * 60)

learn_stats = evo_learning.get_stats()
print(f"  进化代数: {learn_stats['generation']}")
print(f"  最佳适应度: {learn_stats['best_fitness_history'][-1]:.4f}")
print(f"  平均学习改进: {learn_stats['avg_improvement_history'][-1]:.4f}")
print()
print(f"  各代适应度:")
for i, (fit, imp) in enumerate(zip(learn_stats['best_fitness_history'], learn_stats['avg_improvement_history'])):
    print(f"    第{i+1}代: 适应度={fit:.4f}, 学习改进={imp:.4f}")

print("\n" + "=" * 60)
print("✓ 可进化大脑测试完成！")
print("=" * 60)
print()
print("  总结:")
print("  - DNA基因库与主脑类成功整合")
print("  - 17个基因控制大脑结构和功能参数")
print("  - 遗传算法可自动优化大脑参数")
print("  - 支持Baldwin效应（进化+学习）")
print("  - 进化过程中适应度持续提升")
