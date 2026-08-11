"""
进化强化学习测试脚本
测试Evolutionary RL系统
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.evolutionary_rl import EvolutionaryRL, print_evo_rl_report
from brain.genetics import create_default_genome


def main():
    print("\n" + "=" * 70)
    print("  进化强化学习测试")
    print("  Evolutionary RL Test")
    print("=" * 70)
    print()
    
    # 创建进化RL系统
    evo_rl = EvolutionaryRL(
        population_size=12,
        mutation_rate=0.02,
        crossover_rate=0.7,
        elitism=2,
        tournament_size=3,
        rl_episodes=20,
        rl_max_steps=25,
        baldwin_strength=0.5,
        seed=42
    )
    
    # 初始化种群
    template = create_default_genome("template")
    evo_rl.initialize_population(template)
    
    # 执行进化
    result = evo_rl.evolve(num_generations=6)
    
    # 打印报告
    print_evo_rl_report(result)
    
    print("\n" + "=" * 70)
    print("✓ 进化强化学习测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
