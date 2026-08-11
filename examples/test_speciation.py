"""
物种形成模拟测试脚本
测试Speciation系统
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.speciation import SpeciesSystem, multi_niche_task, print_speciation_report
from brain.genetics import create_default_genome


def main():
    print("\n" + "=" * 70)
    print("  物种形成模拟测试")
    print("  Speciation Simulation Test")
    print("=" * 70)
    print()
    
    # 创建物种系统
    species_system = SpeciesSystem(
        population_size=24,
        num_initial_species=3,
        speciation_threshold=0.25,
        min_species_size=2,
        max_species=6,
        mutation_rate=0.02,
        crossover_rate=0.7,
        elitism_per_species=1,
        tournament_size=3,
        niche_based_fitness=True,
        seed=42
    )
    
    # 初始化物种
    template = create_default_genome("template")
    species_system.initialize_species(template)
    
    # 执行进化
    result = species_system.evolve(
        task_function=multi_niche_task,
        num_generations=10,
        num_trials=2,
        max_steps=50,
        migration_rate=0.03
    )
    
    # 打印报告
    print_speciation_report(result)
    
    print("\n" + "=" * 70)
    print("✓ 物种形成模拟测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
