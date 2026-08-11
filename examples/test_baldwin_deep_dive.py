"""
Baldwin效应深化研究测试脚本
测试Baldwin Effect Deep Dive系统
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.baldwin_deep_dive import (
    BaldwinEffectStudy,
    LearningDifficulty,
    create_task_with_difficulty,
    print_baldwin_report
)
from brain.genetics import create_default_genome


def main():
    print("\n" + "=" * 70)
    print("  Baldwin效应深化研究测试")
    print("  Baldwin Effect Deep Dive Test")
    print("=" * 70)
    print()
    
    # 创建基准基因组
    base_genome = create_default_genome("base")
    
    # 研究：中等难度任务
    print("=" * 70)
    print("  研究：中等难度任务")
    print("=" * 70)
    print()
    
    study = BaldwinEffectStudy(
        population_size=15,
        mutation_rate=0.02,
        crossover_rate=0.7,
        elitism=2,
        seed=42
    )
    
    study.initialize_population(base_genome)
    
    medium_task = create_task_with_difficulty(LearningDifficulty.MEDIUM)
    
    result = study.run_study(
        task_function=medium_task,
        num_generations=10,
        learning_steps=30,
        learning_cost_factor=0.1,
        difficulty=LearningDifficulty.MEDIUM
    )
    
    print_baldwin_report(result)
    
    print("\n" + "=" * 70)
    print("✓ Baldwin效应深化研究测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
