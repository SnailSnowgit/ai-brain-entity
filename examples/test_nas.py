"""
神经架构搜索测试脚本
测试BrainNAS系统
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.nas import BrainNAS, multi_task_evaluation, print_nas_report, save_nas_result
from brain.genetics import create_default_genome


def main():
    print("\n" + "=" * 70)
    print("  神经架构搜索测试")
    print("  BrainNAS System Test")
    print("=" * 70)
    print()
    
    # 创建NAS搜索器
    nas = BrainNAS(
        population_size=20,
        mutation_rate=0.02,
        crossover_rate=0.7,
        selection_pressure=0.3,
        elitism=2,
        tournament_size=3,
        adaptive_mutation=True,
        seed=42
    )
    
    # 初始化种群
    template = create_default_genome("template")
    nas.initialize_population(template)
    
    # 执行搜索（10代，快速测试）
    result = nas.search(
        task_function=multi_task_evaluation,
        num_generations=10,
        num_trials=2,
        max_steps=60
    )
    
    # 打印报告
    print_nas_report(result)
    
    # 保存结果
    save_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results',
        'nas_result.json'
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_nas_result(result, save_path)
    
    print("\n" + "=" * 70)
    print("✓ 神经架构搜索测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
