"""
AI认知架构设计系统测试脚本
测试Cognitive Architecture Designer系统
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.cognitive_architecture_designer import (
    CognitiveArchitectureDesigner,
    TaskCategory,
    print_architecture_card,
    print_recommendation_report,
    print_comparison_chart
)


def main():
    print("\n" + "=" * 70)
    print("  AI认知架构设计系统测试")
    print("  Cognitive Architecture Designer Test")
    print("=" * 70)
    print()
    
    # 创建设计器
    designer = CognitiveArchitectureDesigner()
    
    # 1. 展示所有架构
    print("=" * 70)
    print("  1. 所有预设架构")
    print("=" * 70)
    
    for arch in designer.list_architectures():
        print_architecture_card(arch)
    
    # 2. 任务推荐示例
    print("\n" + "=" * 70)
    print("  2. 任务推荐示例")
    print("=" * 70)
    
    tasks = [
        ("实时游戏AI", TaskCategory.REAL_TIME_DECISION),
        ("知识库问答系统", TaskCategory.MEMORY_INTENSIVE),
        ("图像识别系统", TaskCategory.PATTERN_RECOGNITION),
        ("聊天机器人", TaskCategory.EMOTIONAL_INTERACTION),
        ("在线学习系统", TaskCategory.ONLINE_LEARNING),
        ("服务器监控系统", TaskCategory.STABLE_OPERATION),
        ("创意写作助手", TaskCategory.CREATIVE),
        ("通用AI助手", TaskCategory.GENERAL_INTELLIGENCE),
    ]
    
    for task_name, task_cat in tasks:
        rec = designer.recommend_for_task(task_name, task_cat)
        print_recommendation_report(rec)
    
    # 3. 架构对比
    print("\n" + "=" * 70)
    print("  3. 架构对比：记忆型 vs 速度型 vs 平衡型")
    print("=" * 70)
    
    comparison = designer.compare_architectures(
        ["记忆架构师 (Memory Architect)", "速度恶魔 (Speed Demon)", "平衡通才 (Balanced Generalist)"],
        TaskCategory.GENERAL_INTELLIGENCE
    )
    print_comparison_chart(comparison)
    
    # 4. 生成基因组示例
    print("\n" + "=" * 70)
    print("  4. 从架构生成基因组")
    print("=" * 70)
    
    speed_arch = designer.get_architecture("速度恶魔 (Speed Demon)")
    if speed_arch:
        genome = designer.generate_genome(speed_arch)
        print(f"  架构: {speed_arch.name}")
        print(f"  基因组ID: {genome.genome_id}")
        print(f"  基因数: {len(genome.genes)}")
        print()
        print("  关键基因值:")
        for gene_name in ['sensory_neurons', 'association_neurons', 'membrane_time_constant', 
                          'refractory_period', 'plasticity_level', 'attention_strength']:
            if gene_name in genome.genes:
                print(f"    {gene_name}: {genome.genes[gene_name].value:.4f}")
    
    print("\n" + "=" * 70)
    print("✓ AI认知架构设计系统测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
