"""
发育生物学测试脚本
测试Developmental Biology系统
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.developmental import (
    DevelopmentalBrain,
    simulate_normal_development,
    simulate_neurodevelopmental_disorder,
    print_development_report,
    compare_development
)
from brain.genetics import create_default_genome


def main():
    print("\n" + "=" * 70)
    print("  发育生物学测试")
    print("  Developmental Biology Test")
    print("=" * 70)
    print()
    
    # 创建基准基因组
    base_genome = create_default_genome("normal")
    
    # 模拟正常发育
    print("=" * 70)
    print("  正常发育")
    print("=" * 70)
    print()
    
    normal_result = simulate_normal_development(base_genome, seed=42)
    print_development_report(normal_result)
    
    # 模拟自闭症
    print("\n" + "=" * 70)
    print("  自闭症谱系障碍 (ASD) 模拟")
    print("=" * 70)
    print()
    
    asd_result = simulate_neurodevelopmental_disorder(base_genome, "autism", seed=42)
    print_development_report(asd_result)
    
    # 对比
    compare_development(normal_result, asd_result, "自闭症")
    
    print("\n" + "=" * 70)
    print("✓ 发育生物学测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
