"""
个体差异研究测试脚本
测试Individual Differences系统
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.individual_diff import IndividualDifferences, print_individual_diff_report
from brain.genetics import create_default_genome


def main():
    print("\n" + "=" * 70)
    print("  个体差异研究测试")
    print("  Individual Differences Study Test")
    print("=" * 70)
    print()
    
    # 创建研究系统
    study = IndividualDifferences(
        num_individuals=25,
        seed=42
    )
    
    # 运行研究
    result = study.run_study(variation=0.6)
    
    # 打印报告
    print_individual_diff_report(result)
    
    print("\n" + "=" * 70)
    print("✓ 个体差异研究测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
