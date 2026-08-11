"""
预测编码与自由能原理测试脚本
测试Predictive Coding & Free Energy Principle系统
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.predictive_coding import (
    PredictiveCodingNetwork,
    ActiveInference,
    run_prediction_coding_demo,
    run_free_energy_minimization_demo,
    run_active_inference_demo,
    print_predictive_coding_report
)
import numpy as np


def main():
    print("\n" + "=" * 70)
    print("  预测编码与自由能原理测试")
    print("  Predictive Coding & Free Energy Principle Test")
    print("=" * 70)
    print()
    
    # 测试1：预测编码基本功能
    print("=" * 70)
    print("  测试1：预测编码基本功能")
    print("=" * 70)
    
    pc, fe_hist = run_prediction_coding_demo()
    print_predictive_coding_report(pc, fe_hist)
    
    # 测试2：自由能最小化
    print("\n" + "=" * 70)
    print("  测试2：自由能最小化")
    print("=" * 70)
    
    pc2, fe_hist2 = run_free_energy_minimization_demo()
    
    # 测试3：主动推理
    print("\n" + "=" * 70)
    print("  测试3：主动推理")
    print("=" * 70)
    
    ai, fe_hist3 = run_active_inference_demo()
    
    # 测试4：精度加权（注意力）
    print("\n" + "=" * 70)
    print("  测试4：精度加权（注意力机制）")
    print("=" * 70)
    
    pc3 = PredictiveCodingNetwork(
        layer_sizes=[40, 20, 10],
        layer_names=["输入", "特征", "概念"],
        learning_rate=0.01,
        seed=789
    )
    
    # 学习一个清晰模式
    clear_pattern = np.sin(np.linspace(0, 4*np.pi, 40)) * 0.5 + 0.5
    pc3.learn_pattern(clear_pattern, num_steps=80, verbose=False)
    
    # 测试不同噪声水平下的精度
    print("\n  不同噪声水平下的精度变化:")
    print("  " + "-" * 50)
    
    noise_levels = [0.0, 0.1, 0.3, 0.5, 0.8]
    
    for noise in noise_levels:
        noisy_pattern = clear_pattern + np.random.randn(40) * noise
        pc3.step(noisy_pattern)
        
        precision = pc3.layers[0].precision
        error = np.mean(np.abs(pc3.layers[0].prediction_error))
        
        print(f"  噪声 {noise:.1f}: 精度={precision:.4f}, 误差={error:.4f}")
    
    # 测试5：层级预测
    print("\n" + "=" * 70)
    print("  测试5：层级预测结构")
    print("=" * 70)
    
    pc4 = PredictiveCodingNetwork(
        layer_sizes=[64, 32, 16, 8],
        layer_names=["视网膜", "V1", "V2", "IT"],
        learning_rate=0.015,
        time_constant=6.0,
        seed=101
    )
    
    # 学习一个复杂模式
    pattern = np.zeros(64)
    for freq in [1, 3, 5]:
        pattern += np.sin(np.linspace(0, freq*2*np.pi, 64)) / freq
    pattern = pattern / np.max(np.abs(pattern)) * 0.5 + 0.5
    
    print("\n  学习复杂模式（多频率叠加）...")
    fe_hist4 = pc4.learn_pattern(pattern, num_steps=100, verbose=False)
    
    print(f"  初始自由能: {fe_hist4[0]:.4f}")
    print(f"  最终自由能: {fe_hist4[-1]:.4f}")
    print(f"  下降: {fe_hist4[0] - fe_hist4[-1]:.4f}")
    
    print("\n  各层表征维度:")
    for i, layer in enumerate(pc4.layers):
        activity = layer.activity
        print(f"  层 {i} ({layer.name}): {layer.num_units}维, 活动幅度={np.mean(np.abs(activity)):.4f}")
    
    print("\n" + "=" * 70)
    print("✓ 预测编码与自由能原理测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
