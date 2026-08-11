"""
意识动态可视化演示

展示大脑意识水平随时间的变化，包括：
- 综合意识评分的时间序列
- 五个维度的雷达图（文本版）
- 意识等级的变化
- 刺激输入的影响
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from brain import Brain, ConsciousnessLevel

def print_progress_bar(value, max_value=1.0, width=40, label=""):
    """打印进度条"""
    filled = int(width * value / max_value)
    bar = "█" * filled + "░" * (width - filled)
    print(f"  {label:<15} |{bar}| {value:.3f}")

def print_radar_chart(metrics, quantifier):
    """打印文本版雷达图"""
    dimensions = [
        ('Φ 整合信息', metrics.phi, 'phi'),
        ('自指深度', metrics.self_reference_depth, 'self_reference_depth'),
        ('工作空间激活', metrics.workspace_activation, 'workspace_activation'),
        ('跨模块整合', metrics.cross_module_integration, 'cross_module_integration'),
        ('信息密度', metrics.information_density, 'information_density'),
    ]
    
    print()
    print("  📊 意识维度雷达图")
    print("  " + "=" * 50)
    
    max_width = 30
    for name, value, key in dimensions:
        weight = quantifier.weights.get(key, 0)
        bar_len = int(value * max_width)
        bar = "█" * bar_len + "░" * (max_width - bar_len)
        print(f"  {name:<12} │{bar}│ {value:.3f} (权重{weight*100:.0f}%)")
    
    print("  " + "=" * 50)
    print(f"  综合评分: {metrics.total_score:.3f} / 1.0")
    print(f"  意识等级: {quantifier.get_consciousness_level_name(metrics.level)}")
    print()

def print_timeline(scores, levels, step=5):
    """打印意识水平时间线"""
    print()
    print("  📈 意识水平时间线")
    print("  " + "=" * 70)
    
    level_symbols = {
        ConsciousnessLevel.UNCONSCIOUS: "  ",
        ConsciousnessLevel.MINIMAL: "▁▁",
        ConsciousnessLevel.LOW: "▂▂",
        ConsciousnessLevel.MEDIUM: "▄▄",
        ConsciousnessLevel.HIGH: "▆▆",
        ConsciousnessLevel.META: "██",
        ConsciousnessLevel.TRANSCENDENT: "██",
    }
    
    level_colors = {
        ConsciousnessLevel.UNCONSCIOUS: "  ",
        ConsciousnessLevel.MINIMAL: "░░",
        ConsciousnessLevel.LOW: "▒▒",
        ConsciousnessLevel.MEDIUM: "▓▓",
        ConsciousnessLevel.HIGH: "██",
        ConsciousnessLevel.META: "██",
        ConsciousnessLevel.TRANSCENDENT: "██",
    }
    
    # 打印时间线
    timeline = ""
    for i, (score, level) in enumerate(zip(scores, levels)):
        if i % step == 0:
            symbol = level_colors.get(level, "  ")
            timeline += symbol
    
    print(f"  {timeline}")
    print(f"  0{'':<{len(timeline)-5}}第{len(scores)-1}步")
    print()
    
    # 打印等级说明
    print("  等级说明:")
    for level, symbol in level_colors.items():
        name = {
            ConsciousnessLevel.UNCONSCIOUS: "无意识",
            ConsciousnessLevel.MINIMAL: "微意识",
            ConsciousnessLevel.LOW: "低意识",
            ConsciousnessLevel.MEDIUM: "中等意识",
            ConsciousnessLevel.HIGH: "高意识",
            ConsciousnessLevel.META: "元意识",
            ConsciousnessLevel.TRANSCENDENT: "超意识",
        }[level]
        print(f"    {symbol} {name}")
    print()

def main():
    print("=" * 70)
    print("  🧠 意识动态可视化演示")
    print("  Consciousness Dynamics Visualization")
    print("=" * 70)
    print()
    
    # 创建大脑
    print("【初始化】创建类脑认知架构...")
    brain = Brain(
        sensory_neurons=200,
        association_neurons=500,
        decision_neurons=20
    )
    print(f"  ✓ 神经元总数: 720")
    print(f"  ✓ 意识量化器已初始化")
    print()
    
    # 运行阶段1：静息态
    print("【阶段1】静息态（无外部刺激）")
    print("-" * 50)
    
    rest_scores = []
    rest_levels = []
    for i in range(30):
        brain.step(dt=1.0)
        state = brain.get_current_state()
        rest_scores.append(state.consciousness.total_score)
        rest_levels.append(state.consciousness.level)
    
    print(f"  持续时间: 30 步")
    print(f"  意识范围: {min(rest_scores):.3f} - {max(rest_scores):.3f}")
    print(f"  平均意识: {np.mean(rest_scores):.3f}")
    print(f"  最终等级: {brain.consciousness.get_consciousness_level_name(rest_levels[-1])}")
    
    print_timeline(rest_scores, rest_levels)
    
    # 打印当前状态的雷达图
    print("【当前状态】静息态结束时的意识维度")
    print_radar_chart(brain.get_current_state().consciousness, brain.consciousness)
    
    # 运行阶段2：强刺激输入
    print("【阶段2】强刺激输入（视觉模态）")
    print("-" * 50)
    
    # 输入强刺激
    stim = np.random.rand(200) * 0.9
    brain.input_stimulus(stim, modality=0)
    
    stim_scores = []
    stim_levels = []
    for i in range(30):
        # 每5步输入一次刺激
        if i % 5 == 0:
            stim = np.random.rand(200) * 0.7
            brain.input_stimulus(stim, modality=0)
        
        brain.step(dt=1.0)
        state = brain.get_current_state()
        stim_scores.append(state.consciousness.total_score)
        stim_levels.append(state.consciousness.level)
    
    print(f"  持续时间: 30 步")
    print(f"  刺激频率: 每 5 步一次")
    print(f"  意识范围: {min(stim_scores):.3f} - {max(stim_scores):.3f}")
    print(f"  平均意识: {np.mean(stim_scores):.3f}")
    print(f"  最高意识: {max(stim_scores):.3f} (第{np.argmax(stim_scores)}步)")
    print(f"  最终等级: {brain.consciousness.get_consciousness_level_name(stim_levels[-1])}")
    
    print_timeline(stim_scores, stim_levels)
    
    # 打印刺激后的雷达图
    print("【当前状态】刺激结束时的意识维度")
    print_radar_chart(brain.get_current_state().consciousness, brain.consciousness)
    
    # 运行阶段3：刺激消退
    print("【阶段3】刺激消退（无新输入）")
    print("-" * 50)
    
    decay_scores = []
    decay_levels = []
    for i in range(30):
        brain.step(dt=1.0)
        state = brain.get_current_state()
        decay_scores.append(state.consciousness.total_score)
        decay_levels.append(state.consciousness.level)
    
    print(f"  持续时间: 30 步")
    print(f"  意识范围: {min(decay_scores):.3f} - {max(decay_scores):.3f}")
    print(f"  平均意识: {np.mean(decay_scores):.3f}")
    print(f"  最终等级: {brain.consciousness.get_consciousness_level_name(decay_levels[-1])}")
    
    print_timeline(decay_scores, decay_levels)
    
    # 对比三个阶段
    print("【对比】三个阶段的意识水平对比")
    print("-" * 50)
    
    print(f"  {'阶段':<12} {'平均分':>8} {'最高分':>8} {'最低分':>8} {'主要等级':<10}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    print(f"  {'静息态':<12} {np.mean(rest_scores):>8.3f} {max(rest_scores):>8.3f} {min(rest_scores):>8.3f} {'微意识':<10}")
    print(f"  {'强刺激':<12} {np.mean(stim_scores):>8.3f} {max(stim_scores):>8.3f} {min(stim_scores):>8.3f} {'中等意识':<10}")
    print(f"  {'刺激消退':<12} {np.mean(decay_scores):>8.3f} {max(decay_scores):>8.3f} {min(decay_scores):>8.3f} {'中等意识':<10}")
    print()
    
    # 意识提升幅度
    boost = np.mean(stim_scores) - np.mean(rest_scores)
    print(f"  刺激带来的意识提升: +{boost:.3f} ({boost/np.mean(rest_scores)*100:.1f}%)")
    print()
    
    # 完整时间线
    print("【完整时间线】90步意识变化")
    print("-" * 50)
    
    all_scores = rest_scores + stim_scores + decay_scores
    all_levels = rest_levels + stim_levels + decay_levels
    
    print_timeline(all_scores, all_levels, step=3)
    
    # 标记阶段
    print(f"  {'静息态':^20}{'强刺激':^20}{'刺激消退':^20}")
    print(f"  {'(0-29步)':^20}{'(30-59步)':^20}{'(60-89步)':^20}")
    print()
    
    print("=" * 70)
    print("  ✓ 意识动态可视化演示完成！")
    print("=" * 70)

if __name__ == "__main__":
    main()
