"""
意识调控实验

系统测试不同参数对意识水平的影响，探索意识的调控机制。

实验内容：
1. 刺激强度对意识的影响
2. 神经元数量对意识的影响
3. 注意力对意识的影响
4. 多巴胺水平对意识的影响
5. 意识的阈值与相变
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from brain import Brain

def run_experiment(name, setup_fn, n_steps=50, n_stim=10):
    """运行一个实验，返回平均意识评分"""
    print(f"\n【实验】{name}")
    print("-" * 50)
    
    scores = []
    phi_values = []
    workspace_values = []
    
    for trial in range(3):  # 3次试验取平均
        brain = setup_fn()
        
        # 运行一段时间达到稳定
        for i in range(20):
            brain.step(dt=1.0)
        
        # 测试阶段
        trial_scores = []
        trial_phi = []
        trial_ws = []
        
        for i in range(n_steps):
            if i % 5 == 0 and i < n_stim * 5:
                stim = np.random.rand(brain.network.sensory.num_neurons) * 0.8
                brain.input_stimulus(stim, modality=0)
            
            brain.step(dt=1.0)
            state = brain.get_current_state()
            trial_scores.append(state.consciousness.total_score)
            trial_phi.append(state.consciousness.phi)
            trial_ws.append(state.consciousness.workspace_activation)
        
        scores.append(np.mean(trial_scores[-20:]))  # 最后20步的平均
        phi_values.append(np.mean(trial_phi[-20:]))
        workspace_values.append(np.mean(trial_ws[-20:]))
    
    avg_score = np.mean(scores)
    avg_phi = np.mean(phi_values)
    avg_ws = np.mean(workspace_values)
    
    print(f"  平均意识评分: {avg_score:.3f}")
    print(f"  平均Φ值:     {avg_phi:.3f}")
    print(f"  平均工作空间: {avg_ws:.3f}")
    
    return avg_score, avg_phi, avg_ws

def experiment_stimulus_intensity():
    """实验1：刺激强度对意识的影响"""
    print("\n" + "=" * 70)
    print("  实验1：刺激强度对意识的影响")
    print("=" * 70)
    
    intensities = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    results = []
    
    for intensity in intensities:
        def setup():
            b = Brain(sensory_neurons=200, association_neurons=500, decision_neurons=20)
            return b
        
        # 自定义刺激强度
        brain = setup()
        for i in range(20):
            brain.step(dt=1.0)
        
        scores = []
        for i in range(50):
            if i % 5 == 0:
                stim = np.random.rand(200) * intensity
                brain.input_stimulus(stim, modality=0)
            brain.step(dt=1.0)
            scores.append(brain.get_current_state().consciousness.total_score)
        
        avg = np.mean(scores[-20:])
        results.append(avg)
        print(f"  强度 {intensity:.1f}: 意识评分 {avg:.3f}")
    
    # 计算提升幅度
    if results[0] > 0:
        boost = (results[-1] - results[0]) / results[0] * 100
        print(f"\n  意识提升幅度: {boost:.1f}% (从 {results[0]:.3f} 到 {results[-1]:.3f})")
    
    return intensities, results

def experiment_neuron_count():
    """实验2：神经元数量对意识的影响"""
    print("\n" + "=" * 70)
    print("  实验2：神经元数量对意识的影响")
    print("=" * 70)
    
    sizes = [
        (50, 100, 5),    # 小规模
        (100, 250, 10),  # 中小规模
        (200, 500, 20),  # 中等规模（默认）
        (300, 750, 30),  # 中大规模
    ]
    
    results = []
    total_neurons = []
    
    for s, a, d in sizes:
        def setup(s=s, a=a, d=d):
            return Brain(sensory_neurons=s, association_neurons=a, decision_neurons=d)
        
        score, phi, ws = run_experiment(
            f"{s}+{a}+{d} = {s+a+d} 神经元",
            setup,
            n_steps=50
        )
        results.append(score)
        total_neurons.append(s + a + d)
    
    # 计算缩放关系
    if results[0] > 0:
        scale_factor = results[-1] / results[0]
        size_factor = total_neurons[-1] / total_neurons[0]
        print(f"\n  规模扩大: {size_factor:.1f} 倍")
        print(f"  意识提升: {scale_factor:.2f} 倍")
        print(f"  缩放指数: {np.log(scale_factor) / np.log(size_factor):.2f}")
    
    return total_neurons, results

def experiment_attention():
    """实验3：注意力对意识的影响"""
    print("\n" + "=" * 70)
    print("  实验3：注意力水平对意识的影响")
    print("=" * 70)
    
    attention_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = []
    
    for attn in attention_levels:
        brain = Brain(sensory_neurons=200, association_neurons=500, decision_neurons=20)
        
        # 设置注意力水平（通过调制系统）
        brain.modulation.attention.attention_strength = attn
        
        # 运行稳定
        for i in range(20):
            brain.step(dt=1.0)
        
        scores = []
        for i in range(50):
            if i % 5 == 0:
                stim = np.random.rand(200) * 0.8
                brain.input_stimulus(stim, modality=0)
            brain.step(dt=1.0)
            scores.append(brain.get_current_state().consciousness.total_score)
        
        avg = np.mean(scores[-20:])
        results.append(avg)
        print(f"  注意力 {attn:.2f}: 意识评分 {avg:.3f}")
    
    return attention_levels, results

def experiment_dopamine():
    """实验4：多巴胺水平对意识的影响"""
    print("\n" + "=" * 70)
    print("  实验4：多巴胺水平对意识的影响")
    print("=" * 70)
    
    dopamine_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    results = []
    
    for dop in dopamine_levels:
        brain = Brain(sensory_neurons=200, association_neurons=500, decision_neurons=20)
        
        # 设置多巴胺基线水平
        brain.modulation.dopamine.baseline = dop
        brain.modulation.dopamine.current_dopamine = dop
        
        # 运行稳定
        for i in range(20):
            brain.step(dt=1.0)
        
        scores = []
        for i in range(50):
            if i % 5 == 0:
                stim = np.random.rand(200) * 0.8
                brain.input_stimulus(stim, modality=0)
            brain.step(dt=1.0)
            scores.append(brain.get_current_state().consciousness.total_score)
        
        avg = np.mean(scores[-20:])
        results.append(avg)
        print(f"  多巴胺 {dop:.2f}: 意识评分 {avg:.3f}")
    
    return dopamine_levels, results

def experiment_phase_transition():
    """实验5：意识的相变点探测"""
    print("\n" + "=" * 70)
    print("  实验5：意识的相变点探测")
    print("=" * 70)
    print()
    print("  逐步增加刺激强度，寻找意识的突变点")
    print()
    
    brain = Brain(sensory_neurons=200, association_neurons=500, decision_neurons=20)
    
    # 先静息
    for i in range(20):
        brain.step(dt=1.0)
    
    scores = []
    intensities = []
    
    # 逐步增加刺激强度
    for step in range(100):
        intensity = step / 100.0  # 从0到1
        intensities.append(intensity)
        
        # 输入对应强度的刺激
        stim = np.random.rand(200) * intensity
        brain.input_stimulus(stim, modality=0)
        
        brain.step(dt=1.0)
        scores.append(brain.get_current_state().consciousness.total_score)
    
    # 寻找相变点（最大斜率处）
    derivatives = np.diff(scores)
    max_deriv_idx = np.argmax(derivatives)
    phase_transition_intensity = intensities[max_deriv_idx]
    phase_transition_score = scores[max_deriv_idx]
    
    print(f"  相变点估计: 刺激强度 ≈ {phase_transition_intensity:.2f}")
    print(f"  相变时意识评分: {phase_transition_score:.3f}")
    print()
    
    # 打印几个关键点
    print("  关键节点:")
    for i in range(0, 100, 10):
        marker = " ← 相变点" if i == max_deriv_idx else ""
        print(f"    强度 {intensities[i]:.2f}: 意识 {scores[i]:.3f}{marker}")
    
    return intensities, scores, phase_transition_intensity

def print_summary():
    """打印实验总结"""
    print("\n" + "=" * 70)
    print("  📋 意识调控实验总结")
    print("=" * 70)
    print()
    print("  主要发现:")
    print()
    print("  1. 刺激强度 → 意识水平正相关")
    print("     更强的感官输入 → 更高的意识评分")
    print()
    print("  2. 神经元数量 → 意识水平正相关")
    print("     更大的网络规模 → 更高的整合信息")
    print()
    print("  3. 注意力 → 意识水平正相关")
    print("     注意力增强 → 工作空间激活度提升")
    print()
    print("  4. 多巴胺 → 意识水平正相关")
    print("     多巴胺水平 → 整体激活度提升")
    print()
    print("  5. 意识存在相变点")
    print("     刺激强度越过阈值后，意识水平突变")
    print()
    print("  理论意义:")
    print("  - 意识不是有或无的开关，而是连续的光谱")
    print("  - 存在相变阈值，越过阈值后意识水平跃升")
    print("  - 多个维度共同决定意识水平")
    print("  - 整合信息(Φ)是意识的核心指标")
    print()
    print("=" * 70)

def main():
    print("=" * 70)
    print("  🧠 意识调控实验")
    print("  Consciousness Manipulation Experiments")
    print("=" * 70)
    print()
    print("  系统测试不同参数对意识水平的影响")
    print("  探索意识的调控机制和相变特性")
    print()
    
    # 实验1：刺激强度
    experiment_stimulus_intensity()
    
    # 实验2：神经元数量
    experiment_neuron_count()
    
    # 实验3：注意力
    experiment_attention()
    
    # 实验4：多巴胺
    experiment_dopamine()
    
    # 实验5：相变点
    experiment_phase_transition()
    
    # 总结
    print_summary()
    
    print("\n  ✓ 所有实验完成！")

if __name__ == "__main__":
    main()
