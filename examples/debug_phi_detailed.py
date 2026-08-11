"""详细调试Φ值计算"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from brain import Brain

brain = Brain(
    sensory_neurons=200,
    association_neurons=500,
    decision_neurons=20
)

# 检查 spike_rates
print("=== 初始 spike_rates ===")
print(f"sensory spike_rates shape: {brain.network.sensory.spike_rates.shape}")
print(f"sensory spike_rates mean: {np.mean(brain.network.sensory.spike_rates):.4f}")
print(f"sensory spike_rates max: {np.max(brain.network.sensory.spike_rates):.4f}")
print(f"association spike_rates mean: {np.mean(brain.network.association.spike_rates):.4f}")
print(f"decision spike_rates mean: {np.mean(brain.network.decision.spike_rates):.4f}")
print()

# 运行几步，每步检查Φ值
print("=== 逐步运行，检查Φ值 ===")
for i in range(30):
    # 每5步输入一次刺激
    if i % 5 == 0:
        stim = np.random.rand(200) * 0.8
        brain.input_stimulus(stim, modality=0)
    
    brain.step(dt=1.0)
    
    # 检查模块活动收集
    modules = brain.consciousness._collect_module_activities(brain)
    n_modules = len(modules)
    module_shapes = {k: v.shape for k, v in modules.items()}
    module_means = {k: np.mean(v) for k, v in modules.items()}
    
    # 检查滑动窗口
    n_windows = len(brain.consciousness._activity_windows)
    window_sizes = {k: len(v) for k, v in brain.consciousness._activity_windows.items()}
    
    # 检查Φ值
    phi = brain.consciousness.compute_phi(brain)
    
    # 检查详细信息
    details = brain.consciousness._last_phi_details
    
    if i < 10 or i % 5 == 0:
        print(f"Step {i:2d}: phi={phi:.4f}, modules={n_modules}, "
              f"windows={n_windows}, "
              f"sensory_mean={module_means.get('sensory', 0):.4f}, "
              f"assoc_mean={module_means.get('association', 0):.4f}, "
              f"details_keys={list(details.keys())[:3]}")
    
    if i == 29:
        print()
        print("=== 最后一步的详细信息 ===")
        print(f"Phi: {phi:.6f}")
        print(f"Modules: {list(modules.keys())}")
        for k, v in modules.items():
            print(f"  {k}: shape={v.shape}, mean={np.mean(v):.6f}, std={np.std(v):.6f}")
        print()
        print(f"Window sizes: {window_sizes}")
        print()
        print(f"Phi details:")
        for k, v in details.items():
            print(f"  {k}: {v}")
