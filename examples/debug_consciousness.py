"""调试意识量化模块"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from brain import Brain

brain = Brain(
    sensory_neurons=200,
    association_neurons=500,
    decision_neurons=20
)

# 运行一步
brain.step(dt=1.0)

print("=== 检查 brain 属性 ===")
print(f"has network: {hasattr(brain, 'network')}")
print(f"has memory: {hasattr(brain, 'memory')}")
print(f"has modulation: {hasattr(brain, 'modulation')}")
print(f"has thought: {hasattr(brain, 'thought')}")
print(f"has consciousness: {hasattr(brain, 'consciousness')}")
print()

print("=== 检查 network 属性 ===")
if hasattr(brain, 'network'):
    net = brain.network
    print(f"has sensory_layer: {hasattr(net, 'sensory_layer')}")
    print(f"has association_layer: {hasattr(net, 'association_layer')}")
    print(f"has decision_layer: {hasattr(net, 'decision_layer')}")
    print(f"has sensory: {hasattr(net, 'sensory')}")
    print(f"has association: {hasattr(net, 'association')}")
    print(f"has decision: {hasattr(net, 'decision')}")
    
    # 检查神经元状态
    if hasattr(net, 'sensory_layer'):
        sl = net.sensory_layer
        print(f"sensory_layer has neuron_states: {hasattr(sl, 'neuron_states')}")
        if hasattr(sl, 'neuron_states'):
            print(f"  neuron_states shape: {sl.neuron_states.shape}")
            print(f"  mean: {np.mean(sl.neuron_states):.4f}")
    
    if hasattr(net, 'sensory'):
        s = net.sensory
        print(f"sensory has neuron_states: {hasattr(s, 'neuron_states')}")
        if hasattr(s, 'neuron_states'):
            print(f"  neuron_states shape: {s.neuron_states.shape}")
print()

print("=== 检查 thought 属性 ===")
if hasattr(brain, 'thought'):
    ts = brain.thought
    print(f"thought type: {type(ts).__name__}")
    print(f"has thought_space: {hasattr(ts, 'thought_space')}")
    print(f"has thought_sensory: {hasattr(ts, 'thought_sensory')}")
    
    if hasattr(ts, 'thought_space'):
        ws = ts.thought_space
        print(f"  active_thoughts: {len(ws.active_thoughts) if hasattr(ws, 'active_thoughts') else 'N/A'}")
        print(f"  capacity: {ws.capacity if hasattr(ws, 'capacity') else 'N/A'}")
print()

print("=== 手动计算 Φ 值 ===")
phi = brain.consciousness.compute_phi(brain)
print(f"Phi: {phi:.6f}")
print()

print("=== 手动计算自指深度 ===")
self_ref = brain.consciousness.compute_self_reference_depth(brain)
print(f"Self ref: {self_ref:.6f}")
print()

print("=== 手动计算工作空间激活 ===")
ws_act = brain.consciousness.compute_workspace_activation(brain)
print(f"Workspace activation: {ws_act:.6f}")
print()

print("=== 手动计算跨模块整合 ===")
integration = brain.consciousness.compute_cross_module_integration(brain)
print(f"Integration: {integration:.6f}")
print()

print("=== 手动计算信息密度 ===")
density = brain.consciousness.compute_information_density(brain)
print(f"Density: {density:.6f}")
print()

print("=== 综合量化 ===")
metrics = brain.consciousness.quantify(brain)
print(metrics)
