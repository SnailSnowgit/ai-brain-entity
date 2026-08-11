"""
真实度光谱三态实验

在三种诱导模式下各运行 150 步，验证真实度指标能否区分：
  - awake（清醒）：外部刺激驱动，期望高接地、低误归因
  - dream（梦）：感觉门控关闭 + 内部生成增强，期望低接地、门控关闭
  - hallucination（幻觉）：门控开放 + 过强先验，期望高误归因

同时记录意识 Φ，观察"高 Φ 低真实度"（梦）与
"高真实度高 Φ"（清醒）的解离——意识≠真实。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from brain import Brain, RealityState

ZH = {'awake': '清醒', 'dream': '梦', 'hallucination': '幻觉'}


def run_mode(mode, seed=7, steps=150):
    np.random.seed(seed)
    brain = Brain(sensory_neurons=200, association_neurons=500, decision_neurons=20)
    brain.set_reality_mode(mode)

    records = []
    s0 = None
    for i in range(steps):
        if 30 <= i < 140 and (i - 30) % 15 < 5:
            if (i - 30) % 15 == 0:
                # 梦模式下也呈现刺激（很弱也进不来，用于验证门控）
                s0 = np.random.rand(200) * 0.8
            brain.input_stimulus(s0, modality=0)

        brain.step(dt=1.0)
        state = brain.get_current_state()
        r = state.reality
        c = state.consciousness
        records.append({
            'step': i,
            'grounding': r.external_grounding,
            'realness': r.felt_realness,
            'misattribution': r.source_misattribution,
            'gate': r.sensory_gate_open,
            'reality_score': r.reality_score,
            'state': r.state.value,
            'phi': c.phi,
            'consciousness_score': c.total_score,
        })

    tail = records[-60:]
    states = [r['state'] for r in tail]
    dominant = max(set(states), key=states.count)
    print(f"\n【{ZH[mode]}模式】(后60步均值, 主导状态: {ZH.get(dominant, dominant)})")
    print(f"  外部接地度:   {np.mean([r['grounding'] for r in tail]):.3f}")
    print(f"  主观真实感:   {np.mean([r['realness'] for r in tail]):.3f}")
    print(f"  信源误归因:   {np.mean([r['misattribution'] for r in tail]):.3f}")
    print(f"  综合真实度:   {np.mean([r['reality_score'] for r in tail]):.3f}")
    print(f"  意识Φ:        {np.mean([r['phi'] for r in tail]):.3f}")
    print(f"  意识评分:     {np.mean([r['consciousness_score'] for r in tail]):.3f}")
    return tail


print("=" * 60)
print("  真实度光谱三态实验")
print("  Reality Spectrum: Awake / Dream / Hallucination")
print("=" * 60)

awake = run_mode('awake')
dream = run_mode('dream')
hallu = run_mode('hallucination')

print()
print("=" * 60)
print("  区分度检验")
print("=" * 60)
g_awake = np.mean([r['grounding'] for r in awake])
g_dream = np.mean([r['grounding'] for r in dream])
m_hallu = np.mean([r['misattribution'] for r in hallu])
m_awake = np.mean([r['misattribution'] for r in awake])
phi_dream = np.mean([r['phi'] for r in dream])
rs_awake = np.mean([r['reality_score'] for r in awake])
rs_dream = np.mean([r['reality_score'] for r in dream])

print(f"  接地度: 清醒 {g_awake:.3f} vs 梦 {g_dream:.3f}  → {'✓ 可区分' if g_awake > g_dream + 0.1 else '✗ 不可区分'}")
print(f"  误归因: 幻觉 {m_hallu:.3f} vs 清醒 {m_awake:.3f}  → {'✓ 可区分' if m_hallu > m_awake + 0.1 else '✗ 不可区分'}")
print(f"  意识≠真实: 梦态Φ {phi_dream:.3f} 而真实度 {rs_dream:.3f}（清醒真实度 {rs_awake:.3f}）")
