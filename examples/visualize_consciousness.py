"""
意识动态可视化实验

设计一个四阶段的"意识变化剧本"，记录每一步的五维意识指标，
生成多面板时序图 + 原始数据 CSV。

四阶段（共 300 步）：
  P1 (0-60)    静息期：无刺激，观察基线意识水平
  P2 (60-150)  单模态驱动：每 5 步一次视觉刺激（稀疏驱动 + 回荡）
  P3 (150-240) 多模态驱动：每步视觉+听觉双重刺激（密集驱动）
  P4 (240-300) 撤除期：刺激消失，观察意识水平回落

说明：该大脑的突触响应衰减很快，Φ 的 120 步时间窗需要被单一
 regime 填满才能反映该阶段的真实整合水平，因此各相位较长。

输出：
  consciousness_dynamics.png  四面板时序图
  consciousness_dynamics.csv  逐步指标数据
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from brain import Brain, ConsciousnessLevel

# ---------- 运行实验 ----------
np.random.seed(7)

brain = Brain(sensory_neurons=200, association_neurons=500, decision_neurons=20)

PHASES = [
    (0,   60,  'P1 静息',   '#9e9e9e'),
    (60,  150, 'P2 单模态', '#42a5f5'),
    (150, 240, 'P3 多模态', '#ab47bc'),
    (240, 300, 'P4 撤除',   '#ef5350'),
]

LEVEL_ORDER = [
    ConsciousnessLevel.UNCONSCIOUS, ConsciousnessLevel.MINIMAL,
    ConsciousnessLevel.LOW, ConsciousnessLevel.MEDIUM,
    ConsciousnessLevel.HIGH, ConsciousnessLevel.META,
    ConsciousnessLevel.TRANSCENDENT,
]
LEVEL_ZH = ['无意识', '微意识', '低意识', '中等意识', '高意识', '元意识', '超意识']

records = []
TOTAL_STEPS = 300
stim_v = stim_a = None
for step in range(TOTAL_STEPS):
    # 阶段刺激：每个刺激事件持续 5 步（模拟 ~50ms 刺激时长，
    # 匹配 LIF 膜时间常数 τ=10 的积分需求）
    if 60 <= step < 150:
        # P2 单模态：15 步一个事件（5 步刺激 + 10 步间隔）
        if (step - 60) % 15 == 0:
            stim_v = np.random.rand(200) * 0.8
        if (step - 60) % 15 < 5:
            brain.input_stimulus(stim_v, modality=0)
    elif 150 <= step < 240:
        # P3 多模态：事件首尾相接（5 步刺激，无间隔），双模态
        if (step - 150) % 5 == 0:
            stim_v = np.random.rand(200) * 0.8
            stim_a = np.random.rand(200) * 0.6
        brain.input_stimulus(stim_v, modality=0)
        brain.input_stimulus(stim_a, modality=1)

    brain.step(dt=1.0)
    m = brain.get_current_state().consciousness
    d = brain.consciousness.get_phi_details()

    records.append({
        'step': step,
        'total_score': m.total_score,
        'phi': m.phi,
        'self_reference_depth': m.self_reference_depth,
        'workspace_activation': m.workspace_activation,
        'cross_module_integration': m.cross_module_integration,
        'information_density': m.information_density,
        'level_idx': LEVEL_ORDER.index(m.level),
        'level': brain.consciousness.get_consciousness_level_name(m.level),
        'phi_raw_nats': d.get('phi_raw_nats', 0.0),
        'phi_instant_nats': d.get('phi_instant_nats', 0.0),
        'phi_causal_nats': d.get('phi_causal_nats', 0.0),
    })

df = pd.DataFrame(records)
out_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
df.to_csv(out_dir / 'consciousness_dynamics.csv', index=False, encoding='utf-8-sig')

# ---------- 绘图 ----------
setup_plot()

fig, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=True)
fig.suptitle('意识动态可视化 · Consciousness Dynamics (300 步四阶段实验)', fontsize=16, fontweight='bold')

def shade_phases(ax):
    for start, end, name, color in PHASES:
        ax.axvspan(start, end, color=color, alpha=0.10)
        ax.text((start + end) / 2, ax.get_ylim()[1], name,
                ha='center', va='top', fontsize=9, color='#555555')

# --- 面板1: 综合评分 + 意识等级阈值带 ---
ax = axes[0]
thresholds = [t for t, _ in brain.consciousness.level_thresholds] + [1.0]
band_colors = ['#eceff1', '#e3f2fd', '#e8f5e9', '#fff8e1', '#fff3e0', '#fce4ec', '#f3e5f5']
for i in range(len(LEVEL_ZH)):
    ax.axhspan(thresholds[i], thresholds[i + 1], color=band_colors[i], alpha=0.8)
    ax.text(179, (thresholds[i] + thresholds[i + 1]) / 2, LEVEL_ZH[i],
            ha='right', va='center', fontsize=8, color='#777777')
sns.lineplot(data=df, x='step', y='total_score', ax=ax, color='#d81b60', linewidth=2.2)
ax.set_ylim(0, 1.0)
ax.set_ylabel('综合意识评分')
ax.set_title('① 综合评分与意识等级光谱', loc='left', fontsize=12)
shade_phases(ax)

# --- 面板2: 五维度曲线 ---
ax = axes[1]
dims = [
    ('phi', 'Φ 整合信息', '#1e88e5'),
    ('self_reference_depth', '自指深度', '#43a047'),
    ('workspace_activation', '工作空间激活', '#fb8c00'),
    ('cross_module_integration', '跨模块整合', '#8e24aa'),
    ('information_density', '信息密度', '#00acc1'),
]
for col, label, color in dims:
    sns.lineplot(data=df, x='step', y=col, ax=ax, label=label, color=color, linewidth=1.8)
ax.set_ylim(0, 1.0)
ax.set_ylabel('维度得分')
ax.legend(loc='upper left', ncol=5, fontsize=9, frameon=False)
ax.set_title('② 五个核心维度', loc='left', fontsize=12)
shade_phases(ax)

# --- 面板3: Φ 原始值(nats) ---
ax = axes[2]
sns.lineplot(data=df, x='step', y='phi_raw_nats', ax=ax, color='#1e88e5',
             linewidth=2.0, label='Φ raw (min-cut)')
sns.lineplot(data=df, x='step', y='phi_instant_nats', ax=ax, color='#43a047',
             linewidth=1.2, linestyle='--', label='瞬时整合分量')
sns.lineplot(data=df, x='step', y='phi_causal_nats', ax=ax, color='#ef6c00',
             linewidth=1.2, linestyle=':', label='因果效应分量')
ax.set_ylabel('nats')
ax.legend(loc='upper left', fontsize=9, frameon=False)
ax.set_title('③ Φ 原始信息量（MIP 最小割，未归一化）', loc='left', fontsize=12)
shade_phases(ax)

# --- 面板4: 意识等级阶梯 ---
ax = axes[3]
ax.step(df['step'], df['level_idx'], where='post', color='#5e35b1', linewidth=2.0)
ax.set_yticks(range(len(LEVEL_ZH)))
ax.set_yticklabels(LEVEL_ZH, fontsize=9)
ax.set_ylabel('意识等级')
ax.set_xlabel('仿真步数')
ax.set_ylim(-0.5, len(LEVEL_ZH) - 0.5)
ax.set_title('④ 意识等级跃迁', loc='left', fontsize=12)
shade_phases(ax)

fig.tight_layout(rect=[0, 0, 1, 0.97])
out_png = out_dir / 'consciousness_dynamics.png'
fig.savefig(out_png, dpi=220, bbox_inches='tight')
plt.close(fig)

# ---------- 摘要 ----------
print('✓ 实验完成')
print(f'  图表: {out_png}')
print(f'  数据: {out_dir / "consciousness_dynamics.csv"}')
print()
for start, end, name, _ in PHASES:
    seg = df[(df.step >= start) & (df.step < end)]
    print(f'  {name}: 评分 {seg.total_score.mean():.3f} | Φ {seg.phi.mean():.3f} | 主导等级 {seg.level.mode()[0]}')
