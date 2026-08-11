"""
动力学校准扫描（选项 A）— 分批版

用法：
  python examples/calibrate_dynamics.py START END OUT.csv

在小脑模型（100/250/10，约 4 倍速）上扫描，胜出配置再回全尺寸验证。
每批约 4 配置 × 2 种子 × 2 条件，可在几分钟内完成。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import itertools
import numpy as np
import pandas as pd
from brain import Brain

START, END, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]

GRID = list(itertools.product(
    [2.0, 3.0],    # ff_scale 前馈权重缩放
    [0.05, 0.15],  # fb_gain 决策反馈增益
    [0.05, 0.10],  # target_rate 稳态目标
))
SEEDS = [7, 42]
STEPS = 110
REFRACTORY = 5


def run_condition(ff, fb, tr, mode, seed):
    np.random.seed(seed)
    brain = Brain(sensory_neurons=100, association_neurons=250, decision_neurons=10)
    net = brain.network
    net.sensory_to_association_weights = net.sensory_to_association_weights * ff
    net.decision_feedback_gain = fb
    for layer in (net.sensory, net.association, net.decision):
        layer.target_rate = tr
        for n in layer.neurons:
            n.refractory_period = REFRACTORY

    cq = brain.consciousness
    phis, scores, aacts = [], [], []
    s0 = s1 = None
    for i in range(STEPS):
        if mode != 'rest' and 30 <= i < 100 and (i - 30) % 15 < 5:
            if (i - 30) % 15 == 0:
                s0 = np.random.rand(100) * 0.8
                s1 = np.random.rand(100) * 0.7
            brain.input_stimulus(s0, modality=0)
            if mode == 'dual':
                brain.input_stimulus(s1, modality=1)
        brain.step(dt=1.0)
        m = brain.get_current_state().consciousness
        phis.append(m.phi)
        scores.append(m.total_score)
        aacts.append(net.association.get_activity_pattern().mean())

    S = {k: np.stack(v, 0) for k, v in cq._activity_windows.items()}
    corr = 0.0
    if 'sensory' in S and 'association' in S and S['sensory'].std() > 0 and S['association'].std() > 0:
        corrs = [abs(np.corrcoef(S['sensory'][:, a], S['association'][:, b])[0, 1])
                 for a in range(S['sensory'].shape[1]) for b in range(S['association'].shape[1])]
        corr = float(np.nanmax(corrs))

    tail = slice(-40, None)
    return {
        'phi': float(np.mean(phis[tail])),
        'score': float(np.mean(scores[tail])),
        'assoc_act': float(np.mean(aacts[tail])),
        'corr_max': corr,
    }


rows = []
for idx in range(START, min(END, len(GRID))):
    ff, fb, tr = GRID[idx]
    agg = {'phi_discrim': [], 'corr_sa': [], 'rest_quiet': [], 'score_delta': []}
    for seed in SEEDS:
        rest = run_condition(ff, fb, tr, 'rest', seed)
        drv = run_condition(ff, fb, tr, 'dual', seed)
        agg['phi_discrim'].append(drv['phi'] - rest['phi'])
        agg['corr_sa'].append(drv['corr_max'])
        agg['rest_quiet'].append(rest['assoc_act'])
        agg['score_delta'].append(drv['score'] - rest['score'])
    row = {'cfg_idx': idx, 'ff_scale': ff, 'fb_gain': fb, 'refractory': REFRACTORY, 'target_rate': tr,
           **{k: float(np.mean(v)) for k, v in agg.items()}}
    rows.append(row)
    print(f"cfg{idx}: ff={ff} fb={fb} tr={tr} | Φ区分 {row['phi_discrim']:+.4f} "
          f"相关 {row['corr_sa']:.3f} 静息 {row['rest_quiet']:.3f} 评分Δ {row['score_delta']:+.3f}", flush=True)

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f'saved: {OUT}')
