"""
涌现实验平台 (Emergence Experiment Platform)

系统地测试规模/结构/密度的相变点（Scaling Law 与涌现）。

三个一维实验（其余参数固定为动力学校准默认值）：
  E1 规模：总神经元数变化 → Φ 是否出现非线性跳变
  E2 结构：联想层模块化程度 → 模式分离度（概念空间能否区分不同输入）
  E3 密度：层内连接密度 → 临界性（活动自持续性）与 Φ

用法：
  python examples/emergence_platform.py E1   # 单个实验
  python examples/emergence_platform.py all  # 全部
  python examples/emergence_platform.py plot # 汇总绘图

每个配置的探测协议（约 95 步，小脑模型保证速度）：
  1. 静息 15 步
  2. 注入概念簇 0（6 步）→ 记录响应模式 A
  3. 静息 8 步；注入概念簇 3（6 步）→ 记录模式 B
  4. 模式分离度 = 1 - sim(A, B)（中心化余弦）
  5. 稀疏驱动 40 步（5 on / 10 off）→ Φ 时间窗填满
  6. 撤除刺激 → 活动自持续步数（临界性指标）
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from brain import Brain

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

SEEDS = [7, 42]
NUM_CLUSTERS = 4


def rewire_association(net, modularity=0.0, connectivity=0.25):
    """
    按模块化结构重建联想层内连接。
    同簇连接概率 = connectivity；跨簇 = connectivity × (1 - modularity)。
    E/I 平衡因子 4.0（与全局校准一致）。
    """
    assoc = net.association
    n = assoc.num_neurons
    cs = n // NUM_CLUSTERS
    assoc.concept_clusters = {}
    cluster_of = np.zeros(n, dtype=int)
    for i in range(NUM_CLUSTERS):
        end = (i + 1) * cs if i < NUM_CLUSTERS - 1 else n
        assoc.concept_clusters[f'concept_{i}'] = list(range(i * cs, end))
        cluster_of[i * cs:end] = i

    for neuron in assoc.neurons:
        neuron.input_synapses = []
        neuron.output_synapses = []

    for i, neuron in enumerate(assoc.neurons):
        for j, tgt in enumerate(assoc.neurons):
            if i == j:
                continue
            same = cluster_of[i] == cluster_of[j]
            p = connectivity if same else connectivity * (1.0 - modularity)
            if np.random.random() < p:
                w = np.random.normal(0.3, 0.1)
                w = -abs(w) * 4.0 if neuron.type == "inhibitory" else abs(w)
                neuron.add_output_synapse(j, w)
                tgt.add_input_synapse(i, w)


def centered_sim(x, y):
    xc, yc = x - x.mean(), y - y.mean()
    nx, ny = np.linalg.norm(xc), np.linalg.norm(yc)
    return float(xc @ yc / (nx * ny)) if nx > 1e-9 and ny > 1e-9 else 0.0


def probe(sensory_n, assoc_n, decision_n, modularity, connectivity, seed):
    """运行探测协议，返回指标字典"""
    np.random.seed(seed)
    brain = Brain(sensory_neurons=sensory_n, association_neurons=assoc_n,
                  decision_neurons=decision_n)
    net = brain.network
    rewire_association(net, modularity, connectivity)
    assoc = net.association

    # 1. 静息
    for _ in range(15):
        brain.step(dt=1.0)

    # 2-3. 概念响应模式
    def present(concept, steps=6):
        p = assoc.get_concept_pattern(concept)
        for _ in range(steps):
            assoc.apply_external_input(p * 3.0)
            brain.step(dt=1.0)
        return assoc.spike_rates.copy()

    resp_A = present('concept_0')
    for _ in range(8):
        brain.step(dt=1.0)
    resp_B = present(f'concept_{NUM_CLUSTERS - 1}')
    for _ in range(8):
        brain.step(dt=1.0)

    separation = 1.0 - abs(centered_sim(resp_A, resp_B))

    # 4. 稀疏驱动（Φ 窗口填充）
    phis, scores = [], []
    s0 = None
    for i in range(40):
        if i % 15 < 5:
            if i % 15 == 0:
                s0 = np.random.rand(sensory_n) * 0.8
            brain.input_stimulus(s0, modality=0)
        brain.step(dt=1.0)
        m = brain.get_current_state().consciousness
        phis.append(m.phi)
        scores.append(m.total_score)

    # 5. 撤除后自持续步数
    persist = 0
    for _ in range(20):
        brain.step(dt=1.0)
        if net.association.get_activity_pattern().mean() > 0.02:
            persist += 1
        else:
            break

    return {
        'neurons': sensory_n + assoc_n + decision_n,
        'modularity': modularity,
        'connectivity': connectivity,
        'seed': seed,
        'phi': float(np.mean(phis[-15:])),
        'score': float(np.mean(scores[-15:])),
        'separation': separation,
        'persistence': persist,
    }


def run_experiment(name, configs):
    rows = []
    for cfg in configs:
        for seed in SEEDS:
            r = probe(seed=seed, **cfg)
            rows.append(r)
            print(f"  {name} cfg={cfg} seed={seed} | Φ={r['phi']:.3f} "
                  f"分离={r['separation']:.3f} 持续={r['persistence']} 评分={r['score']:.3f}",
                  flush=True)
    df = pd.DataFrame(rows)
    out = os.path.join(DATA_DIR, f'emergence_{name}.csv')
    df.to_csv(out, index=False)
    print(f"  saved: {out}", flush=True)


def e1():
    print("【E1 规模扫描】")
    configs = [
        dict(sensory_n=60, assoc_n=160, decision_n=8, modularity=0.7, connectivity=0.25),
        dict(sensory_n=90, assoc_n=240, decision_n=12, modularity=0.7, connectivity=0.25),
        dict(sensory_n=120, assoc_n=320, decision_n=16, modularity=0.7, connectivity=0.25),
        dict(sensory_n=160, assoc_n=420, decision_n=20, modularity=0.7, connectivity=0.25),
    ]
    run_experiment('E1_scale', configs)


def e2():
    print("【E2 结构扫描（模块化）】")
    configs = [
        dict(sensory_n=90, assoc_n=240, decision_n=12, modularity=m, connectivity=0.25)
        for m in [0.0, 0.4, 0.7, 0.95]
    ]
    run_experiment('E2_structure', configs)


def e3():
    print("【E3 密度扫描】")
    configs = [
        dict(sensory_n=90, assoc_n=240, decision_n=12, modularity=0.7, connectivity=c)
        for c in [0.05, 0.15, 0.25, 0.40]
    ]
    run_experiment('E3_density', configs)


def plot():
    from pathlib import Path
    sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
    from daimon_runtime import setup_plot
    import matplotlib.pyplot as plt
    import seaborn as sns

    setup_plot()
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('涌现实验平台：规模 / 结构 / 密度的相变扫描', fontsize=15, fontweight='bold')

    specs = [
        ('E1_scale', 'neurons', 'E1 规模（总神经元数）'),
        ('E2_structure', 'modularity', 'E2 结构（模块化程度）'),
        ('E3_density', 'connectivity', 'E3 密度（层内连接率）'),
    ]
    metrics = [('phi', 'Φ 整合信息'), ('separation', '模式分离度'), ('persistence', '活动自持续步数')]

    for row, (name, xcol, title) in enumerate(specs):
        path = os.path.join(DATA_DIR, f'emergence_{name}.csv')
        df = pd.read_csv(path)
        for col, (mcol, mlabel) in enumerate(metrics):
            ax = axes[row, col]
            sns.pointplot(data=df, x=xcol, y=mcol, ax=ax, errorbar='sd',
                          color='#d81b60' if col == 0 else '#1e88e5' if col == 1 else '#43a047')
            sns.stripplot(data=df, x=xcol, y=mcol, ax=ax, color='#666666',
                          size=4, alpha=0.6, jitter=0.15)
            ax.set_ylabel(mlabel if col == 0 or row == 2 else mlabel)
            if col == 0:
                ax.set_title(title, loc='left', fontsize=12, fontweight='bold')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(os.path.dirname(DATA_DIR), 'emergence_phase_transitions.png')
    fig.savefig(out, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f'saved: {out}')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if cmd in ('E1', 'all'):
        e1()
    if cmd in ('E2', 'all'):
        e2()
    if cmd in ('E3', 'all'):
        e3()
    if cmd in ('plot', 'all'):
        plot()
