# -*- coding: utf-8 -*-
"""
脉冲思考链可视化：把一次"感知 → 传导 → 回响 → 决策"的脉冲因果链
画成按网络步展开的时序传播图，保存至 figures/thought_chain.png。

左图：脉冲传播光栅（x=网络步，y=神经元 id，标注层间流向）
右图：各层脉冲数逐步衰减（回响衰减曲线）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    # Kimi 托管 Python 运行时自带中文字体配置
    sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
    from daimon_runtime import setup_plot
except ImportError:
    # 普通 Python 环境的降级方案：自行配置中文字体
    def setup_plot():
        import matplotlib
        matplotlib.rcParams["font.sans-serif"] = [
            "Microsoft YaHei", "SimHei", "PingFang SC",
            "Arial Unicode MS", "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ai_brain_entity import AIBrainEntity

BASE = Path(__file__).parent
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)

setup_plot()

STIMULUS = "火焰是危险的"

# 先让大脑反复经历该刺激（强化通路），再追踪一次思考的完整脉冲链
brain = AIBrainEntity("thought-chain", seed=42)
for _ in range(20):
    brain.sensory_input(STIMULUS)
tc = brain.thought_chain(STIMULUS)
steps = tc["steps"]

LAYERS = [("感官层(0-15)", 0), ("联想层(16-47)", 1), ("决策层(48-55)", 2)]
COLORS = ["#4C72B0", "#DD8452", "#55A868"]

# ---- 面板1：按时展开的脉冲传播光栅 ----
rows = []
for step, layer_ids in enumerate(steps):
    for li, ids in enumerate(layer_ids):
        for nid in ids:
            rows.append({"网络步": step, "neuron": nid,
                         "层": LAYERS[li][0]})
raster = pd.DataFrame(rows)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.6),
                         gridspec_kw={"width_ratios": [1.15, 1]})
sns.scatterplot(data=raster, x="网络步", y="neuron", hue="层",
                palette=COLORS, s=55, linewidth=0, ax=axes[0])
axes[0].axhline(15.5, c="black", lw=0.8)
axes[0].axhline(47.5, c="black", lw=0.8)
axes[0].set_yticks([7, 31, 51])
axes[0].set_yticklabels(["感官层", "联想层", "决策层"])
axes[0].set_ylabel("")
axes[0].set_xticks(range(len(steps)))
axes[0].set_xticklabels(["刺激步"] + [f"回响+{i}" for i in range(1, len(steps))])
axes[0].set_title("脉冲逐层传播（层间流向：感官 → 联想 → 决策）")
# 层间流向箭头
for y0, y1 in ((7, 31), (31, 51)):
    axes[0].annotate("", xy=(-0.42, y1), xytext=(-0.42, y0),
                     arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
                     annotation_clip=False)
axes[0].legend(loc="upper right", fontsize=8)

# ---- 面板2：各层脉冲数逐步变化（回响衰减） ----
counts = pd.DataFrame(
    {"网络步": ["刺激步" if s == 0 else f"回响+{s}" for s in range(len(steps))
               for _ in LAYERS],
     "脉冲数": [len(ids) for layer in steps for ids in layer],
     "层": [LAYERS[li][0] for _ in steps for li in range(3)]})
sns.barplot(data=counts, x="网络步", y="脉冲数", hue="层",
            palette=COLORS, ax=axes[1])
axes[1].set_title("回响衰减：各层脉冲数逐网络步变化")
axes[1].legend(fontsize=8)

fig.suptitle(f"脉冲思考链：{STIMULUS!r}\n{tc['output']}", fontsize=11)
fig.tight_layout()
out = FIG / "thought_chain.png"
fig.savefig(out, dpi=220, bbox_inches="tight")
plt.close(fig)
print(f"图表已保存至: {out}")
