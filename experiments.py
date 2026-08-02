# -*- coding: utf-8 -*-
"""
统一实验脚本：一键复现全部 8 组实验，生成真实数据与图表
（基于 ai_brain_entity.py v3.0 核心；可塑性开关同时控制 STDP）

实验1：突触可塑性的有效性（可塑性开/关对照，追踪突触强度演化）
实验2：记忆固化与类艾宾浩斯遗忘（固化竞争 -> 强化 -> 400 轮衰减）
实验3：情绪-注意力闭环的稳定性（高/低强度交替刺激 150 tick）
实验4：脉冲活动模式观察（80 tick 三层神经元脉冲光栅图）
实验5：STDP 因果方向性（A->B 与 B->A 序列训练对照）
实验6：多巴胺奖励调制学习（奖励组 vs 无奖励组）
实验7：群体文化传递（6 代链式传递：温习维系文化 vs 不温习文化灭绝）
实验8：多模态 embedding 感知（embedding 相似度保持）
"""
import json
import math
import random
import sys
from pathlib import Path
from typing import Tuple

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

try:
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    _PLOT_DEPS_OK = True
except ImportError as _deps_err:
    # 优雅降级（与多模态可选依赖同一设计）：缺绘图库时给出明确提示，
    # 而不是裸抛 ModuleNotFoundError；核心模块不受影响
    _PLOT_DEPS_OK = False
    _PLOT_DEPS_ERR = _deps_err
    pd = sns = plt = None

if not _PLOT_DEPS_OK:
    print(f"[跳过] 实验 1-8 复现需要绘图依赖（缺失：{_PLOT_DEPS_ERR.name}）")
    print("       修复：pip install pandas numpy matplotlib seaborn")
    print("       核心模块（ai_brain_entity.py / swarm.py / tests/）零依赖，"
          "不受此影响。")
    sys.exit(0)   # 正常退出：run_all 全流程不因缺可选依赖而中断

from ai_brain_entity import AIBrainEntity, BrainSwarm
from swarm import generation_chain

BASE = Path(__file__).parent
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)

setup_plot()

RESULTS = {}

STIM_A = "事件A：日出东方"
STIM_B = "事件B：鸟鸣枝头"

# ===================== 实验1：突触可塑性的有效性 =====================
print("运行实验1：突触可塑性开/关对照 ...")

STIMS1 = ["神经元脉冲正在传递信号", "记忆是智慧的基石",
          "情绪实时调制着注意力", "突触随经验改变连接强度"]
TICKS1 = 120

def run_plasticity(enabled: bool, seed: int = 42) -> AIBrainEntity:
    """相同种子 = 相同初始突触与噪声序列，唯一变量是可塑性开关"""
    brain = AIBrainEntity(f"exp1-{'on' if enabled else 'off'}",
                          seed=seed, record_history=True)
    brain.hebbian_enabled = enabled
    for i in range(TICKS1):
        brain.sensory_input(STIMS1[i % len(STIMS1)])
    return brain

brain_on = run_plasticity(True)
brain_off = run_plasticity(False)

curve1 = pd.DataFrame(
    [{"tick": t, "突触平均强度": v, "条件": "开启可塑性"}
     for t, v in zip(brain_on.history["tick"], brain_on.history["synapse_mean"])] +
    [{"tick": t, "突触平均强度": v, "条件": "关闭可塑性"}
     for t, v in zip(brain_off.history["tick"], brain_off.history["synapse_mean"])])

bar1 = pd.DataFrame({
    "条件": ["开启可塑性", "关闭可塑性"],
    "强连接数(>0.5)": [brain_on.strong_synapse_count(),
                   brain_off.strong_synapse_count()],
})

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
sns.lineplot(data=curve1, x="tick", y="突触平均强度", hue="条件", ax=axes[0])
axes[0].set_title(f"突触平均强度演化（4 种文本循环刺激 {TICKS1} tick）")
axes[0].set_xlabel("tick")
sns.barplot(data=bar1, x="条件", y="强连接数(>0.5)", ax=axes[1],
            hue="条件", legend=False, palette=["#DD8452", "#4C72B0"])
axes[1].set_title("强突触连接数量（共768条前馈突触）")
for i, v in enumerate(bar1["强连接数(>0.5)"]):
    axes[1].text(i, v + 5, f"{v}", ha="center")
fig.tight_layout()
fig.savefig(FIG / "exp1_hebbian.png", dpi=220, bbox_inches="tight")
plt.close(fig)

RESULTS["exp1"] = {
    "ticks": TICKS1,
    "final_synapse_mean_on": round(brain_on.synapse_mean(), 4),
    "final_synapse_mean_off": round(brain_off.synapse_mean(), 4),
    "strong_synapses_on": brain_on.strong_synapse_count(),
    "strong_synapses_off": brain_off.strong_synapse_count(),
    "interpretation": "循环刺激反复同步激活相同神经通路，LTP 项主导权重演化；"
                      "关闭可塑性后突触始终停留在初始分布——"
                      "验证实体的经验驱动结构可塑性",
}

# ===================== 实验2：记忆固化与类艾宾浩斯遗忘 =====================
print("运行实验2：记忆固化与遗忘 ...")

TOPICS = {
    "天气": ["今天外面的天气很好", "昨夜的暴雨冲垮了小桥", "秋风吹落了一地黄叶",
            "寒冬的第一场雪来了", "春雷唤醒了沉睡的大地", "傍晚的晚霞红似火",
            "清晨的雾气弥漫山谷", "台风即将登陆沿海"],
    "学习": ["神经元脉冲正在传递信号", "记忆是智慧的基石", "熟能生巧勤能补拙",
            "读书破万卷下笔如有神", "学而不思则罔", "温故而知新",
            "知识改变命运", "实践是检验真理的标准"],
    "食物": ["妈妈做的红烧肉很香", "清晨的豆浆配油条", "火锅是冬天的灵魂",
            "茶香沁人心脾", "新鲜的草莓酸甜可口", "烤面包的香气飘满屋",
            "一碗热汤驱散寒冷", "粽子是端午的味道"],
    "运动": ["晨跑让人神清气爽", "篮球比赛进入加时", "游泳是最好的全身运动",
            "登山远眺心旷神怡", "瑜伽让身心平静", "马拉松考验意志力",
            "骑车穿过金色麦田", "太极讲究以柔克刚"],
    "科技": ["人工智能正在改变世界", "量子计算取得新突破", "火箭成功回收复用",
            "芯片制程进入纳米时代", "电动汽车续航破千公里", "卫星组网覆盖全球",
            "基因编辑治疗遗传病", "虚拟现实以假乱真"],
    "自然": ["萤火虫点亮夏夜", "候鸟迁徙飞越千山", "珊瑚礁是海洋热带雨林",
            "沙漠中的胡杨千年不倒", "鲸歌回荡在深海", "蜜蜂跳八字舞传信",
            "竹子在雨后疯长", "极光舞动在极地夜空"],
    "情感": ["朋友的鼓励让我坚持", "故乡的月亮最圆", "离别是为了更好重逢",
            "孩子的笑声治愈一切", "信任是关系的基石", "思念如潮水般涌来",
            "感恩的心常在", "勇气来自被爱"],
    "艺术": ["敦煌壁画历经千年", "贝多芬的月光奏鸣曲", "昆曲水磨腔婉转",
            "青花瓷蓝白相映", "莎士比亚的哈姆雷特", "书法讲究气韵生动",
            "印象派捕捉光影", "编钟之声穿越千年"],
}
STIMULI2 = [s for variants in TOPICS.values() for s in variants]  # 64 条
assert len(STIMULI2) == 64
IMPORTANT = "重要事件：创造者授予我最高荣誉勋章"
DECAY_GAMMA = 0.985
DECAY_ROUNDS = 400

brain2 = AIBrainEntity("exp2", seed=42, record_history=True)

# 阶段 A：64 个不同刺激依次输入（超过 STM 容量 20，触发固化竞争）
for s in STIMULI2:
    brain2.sensory_input(s)
phase_a_end_tick = brain2.tick

# 阶段 B：单一重要事件强化 25 次
for _ in range(25):
    brain2.sensory_input(IMPORTANT)

ltm_after_ab = len(brain2.long_memory)
important_w = next(m.weight for m in brain2.long_memory
                   if m.content == IMPORTANT)

# 阶段 C：不再输入，施加 400 轮衰减
decay_trace = []
for r in range(1, DECAY_ROUNDS + 1):
    brain2.decay_memory(DECAY_GAMMA)
    n = len(brain2.long_memory)
    w = (sum(m.weight for m in brain2.long_memory) / n) if n else 0.0
    decay_trace.append({"衰减轮次": r, "LTM存活条数": n, "LTM平均权重": w})

decay_df = pd.DataFrame(decay_trace)
w0 = decay_df["LTM平均权重"].iloc[0]
half = decay_df[decay_df["LTM平均权重"] <= w0 / 2]
half_life = int(half["衰减轮次"].iloc[0]) if not half.empty else None
n0 = decay_df["LTM存活条数"].iloc[0]
died = decay_df[decay_df["LTM存活条数"] < n0]
first_death = int(died["衰减轮次"].iloc[0]) if not died.empty else None

hist2 = brain2.history
size_df = pd.DataFrame(
    [{"tick": t, "条数": v, "存储": "STM"}
     for t, v in zip(hist2["tick"], hist2["stm_size"])] +
    [{"tick": t, "条数": v, "存储": "LTM"}
     for t, v in zip(hist2["tick"], hist2["ltm_size"])])

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
sns.lineplot(data=size_df, x="tick", y="条数", hue="存储", ax=axes[0])
axes[0].axvline(phase_a_end_tick, c="gray", ls="--", lw=1)
axes[0].text(phase_a_end_tick + 1, axes[0].get_ylim()[1] * 0.95,
             "阶段B: 重要事件×25", fontsize=9, color="gray")
axes[0].set_title("阶段A/B：STM 容量竞争与 LTM 固化")
axes[0].set_xlabel("tick")

ax_r = axes[1].twinx()
sns.lineplot(data=decay_df, x="衰减轮次", y="LTM平均权重", ax=axes[1],
             color="#4C72B0", label="LTM平均权重")
sns.lineplot(data=decay_df, x="衰减轮次", y="LTM存活条数", ax=ax_r,
             color="#DD8452", label="LTM存活条数")
axes[1].set_title(f"阶段C：{DECAY_ROUNDS} 轮指数衰减（γ={DECAY_GAMMA}）下的遗忘")
axes[1].set_ylabel("LTM平均权重", color="#4C72B0")
ax_r.set_ylabel("LTM存活条数", color="#DD8452")
if first_death:
    axes[1].axvline(first_death, c="gray", ls="--", lw=1)
    axes[1].text(first_death + 3, w0 * 0.9,
                 f"第{first_death}轮起条目消亡", fontsize=9, color="gray")
fig.tight_layout()
fig.savefig(FIG / "exp2_memory.png", dpi=220, bbox_inches="tight")
plt.close(fig)

RESULTS["exp2"] = {
    "stimuli_stage_a": len(STIMULI2),
    "reinforce_stage_b": 25,
    "ltm_after_stage_ab": ltm_after_ab,
    "important_event_weight": round(important_w, 3),
    "decay_gamma": DECAY_GAMMA,
    "decay_rounds": DECAY_ROUNDS,
    "ltm_survivors_final": int(decay_df["LTM存活条数"].iloc[-1]),
    "half_life_rounds": half_life,
    "first_death_round": first_death,
    "interpretation": "STM 饱和后权重达标条目持续固化入 LTM；指数衰减下"
                      "LTM 平均权重呈类艾宾浩斯曲线，条目自特定轮次起按权重"
                      "升序消亡——最弱的记忆最先被遗忘，反复强化的记忆存活最久",
}

# ===================== 实验3：情绪-注意力闭环的稳定性 =====================
print("运行实验3：情绪-注意力闭环 ...")

HIGH3 = ("高强度刺激：一场突如其来的风暴席卷了整座城市，"
         "无数神经元同时被激活，信号如洪流般涌入感知系统")
LOW3 = "低强度刺激：远处传来一声微弱的虫鸣"
LOW_SCALE = 0.25   # 低强度 = 输入电流按 0.25 缩放
TICKS3, PERIOD, HIGH_LEN = 150, 30, 15

brain3 = AIBrainEntity("exp3", seed=42, record_history=True)
for i in range(TICKS3 // PERIOD):
    for _ in range(HIGH_LEN):           # 每周期前 15 tick：高强度刺激
        brain3.sensory_input(HIGH3)
    for _ in range(PERIOD - HIGH_LEN):  # 后 15 tick：低强度刺激
        low_currents = [c * LOW_SCALE for c in brain3._str_to_current(LOW3)]
        brain3._perceive(LOW3, low_currents, tag="sensory")

h3 = brain3.history
emo_df = pd.DataFrame(
    [{"tick": t, "强度": e[k], "情绪": name}
     for t, e in zip(h3["tick"], h3["emotion"])
     for k, name in (("curiosity", "好奇"), ("stress", "压力"),
                     ("pleasure", "愉悦"), ("calm", "平静"))])
att_df = pd.DataFrame({"tick": h3["tick"], "注意力因子": h3["attention"],
                       "脉冲发放率": h3["spike_rate"]})
att_min = round(min(h3["attention"]), 3)
att_max = round(max(h3["attention"]), 3)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
sns.lineplot(data=emo_df, x="tick", y="强度", hue="情绪", ax=axes[0])
axes[0].set_title("四维情绪动态（高/低强度交替，30 tick 一周期）")
axes[0].set_xlabel("tick")
for p in range(1, TICKS3 // PERIOD):
    axes[0].axvline(p * PERIOD, c="gray", ls=":", lw=0.8)

sns.lineplot(data=att_df, x="tick", y="注意力因子", ax=axes[1],
             color="#4C72B0", label="注意力因子")
ax_r3 = axes[1].twinx()
sns.lineplot(data=att_df, x="tick", y="脉冲发放率", ax=ax_r3,
             color="#DD8452", alpha=0.6, label="脉冲发放率")
axes[1].set_title(f"注意力受控波动于 [{att_min}, {att_max}]，无发散")
axes[1].set_xlabel("tick")
fig.tight_layout()
fig.savefig(FIG / "exp3_emotion.png", dpi=220, bbox_inches="tight")
plt.close(fig)

final_emo = h3["emotion"][-1]
RESULTS["exp3"] = {
    "ticks": TICKS3,
    "attention_range": [att_min, att_max],
    "final_emotion": {k: round(v, 3) for k, v in final_emo.items()},
    "interpretation": "好奇心随高强度刺激持续上升，压力缓慢积累；"
                      "情绪衰减项（2%/tick）与 clip 约束保证注意力因子"
                      "在窄区间内受控波动，闭环无正反馈发散",
}

# ===================== 实验4：脉冲活动模式观察 =====================
print("运行实验4：脉冲光栅图 ...")

FIXED4 = "固定语句：月光洒在安静的湖面上"
TICKS4 = 80

brain4 = AIBrainEntity("exp4", seed=42, record_history=True)
rng4 = random.Random(0)
for i in range(TICKS4):
    if i % 2 == 0:
        brain4.sensory_input(FIXED4)
    else:
        brain4.sensory_input(f"随机噪声 {rng4.random():.6f}")

h4 = brain4.history
raster_rows = []
for t, (s_ids, a_ids, d_ids) in enumerate(zip(
        h4["sense_spikes"], h4["assoc_spikes"], h4["decision_spikes"])):
    for nid in s_ids:
        raster_rows.append({"tick": t, "neuron": nid, "层": "感官层(0-15)"})
    for nid in a_ids:
        raster_rows.append({"tick": t, "neuron": nid, "层": "联想层(16-47)"})
    for nid in d_ids:
        raster_rows.append({"tick": t, "neuron": nid, "层": "决策层(48-55)"})
raster_df = pd.DataFrame(raster_rows)

total_spikes = len(raster_df)
mean_rate = round(total_spikes / (56 * TICKS4), 3)

fig, ax = plt.subplots(figsize=(13, 6))
sns.scatterplot(data=raster_df, x="tick", y="neuron", hue="层",
                palette=["#4C72B0", "#DD8452", "#55A868"],
                s=8, linewidth=0, ax=ax)
ax.axhline(15.5, c="black", lw=0.8)
ax.axhline(47.5, c="black", lw=0.8)
ax.set_yticks([7, 31, 51])
ax.set_yticklabels(["感官层", "联想层", "决策层"])
ax.set_ylabel("")
ax.set_title(f"实验4：三层神经元集群脉冲光栅图（{TICKS4} tick，"
             f"共 {total_spikes} 次脉冲，平均发放率 {mean_rate}）")
ax.legend(loc="upper right", fontsize=8, markerscale=2)
fig.tight_layout()
fig.savefig(FIG / "exp4_raster.png", dpi=220, bbox_inches="tight")
plt.close(fig)

RESULTS["exp4"] = {
    "ticks": TICKS4,
    "total_spikes": total_spikes,
    "mean_spike_rate": mean_rate,
    "interpretation": "光栅图呈现层间因果链——感官层稀疏放电经突触汇集后"
                      "在联想层诱发密集同步放电，进而驱动决策层激活；"
                      "固定语句与噪声区间在感官层呈现可区分的放电指纹",
}


# ===================== 实验5：STDP 因果方向性 =====================
print("运行实验5：STDP 因果方向性 ...")

def train_sequence(order: str, cycles: int = 60, seed: int = 42) -> AIBrainEntity:
    """按 A,B + 长间歇 的节奏训练：A→B 间隔 2 个网络步（STDP 窗内强 LTP），
    B→下一A 间隔 6 步（弱 LTP），且反因果边受 LTD——制造因果方向差异。
    使用较低学习速率，避免双向权重都在训练中饱和（饱和后方向性不可见）。"""
    brain = AIBrainEntity(f"exp5-{order}", seed=seed, record_history=True)
    brain.settle_ticks = 1
    brain.hebbian_rate = 0.003
    seq = (STIM_A, STIM_B) if order == "AB" else (STIM_B, STIM_A)
    for _ in range(cycles):
        brain.sensory_input(seq[0])
        brain.sensory_input(seq[1])
        brain.free_run(8)
    return brain

def drive_vector(brain: AIBrainEntity, stimulus: str) -> dict:
    """刺激对每个联想层神经元的前馈驱动量 Σ(感官电流 × 突触权重)"""
    currents = brain._str_to_current(stimulus)
    return {an.id: sum(currents[i] * brain.synapse.get((sn.id, an.id), 0.0)
                       for i, sn in enumerate(brain.sense_layer))
            for an in brain.assoc_layer}

def directional_score(brain: AIBrainEntity) -> Tuple[float, float]:
    """连续型因果方向度量：对每条联想层侧向循环突触 (pre→post)，
    按 pre 的 A 偏好度 × post 的 B 偏好度 × 权重 累加（A→B 得分），
    反向同理（B→A 得分）。偏好度 = 该神经元对两刺激驱动量之差的正部。
    得分按偏好总质量归一化，可跨脑比较；覆盖全部循环边，无稀疏性问题。
    """
    da = drive_vector(brain, STIM_A)
    db = drive_vector(brain, STIM_B)
    # 各自归一（消除前馈整体强弱差异），再做均值中心的差分偏好
    na = sum(da.values()) or 1.0
    nb = sum(db.values()) or 1.0
    diff = {j: da[j] / na - db[j] / nb for j in da}
    mu = sum(diff.values()) / len(diff)
    pref_a = {j: max(0.0, d - mu) for j, d in diff.items()}
    pref_b = {j: max(0.0, mu - d) for j, d in diff.items()}
    mass = (sum(pref_a.values()) * sum(pref_b.values())) or 1.0
    ab = sum(w * pref_a.get(pre, 0.0) * pref_b.get(post, 0.0)
             for (pre, post), w in brain.recurrent_synapse.items()) / mass
    ba = sum(w * pref_b.get(pre, 0.0) * pref_a.get(post, 0.0)
             for (pre, post), w in brain.recurrent_synapse.items()) / mass
    return ab, ba

brains5 = {
    "训练A→B": train_sequence("AB"),
    "训练B→A": train_sequence("BA"),
}
untrained = AIBrainEntity("exp5-untrained", seed=42)

# 面板1：学习曲线（A→B 训练）
h5 = brains5["训练A→B"].history
curve_df = pd.DataFrame({"tick": h5["tick"], "synapse_mean": h5["synapse_mean"]})

# 面板2：方向性度量——先分别探出 A/B 各自的联想层响应神经元集合，
# 再比较循环突触 A集合→B集合 与 B集合→A集合 的平均权重。
# STDP 若学到因果方向，训练A→B的脑应是 A→B 方向显著更强，反之亦然。
def asymmetry_index(ab: float, ba: float) -> float:
    """方向不对称指数：(A→B − B→A)/(A→B + B→A)，正=偏 A→B，负=偏 B→A"""
    tot = ab + ba
    return (ab - ba) / tot if tot > 0 else 0.0

dir_rows = []
dir_metrics = {}
for cond, b in brains5.items():
    ab_score, ba_score = directional_score(b)
    idx = asymmetry_index(ab_score, ba_score)
    dir_metrics[cond] = {"A->B_score": round(ab_score, 5),
                         "B->A_score": round(ba_score, 5),
                         "asymmetry_index": round(idx, 3)}
base_ab, base_ba = directional_score(untrained)
base_idx = asymmetry_index(base_ab, base_ba)
dir_metrics["未训练基线"] = {"A->B_score": round(base_ab, 5),
                            "B->A_score": round(base_ba, 5),
                            "asymmetry_index": round(base_idx, 3)}
# 按因果直觉排序：A→B 训练应最正，B→A 训练应最负
dir_df = pd.DataFrame([
    {"条件": "训练A→B", "方向不对称指数": dir_metrics["训练A→B"]["asymmetry_index"]},
    {"条件": "未训练基线", "方向不对称指数": base_idx},
    {"条件": "训练B→A", "方向不对称指数": dir_metrics["训练B→A"]["asymmetry_index"]},
])

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
sns.lineplot(data=curve_df, x="tick", y="synapse_mean", ax=axes[0],
             color="#4C72B0")
axes[0].set_title("STDP 学习曲线（A→B 序列训练 60 轮）")
axes[0].set_xlabel("tick")
axes[0].set_ylabel("前馈突触平均强度")

sns.barplot(data=dir_df, x="条件", y="方向不对称指数", ax=axes[1],
            hue="条件", legend=False,
            palette=["#4C72B0", "#937860", "#DD8452"])
axes[1].axhline(0, c="black", lw=1)
_vmax = dir_df["方向不对称指数"].max()
_vmin = dir_df["方向不对称指数"].min()
axes[1].set_ylim(_vmin - 0.12, _vmax + 0.12)
for i, v in enumerate(dir_df["方向不对称指数"]):
    axes[1].text(i, v + 0.02 if v >= 0 else v - 0.05, f"{v:+.2f}", ha="center")
axes[1].set_title("因果方向不对称指数（正=偏A→B，负=偏B→A）")
axes[1].set_ylabel("(A→B − B→A) / (A→B + B→A)")
fig.tight_layout()
fig.savefig(FIG / "exp5_stdp.png", dpi=220, bbox_inches="tight")
plt.close(fig)

RESULTS["exp5"] = {
    "directional_scores": dir_metrics,
    "final_synapse_mean": {
        cond: round(b.synapse_mean(), 4) for cond, b in brains5.items()},
    "interpretation": "方向不对称指数排序为 训练A→B > 未训练基线 > 训练B→A："
                      "训练时序方向决定循环突触强化的因果方向"
                      "（赫布规则无此方向敏感性）",
}

# ===================== 实验6：多巴胺奖励调制学习 =====================
print("运行实验6：多巴胺奖励调制 ...")

TARGET = "技能训练：投篮动作"
DISTRACTORS = ["路过一只鸽子", "远处传来音乐", "风吹过树叶", "有人在聊天"]

def run_reward_trial(rewarded: bool, seed: int = 42, ticks: int = 80):
    brain = AIBrainEntity(f"exp6-{rewarded}", seed=seed, record_history=True)
    dopa_trace = []
    for i in range(ticks):
        if rewarded:
            brain.reward(0.6)          # 学习前给予奖励，抬升多巴胺
        brain.sensory_input(TARGET)
        # 每 4 tick 插入一个干扰刺激（无奖励），模拟真实环境噪声
        if i % 4 == 3:
            brain.sensory_input(DISTRACTORS[i // 4 % len(DISTRACTORS)])
        dopa_trace.append({"tick": brain.tick, "dopamine": brain.dopamine,
                           "条件": "奖励组" if rewarded else "无奖励组"})
    h = brain.history
    df = pd.DataFrame({"tick": h["tick"], "synapse_mean": h["synapse_mean"],
                       "条件": "奖励组" if rewarded else "无奖励组"})
    target_mem = next((m for m in brain.long_memory if m.content == TARGET), None)
    return df, pd.DataFrame(dopa_trace), brain, target_mem

df_r, dopa_r, brain_r, mem_r = run_reward_trial(True)
df_u, dopa_u, brain_u, mem_u = run_reward_trial(False)
exp6 = pd.concat([df_r, df_u])

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
sns.lineplot(data=exp6, x="tick", y="synapse_mean", hue="条件", ax=axes[0])
axes[0].set_title("突触强度增长：奖励组 vs 无奖励组")
axes[0].set_xlabel("tick")
axes[0].set_ylabel("前馈突触平均强度")

sns.lineplot(data=dopa_r, x="tick", y="dopamine", ax=axes[1], color="#C44E52")
axes[1].set_title("奖励组多巴胺水平（每次学习前 +0.6，逐网络步代谢）")
axes[1].set_xlabel("tick")
axes[1].set_ylabel("多巴胺")

bar_df = pd.DataFrame({
    "条件": ["奖励组", "无奖励组"],
    "强连接数(>0.5)": [brain_r.strong_synapse_count(),
                     brain_u.strong_synapse_count()],
})
sns.barplot(data=bar_df, x="条件", y="强连接数(>0.5)", ax=axes[2],
            hue="条件", legend=False,
            palette=["#DD8452", "#4C72B0"])
axes[2].set_title("强突触连接数量（共768条前馈突触）")
for i, v in enumerate(bar_df["强连接数(>0.5)"]):
    axes[2].text(i, v + 5, f"{v}", ha="center")
fig.tight_layout()
fig.savefig(FIG / "exp6_dopamine.png", dpi=220, bbox_inches="tight")
plt.close(fig)

RESULTS["exp6"] = {
    "final_synapse_mean_rewarded": round(brain_r.synapse_mean(), 4),
    "final_synapse_mean_unrewarded": round(brain_u.synapse_mean(), 4),
    "strong_synapses_rewarded": brain_r.strong_synapse_count(),
    "strong_synapses_unrewarded": brain_u.strong_synapse_count(),
    "target_ltm_weight_rewarded": round(mem_r.weight, 4) if mem_r else None,
    "target_ltm_weight_unrewarded": round(mem_u.weight, 4) if mem_u else None,
    "interpretation": "奖励抬升多巴胺 -> STDP 速率放大，突触学习与强连接显著更多；"
                      "目标记忆权重两组均因高频重复饱和（奖励作用于突触可塑性，"
                      "不直接改变记忆写入权重）",
}

# ===================== 实验7：群体文化传递 =====================
print("运行实验7：群体文化传递（6 代链式传递）...")

MEMES = ["钻木可以取火", "结绳可以记事", "观测星象定节气", "烧制陶器储水",
         "打磨石器狩猎", "口耳相传史诗", "驯化野生谷物", "建造半地穴房屋"]
GEN_NAMES = [f"第{g}代" for g in range(6)]

swarm_rehearse = BrainSwarm(GEN_NAMES, seed=42)
chain_rehearse = generation_chain(swarm_rehearse, MEMES, teach_times=25,
                                  fidelity=0.85, seed=1, rehearse_times=3)

swarm_no = BrainSwarm(GEN_NAMES, seed=42)
chain_no = generation_chain(swarm_no, MEMES, teach_times=25,
                            fidelity=0.85, seed=1, rehearse_times=0)

chain_df = pd.DataFrame(
    [{"世代": r["generation"], "存活文化记忆数": r["survived"],
      "平均记忆权重": r["avg_weight"], "条件": "传递+温习"}
     for r in chain_rehearse] +
    [{"世代": r["generation"], "存活文化记忆数": r["survived"],
      "平均记忆权重": r["avg_weight"], "条件": "仅传递不温习"}
     for r in chain_no])

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
sns.lineplot(data=chain_df, x="世代", y="存活文化记忆数", hue="条件",
             marker="o", ax=axes[0])
axes[0].set_ylim(-0.5, len(MEMES) + 0.5)
axes[0].set_title(f"文化记忆跨代存活（共{len(MEMES)}条，保真度0.85）")
sns.lineplot(data=chain_df, x="世代", y="平均记忆权重", hue="条件",
             marker="s", ax=axes[1])
axes[1].set_title("存活记忆的平均权重逐代变化")
axes[1].set_ylim(0, 1.0)
mut_total = sum(r["mutated"] for r in chain_rehearse)
fig.suptitle(f"实验7：文化沿 6 代链式传递——温习维系文化，失传即灭绝"
             f"（温习组累计变异 {mut_total} 条）")
fig.tight_layout()
fig.savefig(FIG / "exp7_swarm.png", dpi=220, bbox_inches="tight")
plt.close(fig)

RESULTS["exp7"] = {
    "memes_total": len(MEMES),
    "chain_with_rehearsal": chain_rehearse,
    "chain_without_rehearsal": chain_no,
    "final_gen_survival_with_rehearsal": chain_rehearse[-1]["survived"],
    "final_gen_survival_without_rehearsal": chain_no[-1]["survived"],
    "mutated_total_with_rehearsal": mut_total,
    "interpretation": "温习使文化记忆跨代存活并伴随变异；不温习则权重逐代"
                      "衰减跌破固化阈值，文化在数代内灭绝",
}

# ===================== 实验8：多模态 embedding 感知 =====================
print("运行实验8：多模态 embedding 感知 ...")

def cosine(u, v):
    dot = sum(a * b for a, b in zip(u, v))
    nu = math.sqrt(sum(a * a for a in u)) or 1.0
    nv = math.sqrt(sum(b * b for b in v)) or 1.0
    return dot / (nu * nv)

# 实验8说明：感官层把任意长度 embedding 线性插值重采样到 16 维。
# 稠密 iid 高斯 embedding 在插值平均下会"扁平化"（对比度丢失），
# 因此使用分段结构 embedding（16 段 × 32 重复，模拟低频结构占主导的
# 真实 embedding），并直接在"感官电流"层面衡量相似性保持。

rng = random.Random(7)
SEG, REP = 16, 32

def block_vec(seg_values):
    return [s for s in seg_values for _ in range(REP)]

base_segs = [rng.uniform(-1, 1) for _ in range(SEG)]

def variant(src_segs, noise_scale):
    return [x + rng.gauss(0, noise_scale) for x in src_segs]

# 语义标签模拟：猫 / 猫的变体（相似 embedding）/ 不相关对象
stimuli8 = [
    ("猫", block_vec(base_segs)),
    ("猫·变体1", block_vec(variant(base_segs, 0.15))),
    ("猫·变体2", block_vec(variant(base_segs, 0.35))),
    ("狗", block_vec([rng.uniform(-1, 1) for _ in range(SEG)])),
    ("汽车", block_vec([rng.uniform(-1, 1) for _ in range(SEG)])),
    ("随机噪声", block_vec([rng.uniform(-1, 1) for _ in range(SEG)])),
]

# 每个刺激经 sensory_input_vector 的归一化通路生成感官电流（与大脑内一致）
labels8 = [s for s, _ in stimuli8]
vecs8 = {s: v for s, v in stimuli8}
currents8 = {s: AIBrainEntity._normalize_vector(v, 16) for s, v in stimuli8}
cos_mat = pd.DataFrame(
    [[cosine(vecs8[a], vecs8[b]) for b in labels8] for a in labels8],
    index=labels8, columns=labels8)
cur_mat = pd.DataFrame(
    [[cosine(currents8[a], currents8[b]) for b in labels8] for a in labels8],
    index=labels8, columns=labels8)

# 端到端验证：向量刺激经完整感知流水线进入大脑（冒烟测试）
e2e_brain = AIBrainEntity("exp8-e2e", seed=42)
e2e_out = e2e_brain.sensory_input_vector(vecs8["猫"], label="猫(embedding)")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
sns.heatmap(cos_mat, annot=True, fmt=".2f", cmap="vlag", center=0,
            vmin=-1, vmax=1, square=True, ax=axes[0],
            cbar_kws={"label": "余弦相似度"})
axes[0].set_title("输入 embedding 的余弦相似度（512 维）")
sns.heatmap(cur_mat, annot=True, fmt=".2f", cmap="vlag", center=0,
            vmin=-1, vmax=1, square=True, ax=axes[1],
            cbar_kws={"label": "余弦相似度"})
axes[1].set_title("重采样后 16 维感官电流的余弦相似度")
fig.suptitle("实验8：embedding 相似性 -> 感官电流相似性（sensory_input_vector 通路）")
fig.tight_layout()
fig.savefig(FIG / "exp8_multimodal.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# 真实文件端到端冒烟测试：perceive_image（CLIP 未安装时自动降级伪 embedding）
demo_brain = AIBrainEntity("exp8-file", seed=42)
banner = BASE / "figures" / "banner.png"
file_result = demo_brain.perceive_image(str(banner), label="项目封面图") \
    if banner.exists() else None

RESULTS["exp8"] = {
    "current_cos_cat_vs_cat_variant1": round(cur_mat.loc["猫", "猫·变体1"], 3),
    "current_cos_cat_vs_cat_variant2": round(cur_mat.loc["猫", "猫·变体2"], 3),
    "current_cos_cat_vs_dog": round(cur_mat.loc["猫", "狗"], 3),
    "current_cos_cat_vs_random": round(cur_mat.loc["猫", "随机噪声"], 3),
    "input_cos_cat_vs_cat_variant1": round(cos_mat.loc["猫", "猫·变体1"], 3),
    "input_cos_cat_vs_dog": round(cos_mat.loc["猫", "狗"], 3),
    "e2e_vector_perception": e2e_out,
    "perceive_image_smoke_test": file_result,
    "interpretation": "重采样通路保持分段结构 embedding 的相似性排序；"
                      "注1：abs 归一化丢失符号信息，抬高了不相关对的基线相似度；"
                      "注2：稠密 iid embedding 在 16 维插值重采样下会扁平化，"
                      "未来可换可学习投影替代线性插值",
}

# ===================== 汇总 =====================
DATA = BASE / "data"
DATA.mkdir(exist_ok=True)
with open(DATA / "experiment_results.json", "w", encoding="utf-8") as f:
    json.dump(RESULTS, f, ensure_ascii=False, indent=2)

print("\n===== 实验 1-8 结果摘要 =====")
print(json.dumps(RESULTS, ensure_ascii=False, indent=2))
print(f"\n图表已保存至: {FIG}")
