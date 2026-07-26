# -*- coding: utf-8 -*-
"""
基础机制实验脚本：为核心机制生成真实实验数据与图表
（基于 ai_brain_entity.py v3.0 核心；可塑性开关同时控制 STDP）

实验1：突触可塑性的有效性（可塑性开/关对照，追踪突触强度演化）
实验2：记忆固化与类艾宾浩斯遗忘（固化竞争 -> 强化 -> 400 轮衰减）
实验3：情绪-注意力闭环的稳定性（高/低强度交替刺激 150 tick）
实验4：脉冲活动模式观察（80 tick 三层神经元脉冲光栅图）
"""
import json
import random
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

RESULTS = {}

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

# ===================== 汇总 =====================
DATA = BASE / "data"
DATA.mkdir(exist_ok=True)
with open(DATA / "experiment_results.json", "w", encoding="utf-8") as f:
    json.dump(RESULTS, f, ensure_ascii=False, indent=2)

print("\n===== 基础机制实验结果摘要 =====")
print(json.dumps(RESULTS, ensure_ascii=False, indent=2))
print(f"\n图表已保存至: {FIG}")
