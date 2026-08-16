# Brain Simulator v4.0 — 118万参数类脑认知架构

基于全局工作空间理论(GWT)、预测编码、强化学习、Hebbian学习和遗传算法的18模块类脑认知系统。

## 概述

本项目模拟了一个具有意识、情绪、记忆、动机、注意力、睡眠和自我反思能力的类脑认知架构。18个模块通过消息总线互相通信，形成完整的"感知→注意→意识→决策→行动→学习→巩固"认知闭环。

### 理论基础

- **全局工作空间理论(GWT)** — 意识的并行竞争与串行广播
- **预测编码(Predictive Coding)** — 多层预测误差最小化
- **强化学习(Actor-Critic)** — 多巴胺RPE驱动的策略学习
- **自我决定理论(SDT)** — 好奇心/胜任感/自主性等内在动机
- **Hebbian学习** — 在线权重更新
- **双过程理论** — 系统1(快速直觉)/系统2(缓慢推理)
- **突触稳态** — 睡眠中的突触缩放

## 项目结构

```
brain_simulator/
├── main.py                     # 主入口 — 8模块认知演示
├── evolve.py                   # 遗传算法进化
├── export_model.py             # v3模型导出脚本
├── models/
│   ├── brain_v3.1_1m.npz       # v3.1模型(8模块, 100万参数)
│   └── brain_v4.0_18modules.npz # v4.0模型(18模块, 118万参数)
└── brain/
    ├── __init__.py
    └── core/
        ├── language_model.py       # 1. 基础LLM后端
        ├── memory.py               # 2. 三级记忆系统
        ├── emotion.py              # 3. 情绪核心+多巴胺
        ├── consciousness.py        # 4. GWT全局工作空间意识
        ├── motivation.py           # 5. 内在动机(SDT)
        ├── bus.py                  # 6. 模块通信总线
        ├── predictive_coding.py    # 7. 预测编码网络(99.8万参数)
        ├── thought.py              # 8. 思考系统(系统1/2)
        ├── basal_ganglia.py        # 9. 基底神经节(动作选择+习惯)
        ├── attention.py            # 10. 注意力系统
        ├── sleep.py                # 11. 睡眠与记忆重放
        ├── homeostasis.py          # 12. 稳态调节(能量/疲劳/昼夜)
        ├── default_mode.py         # 13. 默认模式网络(想象力)
        ├── metacognition.py        # 14. 元认知与自我模型
        ├── cerebellum.py           # 15. 小脑(时序协调+自动化)
        ├── reinforcement.py        # 16. 强化学习(Actor-Critic)
        ├── goals.py                # 17. 目标管理与规划
        ├── emotion_regulation.py   # 18. 情绪调节
        ├── evolution.py            # 遗传算法
        └── model_io.py             # 模型导入导出
```

## 环境要求

- Python 3.10+
- numpy

```bash
pip install numpy
```

## 快速开始

### 运行认知系统

```bash
python main.py
```

### 遗传算法进化

```bash
python evolve.py
```

### 加载预训练模型

```python
from brain import load_model, restore_modules
from brain import *

pc, state = load_model("models/brain_v4.0_18modules.npz")

emo = EmotionalCore()
mot = MotivationSystem()
con = ConsciousnessSystem()
bg = BasalGanglia()
cer = Cerebellum(128, 64)
rl = ReinforcementLearning(128, 10)
meta = Metacognition()

restore_modules(state, emotion=emo, motivation=mot, consciousness=con,
                basal_ganglia=bg, cerebellum=cer,
                reinforcement=rl, metacognition=meta)

print(state["meta"]["version"])       # 4.0
print(state["meta"]["total_params"])  # 1180320
```

## 18模块详解

### 核心模块 (1-8)

#### 1. 基础LLM (`language_model.py`)

语言模型后端抽象基类，内置MockLanguageModel（关键词模板+哈希词袋）。可继承`LanguageModel`接入真实LLM。

```python
from brain import MockLanguageModel

llm = MockLanguageModel()
resp = llm.generate("你好", temperature=0.7, top_p=0.9)
emb = llm.embed("文本")  # 128维向量
```

#### 2. 三级记忆 (`memory.py`)

模拟人类记忆的三级加工模型：

- **感觉缓冲** (200条, 3秒衰减) — 瞬时感觉登记
- **短期记忆** (100条, 多巴胺门控巩固) — 工作记忆
- **长期记忆** (5000条, 艾宾浩斯遗忘+情绪增强) — 持久存储

```python
from brain import MemorySystem

mem = MemorySystem(sensory_buffer_size=200, stm_size=100, ltm_size=5000)
mem.input_sensory(vector, emotional_valence=0.5)
mem.step(dt=1.0, dopamine_level=0.5)
results = mem.short_term.retrieve(query, top_k=5)
```

#### 3. 情绪核心 (`emotion.py`)

六维情绪模型（喜悦/悲伤/愤怒/恐惧/厌恶/惊讶）+ 多巴胺奖赏预测误差(RPE)系统。情绪状态直接调制LLM采样参数。

```python
from brain import EmotionalCore

emo = EmotionalCore()
emo.evaluate_stimulus("今天真开心")
temp, top_p = emo.get_generation_params(cortisol=0.15, oxytocin=0.2)
rpe = emo.dopamine.compute_rpe(reward=0.8, stimulus_id="turn_1")
print(emo.state.dominant())    # 主导情绪
print(emo.state.valence)       # 效价 -1~1
print(emo.state.arousal)       # 唤醒度 0~1
```

#### 4. 意识GWT (`consciousness.py`)

基于Baars全局工作空间理论：多模块并行产生候选，竞争获胜者广播到全系统。包含Φ信息整合度量和7级意识水平判定。

```python
from brain import ConsciousnessSystem

con = ConsciousnessSystem()
candidates = con.build_candidates(
    user_input="为什么？", emotion_state=emo.state,
    prediction_error=0.4, curiosity=0.7)
winner = con.workspace.compete(candidates)
phi = con.compute_phi({"sensory": 0.8, "memory": 0.5})
level = con.determine_level(arousal=0.5)
```

#### 5. 内在动机 (`motivation.py`)

基于自我决定理论(SDT)的6种驱动力 + 好奇心引擎（金发姑娘效应：偏好适度新颖/复杂的刺激）。

```python
from brain import MotivationSystem

mot = MotivationSystem()
result = mot.evaluate(user_input="为什么？", prediction_error=0.4)
# result: reward, curiosity, dominant_drive, explore_prob
```

6种驱动力：好奇心、胜任感、自主性、社交、确定性、回避。

#### 6. 通信总线 (`bus.py`)

发布-订阅消息总线，支持点对点、广播、中间件过滤、优先级和历史记录。15种消息类型。

```python
from brain import MessageBus, Message, MessageType

bus = MessageBus()
bus.subscribe("my_module", MessageType.THOUGHT, handler)
bus.publish(Message(sender="consciousness", msg_type=MessageType.CONSCIOUS_BROADCAST,
                    content={"text": "..."}, priority=10))
```

#### 7. 预测编码 (`predictive_coding.py`)

3层预测编码网络 `[512→650→256]`，99.8万参数。自上而下预测→计算误差→更新激活→Hebbian权重更新。

```python
from brain import PredictiveCodingNetwork

pc = PredictiveCodingNetwork([512, 650, 256])
result = pc.step(external_input=vector, dt=1.0)
print(result["mean_error"])  # 预测误差
```

#### 8. 思考系统 (`thought.py`)

思考空间（30个并行思维竞争）+ 思维流（100步序列记录）。支持系统1（快速直觉）和系统2（缓慢链式推理）双过程。

```python
from brain import ThoughtSystem

ts = ThoughtSystem(space_capacity=30, vector_dim=512)
ts.input_perceptual(vector, strength=0.7)
ts.input_emotional(vector, strength=0.6)
ts.activate_system2(steps=3)  # 触发深度推理
result = ts.step()
```

7种思维类型：感知、记忆、情绪、目标、推理、创造性、元认知。

---

### 扩展模块 (9-18)

#### 9. 基底神经节 (`basal_ganglia.py`)

动作选择的核心脑区。Go(D1)/NoGo(D2)双通路竞争，Q-learning价值更新，习惯-目标双系统（反复成功的行为自动化为习惯，无需意识参与）。

```python
from brain import BasalGanglia, ActionType

bg = BasalGanglia()
bg.register_action("speak", "说话", ActionType.SPEAK, context=vector)
aid, confidence, is_habit = bg.select_action(state, dopamine=0.5, motivations={...})
bg.update(reward=0.5)  # 多巴胺RPE驱动学习
stats = bg.get_action_stats()
```

#### 10. 注意力系统 (`attention.py`)

双路注意力：自下而上（刺激驱动：新颖性/威胁/变化）和自上而下（目标驱动）。注意力门控决定哪些信息进入GWT竞争，并与预测编码精度联动（注意力=精度分配）。

```python
from brain import AttentionSystem

att = AttentionSystem()
att.set_top_down_bias(goal_vector, strength=0.7)
targets = att.process({"sensory": vec1, "threat": vec2}, threat_level=0.8)
gated = att.gate_for_consciousness(candidates)
precision = att.get_precision_for_layer(layer_idx=0, layer_size=512)
```

#### 11. 睡眠与记忆重放 (`sleep.py`)

完整睡眠周期（N1→N2→N3→REM），各阶段执行不同功能：
- **N3深睡**：陈述性记忆重放、Hebbian重训练、突触缩放（防止权重爆炸）
- **REM**：情绪记忆处理、创造性重组（做梦）、记忆碎片组合
- **N2**：程序性记忆巩固

```python
from brain import SleepSystem

slp = SleepSystem()
slp.add_experience(vector, emotional_valence=0.3, importance=0.7)
result = slp.sleep_cycle(pc_network=pc, memory_system=mem, n_cycles=4)
# result: duration, cycles, total_replays, consolidations
```

#### 12. 稳态调节 (`homeostasis.py`)

能量消耗/恢复、疲劳累积、昼夜节律（双过程模型：过程S睡眠压力+过程C警觉度）、6种需求（能量/休息/社交/认知/安全/自主）。稳态失衡产生内感受驱动信号。

```python
from brain import HomeostaticSystem, NeedType

hom = HomeostaticSystem(start_hour=8.0)
state = hom.step(cognitive_demand=0.5, social_interaction=True, threat=False)
drives = hom.get_drive_signals()      # 传给动机系统
mod = hom.get_modulation()            # 调制认知精度/学习率/情绪基线
hom.satisfy_need(NeedType.SOCIAL, 0.2)
```

#### 13. 默认模式网络 (`default_mode.py`)

大脑空闲时最活跃的网络，负责心智游移、自传体记忆、反事实推理、未来模拟、社会认知和创造性重组。8种思维主题，联想链式生成。

```python
from brain import DefaultModeNetwork

dmn = DefaultModeNetwork()
dmn.add_memory_fragment(vector, valence=0.3)
thought = dmn.step(external_input=False, emotion_valence=0.2, emotion_arousal=0.4)
# thought.content: "想起了以前的事..." / "如果当时..." / "突然有个想法！"
candidates = dmn.get_consciousness_candidates()  # 可进入意识
```

#### 14. 元认知与自我模型 (`metacognition.py`)

对认知过程的认知：元记忆（知道自己知不知道/FOK/TOT）、置信度校准、错误检测、自我效能感、自尊、自我叙事。从"有意识"到"有自我意识"。

```python
from brain import Metacognition

meta = Metacognition()
monitor = meta.monitor(prediction_error=0.3, task_difficulty=0.6)
# monitor: fluency, confidence, confidence_level, error_likelihood
meta.evaluate_response(predicted_confidence=0.8, actual_correct=True, domain="学习")
reflection = meta.reflect()
# reflection: "最近表现不错...", self_efficacy, self_esteem
```

#### 15. 小脑 (`cerebellum.py`)

前馈模型（预测动作结果）、感觉预测误差校正、时序协调、自动化学习（幂律练习曲线：从刻意到自动）。

```python
from brain import Cerebellum

cer = Cerebellum(state_dim=128, command_dim=64)
predicted = cer.predict_outcome(current_state, command)
correction = cer.compute_correction(predicted, actual_outcome)
cer.learn(state, command, actual_next_state)
cer.register_sequence("walk", "走路", commands=[...])
print(cer.get_coordination_quality())
```

#### 16. 强化学习 (`reinforcement.py`)

完整Actor-Critic回路：Critic估计状态价值V(s)，Actor用softmax策略选动作，多巴胺TD误差驱动两者更新，资格迹信用分配。支持5种内在奖励（好奇心/新颖性/稳态/社交/外部）。

```python
from brain import ReinforcementLearning

rl = ReinforcementLearning(state_dim=128, n_actions=10)
result = rl.step(state, extrinsic_reward=0.5, prediction_error=0.3,
                 homeostatic_signals=hom.get_drive_signals())
td_error = rl.learn(state, result["action"], result["total_reward"], next_state)
rl.replay(n_samples=10)  # 经验回放
print(rl.get_dopamine())
```

#### 17. 目标管理 (`goals.py`)

目标栈（按紧急/重要矩阵排序）、子目标分解、进度监控、中断恢复（高优先级目标抢占，完成后恢复）。目标驱动自上而下注意力。

```python
from brain import GoalManager, GoalPriority

gm = GoalManager()
gm.add_goal("learn", "学习新技能", GoalPriority.HIGH,
            subgoals=["理解", "练习", "掌握"], urgency=0.7)
gm.complete_subgoal("learn", "learn_sg0")
current = gm.get_current_goal()
bias_desc, bias_strength = gm.get_attention_bias()
reward = gm.get_reward_signal()  # 传给RL
```

#### 18. 情绪调节 (`emotion_regulation.py`)

10种调节策略（认知重评/注意力转移/表达抑制/正念/社会支持/情境选择/自我安抚/反刍/接纳/分心），根据情绪类型、强度和认知资源自动选择策略并评估效果。

```python
from brain import EmotionRegulation

ereg = EmotionRegulation()
assessment = ereg.assess_emotion("恐惧", intensity=0.7, valence=-0.5)
result = ereg.regulate("恐惧", intensity=0.7, valence=-0.5,
                       cognitive_resources=0.8, social_available=True)
# result: strategy="认知重评", new_intensity=0.43, intensity_reduction=0.27
```

---

### 工具模块

#### 遗传算法 (`evolution.py`)

```python
from brain import GeneticAlgorithm, HyperParams, Genome

# 15个可进化超参数
hyper = HyperParams()
mutated = hyper.mutate(rate=0.2)

# 遗传算法
ga = GeneticAlgorithm(population_size=20, weight_dim=998400,
                      tournament_k=3, elite_count=2)
for individual in ga.population:
    individual.fitness = evaluate(individual)
ga.evolve()
```

#### 模型导入导出 (`model_io.py`)

```python
from brain import export_model, load_model, restore_modules, count_parameters

# 导出
export_model(pc=pc, emotion=emo, basal_ganglia=bg, attention=att,
             sleep=slp, homeostasis=hom, default_mode=dmn,
             metacognition=meta, cerebellum=cer, reinforcement=rl,
             goals=gm, emotion_regulation=ereg,
             filepath="models/my_model.npz")

# 加载
pc, state = load_model("models/my_model.npz")
restore_modules(state, emotion=emo, basal_ganglia=bg, ...)
```

## 认知闭环

```
外部输入
   ↓
稳态调节 → 需求信号 → 内在动机 ←→ 目标管理
   ↓                           ↓
注意力系统(自下而上/自上而下) → 注意力门控
   ↓                           ↓
预测编码(预测/误差) ←→ 三级记忆 ←→ 默认模式网络(空闲时)
   ↓                           ↓
GWT意识竞争 ←→ 思考系统(系统1/2) ←→ 元认知监控
   ↓
基底神经节(Go/NoGo) → 动作选择
   ↓                 ↓
小脑(时序校正)    强化学习(策略更新)
   ↓                 ↓
情绪核心 ←→ 情绪调节     多巴胺RPE
   ↓
LLM生成(情绪调制temperature/top_p)
   ↓
睡眠重放 → 记忆巩固 → 突触缩放
```

## 模型规格

| 项目 | v3.1 | v4.0 |
|------|------|------|
| 模块数 | 8 | 18 |
| 总参数量 | 1,002,656 | 1,180,320 |
| 预测编码层 | [512,650,256] | [512,650,256] |
| 基底神经节 | — | 8,320 |
| 小脑 | — | 167,936 |
| 强化学习 | — | 1,408 |
| 可进化超参数 | 15 | 15 |
| 动作类型 | — | 10 |
| 情绪调节策略 | — | 10 |
| DMN思维主题 | — | 8 |
| 文件大小 | 7.3 MB | 8.6 MB |

## 超参数说明

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| pc_learning_rate | 0.0255 | 0.001-0.05 | 预测编码学习率 |
| pc_precision_input | 0.868 | 0.5-2.0 | 输入层精度 |
| pc_precision_mid | 0.900 | 0.3-1.5 | 中间层精度 |
| pc_precision_top | 0.550 | 0.1-1.0 | 顶层精度 |
| emotion_decay | 0.078 | 0.01-0.15 | 情绪衰减率 |
| dopamine_baseline | 0.293 | 0.1-0.6 | 多巴胺基线 |
| dopamine_learning | 0.327 | 0.05-0.5 | 多巴胺学习率 |
| curiosity_weight | 0.847 | 0.2-1.5 | 好奇心权重 |
| explore_base | 0.225 | 0.05-0.4 | 基础探索率 |
| conscious_noise | 0.155 | 0.02-0.25 | 意识噪声 |
| hebbian_rate | 0.043 | 0.005-0.08 | Hebbian学习率 |

## 扩展开发

### 接入真实LLM

继承`LanguageModel`基类即可，其他17个模块无需修改：

```python
from brain import LanguageModel
import numpy as np

class RealLanguageModel(LanguageModel):
    def __init__(self, model_path):
        super().__init__("real-llm")
        # 加载你的模型...

    def generate(self, prompt, temperature=0.7, top_p=0.9, max_tokens=256):
        # 调用真实LLM...
        return response

    def embed(self, text):
        # 返回512维向量以匹配预测编码输入
        return embedding  # shape (512,)
```

### 添加新模块

1. 在`brain/core/`下创建新文件，文件名对应功能
2. 在`brain/__init__.py`中导出
3. 通过`MessageBus`与其他模块通信
4. 如需保存状态，在`model_io.py`中添加导出/恢复逻辑
