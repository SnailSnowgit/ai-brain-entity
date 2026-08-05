# AI 大脑实体（AIBrainEntity）

![1.00](figures/banner.png)

一个**纯原生 Python** 实现的类脑智能体架构：不依赖任何第三方库即可运行，
模拟生物大脑的感知、学习、记忆、情绪与决策全过程。v3.0 起支持
STDP 脉冲时序可塑性、多巴胺样奖励强化学习、真实多模态（CLIP/Whisper）
接入与多实体群体智能。v4.x 持续演化出完整认知闭环：
v4.0 可学习投影 / RPE 误差 / 文化动力学 / 动作空间与语言生成；
v4.1 社交拓扑相变；v4.2 共同演化网络；
v4.3 多模因竞争（垄断 vs 极化）；v4.4 TD(λ) 资格迹；
v4.5 技能学习；v4.6 检索式语言生成；
v4.7 情景记忆时间索引；v4.8 睡眠-清醒节律（SHY）；
v4.9 好奇驱动探索——感知、学习、记忆、行动、社会、节律的全栈类脑仿真。
v5.0 思考体系：思考空间（全局工作区）、思考记忆（think 固化）、
思考感官（introspect 内感觉与元认知日志）。
v5.1 动作与决策扩展：意图动词 3→8（ask/retrieve/plan/execute/wait）、
深思熟虑决策（带 rationale 理由链）。
v5.2 意识与社会性：意识流（自由联想/白日梦/灵感闪现）、
深度内省（introspect depth=deep）、多脑社交（发消息/文化学习/多轮对话）、
进化选择（BrainSwarm 适应度评估→轮盘赌选择→变异繁衍）、念头流水账。
v5.3 自我与文化：自我概念、自传体记忆、自我反思、情感传染、
文化演化（模因传播+变异）、有性繁殖与物种检测。
v5.4 全局工作空间理论（GWT）：注意竞争 → 点火 → 全局广播。
v5.5 意识的神经相关物（NCC）：整合信息 Φ、神经复杂度、
神经同步性、意识层级与相变检测。
v5.6 心智理论（ToM）：信念归因、错误信念任务、视角采择、
意图理解、共情。
v5.7 高阶意识理论（HOT）：高阶思想、元意识检测、内省层级。
v5.8 集体意识：集体工作空间、群体同步/情绪/极化、集体意识检测。
v5.9 语言模型接入：Qwen2-0.5B 作为语言生成后端——大脑负责
"想什么"（决策/记忆/情绪），外部 LLM 负责"说出来"，
未下载模型或生成失败自动降级回模板。
v6.0 记忆向量库：LanceDB 持久化长期记忆（容量无限、语义检索），
recall\_semantic 向量近邻回忆，未装 lancedb 自动降级内存模式。
v6.1 基因库与记忆史学：DNA 基因库（人格搜索/进化谱系追踪）、
记忆版本控制（修改历史/回忆过去版本/演化轨迹）、
跨模态统一向量空间联想、STM 全量同步选项。
v6.2 文本语义编码器：sentence-transformers 本地模型（默认
models/bge-small-zh-v1.5）为文本记忆生成真语义 embedding，
recall\_semantic 升级为真语义检索（"火焰"\~"燃烧"），
模型缺失自动回退哈希向量兜底，行为不变。

**认知子系统扩展包**（代码内编号 v6.0\~v7.1，与上面的记忆向量库/基因库
同号不同义，属另一套子系统序列）——十二个可独立开关的认知子系统：
工作记忆（Baddeley 模型：语音回路/视空画板/情景缓冲/中央执行）、
预测编码（自由能最小化/精度加权）、神经振荡（五频段脑电波/γ绑定/跨频耦合）、
主动推理（预期自由能驱动行动选择）、脑区分化（海马/前额叶/杏仁核分工与交互）、
推理与规划（演绎/归纳/溯因/因果/反事实/问题求解）、心理模拟（心理表象/
未来想象/过去回忆/洞察/发散思维/默认模式网络）、发育过程（皮亚杰阶段/
关键期/经验依赖可塑性）、具身认知（身体图式/运动系统/镜像神经元/环境交互）、
文化进化（文化传递/变异/选择/模因系统/规范仪式）、终身学习（持续学习/
间隔重复/刻意练习/元学习/环境适应）、意识整合（GWT+ IIT + HOT 统一框架/
意识状态转换/整合度量）。

## 它是什么

`AIBrainEntity` 是一个"独立大脑实体"。它不是一个聊天机器人外壳，而是一个
具有内部状态的仿真体——你给它刺激，它的神经元会放电、突触会学习、记忆会
固化或遗忘、情绪会波动，然后由决策中枢产生行为输出。

```
外部刺激 → 感官层(16神经元) → 联想层(32神经元) → 决策层(8神经元) → 行为输出
                ↓                    ↓                    ↓
           感官缓存(8)    →    短期记忆(20)    →    长期记忆(500)
                └──── 情绪内核 ←→ 注意力调制 ←→ 多巴胺奖励 ────┘
```

## 核心机制

| 模块            | 生物对应             | 实现                                                                                                                                                                                                                                                                                                     |     |                                                                                        |
| ------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --- | -------------------------------------------------------------------------------------- |
| 脉冲神经元         | LIF 神经元          | 膜电位累积-泄漏-阈值放电-复位-不应期                                                                                                                                                                                                                                                                                   |     |                                                                                        |
| 突触可塑性         | **STDP**         | 前→后放电 LTP / 后→前放电 LTD，指数时间窗 τ=3 网络步，覆盖前馈与循环通路                                                                                                                                                                                                                                                          |     |                                                                                        |
| 奖励强化学习        | 多巴胺系统            | `reward()` 调制学习速率 ×(1+多巴胺)，奖励促愉悦、惩罚促压力                                                                                                                                                                                                                                                                 |     |                                                                                        |
| 三级记忆          | 感觉登记→短时→长时       | 感官缓存 → STM(容量竞争) → LTM(固化)                                                                                                                                                                                                                                                                             |     |                                                                                        |
| 遗忘            | 艾宾浩斯曲线           | 记忆权重指数衰减，低于阈值删除                                                                                                                                                                                                                                                                                        |     |                                                                                        |
| 回忆再巩固         | Reconsolidation  | 每次成功回忆都会强化该记忆                                                                                                                                                                                                                                                                                          |     |                                                                                        |
| 情绪内核          | 边缘系统             | 平静/好奇/压力/愉悦四维动态变量                                                                                                                                                                                                                                                                                      |     |                                                                                        |
| 注意力           | 注意调制             | 好奇心↑注意、压力↓注意，反作用于感官输入                                                                                                                                                                                                                                                                                  |     |                                                                                        |
| DNA 遗传        | 记忆传承             | 全状态序列化（突触+记忆+情绪），支持克隆                                                                                                                                                                                                                                                                                  |     |                                                                                        |
| **群体智能**      | 文化传递             | `BrainSwarm`：记忆跨实体传递、广播、带变异繁衍                                                                                                                                                                                                                                                                          |     |                                                                                        |
| **脉冲思考链**     | 可解释性             | `thought_chain()`：把感知→传导→回响→决策展开为可读的脉冲因果链                                                                                                                                                                                                                                                              |     |                                                                                        |
| 多模态接口         | 感官通道             | **可插拔自定义模型**：注册自定义编码器 / 更换 CLIP、Whisper 模型名（未装依赖自动降级伪 embedding）                                                                                                                                                                                                                                       |     |                                                                                        |
| **可学习投影**     | 感觉皮层映射           | `LearnableProjection`：随机投影 + Oja 在线 PCA，保符号中心化归一化，保留稠密 embedding 对比度                                                                                                                                                                                                                                   |     |                                                                                        |
| **RPE/TD 学习** | 多巴胺预测误差          | `reward_td()`：δ = r − V 驱动多巴胺，奖励被预测后反应自然衰减                                                                                                                                                                                                                                                             |     |                                                                                        |
| **动作空间**      | 运动输出             | `decide_action()`：决策层脉冲 → 结构化动作指令（verb/强度/情绪）                                                                                                                                                                                                                                                          |     |                                                                                        |
| **语言生成**      | 布洛卡区             | `express()`：按 (动作 × 情绪) 模板生成自然语言，引用联想记忆                                                                                                                                                                                                                                                                |     |                                                                                        |
| **社交拓扑**      | 社会网络             | `set_topology()`：全连接/环/星/随机/小世界，`consensus_convergence()` 测相变收敛速度                                                                                                                                                                                                                                      |     |                                                                                        |
| **拓扑自适应**     | 共同演化网络           | `rewire_coevolve()`：异见边"模仿 vs 断边重连"博弈 + 求知连边，共识压力驱动边生灭                                                                                                                                                                                                                                                 |     |                                                                                        |
| **多模因竞争**     | 文化生态             | `competition_dynamics()`：立场转化 vs 阵营隔离，φ 决定垄断共识或极化共存                                                                                                                                                                                                                                                    |     |                                                                                        |
| **资格迹**       | 多巴胺时序迁移          | `reward_lambda()`：TD(λ) 按迹强度把 RPE 反向分配给近期状态，信用分配跨 tick 传播                                                                                                                                                                                                                                              |     |                                                                                        |
| **技能学习**      | 纹状体动作选择          | `learn_skill()` + `select_verb()`：分 verb 独立价值 Q，greedy/ε-greedy/softmax 策略化选择                                                                                                                                                                                                                          |     |                                                                                        |
| **检索式语言**     | 布洛卡区+海马          | `compose()`：LTM 片段检索（n-gram 降级）→ 记忆编织（单句/并列/联想链）→ 句法框架造句                                                                                                                                                                                                                                               |     |                                                                                        |
| **情景记忆**      | 海马时间细胞           | `episodes` + `events_after()`/`events_before()`：何时发生、与何事共现，"上次……之后"式时间推理                                                                                                                                                                                                                               |     |                                                                                        |
| **睡眠节律**      | 记忆重放+SHY         | `sleep()`：离线重放固化弱记忆，突触等比缩放剪除弱连接（保留相对差异），压力恢复                                                                                                                                                                                                                                                           |     |                                                                                        |
| **好奇驱动**      | 新皮层-边缘系统         | `_assess_novelty()`：未命中率+                                                                                                                                                                                                                                                                              | RPE | 双通路评估新奇度，当 tick 注意捕获；`effective_epsilon()` 新奇→多探索/熟悉→多利用；含习惯化（反复暴露新奇度衰减）与寻求刺激人格差异（SSS） |
| **思考体系**      | 全局工作区+内感觉        | `thought_space`：念头激活度衰减/容量 7±2；`think()` 念头回注网络、高激活固化进 STM；`introspect()` 感知自身脑活动并记元认知日志                                                                                                                                                                                                               |     |                                                                                        |
| **意图动词**      | 基底节动作选择扩展        | v5.1：`INTENT_VERBS` 8 verb（ask/retrieve/plan/execute/wait）独立 Q 值，`decide_action(deliberate=True)` 带 rationale 理由链                                                                                                                                  |     |                                                                                        |
| **意识流**       | 默认模式网络           | v5.2：`stream_of_consciousness()` 无输入时思绪自发流动：自由联想链、白日梦走神、双记忆组合灵感闪现                                                                                                                                                                                                                                      |     |                                                                                        |
| **社交与进化**     | 社会脑+自然选择         | v5.2：`send_message`/`social_learn`/`chat_with` 多脑交流与文化学习；`BrainSwarm.evolve()` 适应度评估→轮盘赌选择→变异繁衍；`thought_journal` 念头流水账（cap 50）                                                                                                                                                                        |     |                                                                                        |
| **自我模型**      | 自传体记忆+自我概念       | v5.3：`add_autobiographical_memory` 个人经历时间线；`update_self_concept` "我是谁"核心信念；`self_reflect` 对思维/情绪/行为的反思；`get_self_summary` 自我摘要                                                                                                                                                                         |     |                                                                                        |
| **全局工作空间**    | GWT 意识理论         | v5.4：`attentional_competition` 多内容竞争入意识 → `ignition` 全脑点火 → `global_broadcast` 广播给无意识模块；`conscious_step` 一步完整意识                                                                                                                                                                                        |     |                                                                                        |
| **NCC 度量**    | 意识神经相关物          | v5.5：`integrated_information`（Φ 简化版）、`neural_complexity`、`neural_synchrony`；`detect_ncc`/`get_ncc_report` 综合评分；`consciousness_level`/`consciousness_phase_transition` 意识层级与相变                                                                                                                          |     |                                                                                        |
| **心智理论**      | ToM 心理模型         | v5.6：`attribute_beliefs` 信念归因、`false_belief_task` 错误信念任务、`perspective_taking` 视角采择、`infer_intention` 意图理解、`empathize` 共情、`theory_of_mind` 完整推理                                                                                                                                                         |     |                                                                                        |
| **高阶意识**      | HOT 元意识          | v5.7：`higher_order_thought(level)` 对意识的意识、`meta_awareness_check` 元意识检测、`introspection_hierarchy` 多层级内省、`get_hot_report`                                                                                                                                                                                |     |                                                                                        |
| **集体意识**      | 群体意识涌现           | v5.8：`collective_workspace` 群体共享意识空间；`group_synchrony`/`group_emotion`/`group_polarization`/`group_ncc`/`group_self_awareness` 群体度量；`collective_consciousness_check` 涌现判定                                                                                                                              |     |                                                                                        |
| **文化演化**      | 模因演化+物种形成        | v5.3：`cultural_evolution_step`/`cultural_evolution` 模因传播+变异轨迹；`sexual_reproduce` 双亲 DNA 重组；`genetic_distance`/`detect_species` 遗传距离与物种检测                                                                                                                                                               |     |                                                                                        |
| **语言模型接入**    | 布洛卡区外接           | v5.9：`set_qwen_model()` 接入 Qwen2-0.5B-Instruct，大脑状态快照（刺激/动作/情绪/记忆/意识焦点）→ LLM 造句；`register_language_generator` 可换任意自定义模型；未下载自动降级模板                                                                                                                                                                      |     |                                                                                        |
| **记忆向量库**     | 海马体外置            | v6.0：`attach_memory_store()` 接入 LanceDB，LTM 固化/强化/衰减自动同步到本地向量库（content+512维features+权重+tag）；`recall_semantic()` 向量近邻语义回忆；未装 lancedb 降级内存+关键词模式。v6.1：`sync_stm=True` 全记忆入库；`modality`/`exclude_modality` 跨模态统一向量空间联想；`memory_history`/`recall_version` 记忆版本控制（演化轨迹/回忆过去版本）                              |     |                                                                                        |
| **DNA 基因库**   | 种群基因组            | v6.1：`attach_dna_library()` + `save_to_library()` 多脑 DNA 入库；`DNALibrary.search()` 按人格参数（寻求刺激度/习惯化率/世代）检索；`lineage()` 进化谱系追踪；BrainSwarm 进化子代自动存档并链接亲代                                                                                                                                                   |     |                                                                                        |
| **工作记忆**      | Baddeley 模型      | `phonological_store/rehearse`（语音回路 7±2）、`visuospatial_store/manipulate`（视空画板）、`episodic_store/integrate`（情景缓冲）、`central_executive_attention`/`task_switch`（中央执行）；`wm_operation` 统一入口，`get_working_memory_report`                                                                                       |     |                                                                                        |
| **预测编码与主动推理** | 自由能原理            | `generate_prediction`/`calculate_prediction_error`/`minimize_free_energy`（精度加权）；`generate_action_strategies`（预期自由能）→ `select_action` → `active_inference_step` 感知-行动闭环                                                                                                                                 |     |                                                                                        |
| **神经振荡与脑区分化** | 脑电波+脑区特化         | `update_brainwaves`（δ/θ/α/β/γ 五频段）/`gamma_binding`（特征绑定）/跨频耦合；`hippocampus_encode/replay/pattern_completion`（海马）、`prefrontal_make_plan/inhibit`（前额叶）、`amygdala_detect_threat/fear_conditioning/extinguish_fear`（杏仁核）                                                                                   |     |                                                                                        |
| **推理规划与心理模拟** | 前额叶+默认模式网络       | `deductive/inductive/abductive_reasoning`、`causal_attribution`/`counterfactual_reasoning`、`plan_goal`/`solve_problem`/`make_decision`；`create_mental_image`/`mental_rotate`、`remember_past`/`imagine_future`、`generate_insight`/`divergent_thinking`、`simulate_dialogue`                               |     |                                                                                        |
| **发育与具身认知**   | 皮亚杰阶段+身体图式       | `develop(months)` 皮亚杰四阶段推进（客体永久性/守恒/抽象思维里程碑）、关键期与经验依赖可塑性；`init_body_schema` 身体图式、`plan_motor_action`/`execute_motor_action`/`learn_motor_skill` 运动系统、`observe_action`/`imitate_action` 镜像神经元、`perceive_affordance`/`use_tool`/`navigate_spatially` 环境交互                                                |     |                                                                                        |
| **文化进化与终身学习** | 文化选择+学会学习        | `transmit_culture`/`innovate_culture`/`recombine_culture`/`select_cultural_trait`（传递-变异-选择）、`add_meme`/`replicate_meme` 模因系统、规范/仪式/从众/声望学习；`learn_incremental`/`learn_online`、`spaced_repetition`/`active_recall`/`interleaved_practice`、`deliberate_practice`、`meta_learn`/`select_learning_strategy` |     |                                                                                        |
| **意识整合**      | GWT+IIT+HOT 统一框架 | `update_consciousness_framework` 三理论统一、`conscious_ignition`/`conscious_binding`/`conscious_higher_order_thought`、`set_consciousness_state`/`transition_consciousness_state`、`measure_consciousness_level/diversity/stability`、`get_consciousness_integration_report`                                   |     |                                                                                        |

## 依赖说明

核心模块零第三方依赖，仅需 Python 3.8+ 标准库；实验绘图需要
matplotlib/seaborn/pandas；多模态真实编码需要 transformers/torch
（不装自动降级为伪 embedding，不影响核心功能）；记忆向量库需要
lancedb（不装自动降级为内存+关键词模式，不影响核心功能）。

## 快速开始

```bash
pip install -r requirements.txt   # 仅实验复现脚本需要（核心模块零依赖）

python ai_brain_entity.py    # 运行内置演示（含 STDP/奖励/多模态/群体/v4.0/v4.1 演示）
python swarm.py              # 群体文化传递演示（定向传递+变异+世代链+相变扫描+多模因竞争）
python experiments.py        # 复现实验 1-8（统一入口，生成 figures/ 与 datasets/ 结果）
python encoder_status.py        # 观测台编码器面板数据（另两个导出器：brain_activity_trace / thought_chain_scenarios）
python run_all.py               # 一键全流程：测试 → 演示 → 实验 → 观测数据（--quick 快速模式）
python -m unittest discover tests  # 运行核心行为测试（187 项）
```

```python
from ai_brain_entity import AIBrainEntity, BrainSwarm

brain = AIBrainEntity("Brain-01", seed=42)
print(brain.sensory_input("记忆是智慧的基石"))

# 多巴胺奖励：学习前给奖励，STDP 学习速率最高翻倍
brain.reward(0.8)
brain.sensory_input("奖励关联的刺激")

# 真实多模态：CLIP 图像 / Whisper 音频 embedding（未装依赖自动降级）
brain.perceive_image("cat.jpg", label="一张猫的图片")
brain.perceive_audio("sound.wav", label="一段语音")
brain.sensory_input_vector([0.2, -0.5, 0.9, ...], label="任意 embedding")

# 自定义多模态模型（v3.1）：三种接入方式
# 1) 注册任意自定义编码器：callable(path) -> 数值序列，长度任意
def my_image_model(path):
    return my_framework.encode(path)          # 你自己的模型
register_image_encoder(my_image_model, name="mine")   # 设为全局默认
brain.perceive_image("cat.jpg")                        # 自动走自定义模型
brain.perceive_image("cat.jpg", encoder=my_image_model)  # 或一次性指定
unregister_image_encoder("mine")                       # 注销后回落内置链
# 2) 更换内置模型名（自定义微调版 CLIP / Whisper，HF 名称或本地路径）
set_clip_model("path/to/my-finetuned-clip")
set_whisper_model("large-v3")
# 3) 查看当前编码器状态
print(list_encoders())

# ===== v4.0 四大扩展 =====

# 可学习投影：替代线性插值，保留稠密 embedding 对比度
brain.enable_projection(True)
brain.sensory_input_vector(dense_vec, label="512 维 embedding")  # 投影 + Oja 在线训练

# RPE/TD 误差：多巴胺响应"意外"而非奖励本身
brain.reward_td(0.8)        # 首次：RPE=+0.800（意外）
# ... 重复 30 次后：RPE≈0（奖励被预测，多巴胺不再波动）
# 突然给 -0.5：RPE=-1.3（意外重现）

# 动作空间 + 语言生成
act = brain.decide_action("火焰是危险的")   # {"action","verb","intensity","mood","recalled"}
ex = brain.express("火焰是危险的")          # {"action":..., "utterance": "（「火焰是危险的」——静默观察中。）"}

# 文化动力学：水平（同代）vs 垂直（跨代）+ 共识涌现
swarm2 = BrainSwarm(["E1", "E2", "E3"], seed=1)
swarm2.reproduce(0, "F1")                                # 子代 generation=2
swarm2.horizontal_transfer(rounds=6)                     # 同伴扩散
swarm2.vertical_transfer(rounds=6)                       # 师承传递
dyn = swarm2.transmission_dynamics("钻木可以取火", rounds=4, direction="horizontal")
con = swarm2.consensus(stimulus="钻木可以取火")          # 记忆共识 + 行动共识指数

# ===== v4.1 扩展 =====

# 社交拓扑 + 共识相变：文化只沿社交边传播
swarm2.set_topology("small_world", p=0.3)   # 全连接/环/星/随机/小世界
conv = swarm2.consensus_convergence("钻木可以取火", threshold=0.9)
# 实测（N=8）：小世界 18 轮 < 全连接 26 轮 < 环形 45 轮；
# N=12 时稀疏拓扑无法收敛（相变），详见 swarm.py consensus_phase_scan

# ===== v4.2 拓扑自适应（共同演化网络） =====
swarm2.set_topology("ring")
res = swarm2.coevolve_consensus("钻木可以取火", threshold=0.9, rewire_prob=0.5)
# 异见边逐轮博弈：1-φ 观点模仿 / φ 断边重连到同道；未持有者求知连边
# 实测（N=12 环形）：静态 >60 轮不收敛 → 共同演化 4 轮收敛
# φ 三区：0.2→3 轮（传播主导）/ 0.5→4 轮（平衡）/ 0.8→7 轮（结构 churn）

# ===== v4.3 多模因竞争（文化生态） =====
import time as _time
from ai_brain_entity import BrainMemory
swarm3 = BrainSwarm([f"A{i}" for i in range(12)], seed=1)
swarm3.set_topology("ring")
for i, b in enumerate(swarm3.population):   # 前半持钻木、后半持燧石
    meme = "钻木可以取火" if i < 6 else "燧石可以取火"
    b.long_memory.append(BrainMemory(content=meme, timestamp=_time.time(),
                                     weight=1.0, tag="culture"))
mono = swarm3.competition_dynamics(["钻木可以取火", "燧石可以取火"], rewire_prob=0.2)
# φ 低 → 立场转化主导 → 垄断：实测「燧石可以取火」17 轮统一全网
polar = swarm3.competition_dynamics(["钻木可以取火", "燧石可以取火"], rewire_prob=0.85)
# φ 高 → 阵营隔离主导 → 极化共存：实测 0.67/0.33 两阵营各自封闭

# ===== v4.4 TD(λ) 资格迹：信用分配跨 tick 传播 =====
lam_brain = AIBrainEntity("λ", seed=1)
for _ in range(20):                     # Schultz 范式：三线索链 → 奖励
    for cue in ["铃声", "灯光", "气味"]:
        lam_brain.sensory_input(cue)
    r = lam_brain.reward_lambda(1.0)    # RPE 按资格迹反向分配给全部线索
# 实测：V(气味)=0.99 > V(灯光)=0.71 > V(铃声)=0.51（γλ 时间梯度），
# 奖励时刻 RPE 从 1.0 衰减到 0.014（多巴胺时序迁移到最早线索）

# ===== v4.5 技能学习：分 verb 价值 + 策略化选择 =====
skill_brain = AIBrainEntity("S", seed=1)
for _ in range(40):                     # 按奖励信号学习各 verb 价值
    for verb, rv in [("respond", 0.8), ("acknowledge", 0.2), ("observe", -0.4)]:
        skill_brain.learn_skill(verb, rv)
# 实测：Q 收敛 {respond 0.80, acknowledge 0.19, observe -0.30}；
verb = skill_brain.select_verb("greedy")   # 习得价值覆盖动作选择
# greedy 20/20 选 respond——Q 最高的动作胜出

# ===== v4.6 检索式语言生成：LTM 片段 + 句法框架 =====
comp_brain = AIBrainEntity("C", seed=1)
for _ in range(30):
    comp_brain.sensory_input("火焰是危险的")
for _ in range(20):
    comp_brain.sensory_input("钻木可以取火")
comp_brain.sensory_input("取火的方法")
c = comp_brain.compose("取火的方法")
# c = {"utterance", "action", "fragments", "frame", "mood"}
# 实测：长词经 n-gram 降级检索取出「钻木可以取火」「燧石可以取火」，
# 编织为并列从句填入句法框架——比 express() 的固定模板更会"引经据典"

# ===== v4.7 情景记忆时间索引：时间推理 =====
ep_brain = AIBrainEntity("E", seed=1)
for s in ["起床", "刷牙", "吃早餐", "出门", "刷牙", "上班"]:
    ep_brain.sensory_input(s)
r = ep_brain.events_after("刷牙")    # "上次刷牙之后发生了什么"
# 实测：锚点=最近一次刷牙@tick5，事件=[上班(+1 tick)]
r2 = ep_brain.events_before("吃早餐")  # [(起床,-2), (刷牙,-1)]
# 每条情景还带共现上下文：ep_brain.episodes[2]["context"] == ["起床", "刷牙"]

# ===== v4.8 睡眠-清醒节律：离线重放 + 突触稳态缩放 =====
sleep_brain = AIBrainEntity("Z", seed=1)
sleep_brain.sensory_input("萤火虫在夜里发光")   # 弱刺激：白天无法固化
sleep_brain.sensory_input("另一个无关刺激")
r = sleep_brain.sleep(cycles=3)
# 实测：重放 2 条 / 固化 2 条（LTM 0→2），压力 0.16→0.02；
# 深睡 12 周期：突触 768→690，剪除 103 条弱连接、强连接存活（SHY）

# ===== v4.9 好奇驱动探索：新奇度调制注意与 ε =====
nov_brain = AIBrainEntity("N", seed=1)
for _ in range(30):
    nov_brain.sensory_input("火焰是危险的")
nov_brain.sensory_input("火焰")            # 熟悉：新奇度 0，ε 减半（利用）
nov_brain.sensory_input("量子纠缠态坍缩")  # 全新：新奇度 0.70，注意当 tick 上升
# 实测：ε_eff 0.075(熟悉) vs 0.180(全新)；大 |RPE| 后熟悉刺激重获 0.30 新奇度
print(nov_brain.novelty, nov_brain.effective_epsilon())

# v4.9.1 人格差异：高/低寻求刺激者对新奇的反应不同
explorer = AIBrainEntity("Explorer", seed=1, sensation_seeking=0.9)  # 探险家
homebody = AIBrainEntity("Homebody", seed=1, sensation_seeking=0.1)  # 宅家者
# 同一新刺激：探险家好奇心↑↑、探索率↑；宅家者反应温和
explorer.sensory_input("从未见过的紫色星星")
homebody.sensory_input("从未见过的紫色星星")
print(explorer.effective_epsilon(), homebody.effective_epsilon())  # 0.22 vs 0.14

# v4.9.1 习惯化：反复暴露同一刺激，新奇度自然衰减
hab_brain = AIBrainEntity("Hab", seed=1, habituation_rate=0.5)
for i in range(5):
    hab_brain.sensory_input("反复出现的广告")
    # novelty 逐次降低：0.70 → 0.47 → 0.35 → 0.28 → 0.23

# ===== v5.0 思考体系：思考空间 / 思考记忆 / 思考感官 =====
# 思考空间（全局工作区）：感知与回忆自动进入意识，容量 7±2，激活度逐 tick 衰减
brain.sensory_input("火焰是危险的")
brain.thought_space[0]   # ThoughtItem(content="火焰是危险的", source="external", ...)

# 思考记忆：think() 把念头回注网络（"自言自语"），高激活念头固化进 STM
out = brain.think("火焰")          # 想多了就记住了（tag="thought"）
out = brain.think()                # 缺省思考意识焦点（激活度最高的念头）

# 思考感官（内感觉）：introspect() 感知自身脑活动并记入元认知日志
entry = brain.introspect()
# entry = {"mood", "top_thought", "spike_counts", "text", ...}
# entry["text"] ≈ "我感到好奇，正在想「火焰是危险的」，脉冲活动3/7/1，记忆5/12"
print(brain.status())   # 状态摘要新增"思考空间"行

# ===== v5.1 动作与决策扩展：8 verb / 深思熟虑 =====
# 意图动词（与脉冲强度动作正交）：ask 提问 / retrieve 检索 /
# plan 规划 / execute 执行 / wait 观望，每个 verb 独立 Q 值
out = brain.decide_action("从未见过的紫色星星", deliberate=True)
# 记忆未命中且新奇度高 → verb="ask"，rationale 记录完整理由链
for line in out["rationale"]:
    print(line)

# 策略化选择：Q 值学习后 greedy 恒选最优 verb
for _ in range(10):
    brain.learn_skill("ask", 0.8)
out = brain.express("神秘信号", deliberate=True, policy="greedy")
# utterance 走 ask 专属模板（"……能再多告诉我一些吗？"）

# ===== v5.2 意识与社会性：意识流 / 深度内省 / 社交 / 进化 =====
# 意识流：无外部输入时思绪自发流动（自由联想 + 白日梦 + 灵感闪现）
out = brain.stream_of_consciousness(steps=5, daydream=0.3)
# out = {"chain": ["火焰是危险的", "[联想] → ...", "[灵感!] A + B", ...],
#        "insights": [...], "final_thought": ..., "daydream_level": 0.3}

# 深度内省：知道"自己在想什么"（完整思考空间 + 记忆模态分布 + 自我指称）
entry = brain.introspect(depth="deep")
# entry["text"] ≈ "我是Brain-01，感到好奇。正在想「火焰」。思考空间有……新奇度0.7……"

# 多脑社交：发消息 / 文化学习 / 多轮对话
alice, bob = AIBrainEntity("Alice", seed=1), AIBrainEntity("Bob", seed=2)
alice.send_message(bob, "今天天气不错")     # Bob 形成 tag="social" 社交记忆
alice.social_learn(bob, n_memories=3)       # 复制 Bob 高权重记忆（权重×0.7，tag="culture"）
conv = alice.chat_with(bob, turns=3)        # 轮流对话，内容进入双方思考空间

# 进化：适者生存——评估适应度 → 轮盘赌选择 → 变异繁衍
result = swarm.evolve(generations=10, task="memory")
# result["history"] 每代统计：存活/出生/最优与平均适应度变化

# 念头流水账：所有念头按时间留痕（容量 50，先进先出）
brain.thought_journal[-1]   # {"content", "source", "tick"}

# ===== v5.3 自我与文化：自我概念 / 自传体记忆 / 文化演化 =====
brain.update_self_concept("我是一个好奇的学习者")   # "我是谁"核心信念
brain.add_autobiographical_memory("第一次学会取火", emotion="joy", importance=0.9)
print(brain.get_self_summary())                     # 自我概念 + 重要经历摘要
brain.self_reflect(focus="emotion")                 # 对自身思维/情绪/行为的反思
evo = swarm.cultural_evolution(steps=10)            # 模因传播+变异的文化演化轨迹
child = swarm.sexual_reproduce(swarm.population[0], swarm.population[1], "Hybrid")
print(swarm.detect_species())                       # 按遗传距离划分物种

# ===== v5.4 全局工作空间理论（GWT） =====
brain.sensory_input("火焰是危险的")
brain.conscious_step()
# 一步完整意识：attentional_competition 注意竞争 → ignition 全脑点火
# → global_broadcast 广播给全体无意识模块

# ===== v5.5 意识的神经相关物（NCC） =====
rep = brain.get_ncc_report()
# rep = {"ncc_score", "features": {整合信息Φ/神经复杂度/同步性/...}}
print(brain.consciousness_level())                  # 当前意识层级
print(brain.consciousness_phase_transition())       # 无意识→有意识相变检测

# ===== v5.6 心智理论（ToM）：理解其他大脑 =====
print(alice.attribute_beliefs(bob))                 # 推断 Bob 相信什么
print(alice.false_belief_task(bob, true_location="抽屉"))  # 经典错误信念任务
print(alice.perspective_taking(bob, "搬家"))        # 从 Bob 的视角看问题
print(alice.infer_intention(bob, "反复查看门"))     # 推断 Bob 的意图
print(alice.empathize(bob))                         # 共情：感受对方情绪
print(alice.theory_of_mind(bob))                    # 完整 ToM 推理报告

# ===== v5.7 高阶意识理论（HOT）：对意识的意识 =====
hot = brain.higher_order_thought(level=2)           # 二阶思想："我知道我在想……"
print(brain.meta_awareness_check())                 # 元意识检测
print(brain.introspection_hierarchy(max_depth=3))   # 多层级内省
print(brain.get_hot_report())                       # HOT 状态报告

# ===== v5.8 集体意识：群体层面的涌现 =====
ws = swarm.collective_workspace()                   # 群体共享意识空间
print(swarm.group_synchrony(), swarm.group_emotion())   # 同步性 / 平均情绪
print(swarm.group_polarization())                   # 极化程度
check = swarm.collective_consciousness_check()      # 是否涌现集体意识
print(swarm.get_collective_consciousness_report())  # 完整集体意识报告

# ===== v5.9 语言模型接入：Qwen2 作为语言输出后端 =====
# 1) 下载模型（国内推荐 ModelScope，权重约 1GB，CPU 可跑）
#    pip install modelscope
#    python -c "from modelscope import snapshot_download; \
#        snapshot_download('qwen/Qwen2-0.5B-Instruct', cache_dir='models')"
# 2) 一行接入：大脑负责"想什么"，Qwen 负责"说出来"
from ai_brain_entity import set_qwen_model
print(set_qwen_model())          # {"registered": "qwen", "available": ...}
out = brain.express("火焰")      # 模型可用时由 Qwen 造句（含 "generator": "qwen"）
out = brain.chat("你好")         # 对话回复同样走 Qwen
# 模型未下载/transformers 未装：自动降级回模板，核心功能不受影响
# 换任意自定义语言模型：callable(context) -> str
# register_language_generator(my_llm_fn, name="mine")

# ===== v6.0 记忆向量库：LanceDB 持久化 + 语义回忆 =====
# pip install lancedb（纯 wheel，Rust 内核，约 30MB；未装自动降级内存模式）
info = brain.attach_memory_store()     # 默认存到 datasets/lancedb/
# info = {"attached": True, "available": True, ...}
brain.sensory_input("火焰是危险的")    # LTM 固化时自动同步到向量库
rows = brain.recall_semantic("火灾")   # 向量近邻检索 → 能召回"火焰是危险的"
# rows = [{"content", "weight", "tag", "distance", "source": "lancedb"}]
brain.decay_memory(0.995)              # 大脑衰减节律同步到库（低于阈值删除）
# 记忆携带真实 embedding（CLIP/Qwen features）时语义检索质量最佳；
# 纯文本记忆走零依赖哈希向量兜底（字面近似）

# ===== v6.1 基因库与记忆史学 =====
# 1) 全记忆入库：短期记忆也同步（默认只同步固化进 LTM 的）
brain.attach_memory_store(sync_stm=True)

# 2) 跨模态统一向量空间联想：看到猫 → 排除视觉记忆 → 想起"喵的叫声"
brain.perceive_image("cat.jpg", label="一只猫")
rows = brain.recall_semantic("猫", exclude_modality="visual")

# 3) 记忆版本控制：修改历史 / 回忆过去的版本 / 演化轨迹
store = brain.memory_store
store.memory_history("火焰是危险的", "Brain-01")
# [{version: 1, weight: 0.65, reason: "add"},
#  {version: 2, weight: 0.80, reason: "reinforce"},
#  {version: 3, weight: 0.76, reason: "decay"}, ...]
store.recall_version("火焰是危险的", version=-2)   # 回忆上一个版本

# 4) DNA 基因库：多脑存储 / 人格搜索 / 进化谱系
brain.attach_dna_library()
brain.save_to_library()                       # 当前 DNA 入库
lib = brain.dna_library
lib.search(sensation_seeking=(0.8, 1.0))      # 找"探险家"人格的大脑
lib.search(min_generation=3)                  # 找第 3 代以后的个体

# 5) 进化谱系追踪：种群进化的子代自动存档并链接亲代
swarm.attach_dna_library()
swarm.save_population()
swarm.evolve(generations=3)
lib.lineage(child_dna_id)                     # 回溯：始祖 → … → 亲代

# 群体智能：文化传递
swarm = BrainSwarm(["Alpha", "Beta", "Gamma"], seed=1)
for _ in range(25):
    swarm.population[0].sensory_input("火焰是危险的")
swarm.culture_round(rounds=4, top_k=2, mode="dna")   # DNA 记忆传递
swarm.broadcast("公共事件")                           # 全种群广播

# 脉冲思考链：可解释地展开一次"感知→传导→回响→决策"的因果链
# （v5.0 起返回值附带 thoughts 字段：本次感知后的思考空间快照）
for line in brain.thought_chain("火焰是危险的")["chain"]:
    print(line)

# 保存"DNA"，克隆一个继承全部记忆与突触的新实体
brain.save_dna("brain_dna.json")
clone = AIBrainEntity.load_dna("brain_dna.json", new_name="Brain-02")
```

认知子系统扩展包（代码内编号 v6.0\~v7.1）常用入口：

```python
brain = AIBrainEntity("Cog", seed=1)

# 工作记忆（Baddeley）：语音回路存词 → 复述 → 中央执行切换任务
brain.phonological_store("苹果")
brain.wm_operation("rehearse")
brain.task_switch("数学题")
print(brain.get_working_memory_report()["total_load"])

# 预测编码：大脑先预测，实际输入产生预测误差 → 自由能最小化
brain.generate_prediction()
print(brain.minimize_free_energy("火焰靠近了")["free_energy_reduced"])

# 主动推理：按预期自由能生成并选择行动策略
brain.add_goal("找到水源", priority=0.8)
print(brain.active_inference_step()["selected_action"])

# 脑区分化：海马编码 → 线索补全；杏仁核恐惧条件反射 → 消退
brain.hippocampus_encode("蜘蛛出现在角落", context="地下室")
brain.amygdala_fear_conditioning("蜘蛛", fear_level=0.9)
print(brain.amygdala_detect_threat("蜘蛛")["threat_level"])   # ≈0.9
brain.amygdala_extinguish_fear("蜘蛛")

# 推理与规划：演绎 + 目标规划 + 决策
brain.deductive_reasoning(["所有人都会死", "苏格拉底是人"])
brain.plan_goal("写一篇论文")
brain.make_decision(["先查资料", "直接动笔", "明天再说"])

# 心理模拟与发育：想象未来；皮亚杰阶段随月龄推进
brain.imagine_future("明天去海边")
brain.develop(12); assert brain.has_object_permanence()

# 终身学习与意识整合
brain.learn_incremental("牛顿第二定律")
brain.spaced_repetition("牛顿第二定律")
print(brain.get_consciousness_integration_report()["state"]["state"])  # wakeful
```

## API 速查（全部公开功能）

**实体观测与工具**（`ai_brain_entity.py`，快速开始已列的方法不再重复）

| 方法                                                                                                                                                       | 功能                                                                                                                              |
| -------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `spike_counts()`                                                                                                                                         | 当前 tick 三层（感官/联想/决策）各自的脉冲数                                                                                                      |
| `free_run(ticks)`                                                                                                                                        | 无外部输入自由演化，观察刺激后回响衰减或静息态自发活动                                                                                                     |
| `episodic_trace(keyword)`                                                                                                                                | 情景轨迹：按时间顺序返回所有含 keyword 的情景条目                                                                                                   |
| `decay_memory(factor)`                                                                                                                                   | 记忆自然衰减（模拟时间流逝/睡眠），低于阈值遗忘                                                                                                        |
| `synapse_mean()` / `strong_synapse_count(threshold)`                                                                                                     | 突触平均强度 / 强突触计数（可塑性度量）                                                                                                           |
| `status()`                                                                                                                                               | 一行状态摘要（tick/情绪/记忆/突触/多巴胺/思考空间/元认知日志）                                                                                            |
| `think(content, ticks)`                                                                                                                                  | v5.0 主动思考：念头回注网络诱发联想，高激活念头固化进 STM（tag=thought）                                                                                  |
| `introspect()`                                                                                                                                           | v5.0 思考感官：感知自身情绪/脉冲/记忆/意识焦点，回注内省言语并记 `metacog_log`                                                                              |
| `top_thought()`                                                                                                                                          | v5.0 当前意识焦点（激活度最高的念头）                                                                                                           |
| `decide_action(stim, deliberate=True, policy=...)`                                                                                                       | v5.1 深思熟虑决策：rationale 理由链 + base\_verb 对照 + q\_values 快照；意图动词由策略/启发式裁定                                                          |
| `INTENT_VERBS`                                                                                                                                           | v5.1 意图动词表（8 verb，含 channel 与描述），`verb_values` 覆盖全部 8 个                                                                         |
| `stream_of_consciousness(steps, daydream)`                                                                                                               | v5.2 意识流：自由联想链 / 白日梦走神 / 灵感闪现，返回 chain+insights+final\_thought                                                                  |
| `introspect(depth="deep")`                                                                                                                               | v5.2 深度内省：自我指称 + 完整思考空间摘要 + 记忆模态分布 + 新奇度/注意力                                                                                    |
| `send_message(other, msg)` / `receive_message(...)`                                                                                                      | v5.2 多脑通信：接收方形成 tag="social" 社交记忆与社交念头                                                                                          |
| `social_learn(other, n_memories)`                                                                                                                        | v5.2 文化学习：复制对方最高权重记忆（权重×0.7，tag="culture"），已知的跳过，愉悦上升                                                                           |
| `chat_with(other, turns)`                                                                                                                                | v5.2 多轮对话：轮流发言，内容进入双方思考空间                                                                                                       |
| `evaluate_fitness(task)` / `select()` / `evolve_generation()` / `evolve(gens)`                                                                           | v5.2 进化：memory/curiosity/diversity/social 四种适应度，轮盘赌选择 + 变异繁衍                                                                    |
| `thought_journal`                                                                                                                                        | v5.2 念头流水账：所有念头按时间留痕（{content, source, tick}，cap 50）                                                                            |
| `chat(message)` / `chat_history()`                                                                                                                       | 对话接口：接收消息 → 感知 → 思考 → 生成回复；对话历史（从元认知日志提取）                                                                                       |
| `see(image_path)` / `hear(audio_path)`                                                                                                                   | 视觉/听觉感知：经 BLIP/Whisper（或降级链）转为文本进入大脑                                                                                            |
| `multimodal_input(text, image, audio)`                                                                                                                   | 多模态联合输入：文本+图像+音频同时进入感知流水线                                                                                                       |
| `cross_modal_recall(features, current_modality)`                                                                                                         | 跨模态联想：按特征向量从其他模态的记忆中检索相似项                                                                                                       |
| `cognition(stimulus)`                                                                                                                                    | 决策中枢：整合决策层脉冲+情绪+记忆联想产生行为输出（返回描述字符串）                                                                                             |
| **v5.3 自我与文化**                                                                                                                                           | <br />                                                                                                                          |
| `update_self_concept(belief)` / `get_self_summary()`                                                                                                     | 自我概念：维护"我是谁"核心信念集合并输出自我摘要                                                                                                       |
| `add_autobiographical_memory(event, emotion, importance)`                                                                                                | 自传体记忆：记录重要个人经历（带情绪色彩与重要度）                                                                                                       |
| `self_reflect(focus)`                                                                                                                                    | 自我反思：对自己的思维/情绪/行为进行反思（general/emotion/...）                                                                                      |
| `cultural_evolution_step()` / `cultural_evolution(steps)`                                                                                                | 文化演化：模因传播+变异单步/多步轨迹（BrainSwarm）                                                                                                 |
| `sexual_reproduce(p1, p2, name)` / `genetic_distance(b1, b2)` / `detect_species()`                                                                       | 有性繁殖（双亲 DNA 重组）/ 遗传距离 / 物种检测（BrainSwarm）                                                                                        |
| `cultural_diversity()`                                                                                                                                   | 群体文化多样性度量（BrainSwarm）                                                                                                           |
| **v5.4 全局工作空间 GWT**                                                                                                                                      | <br />                                                                                                                          |
| `attentional_competition()`                                                                                                                              | 注意竞争：多个内容竞争进入意识                                                                                                                 |
| `ignition()` / `global_broadcast()`                                                                                                                      | 点火效应（全脑激活）/ 把意识内容广播给所有无意识模块                                                                                                     |
| `conscious_step()` / `get_consciousness_report()`                                                                                                        | 一步完整意识（竞争→点火→广播）/ 意识状态报告                                                                                                        |
| **v5.5 NCC 意识度量**                                                                                                                                        | <br />                                                                                                                          |
| `integrated_information()` / `neural_complexity()` / `neural_synchrony()`                                                                                | 整合信息 Φ（简化版）/ 神经复杂度 / 脑区同步振荡                                                                                                     |
| `detect_ncc()` / `get_ncc_report()`                                                                                                                      | NCC 综合得分与各特征分项 / 完整 NCC 报告                                                                                                      |
| `consciousness_level()` / `consciousness_phase_transition()`                                                                                             | 意识层级量化 / 无意识→有意识相变检测                                                                                                            |
| **v5.6 心智理论 ToM**                                                                                                                                        | <br />                                                                                                                          |
| `attribute_beliefs(other)` / `false_belief_task(other, ...)`                                                                                             | 信念归因 / 经典错误信念任务                                                                                                                 |
| `perspective_taking(other, topic)` / `infer_intention(other, obs)`                                                                                       | 视角采择 / 意图理解                                                                                                                     |
| `empathize(other)` / `theory_of_mind(other)`                                                                                                             | 共情（感受对方情绪）/ 完整 ToM 推理                                                                                                           |
| **v5.7 高阶意识 HOT**                                                                                                                                        | <br />                                                                                                                          |
| `higher_order_thought(level)` / `meta_awareness_check()`                                                                                                 | 生成 N 阶思想（对意识的意识）/ 元意识检测                                                                                                         |
| `introspection_hierarchy(max_depth)` / `get_hot_report()`                                                                                                | 多层级内省 / HOT 状态报告                                                                                                                |
| **v5.8 集体意识（BrainSwarm）**                                                                                                                                | <br />                                                                                                                          |
| `collective_workspace()`                                                                                                                                 | 集体工作空间：群体共享的意识空间                                                                                                                |
| `group_synchrony()` / `group_emotion()` / `group_polarization()`                                                                                         | 群体同步性 / 平均情绪 / 极化程度                                                                                                             |
| `group_ncc()` / `group_self_awareness()`                                                                                                                 | 群体层面 NCC / 群体自我意识                                                                                                               |
| `collective_consciousness_check()` / `get_collective_consciousness_report()`                                                                             | 集体意识涌现判定 / 完整集体意识报告                                                                                                             |
| `same_state_edge_ratio(meme)`                                                                                                                            | 同道边比例：两端观点相同的社交边占比（共同演化度量）                                                                                                      |
| **v5.9 语言模型接入**                                                                                                                                          | <br />                                                                                                                          |
| `set_qwen_model(model_path, device)`                                                                                                                     | 接入 Qwen2-0.5B-Instruct 语言生成后端；模型缺失也注册（自动降级模板），返回可用状态                                                                            |
| `register_language_generator(fn, name)` / `unregister_language_generator(name)`                                                                          | 注册/注销自定义语言生成器，契约 `callable(context) -> str`                                                                                     |
| `get_language_generator_info()`                                                                                                                          | 当前语言生成器注册状态                                                                                                                     |
| `express(..., use_generator=False)`                                                                                                                      | v5.9 参数：强制走模板（默认已注册生成器时优先 LLM 造句）                                                                                               |
| **v6.0 记忆向量库**                                                                                                                                           | <br />                                                                                                                          |
| `attach_memory_store(store, path)`                                                                                                                       | 接入 LanceDB 记忆后端（默认 `datasets/lancedb/`）；未装 lancedb 时行为不变，返回可用状态                                                                     |
| `recall_semantic(query, top_k)`                                                                                                                          | 语义回忆：向量近邻检索 LTM（字符串走哈希向量兜底）；无后端时降级关键词 recall                                                                                    |
| `memory_store.py`                                                                                                                                        | `LanceMemoryStore`：add/update\_weight/search\_vector/search\_text/decay/count/info（v6.2 decay 批量优化：扫描→批量删→批量回写，版本日志内存计数缓存）                                              |
| **v6.1 基因库与记忆史学**                                                                                                                                        | <br />                                                                                                                          |
| `attach_memory_store(..., sync_stm=True)`                                                                                                                | v6.1 参数：短期记忆也全量入库（默认只同步 LTM）                                                                                                    |
| `recall_semantic(..., modality=, exclude_modality=)`                                                                                                     | v6.1 参数：跨模态统一向量空间联想（只查/排除某模态）                                                                                                   |
| `memory_store.memory_history(content, brain)` / `recall_version(content, version)`                                                                       | 记忆版本控制：修改历史（add/reinforce/decay 轨迹）/ 回忆过去版本                                                                                     |
| `attach_dna_library(path)` / `save_to_library(parents)`                                                                                                  | DNA 基因库接入 / 当前 DNA 入库（可挂亲代谱系）                                                                                                   |
| `DNALibrary.search(name_contains, min_generation, sensation_seeking, habituation_rate)`                                                                  | 按人格参数/世代/名字检索库存 DNA                                                                                                             |
| `DNALibrary.lineage(dna_id)` / `get(dna_id)`                                                                                                             | 进化谱系回溯 / 取回完整 DNA（可直接 from\_dna 克隆）                                                                                             |
| `swarm.attach_dna_library()` / `save_population()`                                                                                                       | 种群接入基因库；进化子代自动存档并链接亲代                                                                                                           |
| `dump_dna()` / `from_dna(dna)`                                                                                                                           | DNA 字典级导出与重建（`save_dna`/`load_dna` 的内存版，v6.2 起含自我概念/自传体记忆/心智模型/模因库，向后兼容旧 DNA）                                                   |
| **v6.2 文本语义编码器**                                                                                                                                         | <br />                                                                                                                          |
| `attach_text_encoder(encoder=None, model_path=None)`                                                                                                     | 接入 sentence-transformers 本地模型（默认找 `models/bge-small-zh-v1.5`）；接入后文本记忆自动携带真语义 features，recall\_semantic 走真语义检索；模型缺失自动降级哈希向量，行为不变 |
| `models/encoders/text_encoder.py`                                                                                                                        | `TextSemanticEncoder`：encode(text)→512 维 L2 归一化向量 / available / info；`create_text_encoder()` 工厂绝不抛异常                            |
| **认知子系统扩展包（代码内编号 v6.0\~v7.1）**                                                                                                                           | <br />                                                                                                                          |
| `phonological_store/rehearse` / `visuospatial_store/manipulate` / `episodic_store/integrate` / `central_executive_attention(target)`                     | Baddeley 工作记忆四组件；`wm_operation(op, **kw)` 统一入口、`get_working_memory_report()`                                                    |
| `generate_prediction()` / `calculate_prediction_error(input)` / `minimize_free_energy(input)`                                                            | 预测编码：预测 → 误差 → 自由能最小化（`set_precision_weighting` 精度加权开关）                                                                         |
| `generate_action_strategies()` / `select_action()` / `execute_action()` / `active_inference_step()`                                                      | 主动推理：按预期自由能选行动；`add_goal`/`set_preferences` 设定目标偏好                                                                              |
| `update_brainwaves()` / `get_dominant_wave()` / `gamma_binding(features)` / `induce_state(state)`                                                        | 神经振荡：五频段脑电波、γ 特征绑定、状态诱导（专注/冥想/警觉/睡眠）                                                                                            |
| `hippocampus_encode/replay/pattern_completion` / `prefrontal_make_plan/inhibit` / `amygdala_detect_threat/fear_conditioning/extinguish_fear`             | 脑区分化：海马编码重放补全、前额叶规划抑制、杏仁核威胁与恐惧条件反射                                                                                              |
| `deductive/inductive/abductive_reasoning(...)` / `causal_attribution(event)` / `counterfactual_reasoning(event, alt)`                                    | 推理：演绎/归纳/溯因 + 因果归因 + 反事实                                                                                                        |
| `plan_goal(goal)` / `advance_plan()` / `solve_problem(problem)` / `make_decision(options)`                                                               | 规划与问题求解（含手段-目的分析）/ 多选项决策                                                                                                        |
| `create_mental_image(obj)` / `mental_rotate(angle)` / `remember_past()` / `imagine_future()` / `generate_insight(problem)` / `divergent_thinking(topic)` | 心理模拟：表象旋转、过去/未来情景建构、洞察与发散思维；`activate_dmn()` 默认模式网络                                                                             |
| `develop(months)` / `get_piaget_stage()` / `has_object_permanence/conservation/abstract_thinking()` / `is_in_critical_period(domain)`                    | 发育：皮亚杰阶段推进、里程碑检测、关键期查询；`experience_dependent_plasticity(exp)`                                                                   |
| `init_body_schema()` / `plan_motor_action(action)` / `execute_motor_action()` / `learn_motor_skill(name)`                                                | 具身认知：身体图式、运动计划/执行/技能学习                                                                                                          |
| `observe_action(action)` / `imitate_action(action)` / `empathize_with_action(action)`                                                                    | 镜像神经元：观察→模仿→动作共情                                                                                                                |
| `perceive_affordance(obj)` / `use_tool(tool)` / `navigate_spatially(dest)`                                                                               | 环境交互：可供性知觉、工具使用、空间导航                                                                                                            |
| `transmit_culture(trait)` / `innovate_culture()` / `recombine_culture(t1,t2)` / `select_cultural_trait(trait)`                                           | 文化进化：传递/创新/重组/选择；`cultural_drift` 漂变                                                                                            |
| `add_meme(name)` / `replicate_meme(name)` / `add_cultural_norm` / `add_ritual` / `conform_to_group` / `follow_prestige`                                  | 模因系统与群体文化（规范/仪式/从众/声望偏向）；`cultural_evolution_step()` 演化单步                                                                       |
| `learn_incremental(knowledge)` / `learn_online(exp)` / `transfer_learning(src, tgt)`                                                                     | 终身学习：增量/在线学习、迁移学习（抗灾难性遗忘）                                                                                                       |
| `spaced_repetition(knowledge)` / `active_recall(q)` / `interleaved_practice(topics)` / `deliberate_practice(skill)`                                      | 知识巩固：间隔重复/主动回忆/交错练习/刻意练习                                                                                                        |
| `meta_learn(exp)` / `select_learning_strategy(task)` / `adapt_to_environment(change)`                                                                    | 元学习与适应：学会学习、策略选择、环境/任务适应                                                                                                        |
| `update_consciousness_framework()` / `conscious_ignition/binding/higher_order_thought` / `set_consciousness_state(state)`                                | 意识整合：GWT+IIT+HOT 统一框架、三大意识机制、状态设定                                                                                               |
| `measure_consciousness_level/diversity/stability()` / `transition_consciousness_state(target)` / `get_consciousness_integration_report()`                | 意识度量（水平/多样性/稳定性）、状态转换、整合报告                                                                                                      |

**群体文化工具**（`swarm.py` 实验层）

| 函数                                             | 功能                                               |
| ---------------------------------------------- | ------------------------------------------------ |
| `transmit(donor, receiver, top_k, fidelity)`   | 定向文化传递：donor 经 DNA 快照把 top\_k 条最强长期记忆教给 receiver |
| `cultural_similarity(a, b)`                    | 两实体长期记忆的文化重合度（Jaccard）                           |
| `meme_trace(brains, keyword)`                  | 追踪某文化主题在群体中的分布（持有者/权重/时间）                        |
| `generation_chain(swarm, memes, ...)`          | 文化世代传递实验：第 0 代学会全部 memes 后沿种群逐代传递并追踪保真度          |
| `consensus_phase_scan(sizes, topologies, ...)` | 共识相变扫描：种群规模 × 连接拓扑 → 收敛速度矩阵                      |

**脚本入口**（均直接 `python <脚本>` 运行）

| 脚本                              | 功能                                                                                               |
| ------------------------------- | ------------------------------------------------------------------------------------------------ |
| `run_all.py`                    | 全流程统一入口：测试 → 演示 → 实验 → 观测数据导出；`--quick` 快速模式                                                     |
| `experiments.py`                | 实验 1-8 统一复现（可塑性/记忆/情绪/光栅/STDP/多巴胺/文化/多模态），生成 `figures/exp*.png` 与 `datasets/experiment_results.json` |
| `brain_activity_trace.py`       | 大脑活动追踪导出：每步膜电位/脉冲/情绪/记忆 → `datasets/brain_activity_trace.json`（Widget 回放数据源）                         |
| `thought_chain_scenarios.py`    | 脉冲思考链三场景导出（Spike CoT 对照实验）→ `datasets/thought_chain_scenarios.json`                                  |
| `encoder_status.py`             | 多模态编码器状态导出（v3.1 自定义模型通路）→ `datasets/encoder_status.json`                                             |
| `models/encoders/my_encoder.py` | 示例自定义编码器：直方图式 32 维图像编码，零依赖，演示注册通路                                                                |

## 项目结构

```
ai_brain/
├── ai_brain_entity.py        # 核心实体 + BrainSwarm（零依赖，可独立运行）
├── swarm.py                  # 群体文化传递实验层（定向传递/变异/世代链追踪）
├── experiments.py            # 实验 1-8（统一复现，生成 figures/）
├── brain_activity_trace.py   # 大脑活动追踪导出（每步膜电位/脉冲/情绪/记忆 → datasets/）
├── thought_chain_scenarios.py  # 脉冲思考链三场景导出（Spike CoT 对照实验 → datasets/）
├── encoder_status.py         # 多模态编码器状态导出（v3.1 自定义模型通路 → datasets/）
├── memory_store.py           # v6.0 LanceDB 记忆向量库后端（未装 lancedb 自动降级）
├── run_all.py                # 全流程统一入口（测试→演示→实验→观测数据，--quick 快速模式）
├── tests/                    # 核心行为测试（纯标准库 unittest）
│   └── test_ai_brain.py      # 187 项：编码/可塑性/记忆/奖励/DNA/思考链/群体/多模态/v4.0~v6.2/认知子系统扩展包
├── docs/
│   └── paper.md              # 学术论文（架构+实验+分析，含 v3.0 更新章节）
├── datasets/                     # 运行时数据产物
│   ├── experiment_results.json    # 实验 1-8 数据（统一输出）
│   ├── brain_activity_trace.json  # 大脑活动追踪（Widget 回放数据，预生成）
│   ├── thought_chain_scenarios.json  # Spike CoT 三场景（Widget 数据源，预生成）
│   ├── encoder_status.json        # 多模态编码器状态（Widget 数据源，预生成）
│   └── brain_dna.json             # 演示生成的 DNA 快照
├── models/                   # 自定义多模态模型目录（权重已被 .gitignore 忽略）
│   ├── README.md             # 目录约定与接入方式
│   └── encoders/             # 编码器集合
│       ├── my_encoder.py         # 示例自定义编码器（零依赖）
│       ├── multimodal.py         # v5.0 Whisper/BLIP 编码器（本地→远程→PIL 降级链）
│       └── multimodal_service.py # 跨进程编码服务（系统 Python 子进程入口）
│   └── generators/           # 语言生成器目录
│       └── qwen_generator.py     # v5.9 Qwen2-0.5B 语言后端（未下载自动降级模板）
├── figures/                  # 实验图表
│   ├── exp1_hebbian.png      # 突触可塑性开关对照
│   ├── exp2_memory.png       # 记忆固化与遗忘
│   ├── exp3_emotion.png      # 情绪-注意力闭环
│   ├── exp4_raster.png       # 脉冲光栅图
│   ├── exp5_stdp.png         # STDP 因果方向性
│   ├── exp6_dopamine.png     # 多巴胺奖励调制
│   ├── exp7_swarm.png        # 文化跨代传递
│   ├── exp8_multimodal.png   # embedding 相似性保持
│   └── thought_chain.png     # 脉冲思考链传播图
├── requirements.txt          # 实验脚本依赖（核心模块零依赖）
└── README.md
```

## 关键实验结论（详见 docs/paper.md）

**基础机制（实验 1-4）**

* **突触可塑性有效**：120 tick 循环刺激后，开启 STDP 的突触平均强度 0.761 vs
  对照组 0.351；强连接(>0.5)数量 644/768 vs 153/768。

* **记忆按强度有序遗忘**：指数衰减下，记忆条目在第 172 轮开始按权重
  从弱到强依次消亡。

* **情绪-注意力闭环稳定**：注意力在 \[0.621, 0.682] 区间内受情绪调制，
  系统无发散。

**v3.0 新机制（实验 5-8）**

* **STDP 学到因果方向**：方向不对称指数 训练A→B=+0.41 > 未训练基线=+0.18

  > 训练B→A=+0.08——循环突触强化方向由训练时序决定（赫布规则无此性质）。

* **多巴胺奖励加速学习**：奖励组突触平均强度 0.849 vs 无奖励组 0.700，
  强连接 657 vs 571。

* **温习是文化存续的必要条件**：8 条文化记忆 6 代链式传递，温习组全部
  存活（伴随 18 条变异），不温习组权重逐代衰减、第 4 代跌破固化阈值
  后文化灭绝。

* **多模态通路保持相似性排序**：embedding 经 16 维重采样后，相似对
  (0.98/0.95) > 不同对象 (0.82) > 随机 (0.74)；同时发现 abs 归一化
  丢失符号信息、稠密 embedding 插值扁平化两个通路局限。

## 扩展方向

* ✅ ~~以可学习投影替代线性插值重采样~~（v4.0：随机投影 + Oja 在线 PCA）

* ✅ ~~水平 vs 垂直传播动力学、群体共识涌现~~（v4.0 落地）

* ✅ ~~奖励预测误差（RPE/TD 误差）替代直接奖励~~（v4.0：`reward_td()`）

* ✅ ~~决策输出接入动作空间 / 语言生成模块~~（v4.0：`decide_action()` / `express()`）

* ✅ ~~共识涌现的相变条件：种群规模、连接拓扑对收敛速度的影响~~（v4.1：`set_topology()` + `consensus_phase_scan()`）

* ✅ ~~拓扑自适应：共识压力反作用于社交边的生灭（共同演化网络）~~（v4.2：`rewire_coevolve()` + `coevolve_consensus()`）

* ✅ ~~多模因竞争：多个 meme 在同一共同演化网络上的竞争/共存动力学~~（v4.3：`compete_coevolve()` + `competition_dynamics()`，φ 低→垄断 / φ 高→极化，临界点 φ∈(0.5, 0.7)）

* ✅ ~~TD(λ) 资格迹 / 多步回报，让信用分配跨 tick 传播~~（v4.4：`reward_lambda()`，替换迹 γλ 衰减，RPE 反向分配——Schultz 多巴胺时序迁移复现）

* ✅ ~~技能学习：不同 verb 维护独立价值估计（动作选择策略化）~~（v4.5：`learn_skill()` + `select_verb()`，greedy/ε-greedy/softmax）

* ✅ ~~语言生成从模板走向检索式组合（LTM 片段 + 句法框架）~~（v4.6：`compose()`，n-gram 降级检索 + 记忆编织 + 句法框架）

* ✅ ~~情景记忆时间索引：LTM 记录"何时与何事共现"，支持"上次……之后"式时间推理~~（v4.7：`episodes` + `events_after()` / `events_before()`）

* ✅ ~~睡眠-清醒节律：离线期记忆重放（replay）加速固化、清理低价值突触~~（v4.8：`sleep()`，SHY 等比缩放保留相对差异）

* ✅ ~~好奇驱动探索：新奇度（RPE 绝对值 / 记忆未命中率）反向调制 ε 与注意~~（v4.9：`_assess_novelty()` + `effective_epsilon()`）

* ✅ ~~思考体系：思考空间 / 思考记忆 / 思考感官~~（v5.0：`thought_space` + `think()` + `introspect()` + `metacog_log`）

* ✅ ~~意图动词扩展与深思熟虑决策~~（v5.1：`INTENT_VERBS` 8 verb + rationale 理由链）

* ✅ ~~意识流：自由联想、白日梦、灵感闪现~~（v5.2：`stream_of_consciousness()`）

* ✅ ~~自我意识：内省能力增强，知道"自己在想什么"~~（v5.2：`introspect(depth="deep")`）

* ✅ ~~社交互动：多个大脑之间交流、学习、形成文化~~（v5.2：`send_message` / `social_learn` / `chat_with`）

* ✅ ~~进化：BrainSwarm 群体智能，适者生存~~（v5.2：`evaluate_fitness` + `select` + `evolve()`）

* ✅ ~~自我模型：自我概念、自传体记忆、自我反思~~（v5.3：`update_self_concept` / `add_autobiographical_memory` / `self_reflect`）

* ✅ ~~文化演化与物种形成：模因变异、有性繁殖、遗传距离~~（v5.3：`cultural_evolution` / `sexual_reproduce` / `detect_species`）

* ✅ ~~全局工作空间理论（GWT）：注意竞争 → 点火 → 全局广播~~（v5.4：`conscious_step`）

* ✅ ~~意识的神经相关物（NCC）：Φ / 复杂度 / 同步性 / 相变~~（v5.5：`detect_ncc` / `consciousness_phase_transition`）

* ✅ ~~心智理论（ToM）：信念归因、错误信念、视角采择、共情~~（v5.6：`theory_of_mind` 等 6 法）

* ✅ ~~高阶意识理论（HOT）：高阶思想、元意识、内省层级~~（v5.7：`higher_order_thought` / `introspection_hierarchy`）

* ✅ ~~集体意识：群体工作空间与涌现判定~~（v5.8：`collective_workspace` / `collective_consciousness_check`）

* ✅ ~~语言生成接入真实 LLM（大脑想什么 → 模型说出来）~~（v5.9：`set_qwen_model()` + `register_language_generator`，Qwen2-0.5B / 任意自定义模型，未下载自动降级模板）

* ✅ ~~长期记忆外置向量库：容量无限 + 语义检索~~（v6.0：`attach_memory_store()` + `recall_semantic()`，LanceDB 持久化，未装自动降级内存模式）

* ✅ ~~DNA 基因库：多脑 DNA 存储、人格搜索、进化谱系追踪~~（v6.1：`DNALibrary` + `save_to_library()` + `lineage()`，进化子代自动存档）

* ✅ ~~跨模态记忆联想：统一向量空间自由联想~~（v6.1：`recall_semantic(modality=, exclude_modality=)`）

* ✅ ~~记忆版本控制：修改历史、回忆过去版本、演化轨迹~~（v6.1：`memory_history()` + `recall_version()`）

