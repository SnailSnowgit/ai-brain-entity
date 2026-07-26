# AI 大脑实体（AIBrainEntity）

![项目封面](figures/banner.png)

一个**纯原生 Python** 实现的类脑智能体架构：不依赖任何第三方库即可运行，
模拟生物大脑的感知、学习、记忆、情绪与决策全过程。v3.0 起支持
STDP 脉冲时序可塑性、多巴胺样奖励强化学习、真实多模态（CLIP/Whisper）
接入与多实体群体智能。

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

| 模块 | 生物对应 | 实现 |
|---|---|---|
| 脉冲神经元 | LIF 神经元 | 膜电位累积-泄漏-阈值放电-复位-不应期 |
| 突触可塑性 | **STDP** | 前→后放电 LTP / 后→前放电 LTD，指数时间窗 τ=3 网络步，覆盖前馈与循环通路 |
| 奖励强化学习 | 多巴胺系统 | `reward()` 调制学习速率 ×(1+多巴胺)，奖励促愉悦、惩罚促压力 |
| 三级记忆 | 感觉登记→短时→长时 | 感官缓存 → STM(容量竞争) → LTM(固化) |
| 遗忘 | 艾宾浩斯曲线 | 记忆权重指数衰减，低于阈值删除 |
| 回忆再巩固 | Reconsolidation | 每次成功回忆都会强化该记忆 |
| 情绪内核 | 边缘系统 | 平静/好奇/压力/愉悦四维动态变量 |
| 注意力 | 注意调制 | 好奇心↑注意、压力↓注意，反作用于感官输入 |
| DNA 遗传 | 记忆传承 | 全状态序列化（突触+记忆+情绪），支持克隆 |
| **群体智能** | 文化传递 | `BrainSwarm`：记忆跨实体传递、广播、带变异繁衍 |
| **脉冲思考链** | 可解释性 | `thought_chain()`：把感知→传导→回响→决策展开为可读的脉冲因果链 |
| 多模态接口 | 感官通道 | CLIP 图像 / Whisper 音频 embedding（未装依赖自动降级伪 embedding） |

## 快速开始

```bash
pip install -r requirements.txt   # 仅实验复现脚本需要（核心模块零依赖）

python ai_brain_entity.py    # 运行内置演示（含 STDP/奖励/多模态/群体演示）
python swarm.py              # 群体文化传递演示（定向传递+变异+世代链）
python experiments.py        # 复现实验 1-4
python experiments_v3.py     # 复现 v3.0 新机制实验 5-8
python brain_activity_trace.py  # 录制大脑活动逐帧数据（供可视化回放）
python -m unittest discover tests  # 运行核心行为测试（16 项）
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

# 群体智能：文化传递
swarm = BrainSwarm(["Alpha", "Beta", "Gamma"], seed=1)
for _ in range(25):
    swarm.population[0].sensory_input("火焰是危险的")
swarm.culture_round(rounds=4, top_k=2, mode="dna")   # DNA 记忆传递
swarm.broadcast("公共事件")                           # 全种群广播

# 脉冲思考链：可解释地展开一次"感知→传导→回响→决策"的因果链
for line in brain.thought_chain("火焰是危险的")["chain"]:
    print(line)

# 保存"DNA"，克隆一个继承全部记忆与突触的新实体
brain.save_dna("brain_dna.json")
clone = AIBrainEntity.load_dna("brain_dna.json", new_name="Brain-02")
```

## 项目结构

```
ai_brain/
├── ai_brain_entity.py        # 核心实体 + BrainSwarm（零依赖，可独立运行）
├── swarm.py                  # 群体文化传递实验层（定向传递/变异/世代链追踪）
├── experiments.py            # 实验 1-4（基础机制，生成 figures/）
├── experiments_v3.py         # 实验 5-8（v3.0 新机制验证）
├── thought_chain_figure.py   # 脉冲思考链传播图（生成 figures/thought_chain.png）
├── brain_activity_trace.py   # 大脑活动逐帧追踪导出（供可视化 Widget 回放）
├── tests/                    # 核心行为测试（纯标准库 unittest）
│   └── test_ai_brain.py      # 16 项：编码/可塑性/记忆/奖励/DNA/思考链/群体
├── docs/
│   └── paper.md              # 学术论文（架构+实验+分析，含 v3.0 更新章节）
├── data/                     # 运行时数据产物
│   ├── experiment_results.json    # 实验 1-4 数据
│   ├── experiment_results_v3.json # 实验 5-8 数据
│   ├── brain_activity_trace.json  # 大脑活动追踪（Widget 回放数据）
│   └── brain_dna.json             # 演示生成的 DNA 快照
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

- **突触可塑性有效**：120 tick 循环刺激后，开启 STDP 的突触平均强度 0.761 vs
  对照组 0.351；强连接(>0.5)数量 644/768 vs 153/768。
- **记忆按强度有序遗忘**：指数衰减下，记忆条目在第 172 轮开始按权重
  从弱到强依次消亡。
- **情绪-注意力闭环稳定**：注意力在 [0.621, 0.682] 区间内受情绪调制，
  系统无发散。

**v3.0 新机制（实验 5-8）**

- **STDP 学到因果方向**：方向不对称指数 训练A→B=+0.41 > 未训练基线=+0.18
  > 训练B→A=+0.08——循环突触强化方向由训练时序决定（赫布规则无此性质）。
- **多巴胺奖励加速学习**：奖励组突触平均强度 0.849 vs 无奖励组 0.700，
  强连接 657 vs 571。
- **温习是文化存续的必要条件**：8 条文化记忆 6 代链式传递，温习组全部
  存活（伴随 18 条变异），不温习组权重逐代衰减、第 4 代跌破固化阈值
  后文化灭绝。
- **多模态通路保持相似性排序**：embedding 经 16 维重采样后，相似对
  (0.98/0.95) > 不同对象 (0.82) > 随机 (0.74)；同时发现 abs 归一化
  丢失符号信息、稠密 embedding 插值扁平化两个通路局限。

## 扩展方向

- 以可学习投影替代线性插值重采样，保留稠密 embedding 的对比度
- 文化传递中的水平传播 vs 垂直传播动力学、群体共识涌现
- 奖励预测误差（RPE/TD 误差）替代直接奖励，实现时序差分学习
- 决策输出接入动作空间 / 语言生成模块
