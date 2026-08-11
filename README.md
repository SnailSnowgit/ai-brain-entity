# 类脑认知架构模拟器 (Brain Simulator)

一个模拟大脑信息处理的分层神经网络系统，用 Python + NumPy 实现了完整的感知-认知-决策-记忆-调制闭环，并扩展到进化、发育、意识量化与多智能体研究。

**当前版本**: v1.9.1 · **依赖**: 仅 `numpy>=1.20.0` · **许可证**: MIT

## 架构概览

```
                    外部刺激
                       ↓
            感官层(200神经元) → 联想层(500神经元) → 决策层(20神经元) → 行为输出
                       ↓                  ↓                  ↓
                  感官缓存(50)        短期记忆(100)       长期记忆(5000)
                       └──────── 情绪内核 ←→ 注意力调制 ←→ 多巴胺奖励 ────────┘
                       └──────────── 胶质细胞网络 ────────────────┘
                                    (星形/少突/小胶质)
                       └──────────── 思考系统 ──────────────────┘
                                    (思考空间/思考记忆/思考感官)
```

> 以上括号内为 `Brain()` 的默认参数，均可在构造时自定义。

## 核心模块

### 1. 三层神经网络 (Layers)
- **感官层 (Sensory Layer)**: 默认 200 神经元，接收外部刺激，5 种感官模态，感官适应
- **联想层 (Association Layer)**: 默认 500 神经元，跨模态整合，概念形成，Hebbian 学习
- **决策层 (Decision Layer)**: 默认 20 神经元，动作选择，赢家通吃机制，强化学习

### 2. 三级记忆系统 (Memory)
- **感官缓存 (Sensory Buffer)**: 默认 50 槽位，短时保存原始感官信息（~500ms）
- **短期记忆 (Short-Term Memory)**: 默认 100 槽位，工作记忆，意识可及（秒级）
- **长期记忆 (Long-Term Memory)**: 默认 5000 槽位，持久存储，语义网络，激活扩散

### 3. 三大调制系统 (Modulation)
- **情绪内核 (Emotional Core)**: 效价-唤醒度模型，基本情绪，情绪记忆
- **注意力调制 (Attention Modulator)**: 自下而上显著性，自上而下目标驱动
- **多巴胺奖励 (Dopamine Reward)**: 奖励预测误差，强化学习，动机调节

### 4. 胶质细胞系统 (Glia)
数量随感官层规模自动配置（0.5 / 0.2 / 0.1 比例，默认规模下为 100 / 40 / 20）：
- **星形胶质细胞 (Astrocytes)**: 钙波传播，突触调节，钾缓冲
- **少突胶质细胞 (Oligodendrocytes)**: 髓鞘形成，传导速度调节
- **小胶质细胞 (Microglia)**: 免疫监视，突触修剪，神经营养

### 5. 思考系统 (Thought)
- **思考空间 (Thought Space)**: 全局工作空间，7±2 容量，思维广播
- **思考记忆 (Thought Memory)**: 思维序列记忆，元记忆，思维情节
- **思考感官 (Thought Sensory)**: 元认知感知，流畅度/负荷/确定性，内省

### 6. 脑区模块 (Brain Regions & Advanced Regions)
- **基础脑区**: 海马体（情景记忆）、前额叶（工作记忆/计划）、丘脑（中继门控）、基底节（动作选择）
- **高级脑区**: 小脑（运动学习）、扣带回（冲突监控）、默认模式网络（走神/自我参照）、顶叶（空间注意）、岛叶（内感受）

### 7. 大规模稀疏网络 (Large-Scale)
- **稀疏连接**: 每个神经元只连接 k 个其他神经元（支持百万级）
- **高效存储**: 使用数组存储，避免 Python 对象开销
- **向量化计算**: 利用 NumPy 批量计算，性能优化
- **规模扩展**: 支持 50 万+ 神经元，仅需约 2.5GB 内存

### 8. 感知模块 (Perception)
- **视觉模块 (Visual Module)**: 视觉皮层模拟，特征提取，物体识别，视觉工作记忆
- **语言模块 (Language Module)**: 语言区模拟，词汇表征，语法处理，语义理解，内部言语
- **感知系统 (PerceptualSystem)**: 多模态感知整合，视觉-语言联合表征

### 9. 预测编码与自由能原理 (Predictive Coding & Free Energy)
- **预测编码网络**: 层级预测编码，自上而下预测，自下而上误差
- **自由能最小化**: 变分自由能，预测误差最小化，复杂度-准确率权衡
- **主动推理 (Active Inference)**: 期望自由能，风险-模糊度权衡，感知-行动循环
- **精度加权 (Precision Weighting)**: 注意力的预测编码解释，动态精度调整

### 10. 意识量化 (Consciousness)
- 基于整合信息理论 (IIT Φ)、全局工作空间理论 (GWT) 与元认知理论
- 输出意识等级（无意识 → 微意识 → 低意识 → …）与多维度指标
- 配套 `examples/visualize_consciousness.py` 可生成意识动力学曲线

### 11. 进化与发育研究模块
- **遗传系统 (Genetics)**: 基因编码、基因库、选择/交叉/变异
- **可进化大脑 (Evolvable Brain)**: 大脑结构与参数的代际进化
- **进化强化学习 (Evolutionary RL)**: 外层进化 + 内层 RL 双层优化
- **神经架构搜索 (BrainNAS)**: 自动搜索最优大脑结构与超参数
- **发育生物学 (Developmental)**: 神经发生、细胞迁移、突触修剪
- **物种形成 (Speciation)**: 基于基因组相似度的物种划分与生态位分化
- **Baldwin 效应 (Baldwin Deep Dive)**: 学习引导进化、遗传同化
- **个体差异 (Individual Differences)**: 基因多样性 → 认知多样性研究
- **认知架构设计器 (Cognitive Architecture Designer)**: 按任务自动设计最优认知架构

### 12. 多智能体系统 (Multi-Agent)
- **BrainAgent**: 以完整大脑为核心的智能体
- **MultiAgentSystem**: 多智能体交互、消息传递、社会记忆
- 支持合作 / 竞争等交互模式

## 快速开始

### 安装依赖
```bash
pip install numpy
```

### 运行演示
```bash
cd brain_simulator
python examples/demo.py
```

### 基本使用

```python
from brain import Brain
import numpy as np

# 创建大脑（使用默认规模）
brain = Brain(
    sensory_neurons=200,
    association_neurons=500,
    decision_neurons=8,  # 决策神经元数 = 动作数，可按需调整
    action_names=["前进", "后退", "左转", "右转", "探索", "休息", "进食", "躲避"]
)

# 输入刺激（维度必须与 sensory_neurons 一致）
stimulus = np.random.rand(200) * 2.0
brain.input_stimulus(stimulus, modality=0)

# 运行一步
state = brain.step(dt=1.0)

# 查看状态
print(f"决策: {state.decision[1]}")
print(f"情绪: {state.emotional_state}")
print(f"多巴胺: {state.dopamine_level:.3f}")

# 给予奖励反馈
brain.reward(1.0)
```

> ⚠️ **注意**：输入刺激的维度必须与 `sensory_neurons` 一致（默认 200）。
> 维度不匹配会在注意力调制阶段触发 NumPy 广播错误。

## 项目结构

```
brain_simulator/
├── brain/                           # 核心模块（26 个，约 1.9 万行）
│   ├── __init__.py                  # 包初始化（导出 79 个符号）
│   ├── neuron.py                    # 脉冲神经元 (LIF) 与神经层基础类
│   ├── layers.py                    # 三层网络（感官/联想/决策）
│   ├── memory.py                    # 三级记忆系统
│   ├── modulation.py                # 调制系统（情绪/注意力/多巴胺）
│   ├── glia.py                      # 胶质细胞系统
│   ├── thought.py                   # 思考系统
│   ├── brain.py                     # 主脑整合类 (Brain / BrainState)
│   ├── consciousness.py             # 意识量化 (IIT/GWT/元认知)
│   ├── perception.py                # 感知模块（视觉/语言）
│   ├── brain_regions.py             # 基础脑区（海马体/前额叶/丘脑/基底节）
│   ├── advanced_regions.py          # 高级脑区（小脑/扣带回/默认网络/顶叶/岛叶）
│   ├── large_network.py             # 大规模稀疏网络层
│   ├── large_brain.py               # 大规模类脑整合
│   ├── large_scale.py               # 百万级稀疏网络
│   ├── predictive_coding.py         # 预测编码与自由能原理
│   ├── multi_agent.py               # 多智能体系统
│   ├── genetics.py                  # 遗传与进化
│   ├── evolvable_brain.py           # 可进化大脑
│   ├── evolutionary_rl.py           # 进化 + RL 双层优化
│   ├── nas.py                       # 类脑神经架构搜索 (BrainNAS)
│   ├── developmental.py             # 神经发育模拟
│   ├── speciation.py                # 物种形成模拟
│   ├── baldwin_deep_dive.py         # Baldwin 效应研究
│   ├── individual_diff.py           # 个体差异研究
│   └── cognitive_architecture_designer.py  # AI 认知架构设计器
├── examples/                        # 示例与测试脚本（35 个）
│   ├── demo.py                      # 完整演示
│   ├── demo_large_scale.py          # 大规模网络演示
│   ├── visualize_consciousness.py   # 意识动力学可视化
│   ├── evolution_experiment.py      # 进化实验
│   └── test_*.py                    # 各模块测试
├── docs/
│   └── architecture.md              # 架构设计文档
├── data/                            # （预留）数据目录
├── datasets/                        # （预留）数据集目录
├── models/                          # （预留）模型保存目录
└── README.md
```

## 核心特性

### 脉冲神经元模型 (LIF)
- 漏电积分发放 (Leaky Integrate-and-Fire)
- 膜电位动态变化
- 绝对不应期
- 兴奋性/抑制性神经元

### 记忆机制
- **记忆巩固**: 短期记忆 → 长期记忆的概率性转移
- **记忆提取**: 线索提取，激活扩散
- **情绪增强**: 情绪性记忆更牢固
- **多巴胺调节**: 奖励增强记忆巩固
- **测试效应**: 提取增强记忆强度

### 学习机制
- **Hebbian 学习**: 同时激活的神经元连接增强
- **强化学习**: 多巴胺介导的奖励预测误差
- **联想学习**: 刺激-反应联结形成

### 调制系统
- **情绪-注意交互**: 情绪一致性注意偏向
- **多巴胺-记忆交互**: 奖励增强记忆巩固
- **注意-感知交互**: 注意力增益调制

## 示例脚本一览

### 演示
| 脚本 | 说明 |
|---|---|
| `demo.py` | 完整演示：刺激-反应、记忆、情绪注意、多巴胺学习 |
| `demo_large_scale.py` | 大规模稀疏网络演示 |
| `visualize_consciousness.py` | 意识动力学追踪与可视化 |
| `evolution_experiment.py` | 大脑进化实验 |

### 核心模块测试
| 脚本 | 说明 |
|---|---|
| `test_neuron.py` / `test_network.py` / `test_core.py` | 神经元 / 网络 / 核心通路 |
| `test_glia.py` / `test_thought.py` | 胶质细胞 / 思考系统 |
| `test_perception.py` / `test_predictive_coding.py` | 感知 / 预测编码 |
| `test_brain_regions.py` / `test_advanced_regions.py` | 基础 / 高级脑区 |
| `test_consciousness.py` | 意识量化 |
| `test_multi_agent.py` | 多智能体系统 |

### 进化与发育研究
| 脚本 | 说明 |
|---|---|
| `test_genetics.py` / `test_evolvable_brain.py` | 遗传系统 / 可进化大脑 |
| `test_evolutionary_rl.py` / `test_nas.py` | 进化 RL / 架构搜索 |
| `test_developmental.py` / `test_speciation.py` | 发育 / 物种形成 |
| `test_baldwin_deep_dive.py` / `test_individual_diff.py` | Baldwin 效应 / 个体差异 |
| `test_cognitive_designer.py` / `test_ai_quick.py` | 认知架构设计器 |

### 性能与调试
| 脚本 | 说明 |
|---|---|
| `test_large.py` / `test_500k.py` | 大规模网络（50 万神经元） |
| `test_performance.py` / `test_large_perf.py` / `test_basic_perf.py` | 性能基准 |
| `debug_neuron.py` / `debug_consciousness.py` / `diagnose_pc.py` | 调试与诊断 |
| `verify_scores.py` / `test_ideal_action.py` | 评分验证 / 理想动作测试 |

## 扩展开发

### 添加新的感官模态
```python
brain = Brain(num_modalities=6)  # 增加到 6 种感官模态（默认 5）
```

### 自定义动作集合
```python
brain = Brain(
    decision_neurons=4,
    action_names=["上", "下", "左", "右"]
)
```

### 调整记忆容量
```python
brain = Brain(
    sensory_buffer_size=80,   # 默认 50
    stm_size=150,             # 默认 100
    ltm_size=10000            # 默认 5000
)
```

### 调整网络规模
```python
brain = Brain(
    sensory_neurons=300,
    association_neurons=800,
    decision_neurons=12
)
# 注意：胶质细胞数量按感官层规模自动配置（0.5/0.2/0.1 比例）
```

## 技术细节

### 神经元参数
- 静息电位: 0.0 (归一化)
- 发放阈值: 0.5 (归一化)
- 膜时间常数: 10ms
- 绝对不应期: 10ms

### 时间步长
- 默认 dt = 1.0ms
- 支持自定义时间步长

### 数值方法
- 欧拉法积分膜电位
- 指数移动平均计算发放率

## 参考文献

- Hodgkin, A. L., & Huxley, A. F. (1952). A quantitative description of membrane current and its application to conduction and excitation in nerve.
- Hebb, D. O. (1949). The Organization of Behavior.
- Atkinson, R. C., & Shiffrin, R. M. (1968). Human memory: A proposed system and its control processes.
- Schultz, W., Dayan, P., & Montague, P. R. (1997). A neural substrate of prediction and reward.
- Posner, M. I., & Petersen, S. E. (1990). The attention system of the human brain.
- Friston, K. (2010). The free-energy principle: a unified brain theory?
- Tononi, G. (2008). Consciousness as integrated information: a provisional manifesto.
- Baars, B. J. (1988). A Cognitive Theory of Consciousness.

## 许可证

MIT License
