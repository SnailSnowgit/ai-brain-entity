# BrainCog 学习笔记（2026-08-02，经 GitHub API 研读）

来源：https://github.com/BrainCog-X/Brain-Cog （中科院自动化所曾毅团队，Patterns 2023）
定位：基于脉冲神经网络（SNN）的类脑认知智能引擎，PyTorch 重型框架
（torch/timm/einops/scikit-learn 等依赖；与 ai_brain 零依赖纯标准库路线互补）。

## 体系结构

- `braincog/base/`：基础设施——node（神经元模型）、connection、learningrule（学习规则）、
  brainarea（脑区模块）、encoder、strategy（代理梯度）、conversion（ANN→SNN）
- `examples/`：50+ 算法，分五大认知功能 + 脑模拟 + 硬件 + 具身：
  - Perception_and_Learning（图像/事件分类，BP/代理梯度）
  - Knowledge_Representation_and_Reasoning（CKRGSNN/CRSNN/SPSNN/musicMemory）
  - decision_making（BDM-SNN 基底节决策 / RL / swarm 避碰）
  - MotorControl、Social_Cognition（见下）
  - Structural_Development（DPAP/DSD-SNN/ELSM/SCA-SNN/SD-SNN 结构发育）
  - Structure_Evolution（结构演化）、Snn_safety、TIM、Spiking-Transformers
  - Brain_Cognitive_Function_Simulation（drosophila 果蝇脑）
  - Multiscale_Brain_Structure_Simulation（鼠/猴/人多尺度脑模拟）
  - Embodied_Cognition（Embot：脉冲 Transformer 扩散策略、脉冲世界模型）
  - Hardware_acceleration（FireFly FPGA 加速器系列，TCAS-I/TCAD/TVLSI 论文）

## 学习规则精读（braincog/base/learningrule）

### STDP.py — 迹实现的 STDP 家族
- `STDP`：pre-trace（decay 0.99 指数衰减 + 脉冲累加）× post-spike 产生 dw
  （用 autograd grad 技巧从连接权重取更新量）
- `MutliInputSTDP`：多组输入共用节点，逐连接独立 trace
- `LTD`：post-trace 门控的独立 LTD
- `FullSTDP`：**pre/post 双迹、双时间常数**（decay, decay2）——完整不对称时间窗，
  LTP 与 LTD 由两条独立迹分别计算（对照：ai_brain 用 _last_spike 单配对近似）

### BCM.py — 滑动修饰阈值（元可塑性）
```
θ ← ((τ−1)·θ + s) / τ        # 阈值 = 突触后活动性的滑动平均
dw = s·(s − θ) − (1−wd)·w    # 高于阈值 LTP、低于 LTD，外加权重衰减
```
意义：高频历史活动抬高 LTP 阈值 → 防饱和的稳态调节（与 ai_brain v4.8 SHY
等比缩放互补：BCM 调"学习的方向"，SHY 调"存量的缩放"）。

### RSTDP.py — 奖励调制三因子规则
```
dw_STDP × reward_trace    （reward_trace 按 decay 衰减、随奖励累加）
```
把全局奖励信号转为突触特异门控：只有"近期有 STDP 事件"的突触吃到奖励。
对照：ai_brain v4.4 是状态级资格迹（刺激内容→价值），RSTDP 是突触级
（e_syn × R）——两者可合并为"突触级 e × 状态级 δ"。

### STP.py — 短时程可塑性（Tsodyks-Markram 易化/抑制），存在但未精读

## 脑区模块精读（braincog/base/brainarea）

### basalganglia.py — 基底节三通路动作选择（最值得借鉴）
完整解剖连接：
- 直接通路（Go）：DLPFC→StrD1→GPi（抑制）
- 间接通路（NoGo）：DLPFC→StrD2→GPe→GPi
- 超直接通路（全局制动）：DLPFC→STN→GPi（兴奋）
- STP 只作用于皮层→纹状体突触（DLPFC-StrD1/StrD2/STN），权重 L1 归一化
- GPi 输出即动作选择结果
对照：ai_brain v4.5 是 Q(verb) 表格 + ε-greedy；BG 模型用 Go/NoGo 双通路
竞争 + 超直接通路实现"权衡中的制动"（决策犹豫的神经基础）。

### 其他脑区
PFC（工作记忆/前额叶）、Insula（内感受/共情）、dACC（冲突监控）、IPL（顶下小叶）。

## Social_Cognition（examples/）
ToM（心理理论）、MAToM-SNN、FOToM（一阶 ToM）、ToCM、Intention_Prediction、
affective_empathy（情感共情）、mirror_test（镜像自我识别）、SmashVat。
——BrainSwarm 目前只有文化传递/竞争，没有"个体对他人心智建模"的机制。

## 映射到 ai_brain 的候选方向（v5.x 储备）

1. **BCM 滑动阈值**：STDP 加元可塑性稳态，防权重饱和（与 SHY 互补）
2. **RSTDP 突触级三因子**：v4.4 资格迹下沉到突触级，e_syn × RPE
3. **基底节 Go/NoGo 动作选择**：v4.5 升级为直接/间接/超直接通路竞争，
   支持"决策犹豫/制动"
4. **STP 短时程可塑性**：突触使用的短期易化/抑制动态
5. **心理理论（ToM）**：BrainSwarm 个体维护他者信念模型 → 共情/意图预测
6. **结构发育**：突触自发生成与剪除（目前只有社交边生灭，突触层未发育化）
7. **脉冲世界模型**：model-based RL（v4.4/v4.5 都是 model-free）

## 未覆盖部分（体量所限，可按需续读）

- node.py 神经元模型全族（LIF/IF/HH/多室模型等数千行）
- Structural_Development / Structure_Evolution 五个发育算法细节
- Spiking-Transformers / TIM / Embot 具身智能
- Multiscale_Brain_Structure_Simulation 多尺度脑模拟数据管线
- encoder / conversion / strategy 代理梯度实现
