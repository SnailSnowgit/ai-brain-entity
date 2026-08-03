# -*- coding: utf-8 -*-
"""
AI大脑实体（类脑架构 Python 完整可运行代码）v5.0
====================================================
定位：独立大脑实体，包含感知缓冲区、脉冲神经元集群、短期记忆(STM)、
      长期记忆(LTM)、情绪核、注意力机制、决策中枢、遗忘机制。
依赖：原生 Python（标准库），Python 3.8+。
      多模态真实编码为可选增强（transformers / openai-whisper），
      未安装时自动降级为确定性伪 embedding，核心功能不受影响。

v2.1（动态脉冲活动）：
  - 循环连接：联想层侧向连接 + 决策层→联想层反馈（回声混响）
  - 突触延迟：循环通路的脉冲跨 tick 传播（前馈保持当 tick 级联）
  - 发放不应期：神经元发放后 2 tick 内静默，防止高频锁死
  - 自发背景放电：静息态噪声电流，无输入时也有零星脉冲
  - free_run(ticks)：无外部输入的自由演化，可观察刺激后的回响衰减

v3.0 新增：
  1. 真实多模态接入：CLIP 图像 embedding / Whisper 音频 embedding
     经 sensory_input_vector 进入感官层（未装模型时自动降级）
  2. 群体智能：BrainSwarm 多实体种群，通过 DNA 交换记忆，模拟文化传递
  3. STDP：脉冲时序依赖可塑性替代简化赫布规则
     （pre 先于 post 发放 → LTP；post 先于 pre → LTD，指数时间窗）
  4. 多巴胺样奖励信号：reward(value) 调制学习速率，实现强化学习

v3.1 新增：
  - 可插拔自定义多模态模型：register_*_encoder / set_*_model /
    perceive_*(encoder=...)，编码器优先级链 + 确定性兜底

v4.0 新增（README 四大扩展方向落地）：
  1. 可学习投影（LearnableProjection，Oja 在线 PCA 规则）替代线性插值
     重采样：保符号的中心化归一化，保留稠密 embedding 的对比度
  2. 文化传递动力学：水平（同代同伴）vs 垂直（跨代）传播、
     群体共识涌现度量（consensus）
  3. 奖励预测误差（RPE / TD 误差）：reward_td() 以预测误差驱动多巴胺，
     奖励被预测后多巴胺反应自然衰减（时序差分学习）
  4. 决策输出接入动作空间 / 语言生成：decide_action() 结构化动作 +
     express() 模板化自然语言表达

v4.1 新增：
  1. 动作执行器闭环：register_executor() 注册机器人 / HTTP API 执行器，
     act() 完成"决策→执行→reward_td 奖励回传"的感知-行动-学习闭环
     （make_robot_executor 模拟机器人指令；make_api_executor 零依赖
     urllib POST 结构化动作到外部服务）
  2. 社交拓扑与共识相变：set_topology() 支持全连接/环形/星形/随机/
     小世界，文化传播只沿社交边进行；consensus_convergence() 测量
     模因覆盖率达标所需轮数——拓扑与规模决定收敛速度

v4.2 新增：
  - 拓扑自适应（共同演化网络）：rewire_coevolve() 实现 Holme-Newman
    动力学——异见边上"观点模仿 vs 断边重连"博弈（φ=rewire_prob 控制
    结构主导程度），未持有者向持有者"求知连边"（边之生）；共识压力
    反作用于社交边生灭，静态稀疏拓扑无法收敛的种群在共同演化下
    快速收敛；coevolve_consensus() 记录覆盖率/边数/同道边比例曲线

v4.3 新增：
  - 多模因竞争（文化生态）：compete_coevolve() 让多个模因在同一
    共同演化网络上竞争——个体立场 = 权重最高的模因；阵营内连边
    （边之生），异见边上 φ 断边重连（阵营隔离）vs 1−φ 立场转化
    （教师随机方向）。competition_dynamics() 判定结局：φ 低时立场
    转化主导 → 单一模因垄断共识；φ 高时阵营隔离主导 → 极化共存。
    实测（N=12 环形，两阵营各半）：φ≤0.5 垄断（8-13 轮），
    φ≥0.7 极化（0.5/0.5），临界点 φ∈(0.5, 0.7)

v4.4 新增：
  - TD(λ) 资格迹：每次感知标记状态访问（替换迹，γλ 衰减），
    reward_lambda() 把 RPE 按迹强度反向分配给近期所有状态
    V(s) += α·δ·e(s)——信用分配跨 tick 传播。Schultz 范式实测
    （三线索链→奖励，20 试次）：V 呈时间梯度（0.51/0.71/0.99），
    奖励时刻 RPE 从 1.0 衰减到 0.014（多巴胺时序迁移）

v4.5 新增：
  - 执行器技能学习：learn_skill() 为每个动作 verb 维护独立价值
    估计 Q(verb)（执行后果精确归因到实际采取的动作），select_verb()
    支持 greedy / ε-greedy / softmax 策略；act(policy=...) 由习得
    价值覆盖 verb 选择。实测（respond=0.8/acknowledge=0.2/observe=-0.4
    三执行器，40 轮 ε-greedy）：Q 收敛 0.80/0.19/-0.30，
    greedy 策略 20/20 选高价值动作

v4.6 新增：
  - 检索式语言生成：compose() 把 express() 的固定模板升级为
    "检索→编织→造句"三段式——_retrieve_fragments() 从 LTM 按权重
    取 top_k 相关记忆（长词未命中时退化为 2/3 字 n-gram 子词检索，
    "取火的方法"也能联想到"钻木可以取火"）；_weave_memories()
    按片段数选单句/并列/联想链结构；句法框架填充刺激、记忆从句
    与情绪修饰词，同 tick 输出确定可复现

v4.7 新增：
  - 情景记忆时间索引：每次感知记一条情景 {tick, content, context}
    ——何时发生、与何事共现（感官缓存同期内容）。episodic_trace()
    按时间序取轨迹；events_after() / events_before() 以"最近一次"
    为锚点做时间推理（含 tick 间隔 delta）——支持"上次刷牙之后
    发生了什么"式追问，锚点不存在时安全返回空

v4.8 新增：
  - 睡眠-清醒节律：sleep(cycles) 离线期记忆重放（STM 每条 +0.15/周期，
    达阈值当即固化——白天无法固化的弱刺激在睡眠中转入 LTM）；
    突触稳态缩放（SHY）：全部突触逐周期 ×0.95 等比下调、低于下限
    剪除——保留相对差异（强连接存活），清除噪声弱连接；资格迹
    洗脱、多巴胺归零、压力衰减、平静恢复。实测：3 周期固化 2 条
    弱记忆；12 周期剪除 103/768 条突触

v4.9 新增：
  - 好奇驱动探索：_assess_novelty() 以记忆未命中率为主、近期 |RPE|
    为辅评估新奇度——新奇当 tick 捕获注意（好奇心→注意力即时上升），
    effective_epsilon() 按 ε×(0.5+novelty) 调制探索率：完全新奇 1.5ε
    探索，完全熟悉 0.5ε 利用。实测：熟悉刺激新奇度 0 / ε 减半；
    全新刺激 0.70 / ε 提升；大意外后熟悉刺激重获 0.30 新奇度

v5.0 新增（思考体系）：
  1. 思考空间（全局工作区）：ThoughtItem 念头含来源/激活度/出生 tick，
     容量 7±2（米勒定律），激活度逐 tick 衰减、低于 0.05 退出意识；
     外部感知、联想回忆、主动思考、内省都向思考空间注入念头
  2. 思考记忆：think() 把念头重新编码为电流注入网络（"自言自语"闭环），
     诱发联想；高激活念头（≥0.35）固化进 STM（tag="thought"）——
     想多了就记住了
  3. 思考感官（内感觉 interoception）：introspect() 感知自身脑活动
     （主导情绪/三层脉冲/记忆占用/思考空间顶部念头），生成内省言语
     并回注网络，同时记入 metacog_log 元认知日志

类脑层级：
  1. 感官层神经元：接收外部原始信号（文字 / 任意 embedding 向量）
  2. 联想层神经元：特征提取、关联记忆匹配（含侧向循环连接）
  3. 决策层神经元：整合信息产生行为输出（含反馈投射）
  4. 突触连接：STDP 可塑性（覆盖前馈与循环通路）
  5. 记忆三级结构：感官瞬时缓存 → 短期记忆 → 长期记忆
  6. 自带：情绪动态调节、注意力调制、自然遗忘、回忆再巩固
扩展：
  - DNA 记忆遗传模块（记忆序列化保存、实体克隆继承记忆）
  - 多模态接口（图像/音频 embedding，可选真实模型编码）
  - 群体智能（BrainSwarm：文化传递 / 广播 / 共识涌现）
  - 脉冲活动记录（供可视化/实验分析）
"""

import time
import random
import json
import math
import hashlib
import os
import numpy as np
from dataclasses import dataclass, asdict
from typing import Callable, List, Dict, Tuple, Optional, Union


# ===================== 多模态编码（可插拔自定义模型，自动降级） =====================
#
# 编码器选择优先级（图像 / 音频相同）：
#   1. 调用时显式传入的 encoder（callable 或注册名）
#   2. register_*_encoder 注册的默认自定义编码器
#   3. 内置真实模型（CLIP / Whisper，模型名可通过 set_*_model 换成自定义微调版）
#   4. 确定性伪 embedding（无外部模型时兜底，保证通路可运行、可复现）
#
# 自定义编码器契约：callable(path: str) -> 数值序列（list / tuple / numpy 数组），
# 长度任意（进入感官层前会重采样到 16 维）。

_CLIP_CACHE = None      # (model, processor)
_WHISPER_CACHE = None   # model

_CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
_WHISPER_MODEL_NAME = "base"

# 自定义编码器注册表：name -> callable(path) -> 数值序列
_CUSTOM_IMAGE_ENCODERS: Dict[str, Callable] = {}
_CUSTOM_AUDIO_ENCODERS: Dict[str, Callable] = {}
_DEFAULT_IMAGE_ENCODER: Optional[str] = None
_DEFAULT_AUDIO_ENCODER: Optional[str] = None


def register_image_encoder(fn: Callable, name: str = "custom",
                           default: bool = True) -> str:
    """注册自定义图像编码器。default=True 时设为全局默认（优先于内置 CLIP）。"""
    global _DEFAULT_IMAGE_ENCODER
    if not callable(fn):
        raise TypeError("encoder 必须是 callable(path) -> 数值序列")
    _CUSTOM_IMAGE_ENCODERS[name] = fn
    if default:
        _DEFAULT_IMAGE_ENCODER = name
    return name


def register_audio_encoder(fn: Callable, name: str = "custom",
                           default: bool = True) -> str:
    """注册自定义音频编码器。default=True 时设为全局默认（优先于内置 Whisper）。"""
    global _DEFAULT_AUDIO_ENCODER
    if not callable(fn):
        raise TypeError("encoder 必须是 callable(path) -> 数值序列")
    _CUSTOM_AUDIO_ENCODERS[name] = fn
    if default:
        _DEFAULT_AUDIO_ENCODER = name
    return name


def unregister_image_encoder(name: str) -> None:
    """注销图像编码器；若注销的是当前默认，回落到内置模型/伪 embedding 链。"""
    global _DEFAULT_IMAGE_ENCODER
    _CUSTOM_IMAGE_ENCODERS.pop(name, None)
    if _DEFAULT_IMAGE_ENCODER == name:
        _DEFAULT_IMAGE_ENCODER = None


def unregister_audio_encoder(name: str) -> None:
    """注销音频编码器；若注销的是当前默认，回落到内置模型/伪 embedding 链。"""
    global _DEFAULT_AUDIO_ENCODER
    _CUSTOM_AUDIO_ENCODERS.pop(name, None)
    if _DEFAULT_AUDIO_ENCODER == name:
        _DEFAULT_AUDIO_ENCODER = None


def list_encoders() -> Dict[str, Dict[str, object]]:
    """列出当前已注册的自定义编码器与内置模型名（便于调试/观测台展示）。"""
    return {
        "image": {"custom": sorted(_CUSTOM_IMAGE_ENCODERS),
                  "default": _DEFAULT_IMAGE_ENCODER,
                  "builtin_model": _CLIP_MODEL_NAME},
        "audio": {"custom": sorted(_CUSTOM_AUDIO_ENCODERS),
                  "default": _DEFAULT_AUDIO_ENCODER,
                  "builtin_model": _WHISPER_MODEL_NAME},
    }


def set_clip_model(model_name: str) -> None:
    """更换内置图像模型（如自定义微调版 CLIP 的 HF 名称或本地路径）。"""
    global _CLIP_CACHE, _CLIP_MODEL_NAME
    if model_name != _CLIP_MODEL_NAME:
        _CLIP_MODEL_NAME = model_name
        _CLIP_CACHE = None  # 重置缓存，下次编码时加载新模型


def set_whisper_model(model_name: str) -> None:
    """更换内置音频模型（如 "small"、"large-v3" 或本地微调模型路径）。"""
    global _WHISPER_CACHE, _WHISPER_MODEL_NAME
    if model_name != _WHISPER_MODEL_NAME:
        _WHISPER_MODEL_NAME = model_name
        _WHISPER_CACHE = None


def _resolve_encoder(registry: Dict[str, Callable], default_name: Optional[str],
                     encoder: Union[str, Callable, None]) -> Optional[Callable]:
    """解析本次调用应使用的自定义编码器，无则返回 None（走内置链）。"""
    if encoder is None:
        return registry.get(default_name) if default_name else None
    if callable(encoder):
        return encoder
    if encoder not in registry:
        raise KeyError(
            f"未注册的编码器 {encoder!r}，已注册：{sorted(registry) or '无'}")
    return registry[encoder]


def _coerce_embedding(vec, source: str) -> List[float]:
    """校验并转换自定义编码器输出为 List[float]。

    与内置模型的"静默降级"策略不同：自定义编码器输出非法属于调用方错误，
    直接抛出带上下文的异常，避免错误 embedding 静默污染记忆。
    """
    if hasattr(vec, "tolist"):  # numpy 数组 / torch 张量
        vec = vec.tolist()
    if not isinstance(vec, (list, tuple)):
        raise TypeError(
            f"自定义编码器 {source} 必须返回数值序列，实际返回 {type(vec).__name__}")
    out = []
    for v in vec:
        if not isinstance(v, (int, float)):
            raise TypeError(f"自定义编码器 {source} 返回了非数值元素：{v!r}")
        out.append(float(v))
    return out


def _file_pseudo_embedding(path: str, dim: int = 512) -> List[float]:
    """降级方案：文件内容的确定性哈希伪 embedding。

    不含语义信息，但同一文件在任何进程/平台上编码恒定，
    保证多模态通路在无外部模型时仍可运行、可复现。
    """
    with open(path, "rb") as f:
        data = f.read()
    seed = int.from_bytes(hashlib.md5(data).digest()[:8], "big")
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(dim)]


def encode_image(path: str, dim: int = 512,
                 encoder: Union[str, Callable, None] = None) -> List[float]:
    """图像 → embedding。优先级：自定义编码器 > CLIP（可换模型名）> 伪 embedding。"""
    fn = _resolve_encoder(_CUSTOM_IMAGE_ENCODERS, _DEFAULT_IMAGE_ENCODER, encoder)
    if fn is not None:
        return _coerce_embedding(fn(path), getattr(fn, "__name__", repr(fn)))
    global _CLIP_CACHE
    try:
        if _CLIP_CACHE is None:
            from transformers import CLIPModel, CLIPProcessor
            _CLIP_CACHE = (
                CLIPModel.from_pretrained(_CLIP_MODEL_NAME),
                CLIPProcessor.from_pretrained(_CLIP_MODEL_NAME),
            )
        from PIL import Image
        model, processor = _CLIP_CACHE
        inputs = processor(images=Image.open(path), return_tensors="pt")
        feats = model.get_image_features(**inputs)
        return feats[0].detach().cpu().numpy().tolist()
    except Exception:
        return _file_pseudo_embedding(path, dim)


def encode_audio(path: str, dim: int = 512,
                 encoder: Union[str, Callable, None] = None) -> List[float]:
    """音频 → embedding。优先级：自定义编码器 > Whisper（可换模型名）> 伪 embedding。"""
    fn = _resolve_encoder(_CUSTOM_AUDIO_ENCODERS, _DEFAULT_AUDIO_ENCODER, encoder)
    if fn is not None:
        return _coerce_embedding(fn(path), getattr(fn, "__name__", repr(fn)))
    global _WHISPER_CACHE
    try:
        if _WHISPER_CACHE is None:
            import whisper
            _WHISPER_CACHE = whisper.load_model(_WHISPER_MODEL_NAME)
        import numpy as np
        import torch
        audio = whisper.load_audio(path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(_WHISPER_CACHE.device)
        with torch.no_grad():
            feats = _WHISPER_CACHE.encoder(mel.unsqueeze(0))
        return feats.mean(dim=1)[0].detach().cpu().numpy().tolist()[:dim]
    except Exception:
        return _file_pseudo_embedding(path, dim)


# ===================== 基础数据结构 =====================

@dataclass
class Neuron:
    """脉冲神经元（简化 LIF 神经元模型，v2.1 增加不应期）"""
    id: int
    potential: float = 0.0
    threshold: float = 1.0
    decay: float = 0.92
    spike: bool = False  # 是否发放脉冲
    refractory: int = 0           # 剩余不应期（tick）
    refractory_period: int = 2    # 发放后静默时长

    def step(self, input_current: float):
        # 膜电位更新（不应期内电位照常泄漏，但不发放）
        self.potential = self.potential * self.decay + input_current
        self.spike = False
        if self.refractory > 0:
            self.refractory -= 1
            return
        if self.potential >= self.threshold:
            self.spike = True
            self.potential = 0.0  # 发放后重置
            self.refractory = self.refractory_period


@dataclass
class BrainMemory:
    """记忆单元"""
    content: str
    timestamp: float
    weight: float    # 记忆权重（强度）
    tag: str         # 记忆标签：sensory / emotion / event / culture / thought


@dataclass
class ThoughtItem:
    """思考空间中的一个念头（全局工作区条目，v5.0）。

    思考空间是"心智舞台"——当前被意识到的内容都在这里。
    每个念头有激活度，随时间自然衰减，被注意/联想时重新激活。
    """
    content: str           # 念头内容
    source: str = "internal"  # 来源: external / memory / internal / metacog
    activation: float = 1.0   # 激活度 [0,1]，每 tick 衰减
    birth_tick: int = 0


# ===================== 可学习投影（v4.0） =====================

class LearnableProjection:
    """可学习投影：任意维 embedding → n 维感官电流，替代线性插值重采样。

    解决实验 8 发现的两个通路局限：
      - 线性插值把稠密 iid embedding 扁平化（相邻维平均抹平对比度）；
      - abs 归一化丢失符号信息，抬高不相关对的基线相似度。

    机制：
      - 随机投影矩阵 W（n × in_dim，行向量单位化，Johnson-Lindenstrauss
        风格），对 embedding 做带符号投影，距离/对比度结构得以保留；
      - Oja 在线 PCA 规则训练（w ← w + η·y·(x − y·w)）：投影方向逐步
        对齐输入分布的主成分，方差大的语义方向获得更大表征带宽；
      - 保符号的中心化归一化：以均值为中心线性映射到 [0, 0.8]，
        维度间差异（对比度）等比例保留，不做 abs。

    用法：
        proj = LearnableProjection(in_dim=512, out_dim=16, seed=42)
        cur = proj.project(vec)          # 16 维感官电流
        proj.train(vec)                  # 在线学习（Oja 一步）
    """

    def __init__(self, in_dim: int, out_dim: int = 16,
                 seed: Optional[int] = None, lr: float = 0.01):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lr = lr
        self.train_steps = 0
        rng = random.Random(seed)
        # 随机投影 + 行单位化
        self.W: List[List[float]] = []
        for _ in range(out_dim):
            row = [rng.gauss(0.0, 1.0) for _ in range(in_dim)]
            norm = math.sqrt(sum(v * v for v in row)) or 1.0
            self.W.append([v / norm for v in row])

    def project_raw(self, vec: List[float]) -> List[float]:
        """带符号投影（未归一化）"""
        return [sum(wi * vi for wi, vi in zip(row, vec)) for row in self.W]

    def project(self, vec: List[float]) -> List[float]:
        """投影 → 保符号中心化归一化到 [0, 0.8]（对比度等比例保留）"""
        y = self.project_raw(vec)
        mean = sum(y) / len(y)
        dev = [v - mean for v in y]
        dmax = max(abs(v) for v in dev) or 1.0
        # 0.4 为中心，偏差等比例映射到 ±0.4：维度间差值严格线性保留
        return [self._clip01(0.4 + 0.4 * v / dmax) for v in dev]

    def train(self, vec: List[float], lr: Optional[float] = None):
        """Oja 在线 PCA 一步：投影方向向输入分布主成分收敛"""
        eta = lr if lr is not None else self.lr
        y = self.project_raw(vec)
        for row, yi in zip(self.W, y):
            for i in range(self.in_dim):
                row[i] += eta * yi * (vec[i] - yi * row[i])
            # 轻量再归一化，防权重漂移
            norm = math.sqrt(sum(v * v for v in row)) or 1.0
            for i in range(self.in_dim):
                row[i] /= norm
        self.train_steps += 1

    @staticmethod
    def _clip01(v: float) -> float:
        return max(0.0, min(0.8, v))


# ===================== 动作执行器（v4.1） =====================
#
# 执行器契约：callable(action: Dict) -> Dict，返回至少包含
#   success: bool   — 执行是否成功
#   reward:  float  — 执行结果奖励 [-1, 1]（回传给 reward_td 闭环学习）
#   detail:  str    — 人类可读的执行描述
# action 即 decide_action() 的结构化输出（verb/intensity/mood/recalled）。

def make_robot_executor(strictness: float = 0.3) -> Callable:
    """模拟机器人执行器：把动作动词映射为机器人指令并"执行"。

    verb → 指令：respond→语音+接近，acknowledge→点头示意，observe→原地扫描。
    动作强度低于 strictness 时执行器无法有效驱动（弱信号带不动电机），
    判定失败并给负奖励——强度门槛模拟真实执行器的最小驱动条件。
    全程确定性（无随机），保证实验可复现。
    """
    commands = {"respond": "SPEAK+APPROACH",
                "acknowledge": "NOD",
                "observe": "STAY+SCAN"}

    def executor(action: Dict) -> Dict:
        verb = action.get("verb", "observe")
        cmd = commands.get(verb, "STAY+SCAN")
        intensity = float(action.get("intensity", 0.0))
        if verb != "observe" and intensity < strictness:
            return {"success": False, "reward": -0.4,
                    "detail": f"机器人指令 {cmd} 执行失败：动作强度 "
                              f"{intensity:.2f} 低于驱动门槛 {strictness:.2f}"}
        reward = min(1.0, 0.4 + intensity) if verb != "observe" else 0.2
        return {"success": True, "reward": round(reward, 3),
                "detail": f"机器人执行 {cmd} 完成（强度 {intensity:.2f}）"}

    return executor


def make_api_executor(endpoint: str, timeout: float = 5.0,
                      verify_ssl: bool = True) -> Callable:
    """HTTP API 执行器（零依赖，urllib）：把结构化动作 POST 到外部服务。

    请求体：decide_action() 的完整动作 JSON。
    响应约定：HTTP 200 且 JSON 含 "reward" 字段时采用之，否则默认 +0.5；
    网络/协议错误 → success=False, reward=-0.3（失败也是学习信号）。

    verify_ssl: HTTPS 请求时是否验证服务器证书（默认 True，安全）。
                仅在对接自签名证书的内网服务时设为 False。
    """

    def executor(action: Dict) -> Dict:
        import urllib.request
        import ssl
        payload = json.dumps(action, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            endpoint, data=payload,
            headers={"Content-Type": "application/json"}, method="POST")
        ctx = None
        if endpoint.startswith("https://"):
            ctx = ssl.create_default_context() if verify_ssl \
                else ssl._create_unverified_context()
        try:
            with urllib.request.urlopen(req, timeout=timeout,
                                        context=ctx) as resp:
                body = resp.read().decode("utf-8")
            try:
                data = json.loads(body)
                if not isinstance(data, dict):
                    reward = 0.5
                else:
                    reward = float(data.get("reward", 0.5))
            except (ValueError, TypeError):
                reward = 0.5
            return {"success": True,
                    "reward": max(-1.0, min(1.0, reward)),
                    "detail": f"API {endpoint} 已执行 (HTTP 200)"}
        except Exception as e:
            return {"success": False, "reward": -0.3,
                    "detail": f"API {endpoint} 调用失败：{type(e).__name__}"}

    return executor


# ===================== 大脑实体 =====================

class AIBrainEntity:
    def __init__(self, brain_name: str, seed: Optional[int] = None,
                 record_history: bool = False,
                 sensation_seeking: float = 0.5,
                 habituation_rate: float = 0.3):
        if seed is not None:
            random.seed(seed)
        self.name = brain_name
        self.tick = 0  # 大脑运行时钟周期
        self.step_count = 0  # 网络步计数（STDP 的时间基准）

        # 1. 神经元集群：感知区、联想区、决策区
        self.sense_layer: List[Neuron] = [Neuron(i) for i in range(16)]
        self.assoc_layer: List[Neuron] = [Neuron(i + 16) for i in range(32)]
        self.decision_layer: List[Neuron] = [Neuron(i + 48) for i in range(8)]

        # 前馈突触权重 (前神经元id, 后神经元id) -> 连接强度
        self.synapse: Dict[Tuple[int, int], float] = {}
        self._init_synapse()

        # v2.1 循环突触（带 1 tick 延迟）：联想层侧向 + 决策层反馈
        self.recurrent_synapse: Dict[Tuple[int, int], float] = {}
        self._recurrent_out: Dict[int, List[int]] = {}
        self._recurrent_in: Dict[int, List[int]] = {}
        self._init_recurrent()
        self._pending_recurrent: Dict[int, float] = {}  # 下一 tick 到达的循环电流

        # v2.1 动力学参数
        self.noise_level = 0.06   # 自发背景噪声电流幅度
        self.settle_ticks = 2     # 每次感知后网络自由回响的 tick 数

        # v3.0 STDP 参数（脉冲时序依赖可塑性）
        self._last_spike: Dict[int, int] = {}   # 神经元 id -> 最近一次发放的网络步
        self.stdp_tau = 3.0       # STDP 指数时间窗（网络步）
        self.stdp_window = 50     # 超过此步数差不再更新
        self.ltp_amp = 2.0        # LTP 幅度系数（× 基础速率）
        self.ltd_amp = 2.4        # LTD 幅度系数（略大于 LTP，维持权重稳定）

        # v3.0 多巴胺样奖励信号（调制学习速率）
        self.dopamine = 0.0       # [-1, 1]，正奖励增强学习，负奖励抑制
        self.dopamine_decay = 0.95

        # v4.0 可学习投影（按输入维度惰性创建；None = 沿用线性插值重采样）
        self.use_projection = False
        self._projections: Dict[int, LearnableProjection] = {}
        self.projection_train_on_input = True  # 感知时是否顺带在线训练投影

        # v4.0 奖励预测误差（RPE / TD 学习）
        self.value_estimate = 0.0   # 状态价值 V：对奖励的预期
        self.td_alpha = 0.2         # 价值学习速率
        self.rpe_history: List[float] = []  # 历次奖励预测误差（实验观测用）

        # v4.4 TD(λ) 资格迹：信用分配跨 tick 反向传播
        self.eligibility: Dict[str, float] = {}   # 状态(刺激内容) -> 资格迹强度
        self.state_values: Dict[str, float] = {}  # 状态 -> 价值 V(s)
        self.td_gamma = 0.9         # 折扣因子（未来奖励的现值）
        self.td_lambda = 0.8        # 资格迹衰减率 λ
        self.trace_threshold = 0.01  # 低于此强度的迹被剪除
        self._last_state: Optional[str] = None

        # v4.0 群体动力学：实体所属世代（BrainSwarm.reproduce 会递增）
        self.generation = 1

        # v4.1 动作执行器：verb -> callable(action) -> {success, reward, detail}
        self.executors: Dict[str, Callable] = {}
        self.default_executor: Optional[Callable] = None

        # v4.5 执行器技能学习：每个 verb 独立价值估计 Q(verb)
        self.verb_values: Dict[str, float] = {
            cfg["verb"]: 0.0 for cfg in self.ACTION_SPACE.values()}
        self.skill_alpha = 0.3      # 技能价值学习速率
        self.skill_epsilon = 0.15   # ε-greedy 探索率
        self.skill_temperature = 0.5  # softmax 温度

        # v4.9 好奇驱动探索：新奇度反向调制注意与 ε
        self.novelty = 0.0          # 最近一次刺激的新奇度 [0,1]
        self.novelty_history: List[float] = []
        self.curiosity_drive = 0.25  # 新奇度 → 好奇心增益系数
        # v4.9.1 人格差异：寻求刺激倾向 [0,1]
        #   高值（如0.8）：天生好奇，基础好奇心高，新奇增益强，探索欲旺盛
        #   低值（如0.2）：保守谨慎，偏好熟悉，新奇增益弱，更倾向利用已知
        self.sensation_seeking = self._clip(sensation_seeking, 0.0, 1.0)
        # v4.9.1 习惯化：反复暴露同一刺激后新奇度衰减速度 [0,1]
        #   0 = 不衰减（每次见都像第一次），1 = 第二次就完全习惯
        #   公式：effective_novelty = novelty / (1 + exposure_count × habituation_rate)
        self.habituation_rate = self._clip(habituation_rate, 0.0, 1.0)
        self.exposure_count: Dict[str, int] = {}  # 刺激内容 → 累计暴露次数

        # 2. 记忆系统（三级结构）
        self.sensory_buffer: List[str] = []   # 瞬时感官缓存
        self.short_memory: List[BrainMemory] = []
        self.long_memory: List[BrainMemory] = []

        # v4.7 情景记忆时间索引：每次感知记一条情景（何时 + 与何事共现）
        self.episodes: List[Dict] = []   # {tick, content, context:[共现内容]}

        # v5.0 思考空间（全局工作区）：当前被意识到的念头
        # 容量遵循米勒定律 7±2；激活度逐 tick 衰减，低于 0.05 退出意识
        self.thought_space: List[ThoughtItem] = []
        self.thought_capacity = 9       # 米勒定律上限
        self.thought_decay = 0.9        # 每 tick 激活度衰减系数
        self.thought_salience = 0.35    # 念头固化进 STM 的激活度阈值
        # v5.0 元认知日志（思考感官 introspect 的记录）
        self.metacog_log: List[Dict] = []

        # v4.8 睡眠-清醒节律：离线重放固化 + 突触稳态缩放（SHY）
        self.sleep_replay_gain = 0.15       # 每次重放的 STM 权重增益
        self.sleep_downscale = 0.95         # 突触等比下调系数（保留相对差异）
        self.synapse_prune_threshold = 0.08  # 前馈突触剪除下限
        self.recurrent_prune_threshold = 0.04  # 循环突触剪除下限
        self.max_stm = 20
        self.max_ltm = 500
        self.stm_consolidate_threshold = 0.55   # STM 权重超过此值固化进 LTM
        self.forget_threshold = 0.05            # 权重低于此值被遗忘

        # 3. 情绪内核（动态变量）
        self.emotion = {
            "calm": 0.5,
            "curiosity": 0.3,
            "stress": 0.0,
            "pleasure": 0.2,
        }

        # 4. 注意力权重（受情绪调制）
        self.attention_factor = 0.6

        # 5. 学习开关与基础速率
        self.hebbian_enabled = True   # 可塑性总开关（STDP 走此开关）
        self.hebbian_rate = 0.01      # 基础学习速率（被多巴胺调制）

        # 6. 历史记录（实验/可视化用）
        self.record_history = record_history
        # 最近一次外部刺激步的三层脉冲 id（history 记录刺激响应而非回响残态）
        self._stim_spikes: Optional[Tuple[List[int], List[int], List[int]]] = None
        # 7. 脉冲思考链追踪（thought_chain 用；None 表示不追踪）
        self._step_trace: Optional[
            List[Tuple[List[int], List[int], List[int]]]] = None
        self.history: Dict[str, list] = {
            "tick": [],
            "sense_spikes": [],    # 每 tick 感官层脉冲的神经元 id 列表
            "assoc_spikes": [],
            "decision_spikes": [],
            "spike_rate": [],
            "emotion": [],         # dict 快照
            "attention": [],
            "stm_size": [],
            "ltm_size": [],
            "synapse_mean": [],
        }

    # ------------------ 初始化 ------------------

    def _init_synapse(self):
        """初始化随机前馈突触连接，并建立双向邻接表（STDP 查询用）"""
        for sn in self.sense_layer:
            for an in self.assoc_layer:
                self.synapse[(sn.id, an.id)] = random.uniform(0.1, 0.6)
        for an in self.assoc_layer:
            for dn in self.decision_layer:
                self.synapse[(an.id, dn.id)] = random.uniform(0.1, 0.6)
        self._ff_out: Dict[int, List[int]] = {}
        self._ff_in: Dict[int, List[int]] = {}
        for (pre, post) in self.synapse:
            self._ff_out.setdefault(pre, []).append(post)
            self._ff_in.setdefault(post, []).append(pre)

    def _init_recurrent(self):
        """初始化循环连接：联想层侧向（每神经元4条）+ 决策层反馈（每神经元6条）"""
        for an in self.assoc_layer:
            for t in random.sample(self.assoc_layer, 4):
                if t.id != an.id:
                    self.recurrent_synapse[(an.id, t.id)] = random.uniform(0.05, 0.2)
        for dn in self.decision_layer:
            for t in random.sample(self.assoc_layer, 6):
                self.recurrent_synapse[(dn.id, t.id)] = random.uniform(0.05, 0.15)
        # 双向邻接表（STDP 查询用）
        for (pre, post) in self.recurrent_synapse:
            self._recurrent_out.setdefault(pre, []).append(post)
            self._recurrent_in.setdefault(post, []).append(pre)

    # ------------------ 感官编码 ------------------

    def _str_to_current(self, text: str) -> List[float]:
        """文本输入转为神经元输入电流（感官编码）。

        使用 MD5 派生确定性种子：同一文本在任何进程、任何平台上编码恒定，
        保证实验跨次运行可复现（Python 内置 hash() 每次进程随机加盐，不可用）。
        """
        seed = int.from_bytes(hashlib.md5(text.encode("utf-8")).digest()[:8], "big")
        rng = random.Random(seed)
        return [rng.uniform(0, 0.8) for _ in range(len(self.sense_layer))]

    @staticmethod
    def _normalize_vector(vec: List[float], n: int) -> List[float]:
        """将任意长度 embedding 线性插值重采样到 n 维，并归一化到 [0, 0.8]"""
        if not vec:
            return [0.0] * n
        if len(vec) == 1:
            sampled = vec * n
        else:
            sampled = []
            for i in range(n):
                pos = i * (len(vec) - 1) / max(n - 1, 1)
                lo = int(pos)
                hi = min(lo + 1, len(vec) - 1)
                frac = pos - lo
                sampled.append(vec[lo] * (1 - frac) + vec[hi] * frac)
        vmax = max(abs(v) for v in sampled) or 1.0
        return [abs(v) / vmax * 0.8 for v in sampled]

    # ------------------ 多模态接口（v3.0，v3.1 支持自定义编码器） ------------------

    def perceive_image(self, path: str, label: str = "",
                       encoder: Union[str, Callable, None] = None) -> str:
        """感知一张图片：自定义编码器 / CLIP（或降级伪 embedding）→ 感官层。

        encoder 可传入 callable 或 register_image_encoder 注册的名字；
        不传则走全局默认（注册的默认自定义编码器 → 内置 CLIP → 伪 embedding）。
        """
        vec = encode_image(path, encoder=encoder)
        tag = label or f"<image:{os.path.basename(path)}>"
        return self.sensory_input_vector(vec, label=tag)

    def perceive_audio(self, path: str, label: str = "",
                       encoder: Union[str, Callable, None] = None) -> str:
        """感知一段音频：自定义编码器 / Whisper（或降级伪 embedding）→ 感官层"""
        vec = encode_audio(path, encoder=encoder)
        tag = label or f"<audio:{os.path.basename(path)}>"
        return self.sensory_input_vector(vec, label=tag)

    # ------------------ 奖励信号（v3.0 多巴胺样调制） ------------------

    def reward(self, value: float) -> float:
        """给予奖励/惩罚信号（[-1, 1]）。

        多巴胺水平升高 → 后续 STDP 学习速率最高翻倍；
        负奖励（惩罚）→ 学习速率下降直至完全抑制。
        同时影响情绪：奖励带来愉悦，惩罚带来压力。
        """
        value = self._clip(value, -1.0, 1.0)
        self.dopamine = self._clip(self.dopamine + value, -1.0, 1.0)
        self.emotion["pleasure"] = self._clip(
            self.emotion["pleasure"] + 0.3 * value)
        if value < 0:
            self.emotion["stress"] = self._clip(
                self.emotion["stress"] - 0.2 * value)
        return self.dopamine

    def reward_td(self, reward: float) -> Dict[str, float]:
        """奖励预测误差（RPE / TD 误差）驱动的多巴胺更新（v4.0）。

        与 reward() 的直接注入不同：多巴胺响应的是"意外"——
            RPE δ = 实际奖励 r − 预期价值 V
            V ← V + α·δ          （价值函数学习）
            多巴胺 ← 多巴胺 + δ    （误差驱动，而非奖励驱动）

        学习效果：同一奖励反复出现 → V 收敛到 r → δ→0 → 多巴胺不再波动
        （奖励被完全预测，经典 TD 现象）；奖励突然变化 → 重新出现大 RPE。
        返回本步诊断量。
        """
        reward = self._clip(reward, -1.0, 1.0)
        rpe = reward - self.value_estimate
        self.value_estimate += self.td_alpha * rpe
        self.dopamine = self._clip(self.dopamine + rpe, -1.0, 1.0)
        self.emotion["pleasure"] = self._clip(
            self.emotion["pleasure"] + 0.3 * rpe)
        if rpe < 0:
            self.emotion["stress"] = self._clip(
                self.emotion["stress"] - 0.2 * rpe)
        self.rpe_history.append(rpe)
        return {"reward": reward, "rpe": rpe,
                "value_estimate": self.value_estimate,
                "dopamine": self.dopamine}

    # ------------------ TD(λ) 资格迹（v4.4） ------------------

    def _mark_state(self, state: str):
        """感知到一个状态：所有旧迹按 γλ 衰减，当前状态迹重置为 1（替换迹）"""
        decay = self.td_gamma * self.td_lambda
        self.eligibility = {s: e * decay for s, e in self.eligibility.items()}
        self.eligibility = {s: e for s, e in self.eligibility.items()
                            if e >= self.trace_threshold}
        self.eligibility[state] = 1.0
        self._last_state = state

    def reward_lambda(self, reward: float) -> Dict:
        """TD(λ) 奖励：RPE 按资格迹反向分配给近期所有状态（v4.4）。

        与 reward_td() 的单状态 V 不同，这里维护逐状态价值 V(s)：
            δ = r − V(s_t)              （当前状态的预测误差）
            V(s) ← V(s) + α·δ·e(s)      （所有带迹状态按迹强度分摊信用）

        效果：反复经历"线索A → 线索B → 奖励"后，信用沿迹反向传播，
        较早的线索也获得价值（多巴胺时序迁移，Schultz 经典实验现象）。
        返回本步诊断量（含各状态获得的信用量）。
        """
        reward = self._clip(reward, -1.0, 1.0)
        cur_v = self.state_values.get(self._last_state, 0.0)
        rpe = reward - cur_v
        credited = {}
        for s, e in self.eligibility.items():
            dv = self.td_alpha * rpe * e
            self.state_values[s] = self.state_values.get(s, 0.0) + dv
            credited[s] = dv
        # 同步全局 V（取当前状态），保持与 reward_td 诊断口径一致
        self.value_estimate = self.state_values.get(self._last_state, 0.0)
        self.dopamine = self._clip(self.dopamine + rpe, -1.0, 1.0)
        self.emotion["pleasure"] = self._clip(
            self.emotion["pleasure"] + 0.3 * rpe)
        if rpe < 0:
            self.emotion["stress"] = self._clip(
                self.emotion["stress"] - 0.2 * rpe)
        self.rpe_history.append(rpe)
        return {"reward": reward, "rpe": rpe,
                "value_estimate": self.value_estimate,
                "dopamine": self.dopamine,
                "credited": credited,
                "state_values": dict(self.state_values)}

    def _learning_rate(self) -> float:
        """多巴胺调制后的有效学习速率"""
        return self.hebbian_rate * max(0.0, 1.0 + self.dopamine)

    # ------------------ 网络动力学（v2.1 核心） ------------------

    def _network_step(self, external: Optional[List[float]] = None):
        """推进一个网络步：前馈当步级联，循环信号延迟一步到达，全程噪声。
        每层发放后立即执行 STDP 更新，保证脉冲时序进入可塑性计算。

        external: 感官层外部输入电流（None 表示无外部刺激，自由演化）。
        """
        self.step_count += 1
        now = self.step_count

        # 第一层：感官神经元（外部输入 + 背景噪声）
        for idx, neuron in enumerate(self.sense_layer):
            cur = (external[idx] if external else 0.0) \
                + random.uniform(0, self.noise_level)
            neuron.step(cur)
        if self.hebbian_enabled:
            self._stdp_update(
                [n.id for n in self.sense_layer if n.spike], now)

        # 第二层：联想层（前馈即时 + 上 tick 循环延迟信号 + 噪声）
        pending_next: Dict[int, float] = {}
        for assoc_n in self.assoc_layer:
            total_in = sum(
                self.synapse.get((sense_n.id, assoc_n.id), 0.0)
                for sense_n in self.sense_layer if sense_n.spike
            )
            total_in += self._pending_recurrent.get(assoc_n.id, 0.0)
            total_in += random.uniform(0, self.noise_level)
            assoc_n.step(total_in)
            if assoc_n.spike:
                for post in self._recurrent_out.get(assoc_n.id, ()):
                    w = self.recurrent_synapse[(assoc_n.id, post)]
                    pending_next[post] = pending_next.get(post, 0.0) + w
        if self.hebbian_enabled:
            self._stdp_update(
                [n.id for n in self.assoc_layer if n.spike], now)

        # 第三层：决策层（前馈即时 + 噪声；发放经反馈通路延迟回传联想层）
        for dec_n in self.decision_layer:
            total_in = sum(
                self.synapse.get((assoc_n.id, dec_n.id), 0.0)
                for assoc_n in self.assoc_layer if assoc_n.spike
            )
            total_in += random.uniform(0, self.noise_level)
            dec_n.step(total_in)
            if dec_n.spike:
                for post in self._recurrent_out.get(dec_n.id, ()):
                    w = self.recurrent_synapse[(dec_n.id, post)]
                    pending_next[post] = pending_next.get(post, 0.0) + w
        if self.hebbian_enabled:
            self._stdp_update(
                [n.id for n in self.decision_layer if n.spike], now)

        self._pending_recurrent = pending_next
        # 多巴胺随时间自然代谢
        self.dopamine *= self.dopamine_decay
        # 脉冲思考链追踪：记录本网络步的三层脉冲
        if self._step_trace is not None:
            self._step_trace.append((
                [n.id for n in self.sense_layer if n.spike],
                [n.id for n in self.assoc_layer if n.spike],
                [n.id for n in self.decision_layer if n.spike],
            ))

    def spike_counts(self) -> Tuple[int, int, int]:
        """当前 tick 三层各自的脉冲数"""
        return (sum(1 for n in self.sense_layer if n.spike),
                sum(1 for n in self.assoc_layer if n.spike),
                sum(1 for n in self.decision_layer if n.spike))

    # ------------------ 主入口：感知 -> 认知 ------------------

    def sensory_input(self, data: str) -> str:
        """外部文本感官输入入口，返回大脑的行为输出"""
        currents = self._str_to_current(data)
        return self._perceive(data, currents, tag="sensory")

    def sensory_input_vector(self, vec: List[float], label: str = "") -> str:
        """多模态接口：接收图像/音频 embedding 等任意数值向量。

        v4.0：use_projection=True 时经可学习投影进入感官层（保对比度），
        并顺带对该 embedding 做一步 Oja 在线训练（可关）；否则沿用
        线性插值重采样（向后兼容）。
        """
        if self.use_projection:
            proj = self._get_projection(len(vec))
            if self.projection_train_on_input:
                proj.train(vec)
            currents = proj.project(vec)
        else:
            currents = self._normalize_vector(vec, len(self.sense_layer))
        content = label or f"<vector[{len(vec)}]>"
        return self._perceive(content, currents, tag="sensory")

    # ------------------ 可学习投影（v4.0） ------------------

    def _get_projection(self, in_dim: int) -> LearnableProjection:
        """按输入维度惰性创建投影（每种 embedding 维度各一份，独立学习）"""
        if in_dim not in self._projections:
            self._projections[in_dim] = LearnableProjection(
                in_dim=in_dim, out_dim=len(self.sense_layer), seed=42)
        return self._projections[in_dim]

    def enable_projection(self, on: bool = True):
        """开关可学习投影通路。开启后 sensory_input_vector 走投影。"""
        self.use_projection = on
        return self.use_projection

    # ------------------ 思考空间（v5.0 全局工作区） ------------------

    def _push_thought(self, content: str, source: str = "internal",
                      activation: float = 1.0):
        """念头进入思考空间。同内容念头重新激活而非重复入栈；
        超出容量时挤出激活度最低的念头。"""
        for t in self.thought_space:
            if t.content == content:
                t.activation = 1.0
                t.source = source
                return
        self.thought_space.append(
            ThoughtItem(content, source, self._clip(activation), self.tick))
        if len(self.thought_space) > self.thought_capacity:
            weakest = min(self.thought_space, key=lambda t: t.activation)
            self.thought_space.remove(weakest)

    def _decay_thoughts(self):
        """念头激活度逐 tick 衰减，低于 0.05 退出意识"""
        for t in self.thought_space:
            t.activation *= self.thought_decay
        self.thought_space[:] = [
            t for t in self.thought_space if t.activation >= 0.05]

    def top_thought(self) -> Optional[ThoughtItem]:
        """当前激活度最高的念头（意识焦点）；思考空间为空返回 None"""
        if not self.thought_space:
            return None
        return max(self.thought_space, key=lambda t: t.activation)

    def _perceive(self, content: str, input_currents: List[float], tag: str) -> str:
        """统一的感知-认知流水线"""
        self.tick += 1
        self.sensory_buffer.append(content)
        if len(self.sensory_buffer) > 8:
            self.sensory_buffer.pop(0)
        self._mark_state(content)   # v4.4：感知即状态访问，刷新资格迹
        # v4.7：记一条情景——此刻 tick + 感官缓存里共现的其他内容
        self.episodes.append({
            "tick": self.tick, "content": content,
            "context": [c for c in self.sensory_buffer if c != content]})
        self._assess_novelty(content)   # v4.9：新奇度 → 当 tick 注意捕获
        # v5.0：外部感知进入思考空间（被意识到），旧念头随时间衰减
        self._push_thought(content, source="external")
        self._decay_thoughts()

        # 网络动力学：外部刺激经注意力调制注入，随后自由回响 settle_ticks
        external = [c * self.attention_factor for c in input_currents]
        self._network_step(external)
        # 快照刺激响应步的脉冲（history 应记录对刺激的响应，而非回响后的残态）
        self._stim_spikes = (
            [n.id for n in self.sense_layer if n.spike],
            [n.id for n in self.assoc_layer if n.spike],
            [n.id for n in self.decision_layer if n.spike],
        )
        for _ in range(self.settle_ticks):
            self._network_step()

        # 脉冲活动影响情绪 -> 情绪反过来调制注意力
        # （情绪由刺激响应步的发放率驱动，见 _update_emotion 文档）
        stim_rate = sum(len(ids) for ids in self._stim_spikes) / 56
        self._update_emotion(stim_rate)
        self._modulate_attention()

        # 写入短期记忆（可能触发固化进 LTM / 遗忘）
        self._write_stm(content, tag=tag)

        # 记录历史
        if self.record_history:
            self._record()

        # 思考 + 输出决策
        return self.cognition(content)

    def free_run(self, ticks: int = 1) -> List[Tuple[int, int, int]]:
        """无外部输入自由演化：观察刺激后的回响衰减或静息态自发活动。

        返回每个 tick 的三层脉冲计数列表。
        """
        trace = []
        for _ in range(ticks):
            self.tick += 1
            self._network_step()
            self._update_emotion()
            self._modulate_attention()
            if self.record_history:
                self._record()
            trace.append(self.spike_counts())
        return trace

    # ------------------ STDP（v3.0 脉冲时序依赖可塑性） ------------------

    def _stdp_update(self, spiking: List[int], now: int):
        """STDP 更新：对本次发放的每个神经元——

        LTP：其突触前神经元先于（或同步于）它发放 → 突触增强，
              幅度随时间差指数衰减：Δw = A+ · exp(-Δt/τ)
        LTD：其突触后神经元先于它发放 → 突触减弱：
              Δw = -A- · exp(-Δt/τ)
        速率被多巴胺信号调制（奖励增强学习，惩罚抑制学习）。
        覆盖前馈与循环通路。
        """
        if not spiking:
            return
        lr = self._learning_rate()
        a_plus = self.ltp_amp * lr
        a_minus = self.ltd_amp * lr
        tau = self.stdp_tau
        win = self.stdp_window
        last = self._last_spike

        for j in spiking:
            # LTP：incoming（前馈 + 循环）
            for pre in self._ff_in.get(j, ()):
                dt = now - last.get(pre, -10 ** 9)
                if 0 <= dt < win:
                    k = (pre, j)
                    self.synapse[k] = min(
                        1.0, self.synapse[k] + a_plus * math.exp(-dt / tau))
            for pre in self._recurrent_in.get(j, ()):
                dt = now - last.get(pre, -10 ** 9)
                if 0 <= dt < win:
                    k = (pre, j)
                    self.recurrent_synapse[k] = min(
                        0.5, self.recurrent_synapse[k]
                        + a_plus * math.exp(-dt / tau))
            # LTD：outgoing（前馈 + 循环），post 严格早于 pre 发放
            for post in self._ff_out.get(j, ()):
                dt = now - last.get(post, -10 ** 9)
                if 0 < dt < win:
                    k = (j, post)
                    self.synapse[k] = max(
                        0.0, self.synapse[k] - a_minus * math.exp(-dt / tau))
            for post in self._recurrent_out.get(j, ()):
                dt = now - last.get(post, -10 ** 9)
                if 0 < dt < win:
                    k = (j, post)
                    self.recurrent_synapse[k] = max(
                        0.0, self.recurrent_synapse[k]
                        - a_minus * math.exp(-dt / tau))
            last[j] = now

    # ------------------ 情绪与注意力 ------------------

    def _update_emotion(self, spike_rate: Optional[float] = None):
        """神经元脉冲活动调控情绪。

        spike_rate: 显式给出的网络发放率；缺省时按当前神经元状态计算。
        感知 tick 应传入刺激响应步的发放率——情绪响应的是刺激诱发的
        活动，而非自由回响衰减后的残态。
        """
        if spike_rate is None:
            all_neurons = (self.sense_layer + self.assoc_layer
                           + self.decision_layer)
            spike_count = sum(1 for n in all_neurons if n.spike)
            spike_rate = spike_count / len(all_neurons)
        rate = spike_rate

        self.emotion["curiosity"] = self._clip(
            self.emotion["curiosity"] + (rate - 0.3) * 0.05)
        self.emotion["stress"] = self._clip(
            self.emotion["stress"] + (rate - 0.5) * 0.04)
        self.emotion["pleasure"] = self._clip(
            self.emotion["pleasure"] + (0.35 - abs(rate - 0.35)) * 0.01)
        # 情绪自然衰减
        for k in self.emotion:
            if k != "calm":
                self.emotion[k] *= 0.98
        self.emotion["calm"] = self._clip(
            1.0 - (self.emotion["stress"] + self.emotion["curiosity"]) / 2)

    def _modulate_attention(self):
        """情绪调制注意力：好奇心提高注意，压力降低注意"""
        self.attention_factor = self._clip(
            0.6 + self.emotion["curiosity"] * 0.25 - self.emotion["stress"] * 0.3,
            0.1, 1.0)

    def _assess_novelty(self, content: str) -> float:
        """新奇度评估（v4.9.1）：记忆未命中率为主，|RPE| 为辅，含习惯化与人格。

        1. 认知新奇：对每个关键词试探回忆，未命中比例 = miss
        2. 意外新奇：最近 |RPE|（意料之外的事也是新奇）
        3. 习惯化：同一刺激反复暴露，新奇度按 1/(1+n×rate) 折扣
        4. 人格调制：sensation_seeking 调节好奇心增益强度
        副作用：新奇 → 好奇心上升 → 当 tick 注意捕获；熟悉 → 好奇心回落。
        返回新奇度 [0,1]。
        """
        tokens = [t for t in
                  content.replace("，", " ").replace("。", " ").split()
                  if len(t) >= 2]
        if tokens:
            hits = sum(1 for t in tokens if self.recall(t, top_k=1))
            miss = 1.0 - hits / len(tokens)
        else:
            miss = 0.5
        rpe_part = (min(1.0, abs(self.rpe_history[-1]))
                    if self.rpe_history else 0.0)
        raw_nov = self._clip(0.7 * miss + 0.3 * rpe_part)

        # 习惯化：累计暴露次数折扣新奇度
        exposures = self.exposure_count.get(content, 0)
        self.exposure_count[content] = exposures + 1
        if self.habituation_rate > 0 and exposures > 0:
            habit_factor = 1.0 / (1.0 + exposures * self.habituation_rate)
        else:
            habit_factor = 1.0
        nov = self._clip(raw_nov * habit_factor)
        self.novelty = nov
        self.novelty_history.append(nov)

        # 人格调制：寻求刺激者好奇心增益更强，保守者更弱
        # curiosity_gain 范围 [0.5×, 2.0×]，以 0.5 为中心映射
        personality_gain = 0.5 + self.sensation_seeking * 1.5
        # 好奇驱动：新奇度偏离 0.5 多少，好奇心就增减多少
        self.emotion["curiosity"] = self._clip(
            self.emotion["curiosity"]
            + self.curiosity_drive * personality_gain * (nov - 0.5) * 2)
        self._modulate_attention()      # 当 tick 生效：注意捕获
        return nov

    def effective_epsilon(self) -> float:
        """新奇度+人格调制后的探索率（v4.9.1）。

        基础：ε_eff = ε × (0.5 + novelty)，新奇时 1.5ε，熟悉时 0.5ε。
        人格：sensation_seeking 高者整体探索率上浮，低者下浮。
        """
        base = self.skill_epsilon * (0.5 + self.novelty)
        # 人格探索偏移：高 SSS +0.05，低 SSS -0.05
        personality_shift = (self.sensation_seeking - 0.5) * 0.1
        return self._clip(base + personality_shift, 0.0, 1.0)

    @staticmethod
    def _clip(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, v))

    # ------------------ 记忆系统 ------------------

    def _write_stm(self, content: str, tag: str = "sensory"):
        """写入短期记忆；容量满时：强者固化进 LTM，弱者自然遗忘"""
        mem = BrainMemory(
            content=content,
            timestamp=time.time(),
            weight=self.attention_factor,
            tag=tag,
        )
        self.short_memory.append(mem)

        if len(self.short_memory) > self.max_stm:
            weakest = min(self.short_memory, key=lambda m: m.weight)
            self.short_memory.remove(weakest)
            if weakest.weight >= self.stm_consolidate_threshold:
                self._consolidate_to_ltm(weakest)
            # 否则被自然遗忘

    def _consolidate_to_ltm(self, mem: BrainMemory):
        """短期记忆固化进长期记忆；重复内容则强化已有记忆"""
        for old in self.long_memory:
            if old.content == mem.content:
                old.weight = self._clip(old.weight + 0.15)
                old.timestamp = mem.timestamp
                return
        mem.tag = "event"
        self.long_memory.append(mem)
        if len(self.long_memory) > self.max_ltm:
            self.long_memory.sort(key=lambda m: m.weight)
            self.long_memory.pop(0)  # 遗忘最弱的长期记忆

    def decay_memory(self, factor: float = 0.995):
        """记忆随时间自然衰减（模拟睡眠/时间流逝），低于阈值被遗忘"""
        for store in (self.short_memory, self.long_memory):
            for m in store:
                m.weight *= factor
            store[:] = [m for m in store if m.weight >= self.forget_threshold]

    def sleep(self, cycles: int = 1) -> Dict:
        """睡眠-清醒节律（v4.8）：离线记忆重放 + 突触稳态缩放。

        1. 重放固化：STM 每条记忆离线重放 cycles 次（权重 += 增益/次），
           达固化阈值者当即转入 LTM——白天只出现一两次的弱刺激，
           也能在睡眠中完成固化（海马→新皮层重放）；
        2. 突触稳态缩放（SHY 假说）：全部突触 ×0.95 等比下调，
           低于剪除下限的连接删除——等比缩放保留习得连接间的
           相对差异（选择性不丢失），只清除噪声弱连接；
        3. 清洗：资格迹清零、多巴胺归零、压力衰减、平静恢复。

        返回本轮睡眠报告。
        """
        replayed, consolidated = 0, 0
        pruned = 0
        for _ in range(cycles):
            for mem in list(self.short_memory):
                mem.weight = self._clip(mem.weight + self.sleep_replay_gain)
                replayed += 1
                if mem.weight >= self.stm_consolidate_threshold:
                    self.short_memory.remove(mem)
                    self._consolidate_to_ltm(mem)
                    consolidated += 1
            # 突触稳态缩放随睡眠周期逐轮进行（SHY：越睡越"轻"）
            for table, floor in ((self.synapse, self.synapse_prune_threshold),
                                 (self.recurrent_synapse,
                                  self.recurrent_prune_threshold)):
                for key in list(table):
                    table[key] *= self.sleep_downscale
                    if table[key] < floor:
                        del table[key]
                        pruned += 1

        stress_before = self.emotion["stress"]
        self.eligibility.clear()           # 资格迹在睡眠中洗脱
        self.dopamine = 0.0
        self.emotion["stress"] *= 0.5 ** cycles
        self.emotion["calm"] = self._clip(
            self.emotion["calm"] + 0.2 * cycles)
        return {"cycles": cycles, "replayed": replayed,
                "consolidated": consolidated,
                "pruned_synapses": pruned,
                "synapses": len(self.synapse),
                "recurrent_synapses": len(self.recurrent_synapse),
                "stress_before": round(stress_before, 3),
                "stress_after": round(self.emotion["stress"], 3)}

    def recall(self, keyword: str, top_k: int = 3,
               exclude_latest: bool = True) -> List[BrainMemory]:
        """按关键词联想回忆（LTM 优先，按权重排序），回忆会强化记忆（再巩固）。

        exclude_latest=True 时排除最近一次写入的 STM，避免"回声记忆"——
        即刚听到的内容立即被当作联想结果返回。
        """
        stm = self.short_memory[:-1] if exclude_latest else self.short_memory
        # 去重：同一内容可能在 LTM 与 STM 各有一份，优先保留 LTM 副本——
        # LTM 是规范存储，再巩固应作用于它而非转瞬即逝的 STM 重复件
        seen, unique = set(), []
        for m in self.long_memory + stm:
            if keyword in m.content and m.content not in seen:
                seen.add(m.content)
                unique.append(m)
        unique.sort(key=lambda m: m.weight, reverse=True)
        for m in unique[:top_k]:
            m.weight = self._clip(m.weight + 0.05)  # 回忆强化
        # v5.0：被回忆起的记忆进入思考空间（进入意识）
        for m in unique[:top_k]:
            self._push_thought(m.content, source="memory", activation=0.8)
        return unique[:top_k]

    # ------------------ 情景记忆时间索引（v4.7） ------------------

    def episodic_trace(self, keyword: str) -> List[Dict]:
        """情景轨迹：按时间顺序返回所有含 keyword 的情景条目"""
        return [ep for ep in self.episodes if keyword in ep["content"]]

    def _episodes_relative(self, keyword: str,
                           direction: str) -> Dict:
        """以"最近一次 keyword 情景"为锚点，取之前/之后的事件序列"""
        trace = self.episodic_trace(keyword)
        if not trace:
            return {"anchor": None, "events": []}
        anchor = trace[-1]                     # "上次" = 最近一次
        if direction == "after":
            evs = [ep for ep in self.episodes if ep["tick"] > anchor["tick"]]
        else:
            evs = [ep for ep in self.episodes if ep["tick"] < anchor["tick"]]
        return {"anchor": anchor,
                "events": [{**ep, "delta": ep["tick"] - anchor["tick"]}
                           for ep in evs]}

    def events_after(self, keyword: str) -> Dict:
        """时间推理："上次 keyword 之后发生了什么"（含 tick 间隔 delta）"""
        return self._episodes_relative(keyword, "after")

    def events_before(self, keyword: str) -> Dict:
        """时间推理："上次 keyword 之前经历过什么"（delta 为负）"""
        return self._episodes_relative(keyword, "before")

    # ------------------ 思考体系（v5.0） ------------------

    def think(self, content: Optional[str] = None, ticks: int = 1) -> Dict:
        """主动思考：把念头重新编码为电流注入网络，形成"自言自语"闭环。

        content 缺省时取思考空间中激活度最高的念头继续想（意识焦点）。
        思考活动诱发联想（recall），联想起的记忆也进入思考空间；
        结束时高激活念头（≥ thought_salience）固化进 STM（tag="thought"）
        ——思考记忆：想多了就记住了。
        """
        if content is None:
            top = self.top_thought()
            if top is None:
                return {"thought": None, "spikes": self.spike_counts(),
                        "recalled": [], "consolidated": [],
                        "thought_space": []}
            content = top.content
        self._push_thought(content, source="internal")
        self.tick += 1
        # 内部思考强度低于外部感知（0.6 倍），网络活动驱动联想
        currents = [c * 0.6 for c in self._str_to_current(content)]
        self._network_step(currents)
        for _ in range(max(0, ticks - 1)):
            self._network_step()

        # 思考诱发联想：回忆起的记忆进入意识
        recalled: List[BrainMemory] = []
        for token in content.replace("，", " ").replace("。", " ").split():
            if len(token) >= 2:
                recalled.extend(self.recall(token, top_k=1))
        self._decay_thoughts()

        # 思考记忆固化：高激活念头写入 STM
        consolidated = []
        for t in self.thought_space:
            if t.activation >= self.thought_salience:
                self._write_stm(t.content, tag="thought")
                consolidated.append(t.content)

        if self.record_history:
            self._record()
        return {
            "thought": content,
            "spikes": self.spike_counts(),
            "recalled": [m.content for m in recalled[:2]],
            "consolidated": consolidated,
            "thought_space": [(t.content, t.source, round(t.activation, 2))
                              for t in self.thought_space],
        }

    def introspect(self) -> Dict:
        """思考感官（内感觉 interoception）：感知自己的脑活动。

        读取主导情绪、三层脉冲数、记忆占用、思考空间焦点，生成内省
        言语并把它作为内部刺激回注网络（自我感知回路），同时记入
        metacog_log 元认知日志。外部感官看世界，思考感官看自己。
        """
        s, a, d = self.spike_counts()
        mood = max(self.emotion, key=lambda k: self.emotion[k])
        mood_cn = {"calm": "平静", "curiosity": "好奇",
                   "stress": "紧张", "pleasure": "愉悦"}.get(mood, mood)
        top = self.top_thought()
        top_content = top.content if top else "（空）"
        text = (f"我感到{mood_cn}，正在想「{top_content}」，"
                f"脉冲活动{s}/{a}/{d}，"
                f"记忆{len(self.short_memory)}/{len(self.long_memory)}")

        # 内省言语回注网络（强度 0.5 倍），并作为元认知念头入思考空间
        self.tick += 1
        self._network_step([c * 0.5 for c in self._str_to_current(text)])
        self._push_thought(text, source="metacog")

        entry = {"tick": self.tick, "mood": mood, "top_thought": top_content,
                 "spike_counts": [s, a, d],
                 "stm": len(self.short_memory),
                 "ltm": len(self.long_memory),
                 "text": text}
        self.metacog_log.append(entry)
        if self.record_history:
            self._record()
        return entry

    # ------------------ 决策中枢 ------------------

    def cognition(self, stimulus: str = "") -> str:
        """整合决策层脉冲 + 情绪 + 记忆联想，产生行为输出"""
        decision_spikes = sum(1 for n in self.decision_layer if n.spike)
        e = self.emotion

        # 记忆联想：提取输入中的关键词试探回忆（排除当前输入本身）
        recalled: List[BrainMemory] = []
        if stimulus:
            for token in stimulus.replace("，", " ").replace("。", " ").split():
                if len(token) >= 2:
                    recalled.extend(self.recall(token, top_k=1))

        dominant = max(e, key=lambda k: e[k])
        if decision_spikes >= 4:
            action = "主动响应"
        elif decision_spikes >= 2:
            action = "弱响应"
        else:
            action = "静默观察"

        mood_desc = {"calm": "平静", "curiosity": "好奇",
                     "stress": "紧张", "pleasure": "愉悦"}[dominant]

        reply = (f"[{self.name} | tick={self.tick}] 决策={action} | "
                 f"主导情绪={mood_desc}(cur={e['curiosity']:.2f}, "
                 f"str={e['stress']:.2f}, calm={e['calm']:.2f}) | "
                 f"决策层脉冲={decision_spikes}/8")
        if recalled:
            reply += f" | 联想回忆: {recalled[0].content!r}(w={recalled[0].weight:.2f})"
        return reply

    # ------------------ 动作空间与语言生成（v4.0） ------------------

    # 动作空间：决策层脉冲强度 → 结构化动作（可被外部执行器消费）
    ACTION_SPACE: Dict[str, Dict] = {
        "主动响应": {"min_spikes": 4, "verb": "respond",
                     "description": "决策层强发放，全通道输出"},
        "弱响应":   {"min_spikes": 2, "verb": "acknowledge",
                     "description": "决策层中等发放，低强度跟进"},
        "静默观察": {"min_spikes": 0, "verb": "observe",
                     "description": "决策层弱/无发放，保持监听"},
    }

    # 语言生成模板：按 (动作 × 主导情绪) 选槽位填充
    _UTTER_TEMPLATES: Dict[str, Dict[str, List[str]]] = {
        "主动响应": {
            "curiosity": ["「{stim}」——这个很有意思，我想深入了解。",
                          "我对「{stim}」很感兴趣！{mem_clause}多告诉我一些。"],
            "stress":    ["「{stim}」让我警觉起来了{mem_clause}，需要立刻处理。"],
            "pleasure":  ["「{stim}」太棒了{mem_clause}！我很愿意回应。"],
            "calm":      ["关于「{stim}」，我的回应是：{mem_clause}我在听。"],
        },
        "弱响应": {
            "curiosity": ["「{stim}」……有点意思，但我还看不太清。"],
            "stress":    ["「{stim}」让我有点不安，先观望一下。"],
            "pleasure":  ["「{stim}」还不错。"],
            "calm":      ["「{stim}」，嗯，我注意到了。"],
        },
        "静默观察": {
            "curiosity": ["（「{stim}」……{mem_clause}先记下来，慢慢看。）"],
            "stress":    ["（保持安静，留意「{stim}」的后续。）"],
            "pleasure":  ["（「{stim}」感觉不错，但不必出声。）"],
            "calm":      ["（「{stim}」{mem_clause}——静默观察中。）"],
        },
    }

    def decide_action(self, stimulus: str = "") -> Dict:
        """决策输出 → 结构化动作（动作空间接口，v4.0）。

        返回可供外部执行器消费的动作指令：
          action    — 动作名（ACTION_SPACE 键）
          verb      — 机器可读动作动词
          intensity — 动作强度（决策层脉冲占比 0..1）
          mood      — 主导情绪
          recalled  — 联想记忆内容列表
        """
        spikes = sum(1 for n in self.decision_layer if n.spike)
        if spikes >= 4:
            action = "主动响应"
        elif spikes >= 2:
            action = "弱响应"
        else:
            action = "静默观察"
        recalled: List[BrainMemory] = []
        if stimulus:
            for token in stimulus.replace("，", " ").replace("。", " ").split():
                if len(token) >= 2:
                    recalled.extend(self.recall(token, top_k=1))
        dominant = max(self.emotion, key=lambda k: self.emotion[k])
        return {
            "action": action,
            "verb": self.ACTION_SPACE[action]["verb"],
            "intensity": round(spikes / len(self.decision_layer), 3),
            "mood": dominant,
            "recalled": [m.content for m in recalled[:2]],
            "tick": self.tick,
        }

    def express(self, stimulus: str = "") -> Dict:
        """语言生成模块（v4.0）：决策 → 动作 → 模板化自然语言表达。

        按 (动作 × 主导情绪) 取模板，填充刺激与联想记忆槽位；
        同一 (动作, 情绪) 的多条模板按 tick 轮转，保证确定性可复现。
        返回 {"action": <decide_action 结果>, "utterance": str}。
        """
        act = self.decide_action(stimulus)
        table = self._UTTER_TEMPLATES[act["action"]]
        templates = table.get(act["mood"]) or table["calm"]
        tpl = templates[self.tick % len(templates)]
        mem_clause = (f"这让我想起「{act['recalled'][0]}」，"
                      if act["recalled"] else "")
        utterance = tpl.format(stim=stimulus or "……", mem_clause=mem_clause)
        return {"action": act, "utterance": utterance}

    # ------------------ 检索式语言生成（v4.6） ------------------

    # 句法框架：{stim} 刺激 / {mem_chain} 记忆编织从句 / {mood} 情绪修饰
    _SYNTAX_FRAMES: Dict[str, List[str]] = {
        "主动响应": [
            "「{stim}」——{mem_chain}，{mood}之下我必须作出回应。",
            "面对「{stim}」：{mem_chain}。{mood}地，我给出我的答案。",
        ],
        "弱响应": [
            "「{stim}」……{mem_chain}，{mood}中我先记下一笔。",
            "关于「{stim}」，{mem_chain}。{mood}地看着事态发展。",
        ],
        "静默观察": [
            "（「{stim}」。{mem_chain}——{mood}地静观其变。）",
            "（{mood}中，「{stim}」{mem_chain}。）",
        ],
    }
    _MOOD_ADVERBS: Dict[str, str] = {
        "curiosity": "满怀好奇", "stress": "心存忐忑",
        "pleasure": "带着欣喜", "calm": "平静",
    }

    def _retrieve_fragments(self, stimulus: str,
                            top_k: int = 3) -> List[BrainMemory]:
        """检索式组合的第一步：从 LTM 检索与刺激相关的记忆片段。

        按刺激的关键词逐词回忆，按内容去重、按记忆权重降序取 top_k——
        权重最高的片段排在最前，成为编织从句的主线。
        """
        found: Dict[str, BrainMemory] = {}
        for token in stimulus.replace("，", " ").replace("。", " ").split():
            if len(token) < 2:
                continue
            hits = self.recall(token, top_k=top_k)
            if not hits and len(token) > 3:
                # 长词未命中 → 退化为 2/3 字 n-gram 子词检索
                # （"取火的方法" → "取火"、"的方"…，部分匹配也能联想）
                for n in (3, 2):
                    hits = [m for i in range(len(token) - n + 1)
                            for m in self.recall(token[i:i + n], top_k=2)]
                    if hits:
                        break
            for m in hits:
                if m.content != stimulus and m.content not in found:
                    found[m.content] = m
        ranked = sorted(found.values(), key=lambda m: -m.weight)
        return ranked[:top_k]

    @staticmethod
    def _weave_memories(mems: List[BrainMemory]) -> str:
        """记忆编织：把检索到的片段组合成一条自然语言从句"""
        if not mems:
            return "这是全新的体验，没有记忆可供引用"
        if len(mems) == 1:
            return f"「{mems[0].content}」浮现在脑海"
        if len(mems) == 2:
            return f"「{mems[0].content}」和「{mems[1].content}」一起浮现"
        chain = "」到「".join(m.content for m in mems)
        return f"思绪从「{chain}」一路联想"

    def compose(self, stimulus: str = "", top_k: int = 3) -> Dict:
        """检索式语言生成（v4.6）：LTM 片段 + 句法框架组合。

        与 express() 的固定模板不同：
          1. 检索——_retrieve_fragments() 从 LTM 取出权重最高的
             top_k 条相关记忆片段；
          2. 编织——_weave_memories() 按片段数选择单句/并列/联想链结构；
          3. 造句——句法框架按 (动作 × tick 轮转) 选定，填充刺激、
             记忆从句与情绪修饰词。
        返回 {"utterance", "action", "fragments", "frame", "mood"}。
        """
        act = self.decide_action(stimulus)
        mems = self._retrieve_fragments(stimulus, top_k=top_k)
        mem_chain = self._weave_memories(mems)
        mood = self._MOOD_ADVERBS.get(act["mood"], "平静")
        frames = self._SYNTAX_FRAMES[act["action"]]
        frame = frames[self.tick % len(frames)]
        utterance = frame.format(stim=stimulus or "……",
                                 mem_chain=mem_chain, mood=mood)
        return {"utterance": utterance, "action": act,
                "fragments": [m.content for m in mems],
                "frame": frame, "mood": act["mood"]}

    # ------------------ 动作执行器（v4.1 感知-决策-执行-学习闭环） ------------------

    def register_executor(self, fn: Callable,
                          verb: Optional[str] = None,
                          default: bool = False) -> None:
        """注册动作执行器。

        verb：只处理该动作动词（respond / acknowledge / observe）；
        default=True：作为兜底执行器。执行器契约见 make_robot_executor。
        """
        if not callable(fn):
            raise TypeError("executor 必须是 callable(action) -> "
                            "{success, reward, detail}")
        if verb is not None:
            self.executors[verb] = fn
        if default or verb is None:
            self.default_executor = fn

    def act(self, stimulus: str = "",
            executor: Optional[Callable] = None,
            policy: Optional[str] = None) -> Dict:
        """感知 → 决策 → 执行 → 奖励回传 的完整闭环（v4.1）。

        1. express()：决策 + 语言表达；
        2. 路由执行器：显式传入 > 按 verb 注册 > 默认执行器；
        3. 执行结果（reward）经 reward_td() 回传——执行后果成为
           学习信号，多巴胺响应执行成败的"意外程度"；
        4. 执行器抛异常按失败处理（reward=-0.5），不会中断大脑。

        v4.5：每次执行后按 verb 更新技能价值 Q(verb)；policy 非空时
        （"greedy" / "epsilon" / "softmax"）由习得价值覆盖决策层
        的 verb 选择——动作选择策略化。

        返回 {"utterance", "action", "execution", "feedback", "skill"}；
        无任何执行器时 execution/feedback/skill 为 None（纯决策模式）。
        """
        ex = self.express(stimulus)
        act_dict = ex["action"]
        if policy is not None:
            verb = self.select_verb(policy)
            for name, cfg in self.ACTION_SPACE.items():
                if cfg["verb"] == verb:
                    act_dict = dict(act_dict, action=name, verb=verb)
                    break
        fn = (executor
              or self.executors.get(act_dict["verb"])
              or self.default_executor)
        execution, feedback, skill = None, None, None
        if fn is not None:
            try:
                execution = fn(act_dict)
            except Exception as e:
                execution = {"success": False, "reward": -0.5,
                             "detail": f"执行器异常：{type(e).__name__}: {e}"}
            reward_val = float(execution.get("reward", 0.0))
            feedback = self.reward_td(reward_val)
            skill = self.learn_skill(act_dict["verb"], reward_val)
        return {"utterance": ex["utterance"], "action": act_dict,
                "execution": execution, "feedback": feedback,
                "skill": skill}

    # ------------------ 执行器技能学习（v4.5） ------------------

    def learn_skill(self, verb: str, reward: float) -> Dict:
        """按 verb 更新独立技能价值：Q(verb) += α·(r − Q(verb))。

        与 reward_td 的全局 V 不同：每个动作动词有自己的价值轨道，
        执行后果只记入实际采取的那个动作（信用精确归因）。
        """
        reward = self._clip(reward, -1.0, 1.0)
        q = self.verb_values.get(verb, 0.0)
        rpe = reward - q
        self.verb_values[verb] = q + self.skill_alpha * rpe
        return {"verb": verb, "reward": reward, "rpe": rpe,
                "q": self.verb_values[verb]}

    def select_verb(self, policy: str = "epsilon") -> str:
        """按习得技能价值选择动作动词（动作选择策略化）。

        greedy   — 恒选 Q 最高者；
        epsilon  — 以 ε 概率随机探索，否则 greedy；
        softmax  — 按 exp(Q/T) 比例抽样（温和探索）。
        Q 全零（未学习）时三种策略都退化为均匀随机。
        """
        verbs = list(self.verb_values)
        if policy == "softmax":
            t = max(self.skill_temperature, 1e-3)
            mx = max(self.verb_values[v] for v in verbs)
            ws = [math.exp((self.verb_values[v] - mx) / t) for v in verbs]
            return random.choices(verbs, weights=ws, k=1)[0]
        if policy == "epsilon" and random.random() < self.effective_epsilon():
            return random.choice(verbs)
        best = max(self.verb_values[v] for v in verbs)
        top = [v for v in verbs if self.verb_values[v] == best]
        return random.choice(top)

    # ------------------ 脉冲思考链（可解释追踪） ------------------

    def thought_chain(self, data: str) -> Dict:
        """脉冲思考链：把一次"感知 → 传导 → 回响 → 决策"全过程展开为
        可解释的脉冲因果链（脉冲网络版的 chain-of-thought）。

        返回 dict：
          input  — 原始输入文本
          steps  — 每个网络步的三层脉冲 id 列表
                   （第 0 步为刺激响应步，其余为自由回响步）
          chain  — 人类可读的思考链描述（逐步因果）
          output — 决策中枢的行为输出（同 sensory_input 返回值）
        """
        self._step_trace = []
        output = self.sensory_input(data)
        steps, self._step_trace = self._step_trace, None

        n_s, n_a, n_d = (len(self.sense_layer), len(self.assoc_layer),
                         len(self.decision_layer))
        chain = [f"1. 感官编码：{data!r} → {n_s} 维输入电流"
                 f"（经注意力 ×{self.attention_factor:.2f} 调制）"]
        for i, (s, a, d) in enumerate(steps):
            if i == 0:
                chain.append(
                    f"2. 刺激步：感官层 {len(s)}/{n_s} 脉冲 {s}"
                    f" → 联想层 {len(a)}/{n_a} 脉冲（突触汇集）"
                    f" → 决策层 {len(d)}/{n_d} 脉冲")
            else:
                silent = "（回声衰减，趋于静息）" if not (s or a or d) else ""
                chain.append(
                    f"{i + 2}. 回响+{i}：感官 {len(s)} / 联想 {len(a)} / "
                    f"决策 {len(d)}{silent}")
        chain.append(f"{len(steps) + 2}. 决策输出：{output}")
        return {"input": data, "steps": steps, "chain": chain,
                "output": output}

    # ------------------ 历史记录与统计 ------------------

    def _record(self):
        h = self.history
        h["tick"].append(self.tick)
        # 感知 tick 记录刺激响应步的脉冲快照；free_run tick 记录当前步状态
        if self._stim_spikes is not None:
            s_ids, a_ids, d_ids = self._stim_spikes
            self._stim_spikes = None
        else:
            s_ids = [n.id for n in self.sense_layer if n.spike]
            a_ids = [n.id for n in self.assoc_layer if n.spike]
            d_ids = [n.id for n in self.decision_layer if n.spike]
        h["sense_spikes"].append(s_ids)
        h["assoc_spikes"].append(a_ids)
        h["decision_spikes"].append(d_ids)
        total = len(s_ids) + len(a_ids) + len(d_ids)
        h["spike_rate"].append(total / 56)
        h["emotion"].append(dict(self.emotion))
        h["attention"].append(self.attention_factor)
        h["stm_size"].append(len(self.short_memory))
        h["ltm_size"].append(len(self.long_memory))
        h["synapse_mean"].append(self.synapse_mean())

    def synapse_mean(self) -> float:
        return sum(self.synapse.values()) / len(self.synapse) if self.synapse else 0.0

    def strong_synapse_count(self, threshold: float = 0.5) -> int:
        return sum(1 for w in self.synapse.values() if w > threshold)

    # ------------------ DNA 记忆遗传模块 ------------------

    def dump_dna(self) -> dict:
        """序列化大脑状态（记忆 + 突触 + 情绪 + 人格参数），可用于保存或克隆"""
        return {
            "name": self.name,
            "tick": self.tick,
            "synapse": {f"{a},{b}": w for (a, b), w in self.synapse.items()},
            "recurrent_synapse": {f"{a},{b}": w for (a, b), w in
                                  self.recurrent_synapse.items()},
            "short_memory": [asdict(m) for m in self.short_memory],
            "long_memory": [asdict(m) for m in self.long_memory],
            "emotion": dict(self.emotion),
            "attention_factor": self.attention_factor,
            "personality": {
                "sensation_seeking": self.sensation_seeking,
                "habituation_rate": self.habituation_rate,
            },
            "exposure_count": dict(self.exposure_count),
            # v5.0 思考体系：念头与元认知日志（元认知只保留最近 50 条）
            "thought_space": [asdict(t) for t in self.thought_space],
            "metacog_log": list(self.metacog_log[-50:]),
        }

    @classmethod
    def from_dna(cls, dna: dict, new_name: Optional[str] = None) -> "AIBrainEntity":
        """从 DNA 克隆一个继承记忆与突触的新实体。

        对输入做结构校验与数值边界保护，防止损坏/恶意 JSON 导致异常状态。
        """
        if not isinstance(dna, dict):
            raise TypeError(f"DNA 必须是 dict，实际是 {type(dna).__name__}")
        # 必需字段检查
        for key in ("name", "tick", "synapse", "short_memory",
                    "long_memory", "emotion", "attention_factor"):
            if key not in dna:
                raise ValueError(f"DNA 缺少必需字段: {key!r}")
        if not isinstance(dna["name"], str):
            raise TypeError("DNA.name 必须是字符串")
        if not isinstance(dna["tick"], int):
            raise TypeError("DNA.tick 必须是整数")
        if not isinstance(dna["synapse"], dict):
            raise TypeError("DNA.synapse 必须是 dict")
        if not isinstance(dna.get("recurrent_synapse", {}), dict):
            raise TypeError("DNA.recurrent_synapse 必须是 dict")
        if not isinstance(dna["short_memory"], list):
            raise TypeError("DNA.short_memory 必须是 list")
        if not isinstance(dna["long_memory"], list):
            raise TypeError("DNA.long_memory 必须是 list")
        if not isinstance(dna["emotion"], dict):
            raise TypeError("DNA.emotion 必须是 dict")

        def _parse_synapse(raw: dict, label: str) -> dict:
            """解析突触字典，校验键格式与权重数值范围。"""
            result = {}
            for k, w in raw.items():
                if not isinstance(k, str) or "," not in k:
                    raise ValueError(
                        f"{label} 键格式错误: {k!r}（应为 'pre,post'）")
                try:
                    a_str, b_str = k.split(",", 1)
                    a, b = int(a_str), int(b_str)
                except (ValueError, TypeError):
                    raise ValueError(
                        f"{label} 键 {k!r} 中的神经元 ID 必须是整数")
                try:
                    wf = float(w)
                except (ValueError, TypeError):
                    raise TypeError(
                        f"{label}[{k!r}] 权重必须是数值，实际 {type(w).__name__}")
                # 突触权重边界保护（与 _clip 一致，防止异常值污染网络）
                result[(a, b)] = max(-2.0, min(2.0, wf))
            return result

        brain = cls(new_name or (dna["name"] + "_clone"))
        brain.tick = max(0, dna["tick"])
        brain.synapse = _parse_synapse(dna["synapse"], "synapse")
        brain.recurrent_synapse = _parse_synapse(
            dna.get("recurrent_synapse", {}), "recurrent_synapse")
        brain._recurrent_out = {}
        brain._recurrent_in = {}
        for (pre, post) in brain.recurrent_synapse:
            brain._recurrent_out.setdefault(pre, []).append(post)
            brain._recurrent_in.setdefault(post, []).append(pre)

        def _safe_memory(raw_list: list, label: str) -> list:
            """校验记忆条目结构，跳过格式错误的条目。"""
            result = []
            for i, m in enumerate(raw_list):
                if not isinstance(m, dict):
                    continue  # 跳过非 dict 条目
                try:
                    result.append(BrainMemory(**m))
                except TypeError:
                    continue  # 跳过字段不匹配的条目
            return result

        brain.short_memory = _safe_memory(dna["short_memory"], "short_memory")
        brain.long_memory = _safe_memory(dna["long_memory"], "long_memory")
        # 情绪值边界保护
        brain.emotion = {}
        for k, v in dna["emotion"].items():
            if isinstance(k, str):
                try:
                    brain.emotion[k] = max(0.0, min(1.0, float(v)))
                except (ValueError, TypeError):
                    brain.emotion[k] = 0.5
        try:
            brain.attention_factor = max(0.1, min(3.0,
                                        float(dna["attention_factor"])))
        except (ValueError, TypeError):
            brain.attention_factor = 1.0
        # 人格参数（v4.9.1+ DNA；兼容旧版无此字段的 DNA）
        pers = dna.get("personality", {})
        if isinstance(pers, dict):
            try:
                brain.sensation_seeking = max(0.0, min(1.0,
                        float(pers.get("sensation_seeking", 0.5))))
            except (ValueError, TypeError):
                brain.sensation_seeking = 0.5
            try:
                brain.habituation_rate = max(0.0, min(1.0,
                        float(pers.get("habituation_rate", 0.3))))
            except (ValueError, TypeError):
                brain.habituation_rate = 0.3
        # 暴露计数（习惯化状态）
        ec = dna.get("exposure_count", {})
        if isinstance(ec, dict):
            brain.exposure_count = {
                str(k): int(v) for k, v in ec.items()
                if isinstance(k, str)
                and isinstance(v, (int, float)) and v >= 0
            }
        # v5.0 思考空间（兼容旧版无此字段的 DNA）
        ts = dna.get("thought_space", [])
        if isinstance(ts, list):
            for t in ts:
                if not isinstance(t, dict):
                    continue
                try:
                    item = ThoughtItem(**t)
                    item.activation = max(0.0, min(1.0, float(item.activation)))
                    brain.thought_space.append(item)
                except (TypeError, ValueError):
                    continue
            brain.thought_space = brain.thought_space[-brain.thought_capacity:]
        # v5.0 元认知日志
        mc = dna.get("metacog_log", [])
        if isinstance(mc, list):
            brain.metacog_log = [e for e in mc if isinstance(e, dict)]
        return brain

    def save_dna(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.dump_dna(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_dna(cls, path: str, new_name: Optional[str] = None) -> "AIBrainEntity":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dna(json.load(f), new_name=new_name)

    # ------------------ 状态报告 ------------------

    def status(self) -> str:
        return (f"=== {self.name} 状态 ===\n"
                f"  tick={self.tick}  注意力={self.attention_factor:.2f}  "
                f"多巴胺={self.dopamine:.2f}\n"
                f"  情绪: {', '.join(f'{k}={v:.2f}' for k, v in self.emotion.items())}\n"
                f"  记忆: 感官缓存={len(self.sensory_buffer)} "
                f"STM={len(self.short_memory)}/{self.max_stm} "
                f"LTM={len(self.long_memory)}/{self.max_ltm}\n"
                f"  思考空间: {len(self.thought_space)}/{self.thought_capacity} 个念头"
                f"（焦点: {self.top_thought().content if self.top_thought() else '（空）'}）\n"
                f"  突触: 前馈{len(self.synapse)}条(强连接{self.strong_synapse_count()}), "
                f"循环{len(self.recurrent_synapse)}条, "
                f"前馈平均强度={self.synapse_mean():.3f}")


# ===================== 群体智能（v3.0 BrainSwarm） =====================

class BrainSwarm:
    """多实体种群：通过 DNA 交换记忆，模拟文化传递。

    用法：
        swarm = BrainSwarm(["Alpha", "Beta", "Gamma"], seed=1)
        swarm.population[0].sensory_input("火焰是危险的")   # Alpha 亲身经历
        swarm.culture_round(rounds=2)                       # 文化传递给同伴
        swarm.broadcast("大家注意")                          # 全种群广播
    """

    def __init__(self, names: List[str], seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self.population: List[AIBrainEntity] = [
            AIBrainEntity(name, seed=(seed + i if seed is not None else None))
            for i, name in enumerate(names)
        ]
        self.generation = 1
        # v4.1 社交拓扑：个体索引 -> 邻居索引列表；None = 全连接（默认，向后兼容）
        self.topology: Optional[Dict[int, List[int]]] = None

    # ------------------ 社交拓扑（v4.1） ------------------

    def set_topology(self, kind: str = "fully_connected",
                     p: float = 0.3, seed: Optional[int] = None) -> Dict[int, List[int]]:
        """设置种群社交拓扑（文化只沿边传播）。

        kind:
          fully_connected — 全连接（等价于清空拓扑）
          ring            — 环形：每个体只连左右相邻两个体
          star            — 星形：0 号为中心，其余只连中心
          random          — Erdos-Renyi 随机图（边概率 p）
          small_world     — 小世界：环形 + 以概率 p 随机重连（捷径）
        返回邻接表。设置后 horizontal/vertical_transfer 只在邻居间进行。
        """
        rng = random.Random(seed) if seed is not None else random
        n = len(self.population)
        adj: Dict[int, set] = {i: set() for i in range(n)}

        def link(a: int, b: int):
            if a != b:
                adj[a].add(b)
                adj[b].add(a)

        if kind == "fully_connected":
            self.topology = None
            return {i: [j for j in range(n) if j != i] for i in range(n)}
        if kind == "ring":
            for i in range(n):
                link(i, (i + 1) % n)
        elif kind == "star":
            for i in range(1, n):
                link(0, i)
        elif kind == "random":
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < p:
                        link(i, j)
            # 保证连通性下限：孤立个体并回环
            for i in range(n):
                if not adj[i]:
                    link(i, (i + 1) % n)
        elif kind == "small_world":
            for i in range(n):
                link(i, (i + 1) % n)
            for i in range(n):
                if rng.random() < p:
                    j = rng.randrange(n)
                    link(i, j)
        else:
            raise ValueError(f"未知拓扑 {kind!r}：可选 fully_connected / "
                             f"ring / star / random / small_world")
        self.topology = {i: sorted(s) for i, s in adj.items()}
        return self.topology

    def _neighbors(self, idx: int) -> List[int]:
        """个体的社交邻居索引；无拓扑（全连接）时返回所有其他个体"""
        if self.topology is None:
            return [j for j in range(len(self.population)) if j != idx]
        return self.topology.get(idx, [])

    def _edge_count(self) -> int:
        """当前社交边总数（无拓扑 = 全连接的边数）"""
        n = len(self.population)
        if self.topology is None:
            return n * (n - 1) // 2
        return sum(len(v) for v in self.topology.values()) // 2

    # ------------------ 拓扑自适应：共同演化网络（v4.2） ------------------

    def rewire_coevolve(self, meme: str, rewire_prob: float = 0.5,
                        min_degree: int = 1,
                        birth_prob: float = 0.5) -> Dict[str, int]:
        """Holme-Newman 共同演化一步：共识压力反作用于社交边的生灭。

        "观点" = 是否持有目标模因（二态）。每轮做 N 次尝试（N=种群数）。
        每次随机选个体 i，三种事件：

          1.【异见边博弈】选 i 的一个邻居 j，观点不同则二选一：
             - 以 φ=rewire_prob【断边重连】：断 i−j（道不同不相谋），
               i 重连到同道随机非邻居（向同道靠拢）；
             - 以 1−φ【观点模仿】：持有方当场把模因教给未持有方
               （权重 ×0.8）——文化沿异见边传播。
          2.【求知连边】（边之"生"）：若 i 未持有模因而种群中存在持有者，
             以 birth_prob 概率主动向随机持有者连一条新边——
             共识压力驱动个体向知识靠拢，补充被断边消耗的异见通道。

        φ 是核心参数：φ→0 传播主导、网络近似静态；φ→1 结构主导、
        网络在共识前分裂（模因被封印在持有者社团内）。
        min_degree 防孤立；求知连边使边总数可增长（生灭不守恒）。
        返回 {"rewired", "copied", "born"} 计数。
        """
        if self.topology is None:  # 全连接无演化空间
            return {"rewired": 0, "copied": 0, "born": 0}
        n = len(self.population)

        def holds(k: int) -> bool:
            return any(m.content == meme for m in self.population[k].long_memory)

        rewired = copied = born = 0
        for _ in range(n):
            i = random.randrange(n)
            hi = holds(i)
            # 事件 2：求知连边（未持有者向持有者靠拢）
            if not hi and random.random() < birth_prob:
                cands = [k for k in range(n)
                         if k != i and holds(k)
                         and k not in self.topology[i]]
                if cands:
                    k = random.choice(cands)
                    self.topology[i].append(k)
                    self.topology[i].sort()
                    self.topology[k].append(i)
                    self.topology[k].sort()
                    born += 1
            # 事件 1：异见边博弈
            nbrs = self.topology.get(i, [])
            if not nbrs:
                continue
            j = random.choice(nbrs)
            hj = holds(j)
            if hi == hj:
                continue                      # 同道：边保留
            if random.random() < rewire_prob:
                # 断边重连（保持最小度，防孤立）
                if len(self.topology[i]) <= min_degree:
                    continue
                self.topology[i].remove(j)
                self.topology[j].remove(i)
                cands = [k for k in range(n)
                         if k != i and k not in self.topology[i]
                         and holds(k) == hi]
                if cands:
                    k = random.choice(cands)
                    self.topology[i].append(k)
                    self.topology[i].sort()
                    self.topology[k].append(i)
                    self.topology[k].sort()
                rewired += 1
            else:
                # 观点模仿：持有方把模因当场教给未持有方
                teacher, student = (j, i) if hj else (i, j)
                src = next(m for m in self.population[teacher].long_memory
                           if m.content == meme)
                self.population[student].long_memory.append(BrainMemory(
                    content=meme, timestamp=time.time(),
                    weight=src.weight * 0.8, tag="culture"))
                copied += 1
        return {"rewired": rewired, "copied": copied, "born": born}

    def same_state_edge_ratio(self, meme: str) -> float:
        """同道边比例：两端观点（是否持有模因）相同的边占比。
        共同演化的结构观测量：趋 1 = 网络按观点自组织成团。"""
        if self.topology is None:
            return 1.0
        holds = [any(m.content == meme for m in b.long_memory)
                 for b in self.population]
        total, same = 0, 0
        for i, nbrs in self.topology.items():
            for j in nbrs:
                if j > i:
                    total += 1
                    if holds[i] == holds[j]:
                        same += 1
        return same / total if total else 1.0

    # ------------------ 多模因竞争（v4.3） ------------------

    def _stance(self, idx: int, memes) -> Optional[str]:
        """个体在竞争模因集合中的立场：权重最高的竞争模因；均无则 None"""
        best, best_w = None, -1.0
        for m in self.population[idx].long_memory:
            if m.content in memes and m.weight > best_w:
                best, best_w = m.content, m.weight
        return best

    def compete_coevolve(self, memes: List[str],
                         rewire_prob: float = 0.5,
                         birth_prob: float = 0.5,
                         min_degree: int = 1) -> Dict[str, int]:
        """多模因竞争的共同演化一步（Axelrod/多态投票者风格）。

        立场 = 个体持有的最强竞争模因（或 None）。每轮 N 次尝试：

          1.【阵营连边】（边之生）：有立场者以 birth_prob 概率
             向同立场非邻居连新边——阵营内聚。
          2.【异见边博弈】：随机邻居立场不同则二选一——
             - 以 φ【断边重连】：断边，重连到同立场个体（阵营隔离）；
             - 以 1−φ【立场转化】：强势一方把弱势一方转化为自己的
               立场（学生移除全部竞争模因，改持教师模因 ×0.8）；
               一方无立场时，有立场方恒为教师（学习）。

        φ 低 → 转化主导，一个模因垄断全网（共识）；
        φ 高 → 隔离主导，阵营各自封闭，多模因极化共存。
        返回 {"rewired", "converted", "born"}。
        """
        if self.topology is None:  # 全连接无演化空间
            return {"rewired": 0, "converted": 0, "born": 0}
        n = len(self.population)
        rewired = converted = born = 0
        for _ in range(n):
            i = random.randrange(n)
            si = self._stance(i, memes)
            # 事件 1：阵营连边
            if si is not None and random.random() < birth_prob:
                cands = [k for k in range(n)
                         if k != i and k not in self.topology[i]
                         and self._stance(k, memes) == si]
                if cands:
                    k = random.choice(cands)
                    self.topology[i].append(k)
                    self.topology[i].sort()
                    self.topology[k].append(i)
                    self.topology[k].sort()
                    born += 1
            # 事件 2：异见边博弈
            nbrs = self.topology.get(i, [])
            if not nbrs:
                continue
            j = random.choice(nbrs)
            sj = self._stance(j, memes)
            if si == sj:
                continue                      # 同阵营（含双双无立场）
            if random.random() < rewire_prob:
                if len(self.topology[i]) <= min_degree:
                    continue
                cands = [k for k in range(n)
                         if k != i and k not in self.topology[i]
                         and self._stance(k, memes) == si]
                if not cands:
                    continue
                self.topology[i].remove(j)
                self.topology[j].remove(i)
                k = random.choice(cands)
                self.topology[i].append(k)
                self.topology[i].sort()
                self.topology[k].append(i)
                self.topology[k].sort()
                rewired += 1
            else:
                # 立场转化：教师随机（投票者模型）——
                # 一方无立场时，有立场方恒为教师（学习）
                if si is None and sj is None:
                    continue
                if si is None:
                    teacher, student = j, i
                elif sj is None:
                    teacher, student = i, j
                else:
                    teacher, student = random.choice([(i, j), (j, i)])
                win = self._stance(teacher, memes)
                src = next(m for m in self.population[teacher].long_memory
                           if m.content == win)
                sb = self.population[student]
                sb.long_memory[:] = [m for m in sb.long_memory
                                     if m.content not in memes]
                sb.long_memory.append(BrainMemory(
                    content=win, timestamp=time.time(),
                    weight=src.weight * 0.8, tag="culture"))
                converted += 1
        return {"rewired": rewired, "converted": converted, "born": born}

    def competition_dynamics(self, memes: List[str], max_rounds: int = 60,
                             dominance: float = 0.9,
                             rewire_prob: float = 0.5,
                             birth_prob: float = 0.5) -> Dict:
        """多模因竞争动力学：垄断（共识）vs 极化（共存）的判定实验。

        每轮一次 compete_coevolve，记录各模因立场覆盖率与阵营数曲线；
        某模因覆盖率达 dominance 判为垄断收敛；max_rounds 后仍多阵营
        并存判为极化共存。

        返回 {"memes", "winner", "converged", "rounds",
              "coverage": {meme: [...]}, "camps": [...],
              "same_stance_ratio": [...], "final": {meme: 覆盖率}}。
        """
        n = len(self.population)
        cov: Dict[str, list] = {m: [] for m in memes}
        camps_series, ratio_series = [], []
        converted_total = rewired_total = born_total = 0
        for r in range(1, max_rounds + 1):
            ops = self.compete_coevolve(memes, rewire_prob=rewire_prob,
                                        birth_prob=birth_prob)
            converted_total += ops["converted"]
            rewired_total += ops["rewired"]
            born_total += ops["born"]
            stances = [self._stance(k, memes) for k in range(n)]
            for m in memes:
                cov[m].append(round(
                    sum(1 for s in stances if s == m) / n, 3))
            camps = sum(1 for m in memes if cov[m][-1] > 0)
            camps_series.append(camps)
            # 同立场边比例
            total_e = same_e = 0
            for i, nbrs in self.topology.items():
                for j in nbrs:
                    if j > i:
                        total_e += 1
                        if stances[i] == stances[j]:
                            same_e += 1
            ratio_series.append(round(same_e / total_e, 3) if total_e else 1.0)
            leader = max(memes, key=lambda m: cov[m][-1])
            if cov[leader][-1] >= dominance:
                return {"memes": memes, "winner": leader, "converged": True,
                        "rounds": r, "coverage": cov,
                        "camps": camps_series,
                        "same_stance_ratio": ratio_series,
                        "final": {m: cov[m][-1] for m in memes},
                        "converted_total": converted_total,
                        "rewired_total": rewired_total,
                        "born_total": born_total}
        return {"memes": memes, "winner": None, "converged": False,
                "rounds": None, "coverage": cov, "camps": camps_series,
                "same_stance_ratio": ratio_series,
                "final": {m: cov[m][-1] for m in memes},
                "converted_total": converted_total,
                "rewired_total": rewired_total,
                "born_total": born_total}

    def coevolve_consensus(self, meme: str, max_rounds: int = 60,
                           threshold: float = 0.9,
                           rewire_prob: float = 0.5,
                           birth_prob: float = 0.5) -> Dict:
        """共同演化共识实验：异见边上的"传播 vs 断边"逐轮博弈 + 求知连边。

        每轮执行一次 rewire_coevolve（观点模仿 / 断边重连 / 求知连边），
        记录覆盖率、边数、同道边比例三条曲线，直到覆盖率达 threshold。

        返回 {"rounds", "converged", "coverage", "edges",
              "same_state_ratio", "final_degree",
              "rewired_total", "copied_total", "born_total"}。
        """
        coverage, edges, ratios = [], [], []
        rewired_total = copied_total = born_total = 0
        for r in range(1, max_rounds + 1):
            ops = self.rewire_coevolve(meme, rewire_prob=rewire_prob,
                                       birth_prob=birth_prob)
            rewired_total += ops["rewired"]
            copied_total += ops["copied"]
            born_total += ops["born"]
            have = sum(1 for b in self.population
                       if any(m.content == meme for m in b.long_memory))
            cov = have / len(self.population)
            coverage.append(round(cov, 3))
            edges.append(self._edge_count())
            ratios.append(round(self.same_state_edge_ratio(meme), 3))
            if cov >= threshold:
                return {"rounds": r, "converged": True,
                        "coverage": coverage, "edges": edges,
                        "same_state_ratio": ratios,
                        "rewired_total": rewired_total,
                        "copied_total": copied_total,
                        "born_total": born_total,
                        "final_degree": round(2 * edges[-1] /
                                              len(self.population), 2)}
        return {"rounds": None, "converged": False,
                "coverage": coverage, "edges": edges,
                "same_state_ratio": ratios,
                "rewired_total": rewired_total,
                "copied_total": copied_total,
                "born_total": born_total,
                "final_degree": round(2 * edges[-1] /
                                      len(self.population), 2)}

    def broadcast(self, text: str) -> List[str]:
        """同一刺激广播给全种群（模拟共同经历 / 公共事件）"""
        return [b.sensory_input(text) for b in self.population]

    def culture_round(self, rounds: int = 1, top_k: int = 3,
                      mode: str = "dna") -> int:
        """文化传递：随机选取 (传授者, 学习者) 对，把传授者最强的
        top_k 条记忆传递给学习者。

        mode="dna"：直接经 DNA 复制记忆条目（权重 ×0.8，标记为 culture）；
        mode="teach"：传授者"口述"，学习者以感官输入方式自然学习。
        返回成功传递的记忆条数。
        """
        transfers = 0
        for _ in range(rounds):
            teacher, student = random.sample(self.population, 2)
            dna = teacher.dump_dna()
            pool = sorted(dna["long_memory"],
                          key=lambda m: m["weight"], reverse=True)[:top_k]
            if not pool:  # 没有 LTM 时退而传授 STM 中最强的
                pool = sorted(dna["short_memory"][:-1],
                              key=lambda m: m["weight"], reverse=True)[:top_k]
            for m in pool:
                if mode == "dna":
                    if all(x.content != m["content"]
                           for x in student.long_memory):
                        student.long_memory.append(BrainMemory(
                            content=m["content"],
                            timestamp=time.time(),
                            weight=m["weight"] * 0.8,
                            tag="culture",
                        ))
                        transfers += 1
                else:
                    student.sensory_input(m["content"])
                    transfers += 1
        return transfers

    # ------------------ 水平 vs 垂直传播动力学（v4.0） ------------------

    def horizontal_transfer(self, rounds: int = 1, top_k: int = 3) -> int:
        """水平传播：同代同伴之间的文化传递（同伴学习 / 模因扩散）。

        只在与学习者同世代的个体中选传授者——模拟同侪间的信息扩散，
        速度快、范围广，但易受同代噪声影响。
        """
        transfers = 0
        for _ in range(rounds):
            si = random.randrange(len(self.population))
            student = self.population[si]
            nbrs = set(self._neighbors(si))  # v4.1：只沿社交边传播
            peers = [b for j, b in enumerate(self.population)
                     if j in nbrs and b.generation == student.generation]
            if not peers:
                continue
            teacher = random.choice(peers)
            transfers += self._transfer_top(teacher, student, top_k)
        return transfers

    def vertical_transfer(self, rounds: int = 1, top_k: int = 3) -> int:
        """垂直传播：跨代文化传递（师承 / 传统继承）。

        传授者严格来自比学习者更早的世代——模拟代际传承，
        内容稳定、保真度高，但覆盖面受代际数量限制。
        """
        transfers = 0
        for _ in range(rounds):
            si = random.randrange(len(self.population))
            student = self.population[si]
            nbrs = set(self._neighbors(si))  # v4.1：只沿社交边传播
            # 只有更早世代且确有长期记忆的个体才能充当师承者
            elders = [b for j, b in enumerate(self.population)
                      if j in nbrs and b.generation < student.generation
                      and b.long_memory]
            if not elders:
                continue
            teacher = random.choice(elders)
            transfers += self._transfer_top(teacher, student, top_k)
        return transfers

    @staticmethod
    def _transfer_top(teacher: "AIBrainEntity", student: "AIBrainEntity",
                      top_k: int) -> int:
        """把教师最强的 top_k 条 LTM 以 DNA 方式复制给学生（权重 ×0.8）"""
        pool = sorted(teacher.long_memory,
                      key=lambda m: m.weight, reverse=True)[:top_k]
        n = 0
        for m in pool:
            if all(x.content != m.content for x in student.long_memory):
                student.long_memory.append(BrainMemory(
                    content=m.content, timestamp=time.time(),
                    weight=m.weight * 0.8, tag="culture"))
                n += 1
        return n

    def transmission_dynamics(self, meme: str, rounds: int = 6,
                              direction: str = "horizontal") -> Dict:
        """追踪一个模因在种群中的逐轮扩散曲线（水平 vs 垂直对照用）。

        返回 {"direction": ..., "coverage": [每轮结束时持有该模因的个体比例]}。
        """
        spread = {"horizontal": self.horizontal_transfer,
                  "vertical": self.vertical_transfer}[direction]
        coverage = []
        for _ in range(rounds):
            spread(rounds=1, top_k=3)
            have = sum(1 for b in self.population
                       if any(m.content == meme for m in b.long_memory))
            coverage.append(round(have / len(self.population), 3))
        return {"direction": direction, "meme": meme, "coverage": coverage}

    def consensus_convergence(self, meme: str, max_rounds: int = 40,
                              threshold: float = 0.9,
                              direction: str = "horizontal") -> Dict:
        """共识收敛速度：模因覆盖率首次达到 threshold 所需的传播轮数。

        共识涌现相变研究的核心观测量——拓扑越连通、种群越小，
        收敛轮数越少；达到 max_rounds 仍未收敛则 rounds 为 None。
        返回 {"rounds": int|None, "coverage": [...], "converged": bool}。
        """
        spread = {"horizontal": self.horizontal_transfer,
                  "vertical": self.vertical_transfer}[direction]
        coverage = []
        for r in range(1, max_rounds + 1):
            spread(rounds=1, top_k=3)
            have = sum(1 for b in self.population
                       if any(m.content == meme for m in b.long_memory))
            cov = have / len(self.population)
            coverage.append(round(cov, 3))
            if cov >= threshold:
                return {"rounds": r, "coverage": coverage,
                        "threshold": threshold, "converged": True}
        return {"rounds": None, "coverage": coverage,
                "threshold": threshold, "converged": False}

    def consensus(self, stimulus: Optional[str] = None) -> Dict:
        """群体共识涌现度量（v4.0）。

        记忆共识：种群中最广泛共享的 LTM 内容的持有比例
                  （文化是否收敛到共同记忆）；
        行动共识（传入 stimulus 时）：广播刺激后最多数派动作的比例
                  （群体行为是否一致）。1.0 = 完全共识。
        """
        n = len(self.population)
        # 记忆共识
        holder: Dict[str, int] = {}
        for b in self.population:
            for m in {x.content for x in b.long_memory}:
                holder[m] = holder.get(m, 0) + 1
        mem_top, mem_idx = (None, 0.0)
        if holder:
            mem_top = max(holder, key=lambda k: holder[k])
            mem_idx = round(holder[mem_top] / n, 3)
        result: Dict[str, object] = {
            "population": n,
            "memory_consensus": {"shared_content": mem_top, "index": mem_idx},
        }
        # 行动共识
        if stimulus is not None:
            self.broadcast(stimulus)
            dist: Dict[str, int] = {}
            for b in self.population:
                a = b.decide_action(stimulus)["action"]
                dist[a] = dist.get(a, 0) + 1
            top_action = max(dist, key=lambda k: dist[k])
            result["action_consensus"] = {
                "stimulus": stimulus,
                "distribution": dist,
                "majority_action": top_action,
                "index": round(dist[top_action] / n, 3),
            }
        return result

    def reproduce(self, parent_idx: int, child_name: str,
                  mutation: float = 0.02) -> AIBrainEntity:
        """有性繁衍简化版：克隆父代 DNA，突触权重加小幅变异，
        子代加入种群（模拟代际演化）。子代世代 = 父代 + 1（v4.0）。"""
        parent = self.population[parent_idx]
        dna = parent.dump_dna()
        for k in dna["synapse"]:
            dna["synapse"][k] = min(1.0, max(0.0,
                dna["synapse"][k] + random.uniform(-mutation, mutation)))
        child = AIBrainEntity.from_dna(dna, new_name=child_name)
        child.generation = parent.generation + 1
        self.population.append(child)
        self.generation += 1
        return child


# ===================== 演示 =====================

if __name__ == "__main__":
    brain = AIBrainEntity("Brain-01", seed=42)

    stimuli = [
        "你好，我是你的创造者",
        "今天外面的天气很好",
        "神经元脉冲正在传递信号",
        "你喜欢学习新的知识吗",
        "记忆是智慧的基石",
        "你好，我是你的创造者",   # 重复输入：验证 STDP 强化
        "记忆是智慧的基石",         # 重复输入：验证记忆固化
    ]

    print("--- 感知输入与认知输出 ---")
    for s in stimuli:
        print(f"输入: {s}")
        print(f"  -> {brain.sensory_input(s)}")

    print("\n--- 动态脉冲：刺激后的回响衰减 ---")
    brain.sensory_input("记忆是智慧的基石")
    trace = brain.free_run(8)
    for i, (s, a, d) in enumerate(trace, 1):
        bar = lambda n: "█" * n
        print(f"  +{i} tick: 感官{bar(s)}{s} 联想{bar(a)}{a} 决策{bar(d)}{d}")

    print("\n--- v3.0 多模态感知（CLIP/Whisper 未安装时自动降级伪 embedding）---")
    demo_file = "demo_media.bin"
    with open(demo_file, "wb") as f:
        f.write(bytes(range(256)) * 4)  # 1KB 测试数据
    print(f"  -> {brain.perceive_image(demo_file, label='一张测试图片')}")
    print(f"  -> {brain.perceive_audio(demo_file, label='一段测试音频')}")

    print("\n--- v3.1 自定义多模态模型：注册自定义编码器 ---")
    # 自定义编码器契约：callable(path) -> 数值序列（长度任意）
    def my_image_model(path):
        """示例自定义图像模型：用文件字节直方图构造 32 维特征"""
        with open(path, "rb") as f:
            data = f.read()
        hist = [0.0] * 32
        for b in data:
            hist[b % 32] += 1.0
        total = sum(hist) or 1.0
        return [v / total for v in hist]

    register_image_encoder(my_image_model, name="hist32")
    print(f"  已注册编码器: {list_encoders()['image']}")
    out = brain.perceive_image(demo_file, label="自定义模型编码的图片")
    print(f"  -> {out}")
    print(f"  一次性指定: -> {brain.perceive_image(demo_file, encoder=my_image_model)}")
    unregister_image_encoder("hist32")  # 注销后回落到内置 CLIP / 伪 embedding 链
    os.remove(demo_file)

    print("\n--- v3.0 多巴胺奖励调制学习（强化学习）---")
    rewarded = AIBrainEntity("Rewarded", seed=7)
    control = AIBrainEntity("Control", seed=7)  # 同种子 = 相同的初始突触
    for _ in range(15):
        rewarded.reward(1.0)                     # 每次学习前给予奖励
        rewarded.sensory_input("奖励关联的刺激")
        control.sensory_input("奖励关联的刺激")  # 对照组无奖励
    print(f"  奖励组:  强连接={rewarded.strong_synapse_count()}  "
          f"平均强度={rewarded.synapse_mean():.3f}")
    print(f"  对照组:  强连接={control.strong_synapse_count()}  "
          f"平均强度={control.synapse_mean():.3f}")

    print("\n--- v3.0 群体智能：DNA 文化传递 ---")
    swarm = BrainSwarm(["Alpha", "Beta", "Gamma"], seed=1)
    for _ in range(25):  # Alpha 反复经历，把经验固化进 LTM
        swarm.population[0].sensory_input("火焰是危险的")
    n = swarm.culture_round(rounds=4, top_k=2, mode="dna")
    print(f"  完成 {n} 条记忆的跨个体传递")
    for b in swarm.population:
        hit = b.recall("火焰")
        print(f"  {b.name}: LTM={len(b.long_memory)}  "
              f"回忆'火焰' -> {[m.content for m in hit] or '无'}")

    print("\n--- v4.0 可学习投影：替代线性插值重采样 ---")
    proj_brain = AIBrainEntity("ProjBrain", seed=42)
    proj_brain.enable_projection(True)
    rng42 = random.Random(7)
    cat = [rng42.uniform(-1, 1) for _ in range(512)]
    print(f"  -> {proj_brain.sensory_input_vector(cat, label='稠密 embedding')}")
    print(f"  投影训练步数={proj_brain._projections[512].train_steps}（Oja 在线 PCA）")

    print("\n--- v4.0 RPE/TD 误差：奖励被预测后多巴胺反应衰减 ---")
    td_brain = AIBrainEntity("TDBrain", seed=7)
    r1 = td_brain.reward_td(0.8)
    print(f"  第 1 次奖励 0.8: RPE={r1['rpe']:+.3f} V={r1['value_estimate']:.3f}")
    for _ in range(30):
        td_brain.reward_td(0.8)
    r2 = td_brain.reward_td(0.8)
    print(f"  第 32 次奖励 0.8: RPE={r2['rpe']:+.3f} V={r2['value_estimate']:.3f}（已预测）")
    r3 = td_brain.reward_td(-0.5)
    print(f"  突变为 -0.5: RPE={r3['rpe']:+.3f}（意外重现）")

    print("\n--- v4.0 动作空间与语言生成 ---")
    for s in ["火焰是危险的", "今天天气晴朗"]:
        brain.sensory_input(s)
        ex = brain.express(s)
        a = ex["action"]
        print(f"  输入「{s}」")
        print(f"    动作={a['action']}(verb={a['verb']}, 强度={a['intensity']})"
              f"  情绪={a['mood']}")
        print(f"    表达: {ex['utterance']}")

    print("\n--- v4.0 文化动力学：水平 vs 垂直传播 + 共识涌现 ---")
    swarm2 = BrainSwarm(["E1", "E2", "E3"], seed=1)
    for _ in range(25):
        swarm2.population[0].sensory_input("钻木可以取火")
    child = swarm2.reproduce(0, "F1")
    child.long_memory.clear()
    hor = swarm2.transmission_dynamics("钻木可以取火", rounds=4,
                                       direction="horizontal")
    print(f"  水平传播覆盖率: {hor['coverage']}")
    swarm2.vertical_transfer(rounds=20, top_k=1)
    print(f"  垂直传播后子代 LTM: {[m.content for m in child.long_memory]}")
    con = swarm2.consensus(stimulus="钻木可以取火")
    print(f"  记忆共识指数={con['memory_consensus']['index']}  "
          f"行动共识={con['action_consensus']['majority_action']}"
          f"({con['action_consensus']['index']})")

    print("\n--- v4.1 动作执行器：决策 → 机器人执行 → 奖励回传闭环 ---")
    robot_brain = AIBrainEntity("RobotBrain", seed=42)
    robot_brain.register_executor(make_robot_executor(strictness=0.1),
                                  default=True)
    for s in ["火焰是危险的", "今天天气晴朗"]:
        out = robot_brain.act(s)
        print(f"  输入「{s}」-> {out['action']['verb']}")
        print(f"    执行: {out['execution']['detail']}")
        fb = out["feedback"]
        print(f"    反馈: RPE={fb['rpe']:+.3f} V={fb['value_estimate']:.3f} "
              f"多巴胺={fb['dopamine']:+.2f}")

    print("\n--- v4.1 社交拓扑对共识收敛的影响 ---")
    for topo in ("fully_connected", "ring", "small_world"):
        sw = BrainSwarm([f"T{i}" for i in range(8)], seed=1)
        for _ in range(25):
            sw.population[0].sensory_input("钻木可以取火")
        sw.set_topology(topo, seed=1)
        conv = sw.consensus_convergence("钻木可以取火", max_rounds=60,
                                        threshold=0.9)
        r = conv["rounds"] if conv["converged"] else ">60"
        print(f"  {topo:<16} 收敛轮数={r}")

    print("\n--- v4.4 TD(λ) 资格迹：信用分配跨 tick 反向传播 ---")
    lam_brain = AIBrainEntity("LambdaBrain", seed=1)
    cues = ["铃声", "灯光", "气味"]          # 三线索链 → 奖励
    for trial in range(1, 21):
        for c in cues:
            lam_brain.sensory_input(c)
        r = lam_brain.reward_lambda(1.0)
        if trial in (1, 5, 20):
            vs = {k: round(v, 3) for k, v in r["state_values"].items()}
            print(f"  试次{trial:2d}: RPE={r['rpe']:+.3f}  V={vs}")
    print(f"  -> 越靠近奖励的线索价值越高（γλ 梯度），"
          f"RPE 从 1.0 衰减到 {r['rpe']:+.3f}（奖励被最早线索预测）")

    print("\n--- v4.5 执行器技能学习：分 verb 独立价值 + 策略化选择 ---")
    skill_brain = AIBrainEntity("SkillBrain", seed=1)
    skill_brain.skill_epsilon = 0.4   # 基础 ε 调高：低新奇时 ε_eff=0.2，
                                      # 保证探索能找到最优动作（避免锁死次优）
    mock = {"respond": 0.8, "acknowledge": 0.2, "observe": -0.4}
    for verb, rv in mock.items():
        skill_brain.register_executor(
            (lambda r: (lambda a: {"success": r > 0, "reward": r,
                                   "detail": "mock"}))(rv), verb=verb)
    for _ in range(40):                      # ε-greedy 探索中学习
        skill_brain.act("火焰是危险的", policy="epsilon")
    q = {k: round(v, 3) for k, v in skill_brain.verb_values.items()}
    print(f"  40 轮后技能价值 Q={q}")
    picks = [skill_brain.act("火焰是危险的", policy="greedy")["action"]["verb"]
             for _ in range(20)]
    print(f"  greedy 策略 20 次选择: {picks.count('respond')}/20 选 respond"
          f"（Q 最高的动作胜出）")

    print("\n--- v4.6 检索式语言生成：LTM 片段 + 句法框架 ---")
    comp_brain = AIBrainEntity("CompBrain", seed=1)
    for _ in range(30):
        comp_brain.sensory_input("火焰是危险的")
    for _ in range(20):
        comp_brain.sensory_input("钻木可以取火")
    for _ in range(15):
        comp_brain.sensory_input("燧石可以取火")
    for s in ["取火的方法", "取火 火焰", "完全陌生的东西"]:
        comp_brain.sensory_input(s)
        c = comp_brain.compose(s)
        print(f"  「{s}」片段={len(c['fragments'])}: "
              f"{[f[:6] + '…' for f in c['fragments']]}")
        print(f"    {c['utterance']}")

    print("\n--- v4.7 情景记忆时间索引：'上次……之后' 式时间推理 ---")
    ep_brain = AIBrainEntity("EpBrain", seed=1)
    for s in ["起床", "刷牙", "吃早餐", "出门", "刷牙", "上班"]:
        ep_brain.sensory_input(s)
    r = ep_brain.events_after("刷牙")
    print(f"  锚点「刷牙」@tick{r['anchor']['tick']}（上次，覆盖更早的一次）")
    for e in r["events"]:
        print(f"    +{e['delta']} tick 之后: {e['content']}")
    r2 = ep_brain.events_before("吃早餐")
    print(f"  「吃早餐」之前: "
          f"{[(e['content'], e['delta']) for e in r2['events']]}")
    print(f"  情景共现: tick3 与 {ep_brain.episodes[2]['context']} 同时经历")

    print("\n--- v4.8 睡眠-清醒节律：离线重放固化 + 突触稳态缩放 ---")
    sleep_brain = AIBrainEntity("SleepBrain", seed=1)
    sleep_brain.sensory_input("萤火虫在夜里发光")   # 弱刺激：白天无法固化
    sleep_brain.sensory_input("另一个无关刺激")
    sleep_brain.reward(-0.8)
    print(f"  睡前: STM={len(sleep_brain.short_memory)} "
          f"LTM={len(sleep_brain.long_memory)} "
          f"压力={sleep_brain.emotion['stress']:.2f}")
    r = sleep_brain.sleep(cycles=3)
    print(f"  睡眠 3 周期: 重放{r['replayed']}条 固化{r['consolidated']}条 "
          f"压力 {r['stress_before']}→{r['stress_after']}")
    print(f"  醒后: STM={len(sleep_brain.short_memory)} "
          f"LTM={len(sleep_brain.long_memory)}（弱刺激经重放完成固化）")
    deep = AIBrainEntity("DeepSleep", seed=1)
    before = len(deep.synapse)
    r2 = deep.sleep(cycles=12)
    print(f"  深睡 12 周期: 突触 {before}→{r2['synapses']}"
          f"（剪除 {r2['pruned_synapses']} 条弱连接，强连接存活）")

    print("\n--- v4.9 好奇驱动探索：新奇度反向调制注意与 ε ---")
    nov_brain = AIBrainEntity("NovBrain", seed=1)
    for _ in range(30):
        nov_brain.sensory_input("火焰是危险的")
    for s in ["火焰", "量子纠缠态坍缩", "火焰"]:
        nov_brain.sensory_input(s)
        tag = "熟悉" if s == "火焰" else "全新"
        print(f"  {tag}「{s}」: 新奇度={nov_brain.novelty:.2f} "
              f"注意={nov_brain.attention_factor:.2f} "
              f"ε_eff={nov_brain.effective_epsilon():.3f}")
    nov_brain.reward_lambda(1.0)     # 大意外
    nov_brain.sensory_input("火焰")
    print(f"  大意外后「火焰」: 新奇度={nov_brain.novelty:.2f}"
          f"（|RPE| 让熟悉刺激也重获注意）")

    print("\n--- 反复强化记忆（触发 STM -> LTM 固化）---")
    for _ in range(30):
        brain.sensory_input("记忆是智慧的基石")

    print("\n--- 模拟时间流逝（记忆衰减）---")
    for _ in range(50):
        brain.decay_memory()

    print("\n--- 联想回忆测试 ---")
    for m in brain.recall("记忆"):
        print(f"  回忆到: {m.content!r} 权重={m.weight:.2f} 标签={m.tag}")

    print("\n--- 脉冲思考链：一次感知的完整因果链 ---")
    for line in brain.thought_chain("火焰是危险的")["chain"]:
        print(f"  {line}")

    print()
    print(brain.status())

    print("\n--- DNA 记忆遗传：保存并克隆实体 ---")
    os.makedirs("data", exist_ok=True)
    dna_path = os.path.join("data", "brain_dna.json")
    brain.save_dna(dna_path)
    clone = AIBrainEntity.load_dna(dna_path, new_name="Brain-02")
    print(f"  克隆体 {clone.name} 继承了 LTM={len(clone.long_memory)} 条长期记忆")
    print(f"  克隆体回忆: {[m.content for m in clone.recall('记忆')]}")
