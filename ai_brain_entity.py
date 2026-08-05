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
  1. 社交拓扑与共识相变：set_topology() 支持全连接/环形/星形/随机/
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
  - 技能学习：learn_skill() 为每个动作 verb 维护独立价值
    估计 Q(verb)（后果精确归因到实际采取的动作），select_verb()
    支持 greedy / ε-greedy / softmax 策略。实测
    （respond=0.8/acknowledge=0.2/observe=-0.4，40 轮学习）：
    Q 收敛 0.80/0.19/-0.30，greedy 策略 20/20 选高价值动作

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

v5.1 新增（动作与决策扩展）：
  1. 动作空间 3 verb → 8 verb：INTENT_VERBS 意图动词表（ask 提问澄清 /
     retrieve 检索回忆 / plan 规划分解 / execute 实施行动 / wait 延迟
     观望），与脉冲强度动作正交，每个 verb 独立 Q 值，由 select_verb()
     策略化选择；配套专属语言模板
  2. 深思熟虑决策：decide_action(deliberate=True) 决策前检索记忆、查
     Q 值、看新奇度与资格迹，输出带 rationale 理由链的决策（base_verb
     对照 + q_values 快照）；规划能力从"记忆+价值"涌现而非堆神经元

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


# ------------------ 语言生成器注册（v5.9） ------------------
# 大脑负责"想什么"（决策/记忆/情绪/意识焦点），外部语言模型负责
# "说出来"（自然语言表述）。未注册时 express()/chat() 走内置模板，
# 注册后优先用生成器，生成失败自动降级回模板（与编码器同一风格）。
#
# 语言生成器契约：callable(context: Dict) -> str
#   context = {"brain_name", "stimulus", "verb", "action", "mood",
#              "recalled": [...], "top_thought": str|None}
#   返回空串或抛异常 → 调用方降级回模板。

_LANGUAGE_GENERATORS: Dict[str, Callable] = {}
_DEFAULT_LANGUAGE_GENERATOR: Optional[str] = None


def register_language_generator(fn: Callable, name: str = "custom",
                                default: bool = True) -> str:
    """注册自定义语言生成器。default=True 时设为全局默认。"""
    global _DEFAULT_LANGUAGE_GENERATOR
    if not callable(fn):
        raise TypeError("generator 必须是 callable(context) -> str")
    _LANGUAGE_GENERATORS[name] = fn
    if default:
        _DEFAULT_LANGUAGE_GENERATOR = name
    return name


def unregister_language_generator(name: str) -> None:
    """注销语言生成器；若注销的是当前默认，回落到内置模板。"""
    global _DEFAULT_LANGUAGE_GENERATOR
    _LANGUAGE_GENERATORS.pop(name, None)
    if _DEFAULT_LANGUAGE_GENERATOR == name:
        _DEFAULT_LANGUAGE_GENERATOR = None


def get_language_generator_info() -> Dict[str, object]:
    """当前语言生成器注册状态（便于调试/观测台展示）。"""
    return {"custom": sorted(_LANGUAGE_GENERATORS),
            "default": _DEFAULT_LANGUAGE_GENERATOR}


def set_qwen_model(model_path: Optional[str] = None,
                   device: str = "cpu", name: str = "qwen") -> Dict:
    """接入 Qwen2 语言模型（models/generators/qwen_generator.py）。

    模型不存在或 transformers 未安装时仍然注册——其 __call__ 返回空串，
    express()/chat() 自动降级回模板；模型下载到位后无需改代码即生效。
    返回 {"registered", "available", "model_path", "error"}。
    """
    import sys
    gen_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "models", "generators")
    if gen_dir not in sys.path:
        sys.path.insert(0, gen_dir)
    from qwen_generator import get_qwen_generator
    gen = get_qwen_generator(model_path=model_path, device=device)
    register_language_generator(gen, name=name, default=True)
    return {"registered": name, "available": gen.available,
            "model_path": gen.model_path, "error": gen._error}


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
        import numpy as np
        model, processor = _CLIP_CACHE
        inputs = processor(images=Image.open(path), return_tensors="pt")
        feats = model.get_image_features(**inputs)
        # 兼容：新版 transformers 可能返回 ModelOutput 或多维张量，
        # 统一展平为一维 float 列表（下游投影/重采样要求扁平数值序列）
        arr = getattr(feats, "pooler_output", feats)
        if not hasattr(arr, "detach"):
            arr = feats
        vec = np.asarray(arr.detach().cpu().numpy(), dtype=float).reshape(-1)
        return vec.tolist()
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
    """记忆单元（v5.3 增加情绪色彩）"""
    content: str
    timestamp: float
    weight: float    # 记忆权重（强度）
    tag: str         # 记忆标签：sensory / emotion / event / culture / thought
    modality: str = "text"  # 模态：text / visual / auditory / multimodal
    features: list = None   # 模态特征向量（用于跨模态联想）
    emotion: str = "neutral"  # 情绪色彩：positive / negative / neutral
    emotional_intensity: float = 0.0  # 情绪强度 [0,1]


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

        # v4.5 技能学习：每个 verb 独立价值估计 Q(verb)
        # v5.1：覆盖全部意图动词（INTENT_VERBS），含脉冲三动作
        self.verb_values: Dict[str, float] = {
            v: 0.0 for v in self.INTENT_VERBS}
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
        # v5.2 念头流水账：每个进入意识的念头留痕（自我意识追溯用）
        self.thought_journal: List[Dict] = []   # {content, source, tick}

        # v6.0 LanceDB 记忆后端：attach_memory_store() 接入后，
        # LTM 固化/强化/衰减会同步到本地向量库；None = 纯内存模式
        self.memory_store = None
        # v6.1：STM 全量同步开关（attach_memory_store(sync_stm=True)）；
        # DNA 基因库（attach_dna_library() 接入）
        self.sync_stm = False
        self.dna_library = None
        # v6.2 文本语义编码器：attach_text_encoder() 接入后，
        # 文本记忆携带真语义 embedding（recall_semantic 真语义检索）；
        # None = 哈希向量兜底（字面近似）
        self.text_encoder = None
        # v5.3 自我概念：关于"我是谁"的核心信念集合
        self.self_concept: List[str] = []       # 自我概念条目
        # v5.3 自传体记忆：个人经历的时间线（重要事件）
        self.autobiographical_memory: List[Dict] = []  # {tick, event, emotion, importance}
        # v5.4 全局工作空间理论（GWT）：
        # 专门模块（无意识处理器）：记忆、情绪、感知、语言等
        self.specialized_modules: Dict[str, Dict] = {}  # {模块名: {状态, 输出}}
        # 全局广播历史：意识内容被广播给所有模块的记录
        self.broadcast_history: List[Dict] = []
        # 点火状态：内容进入意识时的全脑激活
        self.ignition_state: bool = False
        self.ignition_count: int = 0  # 点火次数（意识涌现次数）
        # v5.6 心智理论（ToM）：对其他大脑的心理模型
        self.mental_models: Dict[str, Dict] = {}  # {大脑名: {信念, 欲望, 意图, 情绪}}
        self.tom_accuracy: float = 0.7  # 心智理论的准确度（0-1）
        # v5.7 高阶意识理论（HOT）：对意识的意识（元意识）
        self.hot_level: int = 0  # 当前意识层级（0=无意识, 1=一阶, 2=二阶, 3=三阶...）
        self.hot_history: List[Dict] = []  # 高阶思想历史
        self.meta_awareness: float = 0.0  # 元意识程度 [0, 1]

        # v4.8 睡眠-清醒节律：离线重放固化 + 突触稳态缩放（SHY）
        self.sleep_replay_gain = 0.15       # 每次重放的 STM 权重增益
        self.sleep_downscale = 0.95         # 突触等比下调系数（保留相对差异）
        self.synapse_prune_threshold = 0.08  # 前馈突触剪除下限
        self.recurrent_prune_threshold = 0.04  # 循环突触剪除下限
        self.max_stm = 20
        self.max_ltm = 500
        self.stm_consolidate_threshold = 0.55   # STM 权重超过此值固化进 LTM
        self.forget_threshold = 0.05            # 权重低于此值被遗忘

        # v5.9 增强版睡眠系统
        self.sleep_state = "awake"  # 睡眠状态: awake/drowsy/light_sleep/deep_sleep/rem
        self.sleep_cycle_count = 0  # 已完成的睡眠周期数
        self.dream_count = 0        # 做过的梦的数量
        self.dream_log: List[Dict] = []  # 梦境日志
        self.sleep_spindles = 0     # 睡眠纺锤波计数（浅睡）
        self.slow_waves = 0         # 慢波计数（深睡）
        self.rem_duration = 0       # REM睡眠总时长
        self.memory_consolidation_count = 0  # 记忆巩固总次数
        self.ltm_integration_count = 0      # LTM整合次数

        # 睡眠参数
        self.dream_vividness = 0.7  # 梦境生动程度 [0,1]
        self.rem_memory_gain = 0.1  # REM睡眠中记忆整合增益
        self.deep_sleep_replay_rate = 0.8  # 深睡中记忆重放比例
        self.sleep_cycle_length = 5  # 每个睡眠周期的tick数（简化版）

        # v6.0 工作记忆系统（Baddeley模型）
        # 1. 语音回路（Phonological Loop）
        self.phonological_loop: List[Dict] = []  # 语音存储
        self.phonological_capacity = 7  # 语音回路容量（7±2）
        self.phonological_decay_rate = 0.15  # 语音信息衰减率
        self.subvocal_rehearsal_rate = 0.3  # 默读复述速率

        # 2. 视空间模板（Visuospatial Sketchpad）
        self.visuospatial_sketchpad: List[Dict] = []  # 视空间存储
        self.visuospatial_capacity = 4  # 视空间容量
        self.visuospatial_decay_rate = 0.2  # 视空间信息衰减率

        # 3. 情景缓冲器（Episodic Buffer）
        self.episodic_buffer: List[Dict] = []  # 情景缓冲
        self.episodic_capacity = 4  # 情景缓冲容量
        self.episodic_decay_rate = 0.1  # 情景信息衰减率

        # 4. 中央执行系统（Central Executive）
        self.central_executive_load = 0.0  # 认知负荷 [0,1]
        self.task_switching_cost = 0.2  # 任务切换代价
        self.current_task: Optional[str] = None  # 当前任务
        self.task_history: List[str] = []  # 任务历史

        # 工作记忆参数
        self.working_memory_capacity = 7  # 总工作记忆容量（7±2）
        self.rehearsal_effort = 0.1  # 复述需要的认知努力
        self.wm_interference = 0.05  # 不同子系统间的干扰

        # v6.1 预测编码系统（Predictive Coding）
        # 预测状态
        self.current_prediction: Optional[Dict] = None  # 当前预测
        self.prediction_history: List[Dict] = []  # 预测历史
        self.prediction_error_history: List[Dict] = []  # 预测误差历史

        # 预测精度（置信度）
        self.prediction_precision = 0.5  # 预测精度 [0,1]
        self.precision_history: List[float] = []  # 精度历史

        # 内部模型
        self.internal_model: Dict[str, float] = {}  # 内部模型参数
        self.model_uncertainty = 0.5  # 模型不确定性 [0,1]

        # 预测编码参数
        self.prediction_horizon = 3  # 预测时间范围（tick数）
        self.prediction_learning_rate = 0.1  # 预测学习率
        self.error_minimization_rate = 0.05  # 误差最小化速率
        self.precision_weighting = True  # 是否启用精度加权

        # 预测层级
        self.sensory_predictions: List[Dict] = []  # 感官层预测
        self.conceptual_predictions: List[Dict] = []  # 概念层预测
        self.abstract_predictions: List[Dict] = []  # 抽象层预测

        # 自由能（Free Energy）- 预测编码的核心量
        self.variational_free_energy = 0.0  # 变分自由能
        self.free_energy_history: List[float] = []  # 自由能历史

        # v6.2 神经振荡系统（脑电波）
        # 五种脑电波的功率（相对强度 [0,1]）
        self.brainwaves = {
            "delta": 0.1,    # δ波 0.5-4 Hz：深睡、无意识
            "theta": 0.2,    # θ波 4-8 Hz：浅睡、冥想、困倦
            "alpha": 0.4,    # α波 8-13 Hz：清醒放松、闭眼
            "beta": 0.5,     # β波 13-30 Hz：清醒活跃、思考、专注
            "gamma": 0.3,    # γ波 30-100+ Hz：高级认知、意识、感知整合
        }

        # 神经同步性
        self.neural_synchrony = 0.5  # 整体神经同步性 [0,1]
        self.synchrony_history: List[float] = []  # 同步性历史

        # 振荡相位
        self.oscillation_phase = 0.0  # 当前振荡相位 [0, 2π]
        self.oscillation_frequency = 10.0  # 主导频率（Hz，简化）

        # 跨频耦合（相位-幅度耦合）
        self.cross_frequency_coupling = 0.3  # 跨频耦合强度 [0,1]

        # 脑电波历史（用于频谱分析）
        self.brainwave_history: List[Dict[str, float]] = []

        # 神经振荡参数
        self.oscillation_modulation_rate = 0.05  # 振荡调制速率
        self.synchrony_decay = 0.02  # 同步性衰减率
        self.gamma_binding_strength = 0.4  # γ波绑定强度

        # v6.3 主动推理系统（Active Inference）
        # 行动策略
        self.action_space: List[str] = [
            "explore",      # 探索：寻找新信息
            "exploit",      # 利用：使用已知信息
            "wait",         # 等待：观察更多信息
            "focus",        # 专注：集中注意力
            "relax",        # 放松：降低唤醒
            "learn",        # 学习：强化记忆
            "socialize",    # 社交：与他人互动
        ]
        self.current_action: Optional[str] = None  # 当前行动
        self.action_history: List[Dict] = []  # 行动历史

        # 行动预测
        self.action_predictions: Dict[str, Dict] = {}  # 各行动的预测结果
        self.expected_free_energy: Dict[str, float] = {}  # 各行动的预期自由能

        # 目标和偏好
        self.goals: List[Dict] = []  # 当前目标列表
        self.preferences: Dict[str, float] = {
            "novelty": 0.5,      # 对新奇性的偏好
            "certainty": 0.5,    # 对确定性的偏好
            "pleasure": 0.5,     # 对愉悦的偏好
            "social": 0.3,       # 对社交的偏好
            "learning": 0.4,     # 对学习的偏好
        }

        # 主动推理参数
        self.active_inference_enabled = True  # 主动推理开关
        self.action_selection_temperature = 0.3  # 行动选择温度（softmax）
        self.action_learning_rate = 0.1  # 行动学习率
        self.exploration_bonus = 0.2  # 探索奖励

        # v6.4 脑区分化系统（Brain Regionalization）
        # 各脑区状态
        self.brain_regions = {
            "hippocampus": {
                "name": "海马体",
                "function": "记忆形成、情景记忆、空间导航",
                "activity": 0.5,
                "size": 100,  # 神经元数量（相对）
            },
            "prefrontal": {
                "name": "前额叶皮层",
                "function": "决策、规划、工作记忆、执行控制",
                "activity": 0.6,
                "size": 150,
            },
            "amygdala": {
                "name": "杏仁核",
                "function": "情绪处理、恐惧条件反射、情绪记忆",
                "activity": 0.4,
                "size": 50,
            },
            "sensory_cortex": {
                "name": "感觉皮层",
                "function": "视觉、听觉、躯体感觉处理",
                "activity": 0.7,
                "size": 200,
            },
            "association_cortex": {
                "name": "联合皮层",
                "function": "多模态整合、抽象思维、语义",
                "activity": 0.5,
                "size": 180,
            },
        }

        # 海马体专用状态
        self.hippocampus = {
            "episodic_memory": [],  # 情景记忆
            "spatial_map": {},      # 空间地图
            "replay_count": 0,      # 记忆重放次数
            "pattern_separation": 0.5,  # 模式分离能力
            "pattern_completion": 0.5,  # 模式补全能力
        }

        # 前额叶专用状态
        self.prefrontal = {
            "plans": [],            # 计划列表
            "goals": [],            # 目标列表
            "inhibition": 0.5,      # 抑制控制能力
            "cognitive_control": 0.5,  # 认知控制能力
            "task_set": None,       # 当前任务集
        }

        # 杏仁核专用状态
        self.amygdala_state = {
            "fear_conditioning": {},  # 恐惧条件反射
            "emotional_memory": [],   # 情绪记忆
            "threat_detection": 0.5,  # 威胁检测灵敏度
            "emotional_intensity": 0.5,  # 情绪强度
        }

        # 脑区间连接强度
        self.region_connections = {
            ("hippocampus", "prefrontal"): 0.6,  # 海马→前额叶（记忆检索）
            ("prefrontal", "hippocampus"): 0.5,  # 前额叶→海马（记忆编码）
            ("amygdala", "prefrontal"): 0.7,     # 杏仁核→前额叶（情绪影响决策）
            ("prefrontal", "amygdala"): 0.4,     # 前额叶→杏仁核（情绪调节）
            ("sensory_cortex", "hippocampus"): 0.8,  # 感觉→海马（记忆编码）
            ("sensory_cortex", "amygdala"): 0.6,     # 感觉→杏仁核（情绪触发）
            ("sensory_cortex", "association_cortex"): 0.7,  # 感觉→联合
            ("association_cortex", "prefrontal"): 0.8,      # 联合→前额叶
            ("hippocampus", "amygdala"): 0.5,  # 海马→杏仁核（情绪记忆）
        }

        # 脑区激活历史
        self.region_activity_history: Dict[str, List[float]] = {
            region: [] for region in self.brain_regions
        }

        # v6.5 推理与规划系统（Reasoning & Planning）
        # 推理状态
        self.reasoning = {
            "active": False,
            "depth": 0,  # 当前推理深度
            "max_depth": 5,  # 最大推理深度
            "reasoning_count": 0,  # 推理次数
        }

        # 逻辑推理
        self.logic = {
            "rules": [],  # 逻辑规则库
            "facts": [],  # 已知事实
            "deductions": [],  # 演绎结论
            "inductions": [],  # 归纳结论
            "abductions": [],  # 溯因结论
        }

        # 因果推理
        self.causal = {
            "causal_relations": [],  # 因果关系库
            "causal_chains": [],  # 因果链
            "attributions": [],  # 归因结果
            "counterfactuals": [],  # 反事实推理
        }

        # 规划系统
        self.planning = {
            "current_plan": None,  # 当前计划
            "plan_history": [],  # 计划历史
            "subgoals": [],  # 子目标
            "planning_count": 0,  # 规划次数
        }

        # 问题解决
        self.problem_solving = {
            "current_problem": None,  # 当前问题
            "problem_history": [],  # 问题历史
            "strategies": [],  # 策略库
            "solutions": [],  # 解决方案库
        }

        # 决策系统
        self.decision_making = {
            "current_decision": None,  # 当前决策
            "decision_history": [],  # 决策历史
            "options": [],  # 选项列表
            "risk_assessment": {},  # 风险评估
        }

        # 推理参数
        self.reasoning_learning_rate = 0.1  # 推理学习率
        self.logic_confidence_threshold = 0.6  # 逻辑置信度阈值
        self.causal_strength_threshold = 0.5  # 因果强度阈值

        # v6.6 心理模拟系统（Mental Simulation）
        # 心理表象
        self.mental_imagery = {
            "active": False,
            "vividness": 0.5,  # 表象生动度
            "current_image": None,  # 当前心理图像
            "imagery_count": 0,  # 表象次数
            "rotation_angle": 0,  # 心理旋转角度
        }

        # 心理时间旅行
        self.mental_time_travel = {
            "past_remembered": 0,  # 回忆过去的次数
            "future_imagined": 0,  # 想象未来的次数
            "temporal_distance": 0,  # 当前时间距离
            "direction": None,  # past / future
        }

        # 心理模拟
        self.mental_simulation = {
            "active": False,
            "simulation_type": None,  # action / dialogue / problem
            "current_simulation": None,  # 当前模拟内容
            "simulation_count": 0,  # 模拟次数
            "simulation_depth": 0,  # 模拟深度
        }

        # 创造力
        self.creativity = {
            "creative_thoughts": [],  # 创造性想法
            "combinations": [],  # 概念组合
            "analogies": [],  # 类比推理
            "insights": [],  # 灵感洞见
            "creativity_level": 0.5,  # 创造力水平
            "divergent_thinking": 0.5,  # 发散思维
            "convergent_thinking": 0.5,  # 聚合思维
        }

        # 默认模式网络（DMN）
        self.default_mode_network = {
            "active": False,
            "activity_level": 0.3,
            "mind_wandering": False,
            "spontaneous_thoughts": [],
        }

        # 心理模拟参数
        self.imagery_vividness_base = 0.5  # 基础表象生动度
        self.simulation_max_depth = 5  # 最大模拟深度
        self.creativity_bonus = 0.2  # 创造力奖励
        self.dmn_activation_threshold = 0.4  # DMN激活阈值

        # v6.7 发育过程系统（Development）
        # 发育阶段
        self.development = {
            "stage": "newborn",  # 当前发育阶段
            "age": 0,  # 发育年龄（月）
            "developmental_milestones": [],  # 发育里程碑
            "stage_history": [],  # 阶段历史
        }

        # 发育阶段定义
        self.developmental_stages = {
            "newborn": {
                "name": "新生儿期",
                "age_range": (0, 1),  # 0-1个月
                "description": "基本反射和感知觉发展",
                "abilities": ["基本感知", "反射行为", "简单学习"],
            },
            "infant": {
                "name": "婴儿期",
                "age_range": (1, 12),  # 1-12个月
                "description": "依恋形成、客体永久性、运动发展",
                "abilities": ["客体永久性", "依恋形成", "模仿学习"],
            },
            "toddler": {
                "name": "幼儿期",
                "age_range": (12, 36),  # 1-3岁
                "description": "语言爆发、符号思维、自我意识",
                "abilities": ["语言习得", "符号思维", "自我意识"],
            },
            "child": {
                "name": "儿童期",
                "age_range": (36, 132),  # 3-11岁
                "description": "逻辑推理、守恒概念、具体运算",
                "abilities": ["逻辑推理", "守恒概念", "具体运算"],
            },
            "adolescent": {
                "name": "青少年期",
                "age_range": (132, 216),  # 11-18岁
                "description": "抽象思维、元认知、身份认同",
                "abilities": ["抽象思维", "元认知", "假设演绎"],
            },
            "adult": {
                "name": "成人期",
                "age_range": (216, 9999),  # 18岁以上
                "description": "专业技能、智慧、晶体智力",
                "abilities": ["专业知识", "实践智慧", "晶体智力"],
            },
        }

        # 皮亚杰认知发展阶段
        self.piaget_stages = {
            "sensorimotor": {
                "name": "感知运动阶段",
                "age_range": (0, 24),  # 0-2岁
                "description": "通过感知和动作认识世界",
                "key_achievement": "客体永久性",
            },
            "preoperational": {
                "name": "前运算阶段",
                "age_range": (24, 84),  # 2-7岁
                "description": "符号思维、语言发展、自我中心",
                "key_achievement": "符号功能",
            },
            "concrete_operational": {
                "name": "具体运算阶段",
                "age_range": (84, 132),  # 7-11岁
                "description": "逻辑推理、守恒、分类",
                "key_achievement": "守恒概念",
            },
            "formal_operational": {
                "name": "形式运算阶段",
                "age_range": (132, 9999),  # 11岁以上
                "description": "抽象思维、假设演绎、元认知",
                "key_achievement": "抽象推理",
            },
        }

        # 神经发育
        self.neural_development = {
            "neurogenesis_rate": 0.0,  # 神经发生产率
            "synaptogenesis_rate": 0.0,  # 突触发生产率
            "myelination_level": 0.0,  # 髓鞘化水平
            "pruning_rate": 0.0,  # 突触修剪率
            "synaptic_density": 0.5,  # 突触密度
            "neural_complexity": 0.3,  # 神经复杂度
        }

        # 关键期
        self.critical_periods = {
            "language": {
                "name": "语言关键期",
                "start_age": 0,
                "end_age": 84,  # 7岁
                "peak_age": 24,  # 2岁
                "sensitivity": 1.0,  # 当前敏感度
                "active": True,
            },
            "vision": {
                "name": "视觉关键期",
                "start_age": 0,
                "end_age": 36,  # 3岁
                "peak_age": 6,  # 6个月
                "sensitivity": 1.0,
                "active": True,
            },
            "social": {
                "name": "社交关键期",
                "start_age": 6,  # 6个月
                "end_age": 72,  # 6岁
                "peak_age": 36,  # 3岁
                "sensitivity": 1.0,
                "active": True,
            },
        }

        # 发育参数
        self.development_rate = 0.1  # 发育速率
        self.plasticity_level = 1.0  # 可塑性水平
        self.experience_dependent_gain = 0.05  # 经验依赖增益

        # v6.8 具身认知系统（Embodied Cognition）
        # 身体图式
        self.body_schema = {
            "active": False,
            "body_parts": {},  # 身体各部位
            "posture": "standing",  # 当前姿态
            "proprioception": 0.5,  # 本体感觉灵敏度
            "body_awareness": 0.5,  # 身体意识
        }

        # 身体部位定义
        self.body_parts_definition = {
            "head": {"name": "头部", "sensitivity": 0.9, "motor_control": 0.8},
            "left_arm": {"name": "左臂", "sensitivity": 0.7, "motor_control": 0.9},
            "right_arm": {"name": "右臂", "sensitivity": 0.7, "motor_control": 0.9},
            "left_hand": {"name": "左手", "sensitivity": 0.95, "motor_control": 0.95},
            "right_hand": {"name": "右手", "sensitivity": 0.95, "motor_control": 0.95},
            "torso": {"name": "躯干", "sensitivity": 0.5, "motor_control": 0.3},
            "left_leg": {"name": "左腿", "sensitivity": 0.6, "motor_control": 0.8},
            "right_leg": {"name": "右腿", "sensitivity": 0.6, "motor_control": 0.8},
            "left_foot": {"name": "左脚", "sensitivity": 0.8, "motor_control": 0.7},
            "right_foot": {"name": "右脚", "sensitivity": 0.8, "motor_control": 0.7},
        }

        # 运动系统
        self.motor_system = {
            "active": False,
            "current_action": None,  # 当前动作
            "action_history": [],  # 动作历史
            "motor_learning_rate": 0.1,  # 运动学习率
            "motor_skill_level": 0.5,  # 运动技能水平
            "reaction_time": 0.5,  # 反应时间
            "coordination": 0.5,  # 协调性
        }

        # 运动技能库
        self.motor_skills = {}  # 动作技能

        # 感知-行动循环
        self.sensorimotor_loop = {
            "active": False,
            "loop_count": 0,  # 循环次数
            "prediction_error": 0.0,  # 预测误差
            "feedback_delay": 0.1,  # 反馈延迟
            "integration_level": 0.5,  # 感觉运动整合水平
        }

        # 镜像神经元系统
        self.mirror_neuron_system = {
            "active": False,
            "observation_activation": 0.0,  # 观察激活
            "execution_activation": 0.0,  # 执行激活
            "imitation_ability": 0.5,  # 模仿能力
            "empathy_level": 0.5,  # 共情水平
            "mirror_neurons_count": 0,  # 镜像神经元数量
        }

        # 环境交互
        self.environment_interaction = {
            "objects_manipulated": [],  # 操作过的物体
            "tools_used": [],  # 使用过的工具
            "spatial_navigation": 0.5,  # 空间导航能力
            "tool_use_skill": 0.3,  # 工具使用技能
            "affordance_perception": 0.5,  # 功能可供性感知
        }

        # 具身认知参数
        self.embodiment_level = 0.5  # 具身化程度
        self.sensorimotor_continuity = 0.5  # 感觉运动连续性
        self.action_perception_coupling = 0.5  # 行动-感知耦合

        # v6.9 文化进化系统（Cultural Evolution）
        # 文化传递
        self.cultural_transmission = {
            "active": False,
            "transmission_count": 0,  # 传递次数
            "imitation_rate": 0.5,  # 模仿率
            "teaching_rate": 0.3,  # 教学率
            "language_rate": 0.7,  # 语言传播率
            "vertical_transmission": 0,  # 垂直传递（父母→子女）
            "horizontal_transmission": 0,  # 水平传递（同龄人）
            "oblique_transmission": 0,  # 斜向传递（其他长辈）
        }

        # 文化变异
        self.cultural_variation = {
            "active": False,
            "innovation_rate": 0.1,  # 创新率
            "drift_rate": 0.05,  # 漂变速率
            "recombination_rate": 0.2,  # 重组率
            "innovation_count": 0,  # 创新次数
            "drift_count": 0,  # 漂移次数
            "recombination_count": 0,  # 重组次数
        }

        # 文化选择
        self.cultural_selection = {
            "active": False,
            "natural_selection": 0.3,  # 自然选择压力
            "cultural_selection": 0.5,  # 文化选择压力
            "frequency_dependent": 0.2,  # 频率依赖选择
            "selection_count": 0,  # 选择次数
            "selected_traits": [],  # 被选择的特征
        }

        # 模因系统（文化的基本单位）
        self.memes = {}  # 模因库
        self.meme_system = {
            "active": False,
            "total_memes": 0,  # 模因总数
            "active_memes": 0,  # 活跃模因数
            "meme_diversity": 0.0,  # 模因多样性
            "meme_complexity": 0.0,  # 模因复杂性
            "replication_rate": 0.5,  # 复制率
            "mutation_rate": 0.1,  # 突变率
        }

        # 群体文化
        self.group_culture = {
            "active": False,
            "norms": {},  # 群体规范
            "identity": 0.5,  # 群体认同
            "rituals": [],  # 文化仪式
            "shared_values": [],  # 共享价值观
            "cultural_identity": 0.3,  # 文化认同
            "conformity_bias": 0.4,  # 从众偏差
            "prestige_bias": 0.3,  # 声望偏差
        }

        # 文化演化动力学
        self.cultural_dynamics = {
            "generation": 0,  # 文化世代
            "diversity": 0.5,  # 文化多样性
            "complexity": 0.3,  # 文化复杂性
            "change_rate": 0.1,  # 文化变迁速率
            "cumulative_culture": 0.0,  # 累积文化
        }

        # 文化进化参数
        self.cultural_evolution_rate = 0.1  # 文化进化速率
        self.cultural_capacity = 0.5  # 文化容量
        self.social_learning_bias = 0.5  # 社会学习偏差

        # v7.0 终身学习系统（Lifelong Learning）
        # 持续学习
        self.continual_learning = {
            "active": False,
            "learning_count": 0,  # 学习次数
            "incremental_learning": 0,  # 增量学习次数
            "online_learning": 0,  # 在线学习次数
            "transfer_learning": 0,  # 迁移学习次数
            "knowledge_retention": 0.8,  # 知识保留率
            "catastrophic_forgetting": 0.1,  # 灾难性遗忘率
        }

        # 知识巩固
        self.knowledge_consolidation = {
            "active": False,
            "consolidation_count": 0,  # 巩固次数
            "spaced_repetition": 0,  # 间隔重复次数
            "active_recall": 0,  # 主动回忆次数
            "interleaved_practice": 0,  # 交错练习次数
            "consolidation_rate": 0.5,  # 巩固速率
            "retrieval_strength": 0.5,  # 提取强度
            "storage_strength": 0.5,  # 存储强度
        }

        # 技能提升
        self.skill_improvement = {
            "active": False,
            "deliberate_practice": 0,  # 刻意练习次数
            "feedback_loops": 0,  # 反馈循环次数
            "skill_transfer": 0,  # 技能迁移次数
            "practice_count": 0,  # 练习总次数
            "skill_growth_rate": 0.1,  # 技能增长率
            "plateau_resistance": 0.5,  # 平台期抗性
        }

        # 元学习（学习如何学习）
        self.meta_learning = {
            "active": False,
            "meta_learning_count": 0,  # 元学习次数
            "learning_strategy": "default",  # 当前学习策略
            "strategy_effectiveness": 0.5,  # 策略有效性
            "learning_to_learn": 0.3,  # 学习如何学习的能力
            "optimal_strategy": None,  # 最优策略
            "strategy_adaptation_rate": 0.1,  # 策略适应率
        }

        # 学习策略库
        self.learning_strategies = {
            "default": {"effectiveness": 0.5, "description": "默认学习策略"},
            "spaced_repetition": {"effectiveness": 0.7, "description": "间隔重复"},
            "active_recall": {"effectiveness": 0.75, "description": "主动回忆"},
            "deliberate_practice": {"effectiveness": 0.8, "description": "刻意练习"},
            "interleaved": {"effectiveness": 0.65, "description": "交错练习"},
            "elaborative": {"effectiveness": 0.7, "description": "精细加工"},
        }

        # 适应机制
        self.adaptation_mechanism = {
            "active": False,
            "adaptation_count": 0,  # 适应次数
            "environment_adaptation": 0.5,  # 环境适应能力
            "task_adaptation": 0.5,  # 任务适应能力
            "strategy_adjustment": 0.5,  # 策略调整能力
            "adaptation_rate": 0.1,  # 适应速率
            "flexibility": 0.5,  # 灵活性
        }

        # 终身学习参数
        self.lifelong_learning_rate = 0.05  # 终身学习速率
        self.learning_capacity = 0.7  # 学习容量
        self.cognitive_reserve = 0.5  # 认知储备
        self.neuroplasticity_maintenance = 0.8  # 神经可塑性维持

        # v7.1 意识整合系统（Consciousness Integration）
        # 意识统一框架
        self.consciousness_framework = {
            "active": False,
            "integration_level": 0.5,  # 整合水平
            "unity_level": 0.5,  # 统一水平
            "coherence_level": 0.5,  # 连贯水平
            "temporal_depth": 0.5,  # 时间深度
            "self_reference": 0.5,  # 自我参照
        }

        # 意识状态
        self.consciousness_state = {
            "state": "wakeful",  # wakeful, drowsy, sleep, meditation, flow
            "clarity": 0.7,  # 意识清晰度
            "depth": 0.5,  # 意识深度
            "arousal": 0.6,  # 唤醒水平
            "content_load": 0.4,  # 内容负载
            "state_history": [],  # 状态历史
        }

        # 意识整合机制
        self.consciousness_integration = {
            "active": False,
            "global_broadcast_active": False,  # 全局广播激活
            "neural_synchrony_level": 0.5,  # 神经同步水平
            "information_integration": 0.5,  # 信息整合（Φ）
            "higher_order_thought": False,  # 高阶思想
            "binding_level": 0.5,  # 绑定水平
            "ignition_count": 0,  # 点火次数
        }

        # 意识内容
        self.consciousness_content = {
            "current_content": None,  # 当前意识内容
            "content_history": [],  # 内容历史
            "attention_focus": None,  # 注意焦点
            "meta_awareness": 0.5,  # 元意识
            "introspection_depth": 0.3,  # 内省深度
        }

        # 意识测量
        self.consciousness_metrics = {
            "consciousness_level": 0.7,  # 意识水平
            "complexity": 0.5,  # 意识复杂度
            "diversity": 0.5,  # 意识多样性
            "stability": 0.5,  # 意识稳定性
            "flexibility": 0.5,  # 意识灵活性
            "phi_value": 0.5,  # Φ值（整合信息）
        }

        # 意识状态转换
        self.consciousness_transition = {
            "transition_count": 0,  # 转换次数
            "transition_smoothness": 0.5,  # 转换平滑度
            "phase_transition_threshold": 0.7,  # 相变阈值
            "hysteresis": 0.1,  # 滞后效应
        }

        # 意识整合参数
        self.consciousness_integration_rate = 0.1  # 意识整合速率
        self.consciousness_threshold = 0.5  # 意识阈值
        self.global_workspace_capacity = 4  # 全局工作空间容量（4±1）
        self.consciousness_temporal_window = 3  # 意识时间窗口（秒）

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

        # v5.4 初始化全局工作空间的专门模块
        self._init_specialized_modules()

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
                    w = self.recurrent_synapse.get((assoc_n.id, post), 0.0)
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
                    w = self.recurrent_synapse.get((dec_n.id, post), 0.0)
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

    def sensory_input(self, data: str, modality: str = "text",
                      features: list = None) -> str:
        """外部文本感官输入入口，返回大脑的行为输出"""
        currents = self._str_to_current(data)
        return self._perceive(data, currents, tag="sensory",
                              modality=modality, features=features)

    def sensory_input_vector(self, vec: List[float], label: str = "",
                             modality: str = "text",
                             write_memory: bool = True) -> str:
        """多模态接口：接收图像/音频 embedding 等任意数值向量。

        v4.0：use_projection=True 时经可学习投影进入感官层（保对比度），
        并顺带对该 embedding 做一步 Oja 在线训练（可关）；否则沿用
        线性插值重采样（向后兼容）。

        write_memory=False 时只注入感官层，不写入记忆（用于辅助特征通道）。
        """
        if self.use_projection:
            proj = self._get_projection(len(vec))
            if self.projection_train_on_input:
                proj.train(vec)
            currents = proj.project(vec)
        else:
            currents = self._normalize_vector(vec, len(self.sense_layer))
        content = label or f"<vector[{len(vec)}]>"

        if not write_memory:
            # 只注入感官层，不走完整感知流水线
            self.tick += 1
            external = [c * self.attention_factor for c in currents]
            self._network_step(external)
            for _ in range(self.settle_ticks):
                self._network_step()
            return ""

        return self._perceive(content, currents, tag="sensory", modality=modality)

    # ------------------ 多模态感知（v5.0 听觉/视觉） ------------------

    def hear(self, audio_path: str, model_size: str = "base") -> Dict:
        """听觉感知：音频文件 → 语音识别 → 文本进入大脑。

        流程：
        1. Whisper 语音识别，得到文本和音频特征
        2. 文本作为"听到的内容"进入感知流水线
        3. 音频特征向量注入感官层（双通道输入）
        4. 感知内容标记 source=auditory 进入思考空间

        返回:
            text: 识别出的文本
            language: 检测到的语言
            output: 大脑的行为输出
            novelty: 新奇度
            emotion: 当前情绪
        """
        try:
            import sys
            enc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "models", "encoders")
            sys.path.insert(0, enc_dir)
            from multimodal import encode_audio
        except ImportError:
            return {"text": "[听觉模块不可用]", "output": "",
                    "error": "multimodal encoder not found"}

        # 编码音频
        result = encode_audio(audio_path)
        text = result.get("text", "")
        features = result.get("features", [])

        if not text:
            text = "[未识别到语音]"

        # 文本通道：作为听到的内容进入感知（同时携带音频特征）
        output = self.sensory_input(text, modality="auditory",
                                    features=features)

        # 特征通道：音频特征也注入感官层（增强感知，不单独写记忆）
        if features:
            self.sensory_input_vector(features, label=f"[音频特征]",
                                      write_memory=False)

        # 标记听觉来源（重新推入思考空间并标记来源）
        self._push_thought(text, source="auditory")

        # 跨模态联想：听到声音 → 联想起相关的视觉记忆
        cross_modal = []
        if features:
            cross_modal = self.cross_modal_recall(features, "auditory", top_k=2)

        return {
            "text": text,
            "language": result.get("language", "unknown"),
            "duration": result.get("duration", 0),
            "output": output,
            "novelty": round(self.novelty, 3),
            "emotion": dict(self.emotion),
            "thought_space_size": len(self.thought_space),
            "cross_modal_recalled": [m.content for m in cross_modal],
            "meta": result.get("meta", {}),
        }

    def see(self, image_path: str, model_name: str = "auto") -> Dict:
        """视觉感知：图像文件 → 图像理解 → 描述文本进入大脑。

        流程：
        1. 视觉模型生成图像描述 / 提取特征
        2. 描述文本作为"看到的内容"进入感知流水线
        3. 图像特征向量注入感官层（双通道输入）
        4. 感知内容标记 source=visual 进入思考空间

        返回:
            caption: 图像描述文本
            output: 大脑的行为输出
            novelty: 新奇度
            emotion: 当前情绪
        """
        try:
            import sys
            enc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "models", "encoders")
            sys.path.insert(0, enc_dir)
            from multimodal import encode_image
        except ImportError:
            return {"caption": "[视觉模块不可用]", "output": "",
                    "error": "multimodal encoder not found"}

        # 编码图像
        result = encode_image(image_path)
        caption = result.get("text", "")
        features = result.get("features", [])

        if not caption:
            caption = "[未识别出图像内容]"

        # 文本通道：图像描述进入感知（同时携带图像特征）
        output = self.sensory_input(caption, modality="visual",
                                    features=features)

        # 特征通道：图像特征也注入感官层（增强感知，不单独写记忆）
        if features:
            self.sensory_input_vector(features, label=f"[图像特征]",
                                      write_memory=False)

        # 标记视觉来源
        self._push_thought(caption, source="visual")

        # 跨模态联想：看到图像 → 联想起相关的听觉记忆
        cross_modal = []
        if features:
            cross_modal = self.cross_modal_recall(features, "visual", top_k=2)

        return {
            "caption": caption,
            "output": output,
            "novelty": round(self.novelty, 3),
            "emotion": dict(self.emotion),
            "thought_space_size": len(self.thought_space),
            "cross_modal_recalled": [m.content for m in cross_modal],
            "meta": result.get("meta", {}),
        }

    def understand(self, text: str, model_name: str = "Qwen/Qwen2-0.5B") -> Dict:
        """语言深度理解：文本 → Qwen2语义编码 → 语义向量注入大脑。

        与普通 sensory_input 的区别：
        - sensory_input：简单的字符哈希 → 感官层（浅层感知）
        - understand：Qwen2 语义编码 → 感官层（深度理解）

        流程：
        1. Qwen2 编码文本，得到 896 维语义向量
        2. 文本作为"读到的内容"进入感知流水线
        3. 语义特征向量注入感官层（双通道输入）
        4. 感知内容标记 source=language 进入思考空间

        返回:
            text: 输入文本
            output: 大脑的行为输出
            novelty: 新奇度
            emotion: 当前情绪
            thought_space_size: 思考空间大小
            cross_modal_recalled: 跨模态联想结果
            meta: 编码器元信息
        """
        try:
            import sys
            enc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "models", "encoders")
            sys.path.insert(0, enc_dir)
            from multimodal import encode_text
        except ImportError:
            return {"text": text, "output": "",
                    "error": "language encoder not found"}

        # 编码文本
        result = encode_text(text)
        encoded_text = result.get("text", text)
        features = result.get("features", [])

        if not encoded_text:
            encoded_text = text

        # 文本通道：作为读到的内容进入感知（同时携带语义特征）
        output = self.sensory_input(encoded_text, modality="language",
                                    features=features)

        # 特征通道：语义特征也注入感官层（增强理解，不单独写记忆）
        if features:
            self.sensory_input_vector(features, label=f"[语义特征]",
                                      write_memory=False)

        # 标记语言来源（重新推入思考空间并标记来源）
        self._push_thought(encoded_text, source="language")

        # 跨模态联想：理解文本 → 联想起相关的视觉/听觉记忆
        cross_modal = []
        if features:
            cross_modal = self.cross_modal_recall(features, "language", top_k=2)

        return {
            "text": encoded_text,
            "output": output,
            "novelty": round(self.novelty, 3),
            "emotion": dict(self.emotion),
            "thought_space_size": len(self.thought_space),
            "cross_modal_recalled": [m.content for m in cross_modal],
            "meta": result.get("meta", {}),
        }

    def reply(self, message: str, think_ticks: int = 2,
              max_length: int = 100, temperature: float = 0.7) -> Dict:
        """对话回复：理解消息 → 思考 → 生成回复。

        流程：
        1. 理解用户消息（Qwen2 语义编码）
        2. 思考几步（激活相关记忆）
        3. 基于思考空间内容 + 情绪状态生成提示词
        4. 调用 Qwen2 生成自然语言回复
        5. 把回复也推入思考空间

        Args:
            message: 用户消息
            think_ticks: 思考步数
            max_length: 回复最大长度
            temperature: 生成温度

        Returns:
            {reply, emotion, thought, recalled, novelty, ...}
        """
        # 1. 理解消息
        understand_result = self.understand(message)

        # 2. 思考几步
        thought_result = None
        for _ in range(think_ticks):
            thought_result = self.think()

        # 3. 构建生成提示词
        top_thought = self.top_thought()
        current_thought = top_thought.content if top_thought else ""

        # 获取当前情绪
        mood = max(self.emotion, key=self.emotion.get)
        mood_val = self.emotion[mood]

        # 构建提示词
        prompt = self._build_reply_prompt(message, current_thought, mood, mood_val)

        # 4. 生成回复
        try:
            import sys
            enc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "models", "encoders")
            sys.path.insert(0, enc_dir)
            from multimodal import generate_text
            gen_result = generate_text(prompt, max_length=max_length,
                                       temperature=temperature)
            reply_text = gen_result.get("text", "")
            gen_meta = gen_result.get("meta", {})
        except Exception as e:
            reply_text = f"（思考中...）"
            gen_meta = {"error": str(e)}

        # 5. 把回复推入思考空间
        if reply_text:
            self._push_thought(reply_text, source="self")

        return {
            "reply": reply_text,
            "emotion": dict(self.emotion),
            "dominant_mood": mood,
            "thought": current_thought,
            "novelty": understand_result.get("novelty", 0),
            "thought_space_size": len(self.thought_space),
            "recalled": thought_result.get("recalled", []) if thought_result else [],
            "gen_meta": gen_meta,
        }

    def _build_reply_prompt(self, user_message: str, current_thought: str,
                            mood: str, mood_val: float) -> str:
        """构建回复生成的提示词

        基于大脑当前状态生成个性化的提示词，
        让回复符合大脑的"性格"和当前状态。
        """
        # 情绪描述
        mood_desc = {
            "calm": "平静的",
            "curiosity": "好奇的",
            "stress": "有点紧张的",
            "pleasure": "愉悦的",
        }.get(mood, "平静的")

        # 构建提示词
        prompt = f"""{self.name}。

你当前的状态：
- 情绪：{mood_desc}（强度：{mood_val:.2f}）
- 你正在思考：{current_thought[:50] if current_thought else '（空）'}

用户对你说：{user_message}

请用自然、简洁的方式回复用户，表达你的想法和感受。
回复："""

        return prompt

    def multimodal_input(self,
                         text: str = "",
                         audio_path: str = "",
                         image_path: str = "") -> Dict:
        """多模态联合输入：同时接收文本、音频、图像。

        所有模态的感知结果都进入同一个思考空间，
        大脑会跨模态联想（比如看到猫+听到喵叫 → 联想到"猫"）。
        """
        results = {}

        if text:
            results["text"] = self.sensory_input(text)
            self._push_thought(text, source="text")

        if audio_path:
            results["audio"] = self.hear(audio_path)

        if image_path:
            results["image"] = self.see(image_path)

        # 多模态整合：思考空间里同时有来自不同感官的念头
        thoughts = self.current_thoughts() if hasattr(self, 'current_thoughts') else []

        return {
            "results": results,
            "thought_space": [(t.content, t.source, round(t.activation, 2))
                              for t in self.thought_space],
            "dominant_thought": (self.top_thought().content
                                 if self.top_thought() else None),
            "emotion": dict(self.emotion),
        }

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
        超出容量时挤出激活度最低的念头。v5.2：同时记入念头流水账。"""
        self.thought_journal.append(
            {"content": content, "source": source, "tick": self.tick})
        if len(self.thought_journal) > 50:
            self.thought_journal.pop(0)
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

    # ------------------ 全局工作空间理论 GWT（v5.4） ------------------

    def _init_specialized_modules(self):
        """初始化专门模块（无意识处理器）。

        根据全局工作空间理论，大脑由许多专门的无意识模块组成，
        它们竞争进入全局工作空间（意识）。
        获胜的模块会把内容广播给所有其他模块。
        """
        self.specialized_modules = {
            "perception": {
                "name": "感知模块",
                "description": "处理外部感官输入",
                "activation": 0.0,
                "output": None,
            },
            "memory": {
                "name": "记忆模块",
                "description": "检索和存储记忆",
                "activation": 0.0,
                "output": None,
            },
            "emotion": {
                "name": "情绪模块",
                "description": "评估情绪意义",
                "activation": 0.0,
                "output": None,
            },
            "language": {
                "name": "语言模块",
                "description": "语言理解和生成",
                "activation": 0.0,
                "output": None,
            },
            "attention": {
                "name": "注意力模块",
                "description": "选择和聚焦",
                "activation": 0.0,
                "output": None,
            },
            "metacognition": {
                "name": "元认知模块",
                "description": "监控和调节认知过程",
                "activation": 0.0,
                "output": None,
            },
            "motor": {
                "name": "运动模块",
                "description": "计划和执行动作",
                "activation": 0.0,
                "output": None,
            },
            "value": {
                "name": "价值评估模块",
                "description": "评估奖励和价值",
                "activation": 0.0,
                "output": None,
            },
        }

    def attentional_competition(self) -> Dict:
        """注意力竞争：多个内容竞争进入意识。

        根据全局工作空间理论，无意识模块不断产生候选内容，
        它们竞争进入全局工作空间（意识）。
        只有激活度最高的内容才能进入意识。

        返回：
            winner: 获胜的内容
            competitors: 所有竞争者列表
            winner_activation: 获胜者的激活度
        """
        if not self.thought_space:
            return {"winner": None, "competitors": [], "winner_activation": 0.0}

        # 按激活度排序
        competitors = sorted(self.thought_space,
                             key=lambda t: t.activation, reverse=True)
        winner = competitors[0]

        return {
            "winner": winner.content,
            "winner_source": winner.source,
            "winner_activation": round(winner.activation, 3),
            "competitors": [
                {"content": t.content[:20], "source": t.source,
                 "activation": round(t.activation, 3)}
                for t in competitors
            ],
            "n_competitors": len(competitors),
        }

    def global_broadcast(self) -> Dict:
        """全局广播：把意识内容广播给所有无意识模块。

        根据全局工作空间理论，进入意识的内容会被全局广播，
        所有专门模块都能接收这个信息。
        这就是意识的"全局可用性"。

        返回：
            broadcast_content: 广播的内容
            broadcast_to: 接收广播的模块列表
            module_reactions: 各模块的反应
        """
        top = self.top_thought()
        if top is None:
            return {"broadcast_content": None, "broadcast_to": [],
                    "module_reactions": {}}

        content = top.content
        broadcast_to = list(self.specialized_modules.keys())

        # 各模块对广播内容的反应（模拟）
        module_reactions = {}
        for mod_name, mod in self.specialized_modules.items():
            # 模块激活度提升（因为接收到了广播）
            mod["activation"] = min(1.0, mod["activation"] + 0.2)
            mod["output"] = content
            module_reactions[mod_name] = {
                "activation": round(mod["activation"], 3),
                "received": True,
            }

        # 记录广播历史
        broadcast_record = {
            "tick": self.tick,
            "content": content,
            "source": top.source,
            "broadcast_to": broadcast_to,
            "n_modules": len(broadcast_to),
        }
        self.broadcast_history.append(broadcast_record)
        if len(self.broadcast_history) > 100:
            self.broadcast_history.pop(0)

        return {
            "broadcast_content": content,
            "broadcast_source": top.source,
            "broadcast_to": broadcast_to,
            "n_modules": len(broadcast_to),
            "module_reactions": module_reactions,
        }

    def ignition(self) -> Dict:
        """点火效应（Ignition）：内容进入意识时的全脑激活。

        根据全局工作空间理论，当内容进入意识时，会触发"点火"——
        全脑范围的激活放大，就像点燃了一场野火。
        这是意识的神经相关物（NCC）之一。

        点火的特征：
        1. 突然的、非线性的激活跃升
        2. 全脑范围的传播
        3. 持续一段时间
        4. 之后逐渐衰减

        返回：
            ignited: 是否发生了点火
            ignition_strength: 点火强度
            spike_increase: 脉冲数增加量
        """
        top = self.top_thought()
        if top is None:
            return {"ignited": False, "ignition_strength": 0.0,
                    "spike_increase": 0}

        # 点火条件：激活度超过阈值，且比第二名高很多（赢者通吃）
        if len(self.thought_space) >= 2:
            sorted_thoughts = sorted(self.thought_space,
                                     key=lambda t: t.activation, reverse=True)
            top_activation = sorted_thoughts[0].activation
            second_activation = sorted_thoughts[1].activation
            # 点火条件：第一名激活度 > 0.7，且比第二名高 50% 以上
            should_ignite = (top_activation > 0.7
                             and top_activation > second_activation * 1.5)
        else:
            # 只有一个念头，且激活度足够高
            should_ignite = top.activation > 0.8

        if should_ignite and not self.ignition_state:
            # 点火发生！
            self.ignition_state = True
            self.ignition_count += 1

            # 模拟全脑激活：脉冲数增加
            s, a, d = self.spike_counts()
            baseline_spikes = s + a + d
            ignition_strength = top.activation

            # 记录点火事件
            ignition_event = {
                "tick": self.tick,
                "content": top.content,
                "source": top.source,
                "strength": round(ignition_strength, 3),
                "baseline_spikes": baseline_spikes,
            }
            self.broadcast_history.append({"type": "ignition", **ignition_event})

            return {
                "ignited": True,
                "ignition_strength": round(ignition_strength, 3),
                "content": top.content,
                "source": top.source,
                "ignition_count": self.ignition_count,
            }
        elif not should_ignite and self.ignition_state:
            # 点火结束
            self.ignition_state = False
            return {"ignited": False, "ignition_strength": 0.0,
                    "message": "点火结束"}
        else:
            return {
                "ignited": self.ignition_state,
                "ignition_strength": round(top.activation, 3) if top else 0,
                "content": top.content if top else None,
                "ignition_count": self.ignition_count,
            }

    def conscious_step(self) -> Dict:
        """意识的一个完整步骤：竞争 → 点火 → 广播。

        这是全局工作空间理论的完整意识循环：
        1. 多个无意识内容竞争进入意识
        2. 获胜者触发点火（全脑激活）
        3. 内容被全局广播给所有模块
        4. 各模块接收并处理，产生新的候选内容

        返回：
            competition: 注意力竞争结果
            ignition: 点火状态
            broadcast: 全局广播结果
        """
        # 1. 注意力竞争
        competition = self.attentional_competition()

        # 2. 点火检测
        ignition = self.ignition()

        # 3. 全局广播（只有点火时才广播）
        if ignition["ignited"]:
            broadcast = self.global_broadcast()
        else:
            broadcast = {"broadcast_content": None, "broadcast_to": []}

        # 4. 衰减思考空间（旧念头淡出）
        self._decay_thoughts()

        return {
            "tick": self.tick,
            "competition": competition,
            "ignition": ignition,
            "broadcast": broadcast,
            "thought_space_size": len(self.thought_space),
            "conscious": self.ignition_state,
        }

    def get_consciousness_report(self) -> Dict:
        """获取意识状态报告。

        这是对当前意识状态的完整描述，
        就像在问："你现在意识到了什么？"
        """
        top = self.top_thought()
        competition = self.attentional_competition()

        return {
            "tick": self.tick,
            "name": self.name,
            "conscious": self.ignition_state,
            "ignition_count": self.ignition_count,
            "current_content": top.content if top else None,
            "current_source": top.source if top else None,
            "current_activation": round(top.activation, 3) if top else 0,
            "thought_space_size": len(self.thought_space),
            "thought_capacity": self.thought_capacity,
            "competitors": competition["n_competitors"],
            "mood": max(self.emotion, key=lambda k: self.emotion[k]),
            "attention": round(self.attention_factor, 3),
            "novelty": round(self.novelty, 3),
            "broadcast_count": len(self.broadcast_history),
            "modules_active": sum(1 for m in self.specialized_modules.values()
                                  if m["activation"] > 0.3),
        }

    # ------------------ 意识的神经相关物 NCC（v5.5） ------------------

    def neural_synchrony(self) -> Dict:
        """计算神经同步性（不同脑区之间的同步振荡）。

        意识的神经相关物之一是gamma频段同步（30-80Hz）。
        当意识出现时，不同脑区的神经元活动会变得同步。

        这里我们计算三层网络之间的脉冲同步性。

        返回：
            synchrony: 整体同步性 [0, 1]
            sense_assoc_sync: 感官层-联想层同步性
            assoc_decision_sync: 联想层-决策层同步性
            sense_decision_sync: 感官层-决策层同步性
        """
        # 获取各层脉冲状态
        sense_spikes = set(n.id for n in self.sense_layer if n.spike)
        assoc_spikes = set(n.id for n in self.assoc_layer if n.spike)
        decision_spikes = set(n.id for n in self.decision_layer if n.spike)

        # 计算两层之间的同步性（基于共同激活的神经元比例）
        def layer_sync(layer1_spikes, layer2_spikes, n1, n2):
            """计算两层之间的同步性"""
            if not layer1_spikes and not layer2_spikes:
                return 0.0
            # 用脉冲数的相似度来衡量同步性
            ratio1 = len(layer1_spikes) / max(n1, 1)
            ratio2 = len(layer2_spikes) / max(n2, 1)
            # 同步性 = 1 - |ratio1 - ratio2|（激活比例越接近，越同步）
            return 1.0 - abs(ratio1 - ratio2)

        sense_assoc = layer_sync(sense_spikes, assoc_spikes,
                                 len(self.sense_layer), len(self.assoc_layer))
        assoc_decision = layer_sync(assoc_spikes, decision_spikes,
                                    len(self.assoc_layer), len(self.decision_layer))
        sense_decision = layer_sync(sense_spikes, decision_spikes,
                                    len(self.sense_layer), len(self.decision_layer))

        # 整体同步性 = 三层之间同步性的平均值
        overall = (sense_assoc + assoc_decision + sense_decision) / 3

        return {
            "synchrony": round(overall, 3),
            "sense_assoc_sync": round(sense_assoc, 3),
            "assoc_decision_sync": round(assoc_decision, 3),
            "sense_decision_sync": round(sense_decision, 3),
            "sense_spikes": len(sense_spikes),
            "assoc_spikes": len(assoc_spikes),
            "decision_spikes": len(decision_spikes),
        }

    def neural_complexity(self) -> Dict:
        """计算神经复杂度（大脑活动的复杂程度）。

        意识的神经相关物之一是神经活动的复杂度。
        有意识的大脑活动既不是完全有序的（癫痫），
        也不是完全无序的（深度睡眠），而是处于"混沌边缘"。

        复杂度 = 整合度 × 分化度
        - 整合度：各脑区之间的连接程度
        - 分化度：各脑区活动的差异程度

        返回：
            complexity: 神经复杂度
            integration: 整合度
            differentiation: 分化度
            entropy: 熵（无序程度）
        """
        # 计算各层的脉冲率
        sense_rate = sum(1 for n in self.sense_layer if n.spike) / len(self.sense_layer)
        assoc_rate = sum(1 for n in self.assoc_layer if n.spike) / len(self.assoc_layer)
        decision_rate = sum(1 for n in self.decision_layer if n.spike) / len(self.decision_layer)

        rates = [sense_rate, assoc_rate, decision_rate]

        # 分化度：各层活动的差异程度（标准差）
        mean_rate = sum(rates) / len(rates)
        variance = sum((r - mean_rate) ** 2 for r in rates) / len(rates)
        differentiation = min(1.0, variance * 10)  # 缩放

        # 整合度：各层之间的连接强度（突触权重的平均值）
        ff_weights = list(self.synapse.values())
        integration = sum(ff_weights) / len(ff_weights) if ff_weights else 0

        # 熵：活动的无序程度
        import math
        def safe_log(x):
            return math.log(x) if x > 0 else 0

        entropy = 0
        for r in rates:
            if r > 0 and r < 1:
                entropy -= r * safe_log(r) + (1 - r) * safe_log(1 - r)
        entropy = entropy / len(rates)  # 归一化

        # 复杂度 = 整合度 × 分化度 × 熵
        complexity = integration * differentiation * (1 + entropy)

        return {
            "complexity": round(complexity, 4),
            "integration": round(integration, 4),
            "differentiation": round(differentiation, 4),
            "entropy": round(entropy, 4),
            "sense_rate": round(sense_rate, 3),
            "assoc_rate": round(assoc_rate, 3),
            "decision_rate": round(decision_rate, 3),
        }

    def integrated_information(self) -> Dict:
        """计算整合信息（Φ值的简化版本）。

        整合信息理论（IIT）认为，意识 = 整合信息（Φ）。
        Φ 衡量的是系统作为整体产生的信息，
        大于各部分单独产生的信息之和。

        这里我们用简化的方法计算 Φ：
        Φ = 整体熵 - 各部分熵之和

        返回：
            phi: 整合信息 Φ 值
            whole_entropy: 整体熵
            parts_entropy_sum: 各部分熵之和
            phi_ratio: Φ / 整体熵
        """
        import math

        def safe_log(x):
            return math.log(x) if x > 0 else 0

        def layer_entropy(layer):
            """计算一层神经元的熵"""
            n = len(layer)
            n_spike = sum(1 for neuron in layer if neuron.spike)
            p = n_spike / n  # 发放概率
            if p <= 0 or p >= 1:
                return 0.0
            return -(p * safe_log(p) + (1 - p) * safe_log(1 - p))

        # 各部分熵
        sense_ent = layer_entropy(self.sense_layer)
        assoc_ent = layer_entropy(self.assoc_layer)
        decision_ent = layer_entropy(self.decision_layer)
        parts_sum = sense_ent + assoc_ent + decision_ent

        # 整体熵（近似：用所有神经元的平均发放率计算）
        all_neurons = self.sense_layer + self.assoc_layer + self.decision_layer
        n_all = len(all_neurons)
        n_spike_all = sum(1 for n in all_neurons if n.spike)
        p_all = n_spike_all / n_all
        if p_all <= 0 or p_all >= 1:
            whole_ent = 0.0
        else:
            whole_ent = -(p_all * safe_log(p_all) + (1 - p_all) * safe_log(1 - p_all))

        # Φ = 整体熵 - 各部分熵之和
        # （简化版本，真正的Φ需要考虑最小信息分割）
        phi = max(0, whole_ent * 3 - parts_sum)  # 乘以3是因为有3层

        phi_ratio = phi / whole_ent if whole_ent > 0 else 0

        return {
            "phi": round(phi, 4),
            "whole_entropy": round(whole_ent, 4),
            "parts_entropy_sum": round(parts_sum, 4),
            "phi_ratio": round(phi_ratio, 4),
            "sense_entropy": round(sense_ent, 4),
            "assoc_entropy": round(assoc_ent, 4),
            "decision_entropy": round(decision_ent, 4),
        }

    def detect_ncc(self) -> Dict:
        """检测意识的神经相关物（NCC）。

        综合多个指标，判断当前的意识状态，
        并提取意识的神经标志。

        NCC 特征：
        1. 高神经同步性（gamma同步）
        2. 适度的神经复杂度（混沌边缘）
        3. 高整合信息（Φ值）
        4. 点火效应（全脑激活）
        5. 全局广播（信息传遍全脑）

        返回：
            conscious: 是否有意识
            ncc_score: NCC 综合得分 [0, 1]
            features: 各NCC特征的得分
            markers: 意识的神经标志列表
        """
        # 获取各个指标
        sync = self.neural_synchrony()
        comp = self.neural_complexity()
        info = self.integrated_information()

        # 各特征得分（0-1）
        features = {
            "synchrony": sync["synchrony"],  # 神经同步性
            "complexity": min(1.0, comp["complexity"] * 5),  # 神经复杂度（缩放）
            "integration": min(1.0, info["phi"] * 10),  # 整合信息（缩放）
            "ignition": 1.0 if self.ignition_state else 0.0,  # 点火效应
            "broadcast": min(1.0, len(self.broadcast_history) / 10),  # 全局广播
        }

        # NCC 综合得分（加权平均）
        weights = {
            "synchrony": 0.2,
            "complexity": 0.2,
            "integration": 0.2,
            "ignition": 0.25,
            "broadcast": 0.15,
        }
        ncc_score = sum(features[k] * weights[k] for k in features)

        # 判断是否有意识（阈值 0.5）
        conscious = ncc_score > 0.5

        # 提取意识的神经标志
        markers = []
        if features["synchrony"] > 0.6:
            markers.append("高神经同步性")
        if features["complexity"] > 0.5:
            markers.append("适度神经复杂度")
        if features["integration"] > 0.5:
            markers.append("高整合信息")
        if features["ignition"] > 0:
            markers.append("点火效应")
        if features["broadcast"] > 0.3:
            markers.append("全局广播")

        return {
            "conscious": conscious,
            "ncc_score": round(ncc_score, 3),
            "features": {k: round(v, 3) for k, v in features.items()},
            "markers": markers,
            "n_markers": len(markers),
        }

    def consciousness_phase_transition(self, history_length: int = 20) -> Dict:
        """检测意识的相变（从无意识到有意识的临界转换）。

        相变的特征：
        1. 突然的、非线性的变化
        2. 临界慢化（变化速度变慢）
        3. 方差增加
        4. 自相关增加

        返回：
            phase: 当前阶段（unconscious / transition / conscious）
            transition_detected: 是否检测到相变
            criticality: 临界程度 [0, 1]
            trend: 变化趋势
        """
        # 这里简化实现：用最近几tick的NCC得分变化来判断
        # 由于我们没有完整的历史记录，用当前状态来估计

        ncc = self.detect_ncc()
        score = ncc["ncc_score"]

        # 判断阶段
        if score < 0.3:
            phase = "unconscious"  # 无意识
        elif score < 0.7:
            phase = "transition"   # 过渡态
        else:
            phase = "conscious"    # 有意识

        # 临界程度：离0.5越近，越接近临界点
        criticality = 1.0 - abs(score - 0.5) * 2

        # 变化趋势（简化：用思考空间大小变化来估计）
        if len(self.thought_space) > self.thought_capacity * 0.8:
            trend = "increasing"  # 增加
        elif len(self.thought_space) < self.thought_capacity * 0.3:
            trend = "decreasing"  # 减少
        else:
            trend = "stable"      # 稳定

        return {
            "phase": phase,
            "transition_detected": phase == "transition",
            "criticality": round(criticality, 3),
            "ncc_score": score,
            "trend": trend,
            "thought_space_fill": round(
                len(self.thought_space) / self.thought_capacity, 3),
        }

    def get_ncc_report(self) -> Dict:
        """获取完整的 NCC 报告。

        这是对意识神经状态的完整分析，
        就像在做"意识脑电图"。
        """
        sync = self.neural_synchrony()
        comp = self.neural_complexity()
        info = self.integrated_information()
        ncc = self.detect_ncc()
        phase = self.consciousness_phase_transition()

        return {
            "tick": self.tick,
            "name": self.name,
            "conscious": ncc["conscious"],
            "ncc_score": ncc["ncc_score"],
            "phase": phase["phase"],
            "criticality": phase["criticality"],
            "markers": ncc["markers"],
            "synchrony": sync,
            "complexity": comp,
            "integrated_information": info,
            "ncc_features": ncc["features"],
            "ignition_count": self.ignition_count,
            "broadcast_count": len(self.broadcast_history),
            "thought_space_size": len(self.thought_space),
        }

    def _perceive(self, content: str, input_currents: List[float],
                  tag: str, modality: str = "text",
                  features: list = None) -> str:
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
        self._write_stm(content, tag=tag, modality=modality, features=features)

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
                    if k in self.recurrent_synapse:
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
                    if k in self.recurrent_synapse:
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

    def _write_stm(self, content: str, tag: str = "sensory",
                   modality: str = "text", features: list = None):
        """写入短期记忆；容量满时：强者固化进 LTM，弱者自然遗忘"""
        # v6.2：文本记忆在语义检索通路存在时（编码器或向量库已接入）
        # 自动生成 features；否则保持空列表（行为与 DNA 体积不变）
        if features is None and modality == "text" and (
                (self.text_encoder is not None
                 and getattr(self.text_encoder, "available", False))
                or (self.memory_store is not None
                    and getattr(self.memory_store, "available", False))):
            features = self._encode_text_features(content)
        mem = BrainMemory(
            content=content,
            timestamp=time.time(),
            weight=self.attention_factor,
            tag=tag,
            modality=modality,
            features=features if features is not None else [],
        )
        self.short_memory.append(mem)

        # v6.1：可选 STM 全量同步（"所有记忆入库"）；默认关闭，
        # 只同步固化进 LTM 的记忆（attach_memory_store(sync_stm=True) 开启）
        store = self.memory_store
        if self.sync_stm and store is not None and \
                getattr(store, "available", False):
            try:
                store.add(mem, brain_name=self.name)
            except Exception:
                pass

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
                self._store_sync_weight(old)      # v6.0 同步到 LanceDB
                return
        mem.tag = "event"
        self.long_memory.append(mem)
        if len(self.long_memory) > self.max_ltm:
            self.long_memory.sort(key=lambda m: m.weight)
            self.long_memory.pop(0)  # 遗忘最弱的长期记忆
        self._store_sync_add(mem)                 # v6.0 同步到 LanceDB

    # ------------------ LanceDB 记忆后端（v6.0） ------------------

    def attach_memory_store(self, store=None,
                            path: str = "datasets/lancedb",
                            sync_stm: bool = False) -> Dict:
        """接入 LanceDB 记忆后端：LTM 持久化到本地向量库。

        sync_stm=True 时短期记忆也全量入库（"所有记忆存入 LanceDB"）；
        默认只同步固化进 LTM 的记忆（STM 转瞬即逝，不同步更轻快）。
        lancedb 未安装时 store.available=False，大脑行为完全不变
        （纯内存 + 关键词 recall）。返回后端状态。
        """
        if store is None:
            from memory_store import LanceMemoryStore
            store = LanceMemoryStore(path)
        self.memory_store = store
        self.sync_stm = sync_stm
        return {"attached": True,
                "available": getattr(store, "available", False),
                "sync_stm": self.sync_stm,
                "error": getattr(store, "_error", None)}

    def attach_text_encoder(self, encoder=None,
                            model_path: Optional[str] = None,
                            device: str = "cpu") -> Dict:
        """接入文本语义编码器（v6.2）：文本记忆携带真语义 embedding。

        接入后 _write_stm 自动为文本记忆生成语义 features，
        recall_semantic 的字符串查询也走真语义向量（不再是哈希字面近似）。
        模型缺失 / sentence-transformers 未安装时 encoder.available=False，
        大脑行为完全不变（哈希向量兜底），返回后端状态。
        也可直接传入自定义 encoder（需有 encode(text)->list 与 available）。
        """
        if encoder is None:
            from models.encoders.text_encoder import create_text_encoder
            encoder = create_text_encoder(model_path=model_path, device=device)
        self.text_encoder = encoder
        return {"attached": True,
                "available": getattr(encoder, "available", False),
                "info": (encoder.info() if hasattr(encoder, "info")
                         else {"available": getattr(encoder, "available", False)})}

    def _encode_text_features(self, content: str) -> list:
        """v6.2：文本记忆的特征生成——语义编码器优先，哈希向量兜底。

        哈希兜底保证无论是否接入编码器，记忆始终携带可检索的 features。
        """
        enc = self.text_encoder
        if enc is not None and getattr(enc, "available", False):
            try:
                vec = enc.encode(content)
                if vec:
                    return vec
            except Exception:
                pass
        from memory_store import text_to_vector
        return text_to_vector(content)

    # ------------------ DNA 基因库（v6.1） ------------------

    def attach_dna_library(self, library=None,
                           path: str = "datasets/lancedb") -> Dict:
        """接入 DNA 基因库：多大脑 DNA 存储 / 人格参数搜索 / 进化谱系追踪"""
        if library is None:
            from memory_store import DNALibrary
            library = DNALibrary(path)
        self.dna_library = library
        return {"attached": True,
                "available": getattr(library, "available", False),
                "error": getattr(library, "_error", None)}

    def save_to_library(self, parents: Optional[List[str]] = None,
                        library=None) -> Dict:
        """把当前 DNA 存入基因库，返回 {"saved", "dna_id"}。

        parents 为亲代 dna_id 列表（进化谱系追踪用）。
        未接入基因库时返回 {"saved": False, "error"}。
        """
        lib = library or self.dna_library
        if lib is None or not getattr(lib, "available", False):
            return {"saved": False,
                    "error": "未接入 DNA 基因库（attach_dna_library）"}
        dna_id = lib.save(self.dump_dna(),
                          generation=getattr(self, "generation", 1),
                          parents=parents or [])
        return {"saved": bool(dna_id), "dna_id": dna_id}

    def _store_sync_add(self, mem: BrainMemory) -> None:
        """固化同步：新 LTM 写入向量库（尽力而为，失败不影响大脑）"""
        store = self.memory_store
        if store is not None and getattr(store, "available", False):
            try:
                store.add(mem, brain_name=self.name)
            except Exception:
                pass

    def _store_sync_weight(self, mem: BrainMemory) -> None:
        """强化同步：已有 LTM 权重更新到向量库"""
        store = self.memory_store
        if store is not None and getattr(store, "available", False):
            try:
                store.update_weight(mem.content, mem.weight,
                                    brain_name=self.name)
            except Exception:
                pass

    def recall_semantic(self, query, top_k: int = 3,
                        modality: Optional[str] = None,
                        exclude_modality: Optional[str] = None) -> List[Dict]:
        """语义回忆（v6.0）：向量近邻检索长期记忆。

        query 为数值序列时直接作为查询向量；为字符串时经零依赖
        哈希向量编码后检索（字面近似；真正语义相似需记忆携带
        CLIP/Qwen 真实 embedding）。
        v6.1 跨模态联想：modality="visual" 只查该模态；
        exclude_modality="text" 排除该模态（统一向量空间自由联想）。
        未接入 LanceDB 时自动降级为关键词 recall（同样返回字典列表）。
        """
        store = self.memory_store
        if store is None or not getattr(store, "available", False):
            kw = query if isinstance(query, str) else ""
            return [{"content": m.content, "weight": round(m.weight, 4),
                     "tag": m.tag, "modality": m.modality,
                     "source": "memory-fallback"}
                    for m in self.recall(kw, top_k=top_k)]
        if isinstance(query, str):
            query = self._encode_text_features(query)
        rows = store.search_vector(query, top_k=top_k,
                                   brain_name=self.name,
                                   modality=modality,
                                   exclude_modality=exclude_modality)
        for r in rows:
            r["source"] = "lancedb"
            # 语义命中也进入思考空间（与关键词 recall 一致）
            self._push_thought(r["content"], source="memory",
                               activation=0.8)
        return rows

    def decay_memory(self, factor: float = 0.995):
        """记忆随时间自然衰减（模拟睡眠/时间流逝），低于阈值被遗忘"""
        for store in (self.short_memory, self.long_memory):
            for m in store:
                m.weight *= factor
            store[:] = [m for m in store if m.weight >= self.forget_threshold]
        # v6.0：衰减节律同步到 LanceDB（尽力而为）
        store = self.memory_store
        if store is not None and getattr(store, "available", False):
            try:
                store.decay(factor, self.forget_threshold)
            except Exception:
                pass

    def sleep(self, cycles: int = 1) -> Dict:
        """增强版睡眠系统（v5.9）：完整睡眠周期 + 记忆巩固 + 梦境。

        睡眠阶段：
        1. 昏昏欲睡（Drowsy）：注意力下降，思维松散
        2. 浅睡（Light Sleep）：睡眠纺锤波，记忆初步巩固
        3. 深睡（Deep Sleep / SWS）：慢波睡眠，记忆重放，突触稳态下调
        4. REM睡眠：做梦，记忆整合，情绪调节，创造性联想

        每个睡眠周期约90分钟（简化为若干tick），
        深睡在前半夜多，REM在后半夜多。

        返回完整的睡眠报告。
        """
        import random

        total_replayed = 0
        total_consolidated = 0
        total_pruned = 0
        total_dreams = 0
        total_ltm_integrated = 0
        stress_before = self.emotion["stress"]
        attention_before = self.attention_factor  # 记录睡前注意力

        # 设置睡眠开始状态
        self.sleep_state = "drowsy"
        self._update_emotion()

        for cycle in range(cycles):
            self.sleep_cycle_count += 1

            # ---- 阶段1：昏昏欲睡 ----
            self.sleep_state = "drowsy"
            self.attention_factor *= 0.7  # 注意力下降
            self.emotion["calm"] = self._clip(self.emotion["calm"] + 0.1)
            # 思考空间开始松散，念头衰减加快
            for t in self.thought_space:
                t.activation *= 0.8

            # ---- 阶段2：浅睡 ----
            self.sleep_state = "light_sleep"
            self.sleep_spindles += random.randint(3, 8)  # 睡眠纺锤波
            # 记忆初步巩固（STM权重小幅提升）
            for mem in self.short_memory:
                mem.weight = self._clip(mem.weight + self.sleep_replay_gain * 0.3)
                total_replayed += 1

            # ---- 阶段3：深睡（SWS）----
            self.sleep_state = "deep_sleep"
            self.slow_waves += random.randint(5, 15)  # 慢波

            # 深睡记忆重放（海马→新皮层）
            stm_to_replay = list(self.short_memory)
            random.shuffle(stm_to_replay)
            replay_count = max(1, int(len(stm_to_replay) *
                                      self.deep_sleep_replay_rate))

            for mem in stm_to_replay[:replay_count]:
                mem.weight = self._clip(mem.weight + self.sleep_replay_gain)
                total_replayed += 1
                # 达到阈值则固化进LTM
                if mem.weight >= self.stm_consolidate_threshold:
                    if mem in self.short_memory:
                        self.short_memory.remove(mem)
                    self._consolidate_to_ltm(mem)
                    total_consolidated += 1
                    self.memory_consolidation_count += 1

            # 突触稳态缩放（SHY假说）
            for table, floor in ((self.synapse, self.synapse_prune_threshold),
                                 (self.recurrent_synapse,
                                  self.recurrent_prune_threshold)):
                for key in list(table):
                    table[key] *= self.sleep_downscale
                    if table[key] < floor:
                        del table[key]
                        total_pruned += 1

            # ---- 阶段4：REM睡眠 ----
            # 后半夜REM更多（周期越靠后，REM越长）
            rem_intensity = 0.3 + (cycle / max(cycles, 1)) * 0.5
            if random.random() < rem_intensity:
                self.sleep_state = "rem"
                self.rem_duration += 1

                # REM睡眠：做梦
                dream = self._generate_dream()
                if dream:
                    total_dreams += 1
                    self.dream_count += 1
                    self.dream_log.append(dream)

                    # 梦境记忆进入思考空间（醒来后可能记得）
                    self._push_thought(dream["content"], source="dream",
                                       activation=0.6 * self.dream_vividness)

                # REM睡眠：记忆整合（LTM内部的关联和强化）
                integrated = self._rem_memory_integration()
                total_ltm_integrated += integrated
                self.ltm_integration_count += integrated

                # REM睡眠：情绪调节
                self.emotion["stress"] *= 0.7
                self.emotion["pleasure"] = self._clip(
                    self.emotion["pleasure"] + 0.05)

            # 周期结束，回到浅睡过渡
            self.sleep_state = "light_sleep"

        # ---- 醒来 ----
        self.sleep_state = "awake"
        self.attention_factor = attention_before  # 恢复注意力

        # 清洗：资格迹清零、多巴胺归零
        self.eligibility.clear()
        self.dopamine = 0.0

        # 情绪恢复
        self.emotion["stress"] *= 0.5 ** cycles
        self.emotion["calm"] = self._clip(
            self.emotion["calm"] + 0.2 * cycles)

        # 记录睡眠历史
        if self.record_history:
            self.history.setdefault("sleep_cycles", []).append(cycles)
            self.history.setdefault("dreams", []).append(total_dreams)

        return {
            "cycles": cycles,
            "sleep_stages": ["drowsy", "light_sleep", "deep_sleep", "rem"],
            "replayed": total_replayed,
            "consolidated": total_consolidated,
            "pruned_synapses": total_pruned,
            "dreams": total_dreams,
            "ltm_integrated": total_ltm_integrated,
            "sleep_spindles": self.sleep_spindles,
            "slow_waves": self.slow_waves,
            "rem_duration": self.rem_duration,
            "synapses": len(self.synapse),
            "recurrent_synapses": len(self.recurrent_synapse),
            "stress_before": round(stress_before, 3),
            "stress_after": round(self.emotion["stress"], 3),
            "final_state": self.sleep_state,
        }

    def _generate_dream(self) -> Optional[Dict]:
        """生成一个梦（REM睡眠中）。

        梦的特点：
        - 由记忆碎片组合而成
        - 内容常常荒诞、不合逻辑
        - 与近期经历和情绪相关
        - 可能有创造性的新联想

        返回梦境字典，或None（不做梦）
        """
        import random

        # 不是每次REM都做梦
        if random.random() > 0.8:
            return None

        # 从LTM和STM中随机选取记忆碎片
        all_memories = list(self.long_memory) + list(self.short_memory)
        if len(all_memories) < 2:
            return None

        # 选取2-4个记忆碎片
        n_fragments = random.randint(2, min(4, len(all_memories)))
        fragments = random.sample(all_memories, n_fragments)

        # 组合成梦境（记忆的奇怪组合）
        dream_elements = [m.content[:15] for m in fragments]
        dream_content = " + ".join(dream_elements)

        # 加入一些荒诞元素
        absurdities = [
            "在飞",
            "变得很小",
            "时间倒流",
            "所有人都变成了猫",
            "天空是紫色的",
            "在水下呼吸",
            "突然会说所有语言",
            "东西都在融化",
        ]
        if random.random() < 0.5:
            dream_content += "，" + random.choice(absurdities)

        # 情绪基调（受当前情绪影响）
        mood = max(self.emotion, key=lambda k: self.emotion[k])
        mood_intensity = self.emotion[mood]

        # 梦境生动程度
        vividness = self.dream_vividness * random.uniform(0.5, 1.5)
        vividness = self._clip(vividness, 0.1, 1.0)

        dream = {
            "tick": self.tick,
            "cycle": self.sleep_cycle_count,
            "content": dream_content,
            "fragments": [m.content for m in fragments],
            "mood": mood,
            "mood_intensity": round(mood_intensity, 3),
            "vividness": round(vividness, 3),
            "type": random.choice(["normal", "nightmare", "lucid", "creative"]),
        }

        # 噩梦：压力大时更容易做噩梦
        if self.emotion["stress"] > 0.5 and random.random() < 0.4:
            dream["type"] = "nightmare"
            dream["content"] += "（很可怕）"
            self.emotion["stress"] = self._clip(
                self.emotion["stress"] + 0.1)

        # 清醒梦：元意识高时可能做清醒梦
        if self.meta_awareness > 0.5 and random.random() < 0.3:
            dream["type"] = "lucid"
            dream["content"] += "（我知道我在做梦）"

        # 创造性梦：可能产生新的想法
        if random.random() < 0.2:
            dream["type"] = "creative"
            # 两个记忆的组合可能产生新想法
            if len(fragments) >= 2:
                insight = f"{fragments[0].content[:10]} × {fragments[1].content[:10]}"
                dream["insight"] = insight
                # 创造性洞见写入记忆
                self._write_stm(f"梦中灵感：{insight}", tag="dream_insight")

        return dream

    def _rem_memory_integration(self) -> int:
        """REM睡眠中的记忆整合。

        REM睡眠的功能：
        1. 整合新旧记忆
        2. 建立记忆之间的关联
        3. 情绪记忆的调节
        4. 创造性联想

        返回整合的记忆对数
        """
        import random

        if len(self.long_memory) < 2:
            return 0

        integrated = 0

        # 随机选取一些记忆对，建立关联
        n_pairs = min(5, len(self.long_memory) // 2)
        memories = list(self.long_memory)
        random.shuffle(memories)

        for i in range(0, n_pairs * 2, 2):
            if i + 1 >= len(memories):
                break
            m1 = memories[i]
            m2 = memories[i + 1]

            # 计算语义相似度（简化：用关键词重叠）
            words1 = set(m1.content.replace("，", " ").replace("。", " ").split())
            words2 = set(m2.content.replace("，", " ").replace("。", " ").split())
            overlap = len(words1 & words2)
            similarity = overlap / max(len(words1 | words2), 1)

            # 如果有一定相似度，强化两者的关联
            if similarity > 0.1 or random.random() < 0.3:
                # 互相强化（通过权重提升来表示关联增强）
                gain = self.rem_memory_gain * (1 + similarity)
                m1.weight = self._clip(m1.weight + gain * 0.5)
                m2.weight = self._clip(m2.weight + gain * 0.5)
                integrated += 1

                # 标记为关联记忆
                if not hasattr(m1, 'associated'):
                    m1.associated = []
                if not hasattr(m2, 'associated'):
                    m2.associated = []
                if m2.content not in m1.associated:
                    m1.associated.append(m2.content)
                if m1.content not in m2.associated:
                    m2.associated.append(m1.content)

        return integrated

    def get_sleep_report(self) -> Dict:
        """获取睡眠状态报告。"""
        return {
            "sleep_state": self.sleep_state,
            "sleep_cycle_count": self.sleep_cycle_count,
            "dream_count": self.dream_count,
            "sleep_spindles": self.sleep_spindles,
            "slow_waves": self.slow_waves,
            "rem_duration": self.rem_duration,
            "memory_consolidation_count": self.memory_consolidation_count,
            "ltm_integration_count": self.ltm_integration_count,
            "recent_dreams": self.dream_log[-5:] if self.dream_log else [],
        }

    # ------------------ 工作记忆系统（v6.0 Baddeley模型） ------------------

    # ===== 语音回路（Phonological Loop） =====

    def phonological_store(self, item: str, weight: float = 0.8) -> bool:
        """语音存储：将语音/语言信息存入语音回路。

        语音回路是工作记忆的子系统，负责处理语音和语言信息。
        容量约7±2个项目，信息会随时间衰减。

        参数：
            item: 要存储的语音/语言信息
            weight: 初始权重 [0,1]

        返回：
            是否成功存储
        """
        # 检查容量
        if len(self.phonological_loop) >= self.phonological_capacity:
            # 移除权重最低的
            weakest = min(self.phonological_loop, key=lambda x: x["weight"])
            self.phonological_loop.remove(weakest)

        # 添加新项目
        self.phonological_loop.append({
            "content": item,
            "weight": self._clip(weight, 0, 1),
            "tick": self.tick,
            "rehearsed": 0,  # 被复述的次数
        })

        # 增加认知负荷
        self.central_executive_load = self._clip(
            self.central_executive_load + 0.05)

        return True

    def phonological_rehearse(self, index: int = -1) -> bool:
        """发音复述：默读复述语音回路中的项目，防止衰减。

        这是语音回路的发音控制过程，通过默读来刷新记忆痕迹。

        参数：
            index: 要复述的项目索引（-1表示全部复述）

        返回：
            是否成功复述
        """
        if not self.phonological_loop:
            return False

        # 检查认知负荷
        if self.central_executive_load > 0.8:
            return False  # 认知负荷太高，无法复述

        if index == -1:
            # 全部复述
            for item in self.phonological_loop:
                item["weight"] = self._clip(item["weight"] + self.subvocal_rehearsal_rate)
                item["rehearsed"] += 1
        else:
            # 复述指定项目
            if 0 <= index < len(self.phonological_loop):
                item = self.phonological_loop[index]
                item["weight"] = self._clip(item["weight"] + self.subvocal_rehearsal_rate)
                item["rehearsed"] += 1
            else:
                return False

        # 复述需要认知努力
        self.central_executive_load = self._clip(
            self.central_executive_load + self.rehearsal_effort)

        return True

    def phonological_decay(self):
        """语音信息衰减：语音回路中的信息随时间自然衰减。"""
        for item in list(self.phonological_loop):
            item["weight"] -= self.phonological_decay_rate
            if item["weight"] <= 0:
                self.phonological_loop.remove(item)

    def get_phonological_loop(self) -> List[Dict]:
        """获取语音回路的内容。"""
        return sorted(self.phonological_loop, key=lambda x: x["weight"], reverse=True)

    # ===== 视空间模板（Visuospatial Sketchpad） =====

    def visuospatial_store(self, item: str, spatial: bool = False,
                           weight: float = 0.8) -> bool:
        """视空间存储：将视觉/空间信息存入视空间模板。

        视空间模板负责处理视觉和空间信息，
        包括心理表象、空间操作、心理旋转等。

        参数：
            item: 要存储的视觉/空间信息
            spatial: 是否是空间信息（vs 视觉信息）
            weight: 初始权重 [0,1]

        返回：
            是否成功存储
        """
        # 检查容量
        if len(self.visuospatial_sketchpad) >= self.visuospatial_capacity:
            # 移除权重最低的
            weakest = min(self.visuospatial_sketchpad, key=lambda x: x["weight"])
            self.visuospatial_sketchpad.remove(weakest)

        # 添加新项目
        self.visuospatial_sketchpad.append({
            "content": item,
            "type": "spatial" if spatial else "visual",
            "weight": self._clip(weight, 0, 1),
            "tick": self.tick,
            "manipulated": 0,  # 被操作的次数
        })

        # 增加认知负荷
        self.central_executive_load = self._clip(
            self.central_executive_load + 0.08)

        return True

    def visuospatial_manipulate(self, index: int = 0,
                                operation: str = "rotate") -> bool:
        """视空间操作：对视空间模板中的信息进行心理操作。

        比如心理旋转、空间转换、表象扫描等。

        参数：
            index: 要操作的项目索引
            operation: 操作类型（rotate/scale/translate/scan）

        返回：
            是否成功操作
        """
        if not self.visuospatial_sketchpad:
            return False

        # 检查认知负荷（空间操作需要更多认知资源）
        if self.central_executive_load > 0.7:
            return False

        if 0 <= index < len(self.visuospatial_sketchpad):
            item = self.visuospatial_sketchpad[index]
            item["weight"] = self._clip(item["weight"] + 0.1)  # 操作强化记忆
            item["manipulated"] += 1
            item["last_operation"] = operation

            # 空间操作需要更多认知努力
            effort = 0.15 if item["type"] == "spatial" else 0.1
            self.central_executive_load = self._clip(
                self.central_executive_load + effort)

            return True
        return False

    def visuospatial_decay(self):
        """视空间信息衰减：视空间模板中的信息随时间自然衰减。"""
        for item in list(self.visuospatial_sketchpad):
            item["weight"] -= self.visuospatial_decay_rate
            if item["weight"] <= 0:
                self.visuospatial_sketchpad.remove(item)

    def get_visuospatial_sketchpad(self) -> List[Dict]:
        """获取视空间模板的内容。"""
        return sorted(self.visuospatial_sketchpad, key=lambda x: x["weight"], reverse=True)

    # ===== 情景缓冲器（Episodic Buffer） =====

    def episodic_store(self, content: str,
                       sources: Optional[List[str]] = None,
                       weight: float = 0.9) -> bool:
        """情景存储：将整合后的信息存入情景缓冲器。

        情景缓冲器负责整合来自语音回路、视空间模板和长时记忆的信息，
        形成连贯的情景或事件表征。

        参数：
            content: 整合后的情景内容
            sources: 信息来源列表（phonological/visuospatial/ltm）
            weight: 初始权重 [0,1]

        返回：
            是否成功存储
        """
        if sources is None:
            sources = []

        # 检查容量
        if len(self.episodic_buffer) >= self.episodic_capacity:
            # 移除权重最低的
            weakest = min(self.episodic_buffer, key=lambda x: x["weight"])
            self.episodic_buffer.remove(weakest)

        # 添加新情景
        self.episodic_buffer.append({
            "content": content,
            "sources": sources,
            "weight": self._clip(weight, 0, 1),
            "tick": self.tick,
            "integrated": len(sources),  # 整合的来源数量
        })

        # 整合需要较多认知资源
        self.central_executive_load = self._clip(
            self.central_executive_load + 0.1 * len(sources))

        return True

    def episodic_integrate(self) -> Optional[Dict]:
        """情景整合：将语音、视空间信息整合为连贯的情景。

        这是情景缓冲器的核心功能：将不同来源的信息整合在一起。

        返回：
            整合后的情景，失败返回None
        """
        # 需要至少两个子系统都有内容
        if len(self.phonological_loop) == 0 or len(self.visuospatial_sketchpad) == 0:
            return None

        # 检查认知负荷（整合需要大量资源）
        if self.central_executive_load > 0.6:
            return None

        # 取两个子系统中权重最高的项目进行整合
        phon_item = max(self.phonological_loop, key=lambda x: x["weight"])
        vis_item = max(self.visuospatial_sketchpad, key=lambda x: x["weight"])

        # 整合内容
        integrated_content = f"{vis_item['content']}（{phon_item['content']}）"

        # 存入情景缓冲器
        self.episodic_store(
            integrated_content,
            sources=["phonological", "visuospatial"],
            weight=min(phon_item["weight"], vis_item["weight"]) * 0.9
        )

        # 整合消耗认知资源
        self.central_executive_load = self._clip(
            self.central_executive_load + 0.15)

        return {
            "content": integrated_content,
            "phonological_source": phon_item["content"],
            "visuospatial_source": vis_item["content"],
            "weight": min(phon_item["weight"], vis_item["weight"]) * 0.9,
        }

    def episodic_decay(self):
        """情景信息衰减：情景缓冲器中的信息随时间自然衰减。"""
        for item in list(self.episodic_buffer):
            item["weight"] -= self.episodic_decay_rate
            if item["weight"] <= 0:
                self.episodic_buffer.remove(item)

    def get_episodic_buffer(self) -> List[Dict]:
        """获取情景缓冲器的内容。"""
        return sorted(self.episodic_buffer, key=lambda x: x["weight"], reverse=True)

    # ===== 中央执行系统（Central Executive） =====

    def central_executive_attention(self, target: str) -> bool:
        """注意力控制：中央执行系统分配注意力到目标。

        中央执行系统是工作记忆的控制中心，负责注意力分配、
        任务切换、计划制定等高级认知功能。

        参数：
            target: 注意力目标

        返回：
            是否成功分配注意力
        """
        # 检查认知负荷
        if self.central_executive_load > 0.9:
            return False

        # 注意力分配
        self.current_task = target
        self.task_history.append(target)
        if len(self.task_history) > 20:
            self.task_history.pop(0)

        # 注意力分配需要认知努力
        self.central_executive_load = self._clip(
            self.central_executive_load + 0.1)

        # 注意力影响感知
        self.attention_factor = self._clip(
            self.attention_factor + 0.05, 0.1, 1.0)

        return True

    def task_switch(self, new_task: str) -> Dict:
        """任务切换：从当前任务切换到新任务。

        任务切换会产生切换代价（switch cost），
        表现为反应时增加、错误率上升。

        参数：
            new_task: 新任务

        返回：
            切换结果
        """
        old_task = self.current_task

        # 计算切换代价
        if old_task and old_task != new_task:
            switch_cost = self.task_switching_cost
            self.central_executive_load = self._clip(
                self.central_executive_load + switch_cost)
        else:
            switch_cost = 0.0

        # 切换到新任务
        self.current_task = new_task
        self.task_history.append(new_task)
        if len(self.task_history) > 20:
            self.task_history.pop(0)

        return {
            "old_task": old_task,
            "new_task": new_task,
            "switch_cost": round(switch_cost, 3),
            "cognitive_load_after": round(self.central_executive_load, 3),
        }

    def update_working_memory(self):
        """更新工作记忆：所有子系统的衰减和认知负荷恢复。

        每个tick调用一次，模拟工作记忆的动态变化。
        """
        # 各子系统衰减
        self.phonological_decay()
        self.visuospatial_decay()
        self.episodic_decay()

        # 认知负荷自然恢复
        self.central_executive_load = self._clip(
            self.central_executive_load - 0.05, 0, 1)

        # 子系统间的干扰
        if (self.phonological_loop and self.visuospatial_sketchpad):
            # 同时使用两个子系统会产生干扰
            for item in self.phonological_loop:
                item["weight"] -= self.wm_interference
            for item in self.visuospatial_sketchpad:
                item["weight"] -= self.wm_interference

    def clear_working_memory(self):
        """清空工作记忆的所有内容。"""
        self.phonological_loop.clear()
        self.visuospatial_sketchpad.clear()
        self.episodic_buffer.clear()
        self.central_executive_load = 0.0
        self.current_task = None

    # ===== 工作记忆操作 =====

    def wm_operation(self, operation: str, **kwargs) -> Dict:
        """工作记忆操作：统一的工作记忆操作接口。

        参数：
            operation: 操作类型
                - "store_phonological": 存储语音信息
                - "rehearse": 复述语音信息
                - "store_visuospatial": 存储视空间信息
                - "manipulate": 操作视空间信息
                - "integrate": 整合情景
                - "attention": 分配注意力
                - "switch_task": 任务切换
                - "update": 更新工作记忆
                - "clear": 清空工作记忆

        返回：
            操作结果
        """
        if operation == "store_phonological":
            item = kwargs.get("item", "")
            weight = kwargs.get("weight", 0.8)
            success = self.phonological_store(item, weight)
            return {"operation": operation, "success": success,
                    "item": item}

        elif operation == "rehearse":
            index = kwargs.get("index", -1)
            success = self.phonological_rehearse(index)
            return {"operation": operation, "success": success}

        elif operation == "store_visuospatial":
            item = kwargs.get("item", "")
            spatial = kwargs.get("spatial", False)
            weight = kwargs.get("weight", 0.8)
            success = self.visuospatial_store(item, spatial, weight)
            return {"operation": operation, "success": success,
                    "item": item, "spatial": spatial}

        elif operation == "manipulate":
            index = kwargs.get("index", 0)
            op = kwargs.get("operation_type", "rotate")
            success = self.visuospatial_manipulate(index, op)
            return {"operation": operation, "success": success,
                    "manipulation": op}

        elif operation == "integrate":
            result = self.episodic_integrate()
            return {"operation": operation, "success": result is not None,
                    "result": result}

        elif operation == "attention":
            target = kwargs.get("target", "")
            success = self.central_executive_attention(target)
            return {"operation": operation, "success": success,
                    "target": target}

        elif operation == "switch_task":
            new_task = kwargs.get("new_task", "")
            result = self.task_switch(new_task)
            return {"operation": operation, **result}

        elif operation == "update":
            self.update_working_memory()
            return {"operation": operation, "success": True}

        elif operation == "clear":
            self.clear_working_memory()
            return {"operation": operation, "success": True}

        else:
            return {"operation": operation, "success": False,
                    "error": "Unknown operation"}

    def get_working_memory_report(self) -> Dict:
        """获取工作记忆状态报告。"""
        return {
            "central_executive": {
                "cognitive_load": round(self.central_executive_load, 3),
                "current_task": self.current_task,
                "task_history": self.task_history[-5:],
            },
            "phonological_loop": {
                "count": len(self.phonological_loop),
                "capacity": self.phonological_capacity,
                "items": self.get_phonological_loop()[:3],
            },
            "visuospatial_sketchpad": {
                "count": len(self.visuospatial_sketchpad),
                "capacity": self.visuospatial_capacity,
                "items": self.get_visuospatial_sketchpad()[:3],
            },
            "episodic_buffer": {
                "count": len(self.episodic_buffer),
                "capacity": self.episodic_capacity,
                "items": self.get_episodic_buffer()[:3],
            },
            "total_load": round(
                (len(self.phonological_loop) / self.phonological_capacity +
                 len(self.visuospatial_sketchpad) / self.visuospatial_capacity +
                 len(self.episodic_buffer) / self.episodic_capacity) / 3, 3
            ),
        }

    # ------------------ 预测编码系统（v6.1 Predictive Coding） ------------------

    # ===== 预测生成 =====

    def generate_prediction(self, horizon: Optional[int] = None) -> Dict:
        """生成预测：基于当前状态和历史，预测未来的感知。

        预测编码的核心：大脑不断地对未来进行预测，
        然后用实际感知来验证和更新这些预测。

        参数：
            horizon: 预测时间范围（tick数），默认使用prediction_horizon

        返回：
            预测结果字典
        """
        if horizon is None:
            horizon = self.prediction_horizon

        # 基于历史感知生成预测
        recent_inputs = [m.content for m in self.short_memory[-5:]]
        if not recent_inputs:
            recent_inputs = [""]

        # 简单预测：基于最近输入的模式推断
        predicted_input = self._extrapolate_pattern(recent_inputs)

        # 预测的置信度（精度）
        precision = self._calculate_prediction_precision(recent_inputs)

        # 层级预测
        sensory_pred = self._generate_sensory_prediction(predicted_input)
        conceptual_pred = self._generate_conceptual_prediction(predicted_input)
        abstract_pred = self._generate_abstract_prediction(predicted_input)

        # 构建预测结果
        prediction = {
            "tick": self.tick,
            "horizon": horizon,
            "predicted_input": predicted_input,
            "precision": precision,
            "confidence": precision,  # 别名
            "sensory": sensory_pred,
            "conceptual": conceptual_pred,
            "abstract": abstract_pred,
            "based_on": len(recent_inputs),
        }

        # 保存当前预测
        self.current_prediction = prediction
        self.prediction_history.append(prediction)
        if len(self.prediction_history) > 50:
            self.prediction_history.pop(0)

        # 记录精度历史
        self.precision_history.append(precision)
        if len(self.precision_history) > 50:
            self.precision_history.pop(0)

        return prediction

    def _extrapolate_pattern(self, recent_inputs: List[str]) -> str:
        """模式外推：基于最近的输入模式，预测下一个输入。

        这是一个简化的预测机制，基于最近输入的特征来推断。
        """
        if not recent_inputs:
            return ""

        # 取最近的输入作为预测基础
        latest = recent_inputs[-1]

        # 如果有多个输入，尝试找模式
        if len(recent_inputs) >= 2:
            # 简单的模式：如果有重复，预测会继续
            if len(set(recent_inputs)) == 1:
                # 所有输入都一样，预测会继续一样
                return latest

        # 默认预测：和最近的相似
        return latest

    def _calculate_prediction_precision(self, recent_inputs: List[str]) -> float:
        """计算预测精度（置信度）。

        精度取决于：
        1. 历史输入的一致性（越一致，精度越高）
        2. 模型的不确定性（越低，精度越高）
        3. 注意力水平（越高，精度越高）
        """
        if len(recent_inputs) < 2:
            return 0.3  # 数据太少，精度低

        # 计算输入的一致性
        unique_inputs = len(set(recent_inputs))
        consistency = 1.0 - (unique_inputs / len(recent_inputs))

        # 模型不确定性的影响
        uncertainty_factor = 1.0 - self.model_uncertainty

        # 注意力的影响
        attention_factor = self.attention_factor

        # 综合计算精度
        precision = (consistency * 0.5 +
                     uncertainty_factor * 0.3 +
                     attention_factor * 0.2)

        return self._clip(precision, 0.0, 1.0)

    def _generate_sensory_prediction(self, predicted_input: str) -> Dict:
        """生成感官层预测（低级预测）。

        预测具体的感官输入模式。
        """
        # 基于预测输入生成感官电流模式
        predicted_currents = self._str_to_current(predicted_input)

        return {
            "type": "sensory",
            "input": predicted_input,
            "currents": predicted_currents,
            "expected_spikes": int(sum(1 for c in predicted_currents if c > 0.5)),
        }

    def _generate_conceptual_prediction(self, predicted_input: str) -> Dict:
        """生成概念层预测（中级预测）。

        预测输入的语义/概念内容。
        """
        # 从LTM中找相关的概念
        related_concepts = []
        for mem in self.long_memory:
            if any(word in mem.content for word in predicted_input.split()):
                related_concepts.append(mem.content[:20])
                if len(related_concepts) >= 3:
                    break

        return {
            "type": "conceptual",
            "input": predicted_input,
            "related_concepts": related_concepts,
            "expected_category": "unknown" if not related_concepts else "known",
        }

    def _generate_abstract_prediction(self, predicted_input: str) -> Dict:
        """生成抽象层预测（高级预测）。

        预测更抽象的模式、规律、意义。
        """
        # 基于情绪和上下文的抽象预测
        mood = max(self.emotion, key=lambda k: self.emotion[k])

        return {
            "type": "abstract",
            "input": predicted_input,
            "expected_mood": mood,
            "expected_novelty": self.novelty,
            "expected_attention": self.attention_factor,
        }

    # ===== 预测误差 =====

    def calculate_prediction_error(self, actual_input: str) -> Dict:
        """计算预测误差：实际感知与预测之间的差异。

        预测误差是预测编码的核心驱动力——
        大脑的目标就是最小化预测误差。

        参数：
            actual_input: 实际的感知输入

        返回：
            预测误差字典
        """
        # 如果没有预测，先生成一个
        if self.current_prediction is None:
            self.generate_prediction()

        predicted = self.current_prediction["predicted_input"]
        precision = self.current_prediction["precision"]

        # 计算误差
        error_magnitude = self._calculate_error_magnitude(predicted, actual_input)

        # 精度加权误差
        if self.precision_weighting:
            weighted_error = error_magnitude * precision
        else:
            weighted_error = error_magnitude

        # 层级误差
        sensory_error = self._calculate_sensory_error(actual_input)
        conceptual_error = self._calculate_conceptual_error(actual_input)
        abstract_error = self._calculate_abstract_error(actual_input)

        # 计算变分自由能（Variational Free Energy）
        free_energy = self._calculate_free_energy(error_magnitude, precision)

        # 构建误差结果
        error_result = {
            "tick": self.tick,
            "predicted": predicted,
            "actual": actual_input,
            "error_magnitude": error_magnitude,
            "weighted_error": weighted_error,
            "precision": precision,
            "precision_weighted": self.precision_weighting,
            "sensory_error": sensory_error,
            "conceptual_error": conceptual_error,
            "abstract_error": abstract_error,
            "free_energy": free_energy,
            "surprise": error_magnitude,  # 惊讶度 = 误差大小
        }

        # 记录误差历史
        self.prediction_error_history.append(error_result)
        if len(self.prediction_error_history) > 50:
            self.prediction_error_history.pop(0)

        # 更新自由能
        self.variational_free_energy = free_energy
        self.free_energy_history.append(free_energy)
        if len(self.free_energy_history) > 50:
            self.free_energy_history.pop(0)

        # 更新预测精度
        self._update_precision(error_magnitude)

        return error_result

    def _calculate_error_magnitude(self, predicted: str, actual: str) -> float:
        """计算预测误差的大小。

        基于字符串相似度来计算误差。
        """
        if not predicted and not actual:
            return 0.0
        if not predicted or not actual:
            return 1.0

        # 简单的字符级差异
        pred_chars = set(predicted)
        actual_chars = set(actual)

        intersection = len(pred_chars & actual_chars)
        union = len(pred_chars | actual_chars)

        if union == 0:
            return 0.0

        similarity = intersection / union
        error = 1.0 - similarity

        return error

    def _calculate_sensory_error(self, actual_input: str) -> float:
        """计算感官层预测误差。"""
        actual_currents = self._str_to_current(actual_input)
        predicted_currents = self.current_prediction["sensory"]["currents"]

        # 计算电流模式的差异
        diff = sum(abs(a - b) for a, b in zip(actual_currents, predicted_currents))
        max_diff = len(actual_currents) * 0.8  # 最大可能差异

        return min(diff / max_diff, 1.0)

    def _calculate_conceptual_error(self, actual_input: str) -> float:
        """计算概念层预测误差。"""
        predicted_concepts = self.current_prediction["conceptual"]["related_concepts"]

        if not predicted_concepts:
            return 0.5  # 没有预测，中等误差

        # 检查实际输入是否与预测的概念相关
        actual_words = set(actual_input.replace("，", " ").replace("。", " ").split())

        related_count = 0
        for concept in predicted_concepts:
            concept_words = set(concept.replace("，", " ").replace("。", " ").split())
            if actual_words & concept_words:
                related_count += 1

        error = 1.0 - (related_count / len(predicted_concepts))
        return error

    def _calculate_abstract_error(self, actual_input: str) -> float:
        """计算抽象层预测误差。"""
        # 比较情绪和注意力的预期与实际
        expected_mood = self.current_prediction["abstract"]["expected_mood"]
        actual_mood = max(self.emotion, key=lambda k: self.emotion[k])

        mood_error = 0.0 if expected_mood == actual_mood else 0.5

        # 新奇度的预期与实际
        expected_novelty = self.current_prediction["abstract"]["expected_novelty"]
        novelty_error = abs(self.novelty - expected_novelty)

        return (mood_error + novelty_error) / 2

    def _calculate_free_energy(self, error: float, precision: float) -> float:
        """计算变分自由能（Variational Free Energy）。

        自由能 = 复杂度 - 准确度
        在简化形式下：F = 误差 / 精度 - log(精度)

        自由能最小化是预测编码和主动推理的核心原则。
        """
        if precision < 0.01:
            precision = 0.01  # 避免除以零

        # 简化的自由能计算
        free_energy = (error / precision) - (0.5 * (1 + precision))

        return self._clip(free_energy, -2.0, 2.0)

    # ===== 精度加权 =====

    def _update_precision(self, error: float):
        """更新预测精度。

        精度会根据预测误差动态调整：
        - 误差小 → 精度提高（预测很准，增加置信度）
        - 误差大 → 精度降低（预测不准，降低置信度）
        """
        # 精度更新：误差越小，精度越高
        precision_change = (1.0 - error) * self.prediction_learning_rate
        self.prediction_precision = self._clip(
            self.prediction_precision + precision_change - 0.05,  # 基础衰减
            0.0, 1.0)

        # 更新模型不确定性
        self.model_uncertainty = self._clip(
            self.model_uncertainty + error * 0.02 - 0.01,
            0.0, 1.0)

    def set_precision_weighting(self, enabled: bool):
        """设置是否启用精度加权。

        精度加权是预测编码的重要机制：
        高精度的预测误差对模型更新的影响更大。
        """
        self.precision_weighting = enabled

    # ===== 模型更新 =====

    def update_model(self, error: Dict) -> Dict:
        """更新内部模型：用预测误差来最小化未来误差。

        这是预测编码的学习过程：
        大脑根据预测误差来更新内部模型，
        使未来的预测更加准确。

        参数：
            error: 预测误差字典

        返回：
            更新结果字典
        """
        error_magnitude = error["error_magnitude"]
        precision = error["precision"]

        # 误差越大，更新越多（但受精度加权）
        if self.precision_weighting:
            update_amount = error_magnitude * precision * self.error_minimization_rate
        else:
            update_amount = error_magnitude * self.error_minimization_rate

        # 更新内部模型参数
        # 这里简化为更新一些关键参数
        updates = {}

        # 1. 更新学习率（误差大时学习更快）
        old_hebbian_rate = self.hebbian_rate
        self.hebbian_rate = self._clip(
            self.hebbian_rate + update_amount * 0.1,
            0.001, 0.1)
        updates["hebbian_rate"] = {
            "before": old_hebbian_rate,
            "after": self.hebbian_rate,
        }

        # 2. 更新注意力（误差大时注意力提高）
        old_attention = self.attention_factor
        self.attention_factor = self._clip(
            self.attention_factor + error_magnitude * 0.1,
            0.1, 1.0)
        updates["attention_factor"] = {
            "before": old_attention,
            "after": self.attention_factor,
        }

        # 3. 更新好奇心（误差大时好奇心增加）
        old_curiosity = self.emotion["curiosity"]
        self.emotion["curiosity"] = self._clip(
            self.emotion["curiosity"] + error_magnitude * 0.15,
            0.0, 1.0)
        updates["curiosity"] = {
            "before": old_curiosity,
            "after": self.emotion["curiosity"],
        }

        # 4. 更新内部模型字典
        actual = error["actual"]
        if actual in self.internal_model:
            self.internal_model[actual] = self._clip(
                self.internal_model[actual] + update_amount, 0, 1)
        else:
            self.internal_model[actual] = update_amount

        # 记录更新
        update_result = {
            "tick": self.tick,
            "error": error_magnitude,
            "precision": precision,
            "update_amount": update_amount,
            "updates": updates,
            "model_size": len(self.internal_model),
        }

        return update_result

    def minimize_free_energy(self, actual_input: str) -> Dict:
        """自由能最小化：完整的预测编码循环。

        1. 生成预测
        2. 计算误差
        3. 更新模型

        这是主动推理（Active Inference）的核心：
        大脑通过改变感知或改变模型来最小化自由能。

        参数：
            actual_input: 实际的感知输入

        返回：
            完整的预测编码循环结果
        """
        # 1. 生成预测
        prediction = self.generate_prediction()

        # 2. 计算预测误差
        error = self.calculate_prediction_error(actual_input)

        # 3. 更新模型（最小化误差）
        update = self.update_model(error)

        # 综合结果
        result = {
            "tick": self.tick,
            "prediction": prediction,
            "error": error,
            "update": update,
            "free_energy": self.variational_free_energy,
            "free_energy_reduced": error["error_magnitude"] > 0,  # 简化判断
        }

        return result

    # ===== 预测历史和报告 =====

    def get_prediction_report(self) -> Dict:
        """获取预测编码状态报告。"""
        # 计算平均误差
        if self.prediction_error_history:
            avg_error = sum(e["error_magnitude"] for e in self.prediction_error_history[-10:]) / min(10, len(self.prediction_error_history))
        else:
            avg_error = 0.0

        # 计算平均精度
        if self.precision_history:
            avg_precision = sum(self.precision_history[-10:]) / min(10, len(self.precision_history))
        else:
            avg_precision = 0.0

        # 计算平均自由能
        if self.free_energy_history:
            avg_free_energy = sum(self.free_energy_history[-10:]) / min(10, len(self.free_energy_history))
        else:
            avg_free_energy = 0.0

        return {
            "current_prediction": self.current_prediction,
            "prediction_precision": round(self.prediction_precision, 3),
            "model_uncertainty": round(self.model_uncertainty, 3),
            "variational_free_energy": round(self.variational_free_energy, 3),
            "avg_error": round(avg_error, 3),
            "avg_precision": round(avg_precision, 3),
            "avg_free_energy": round(avg_free_energy, 3),
            "prediction_count": len(self.prediction_history),
            "error_count": len(self.prediction_error_history),
            "model_size": len(self.internal_model),
            "precision_weighting": self.precision_weighting,
            "prediction_horizon": self.prediction_horizon,
            "recent_errors": [
                {
                    "tick": e["tick"],
                    "error": round(e["error_magnitude"], 3),
                    "precision": round(e["precision"], 3),
                }
                for e in self.prediction_error_history[-5:]
            ],
        }

    # ------------------ 神经振荡系统（v6.2 脑电波） ------------------

    # ===== 脑电波基础 =====

    def update_brainwaves(self):
        """更新脑电波状态：根据当前意识状态调整各波段功率。

        脑电波会随意识状态自然变化：
        - 清醒思考：β波主导
        - 放松冥想：α波增强
        - 困倦浅睡：θ波增强
        - 深睡：δ波主导
        - 高级认知：γ波增强
        """
        # 基于当前状态计算各波段功率
        # 1. β波：清醒活跃、思考、注意力集中
        beta_power = 0.3 + self.attention_factor * 0.4
        if self.current_task:
            beta_power += 0.2

        # 2. α波：放松、平静、闭眼
        alpha_power = 0.2 + self.emotion["calm"] * 0.3
        if self.emotion["stress"] < 0.2:
            alpha_power += 0.1

        # 3. θ波：困倦、冥想、浅睡
        theta_power = 0.1 + (1 - self.attention_factor) * 0.2
        if self.sleep_state in ["drowsy", "light_sleep"]:
            theta_power += 0.4

        # 4. δ波：深睡、无意识
        delta_power = 0.05
        if self.sleep_state in ["deep_sleep"]:
            delta_power += 0.6
        elif self.sleep_state in ["rem"]:
            delta_power += 0.1

        # 5. γ波：高级认知、意识、感知整合
        gamma_power = 0.2 + self.attention_factor * 0.2
        if self.ignition_state:
            gamma_power += 0.2  # 意识点火时γ波增强
        if len(self.thought_space) > 5:
            gamma_power += 0.1  # 思考活跃时γ波增强

        # 应用调制速率（平滑过渡）
        self.brainwaves["beta"] = self._clip(
            self.brainwaves["beta"] + (beta_power - self.brainwaves["beta"]) * self.oscillation_modulation_rate,
            0, 1)
        self.brainwaves["alpha"] = self._clip(
            self.brainwaves["alpha"] + (alpha_power - self.brainwaves["alpha"]) * self.oscillation_modulation_rate,
            0, 1)
        self.brainwaves["theta"] = self._clip(
            self.brainwaves["theta"] + (theta_power - self.brainwaves["theta"]) * self.oscillation_modulation_rate,
            0, 1)
        self.brainwaves["delta"] = self._clip(
            self.brainwaves["delta"] + (delta_power - self.brainwaves["delta"]) * self.oscillation_modulation_rate,
            0, 1)
        self.brainwaves["gamma"] = self._clip(
            self.brainwaves["gamma"] + (gamma_power - self.brainwaves["gamma"]) * self.oscillation_modulation_rate,
            0, 1)

        # 更新主导频率
        dominant = max(self.brainwaves, key=lambda k: self.brainwaves[k])
        freq_map = {"delta": 2, "theta": 6, "alpha": 10, "beta": 20, "gamma": 40}
        self.oscillation_frequency = freq_map.get(dominant, 10)

        # 更新振荡相位
        self.oscillation_phase += 0.1
        if self.oscillation_phase > 6.283:  # 2π
            self.oscillation_phase -= 6.283

        # 记录历史
        self.brainwave_history.append(dict(self.brainwaves))
        if len(self.brainwave_history) > 100:
            self.brainwave_history.pop(0)

    def get_dominant_wave(self) -> str:
        """获取当前主导脑电波。"""
        return max(self.brainwaves, key=lambda k: self.brainwaves[k])

    def get_consciousness_state(self) -> str:
        """根据脑电波判断意识状态。"""
        dominant = self.get_dominant_wave()
        states = {
            "delta": "深睡/无意识",
            "theta": "浅睡/困倦/冥想",
            "alpha": "放松/清醒闭眼",
            "beta": "清醒/思考/专注",
            "gamma": "高级认知/意识整合",
        }
        return states.get(dominant, "未知")

    # ===== 神经同步性 =====

    def update_neural_synchrony(self):
        """更新神经同步性：神经元群同步放电的程度。

        神经同步性与意识、注意力、感知整合密切相关。
        γ波同步被认为是"绑定问题"的解决方案。
        """
        # 基于γ波功率计算同步性
        gamma_sync = self.brainwaves["gamma"] * 0.5

        # 基于注意力计算同步性
        attention_sync = self.attention_factor * 0.3

        # 基于点火状态计算同步性
        ignition_sync = 0.2 if self.ignition_state else 0.0

        # 总同步性
        synchrony = gamma_sync + attention_sync + ignition_sync

        # 应用衰减和平滑
        self.neural_synchrony = self._clip(
            self.neural_synchrony + (synchrony - self.neural_synchrony) * 0.1,
            0, 1)

        # 记录历史
        self.synchrony_history.append(self.neural_synchrony)
        if len(self.synchrony_history) > 100:
            self.synchrony_history.pop(0)

    def gamma_binding(self, features: List[str]) -> Dict:
        """γ波绑定：将分散的特征整合为统一的感知。

        这是"绑定问题"（Binding Problem）的神经振荡解释：
        不同脑区处理的特征通过γ波同步整合为统一的意识体验。

        参数：
            features: 要绑定的特征列表

        返回：
            绑定结果
        """
        if not features:
            return {"bound": False, "content": "", "strength": 0.0}

        # γ波强度影响绑定质量
        binding_strength = self.brainwaves["gamma"] * self.gamma_binding_strength

        # 同步性影响绑定质量
        binding_strength *= self.neural_synchrony

        # 绑定后的内容
        bound_content = " + ".join(features)

        # 绑定成功的概率
        binding_success = binding_strength > 0.2

        result = {
            "bound": binding_success,
            "content": bound_content if binding_success else "",
            "features": features,
            "strength": round(binding_strength, 3),
            "gamma_power": round(self.brainwaves["gamma"], 3),
            "synchrony": round(self.neural_synchrony, 3),
        }

        # 如果绑定成功，推入思考空间（进入意识）
        if binding_success:
            self._push_thought(bound_content, source="perceptual",
                               activation=binding_strength)

        return result

    # ===== 跨频耦合 =====

    def update_cross_frequency_coupling(self):
        """更新跨频耦合：不同频率脑电波之间的相互作用。

        相位-幅度耦合（PAC）是最常见的跨频耦合：
        低频波的相位调制高频波的幅度。
        例如：θ波相位调制γ波幅度。
        """
        # θ-γ耦合：记忆和认知中的重要机制
        theta_gamma_coupling = self.brainwaves["theta"] * self.brainwaves["gamma"] * 0.5

        # α-β耦合：注意力调节
        alpha_beta_coupling = self.brainwaves["alpha"] * self.brainwaves["beta"] * 0.3

        # 总耦合强度
        coupling = theta_gamma_coupling + alpha_beta_coupling

        # 平滑更新
        self.cross_frequency_coupling = self._clip(
            self.cross_frequency_coupling + (coupling - self.cross_frequency_coupling) * 0.1,
            0, 1)

    # ===== 振荡调制 =====

    def modulate_oscillation(self, wave: str, amount: float):
        """调制特定脑电波的功率。

        参数：
            wave: 脑电波类型（delta/theta/alpha/beta/gamma）
            amount: 调制量（正为增强，负为减弱）
        """
        if wave in self.brainwaves:
            self.brainwaves[wave] = self._clip(
                self.brainwaves[wave] + amount, 0, 1)

    def induce_state(self, state: str):
        """诱导特定意识状态（通过调节脑电波）。

        参数：
            state: 目标状态
                - "relax": 放松（增强α波）
                - "focus": 专注（增强β波）
                - "meditate": 冥想（增强θ波）
                - "sleep": 睡眠（增强δ波）
                - "insight": 洞见（增强γ波）
        """
        if state == "relax":
            self.modulate_oscillation("alpha", 0.3)
            self.modulate_oscillation("beta", -0.2)
            self.emotion["calm"] = self._clip(self.emotion["calm"] + 0.2)
            self.attention_factor = self._clip(self.attention_factor - 0.1, 0.1, 1.0)

        elif state == "focus":
            self.modulate_oscillation("beta", 0.3)
            self.modulate_oscillation("alpha", -0.1)
            self.attention_factor = self._clip(self.attention_factor + 0.2, 0.1, 1.0)

        elif state == "meditate":
            self.modulate_oscillation("theta", 0.3)
            self.modulate_oscillation("alpha", 0.2)
            self.modulate_oscillation("beta", -0.3)
            self.emotion["calm"] = self._clip(self.emotion["calm"] + 0.3)
            self.emotion["stress"] = self._clip(self.emotion["stress"] - 0.3)

        elif state == "sleep":
            self.modulate_oscillation("delta", 0.4)
            self.modulate_oscillation("theta", 0.3)
            self.modulate_oscillation("beta", -0.4)
            self.attention_factor = self._clip(self.attention_factor - 0.3, 0.1, 1.0)

        elif state == "insight":
            self.modulate_oscillation("gamma", 0.3)
            self.modulate_oscillation("alpha", 0.1)
            self.emotion["pleasure"] = self._clip(self.emotion["pleasure"] + 0.2)

    # ===== 脑电波分析 =====

    def get_brainwave_spectrum(self) -> Dict:
        """获取脑电波频谱分析。"""
        total = sum(self.brainwaves.values())
        if total == 0:
            total = 1

        spectrum = {
            wave: {
                "power": round(power, 3),
                "ratio": round(power / total, 3),
            }
            for wave, power in self.brainwaves.items()
        }

        return {
            "dominant": self.get_dominant_wave(),
            "consciousness_state": self.get_consciousness_state(),
            "spectrum": spectrum,
            "total_power": round(total, 3),
            "dominant_frequency": self.oscillation_frequency,
            "phase": round(self.oscillation_phase, 3),
        }

    def get_neural_oscillation_report(self) -> Dict:
        """获取神经振荡系统状态报告。"""
        spectrum = self.get_brainwave_spectrum()

        return {
            "brainwaves": spectrum["spectrum"],
            "dominant_wave": spectrum["dominant"],
            "consciousness_state": spectrum["consciousness_state"],
            "dominant_frequency": spectrum["dominant_frequency"],
            "oscillation_phase": round(self.oscillation_phase, 3),
            "neural_synchrony": round(self.neural_synchrony, 3),
            "cross_frequency_coupling": round(self.cross_frequency_coupling, 3),
            "gamma_binding_strength": round(
                self.brainwaves["gamma"] * self.gamma_binding_strength * self.neural_synchrony, 3),
            "sleep_state": self.sleep_state,
            "attention": round(self.attention_factor, 3),
            "ignition": self.ignition_state,
        }

    # ------------------ 主动推理系统（v6.3 Active Inference） ------------------

    # ===== 行动策略生成 =====

    def generate_action_strategies(self) -> List[Dict]:
        """生成行动策略：生成可能的行动选项及其预期结果。

        主动推理的第一步：考虑所有可能的行动，
        并预测每个行动的结果。
        """
        strategies = []

        for action in self.action_space:
            # 预测行动结果
            prediction = self._predict_action_outcome(action)

            # 计算预期自由能
            expected_fe = self._calculate_expected_free_energy(action, prediction)

            strategy = {
                "action": action,
                "prediction": prediction,
                "expected_free_energy": expected_fe,
                "value": -expected_fe,  # 价值 = -自由能
            }
            strategies.append(strategy)

            # 保存预测
            self.action_predictions[action] = prediction
            self.expected_free_energy[action] = expected_fe

        # 按预期自由能排序（越低越好）
        strategies.sort(key=lambda s: s["expected_free_energy"])

        return strategies

    def _predict_action_outcome(self, action: str) -> Dict:
        """预测行动的结果。

        基于当前状态和内部模型，预测执行某个行动后的结果。
        """
        # 基于当前状态的基线预测
        base_prediction = {
            "novelty": self.novelty,
            "certainty": 1.0 - self.model_uncertainty,
            "pleasure": self.emotion["pleasure"],
            "stress": self.emotion["stress"],
            "learning": 0.0,
            "social": 0.0,
            "energy": self.attention_factor,
        }

        # 根据行动类型调整预测
        if action == "explore":
            # 探索：新奇度增加，不确定性减少
            base_prediction["novelty"] += 0.2
            base_prediction["certainty"] += 0.1
            base_prediction["learning"] += 0.3
            base_prediction["energy"] -= 0.1

        elif action == "exploit":
            # 利用：使用已知知识，确定性高
            base_prediction["certainty"] += 0.2
            base_prediction["pleasure"] += 0.1
            base_prediction["energy"] -= 0.05

        elif action == "wait":
            # 等待：观察更多信息
            base_prediction["certainty"] += 0.05
            base_prediction["energy"] += 0.05

        elif action == "focus":
            # 专注：注意力提高，学习增加
            base_prediction["energy"] += 0.2
            base_prediction["learning"] += 0.2
            base_prediction["certainty"] += 0.1

        elif action == "relax":
            # 放松：压力减少，愉悦增加
            base_prediction["stress"] -= 0.2
            base_prediction["pleasure"] += 0.15
            base_prediction["energy"] -= 0.2

        elif action == "learn":
            # 学习：强化记忆，确定性增加
            base_prediction["learning"] += 0.4
            base_prediction["certainty"] += 0.15
            base_prediction["energy"] -= 0.15

        elif action == "socialize":
            # 社交：社交增加，愉悦增加
            base_prediction["social"] += 0.5
            base_prediction["pleasure"] += 0.2
            base_prediction["learning"] += 0.1

        # 裁剪到 [0,1]
        for key in base_prediction:
            base_prediction[key] = self._clip(base_prediction[key], 0.0, 1.0)

        return base_prediction

    # ===== 预期自由能 =====

    def _calculate_expected_free_energy(self, action: str, prediction: Dict) -> float:
        """计算预期自由能（Expected Free Energy, EFE）。

        预期自由能 = 风险 + 模糊度
        - 风险：偏离偏好的程度
        - 模糊度：预期的不确定性

        主动推理选择预期自由能最小的行动。
        """
        # 1. 风险项（Risk）：预测结果与偏好的差异
        risk = 0.0
        for pref_key, pref_value in self.preferences.items():
            if pref_key in prediction:
                diff = abs(prediction[pref_key] - pref_value)
                risk += diff * pref_value

        # 2. 模糊度（Ambiguity）：预期的不确定性
        ambiguity = 1.0 - prediction.get("certainty", 0.5)

        # 3. 探索奖励（Novelty bonus）
        novelty_bonus = prediction.get("novelty", 0) * self.exploration_bonus

        # 预期自由能 = 风险 + 模糊度 - 探索奖励
        expected_fe = risk + ambiguity - novelty_bonus

        return self._clip(expected_fe, 0.0, 2.0)

    # ===== 行动选择 =====

    def select_action(self, strategies: Optional[List[Dict]] = None) -> Dict:
        """选择行动：基于预期自由能选择最优行动。

        使用 softmax 选择，温度参数控制探索-利用平衡。
        """
        if strategies is None:
            strategies = self.generate_action_strategies()

        # 计算 softmax 概率
        values = [-s["expected_free_energy"] for s in strategies]
        max_val = max(values)

        # softmax
        exp_values = [np.exp((v - max_val) / self.action_selection_temperature)
                      for v in values]
        total = sum(exp_values)
        probabilities = [ev / total for ev in exp_values]

        # 按概率选择
        import random
        r = random.random()
        cumulative = 0.0
        selected_idx = 0
        for i, prob in enumerate(probabilities):
            cumulative += prob
            if r <= cumulative:
                selected_idx = i
                break

        selected = strategies[selected_idx]
        self.current_action = selected["action"]

        return {
            "selected_action": selected["action"],
            "expected_free_energy": selected["expected_free_energy"],
            "probability": probabilities[selected_idx],
            "all_probabilities": {
                s["action"]: round(p, 3)
                for s, p in zip(strategies, probabilities)
            },
            "strategies": strategies,
        }

    # ===== 行动执行 =====

    def execute_action(self, action: Optional[str] = None) -> Dict:
        """执行行动：执行选择的行动并返回结果。

        行动会改变大脑的内部状态，
        然后用实际结果更新内部模型。
        """
        if action is None:
            action = self.current_action or "wait"

        # 执行行动，改变状态
        result = self._perform_action(action)

        # 记录行动历史
        action_record = {
            "tick": self.tick,
            "action": action,
            "result": result,
            "expected_free_energy": self.expected_free_energy.get(action, 0.0),
        }
        self.action_history.append(action_record)
        if len(self.action_history) > 50:
            self.action_history.pop(0)

        # 用实际结果更新模型（主动推理的学习部分）
        self._update_action_model(action, result)

        return action_record

    def _perform_action(self, action: str) -> Dict:
        """执行具体的行动，改变大脑状态。"""
        result = {
            "action": action,
            "success": True,
            "effects": {},
        }

        if action == "explore":
            # 探索：增加新奇度，提高好奇心
            self.novelty = self._clip(self.novelty + 0.1, 0, 1)
            self.emotion["curiosity"] = self._clip(
                self.emotion["curiosity"] + 0.1, 0, 1)
            self.attention_factor = self._clip(
                self.attention_factor + 0.05, 0.1, 1.0)
            result["effects"] = {
                "novelty": +0.1,
                "curiosity": +0.1,
                "attention": +0.05,
            }

        elif action == "exploit":
            # 利用：使用已知知识，增加愉悦
            self.emotion["pleasure"] = self._clip(
                self.emotion["pleasure"] + 0.05, 0, 1)
            self.model_uncertainty = self._clip(
                self.model_uncertainty - 0.05, 0, 1)
            result["effects"] = {
                "pleasure": +0.05,
                "uncertainty": -0.05,
            }

        elif action == "wait":
            # 等待：恢复能量
            self.attention_factor = self._clip(
                self.attention_factor + 0.02, 0.1, 1.0)
            result["effects"] = {
                "attention": +0.02,
            }

        elif action == "focus":
            # 专注：提高注意力
            self.attention_factor = self._clip(
                self.attention_factor + 0.15, 0.1, 1.0)
            self.emotion["stress"] = self._clip(
                self.emotion["stress"] + 0.05, 0, 1)
            result["effects"] = {
                "attention": +0.15,
                "stress": +0.05,
            }

        elif action == "relax":
            # 放松：减少压力，增加平静
            self.emotion["stress"] = self._clip(
                self.emotion["stress"] - 0.15, 0, 1)
            self.emotion["calm"] = self._clip(
                self.emotion["calm"] + 0.1, 0, 1)
            self.attention_factor = self._clip(
                self.attention_factor - 0.1, 0.1, 1.0)
            # 诱导α波
            self.modulate_oscillation("alpha", 0.1)
            result["effects"] = {
                "stress": -0.15,
                "calm": +0.1,
                "attention": -0.1,
            }

        elif action == "learn":
            # 学习：强化记忆
            # 随机选择一些记忆强化
            if self.short_memory:
                for mem in self.short_memory[:3]:
                    mem.weight = self._clip(mem.weight + 0.1, 0, 1)
            self.model_uncertainty = self._clip(
                self.model_uncertainty - 0.1, 0, 1)
            result["effects"] = {
                "memory_strength": +0.1,
                "uncertainty": -0.1,
            }

        elif action == "socialize":
            # 社交：增加社交相关的
            self.emotion["pleasure"] = self._clip(
                self.emotion["pleasure"] + 0.1, 0, 1)
            result["effects"] = {
                "pleasure": +0.1,
                "social": +0.5,
            }

        else:
            result["success"] = False
            result["effects"] = {}

        return result

    def _update_action_model(self, action: str, result: Dict):
        """用行动结果更新内部模型。

        这是主动推理的学习部分：
        比较预期和实际结果，更新预测模型。
        """
        # 获取预期结果
        expected = self.action_predictions.get(action, {})

        # 计算预测误差
        if expected and result.get("success"):
            # 简单的误差计算：预期与实际的差异
            error = 0.0
            for key in result.get("effects", {}):
                if key in expected:
                    expected_val = expected.get(key, 0)
                    actual_val = result["effects"][key]
                    error += abs(actual_val - expected_val)

            # 更新模型不确定性
            self.model_uncertainty = self._clip(
                self.model_uncertainty + error * 0.01 - 0.005,
                0.0, 1.0)

    # ===== 主动推理完整循环 =====

    def active_inference_step(self) -> Dict:
        """主动推理完整循环：感知 → 预测 → 选择 → 行动 → 学习。

        这是主动推理的完整流程：
        1. 生成行动策略
        2. 选择最优行动
        3. 执行行动
        4. 用结果更新模型

        同时也进行知觉推理（预测编码）。
        """
        if not self.active_inference_enabled:
            return {"enabled": False}

        # 1. 生成行动策略
        strategies = self.generate_action_strategies()

        # 2. 选择行动
        selection = self.select_action(strategies)

        # 3. 执行行动
        action_result = self.execute_action(selection["selected_action"])

        # 4. 知觉推理（预测编码）
        # 用当前状态进行预测编码更新
        current_input = self.short_memory[-1].content if self.short_memory else ""
        if current_input:
            self.minimize_free_energy(current_input)

        result = {
            "tick": self.tick,
            "selected_action": selection["selected_action"],
            "expected_free_energy": selection["expected_free_energy"],
            "action_result": action_result,
            "selection_probabilities": selection["all_probabilities"],
            "current_free_energy": self.variational_free_energy,
        }

        return result

    def set_preferences(self, **kwargs):
        """设置偏好：调整大脑的目标和偏好。

        偏好决定了大脑认为什么是"好的"结果，
        主动推理会选择能实现这些偏好的行动。
        """
        for key, value in kwargs.items():
            if key in self.preferences:
                self.preferences[key] = self._clip(value, 0.0, 1.0)

    def add_goal(self, goal: str, priority: float = 0.5):
        """添加目标。"""
        self.goals.append({
            "goal": goal,
            "priority": self._clip(priority, 0, 1),
            "tick": self.tick,
        })
        # 按优先级排序
        self.goals.sort(key=lambda g: g["priority"], reverse=True)

    # ===== 主动推理报告 =====

    def get_active_inference_report(self) -> Dict:
        """获取主动推理系统状态报告。"""
        # 生成当前策略（如果还没有）
        if not self.expected_free_energy:
            self.generate_action_strategies()

        return {
            "enabled": self.active_inference_enabled,
            "current_action": self.current_action,
            "action_count": len(self.action_history),
            "preferences": {k: round(v, 3) for k, v in self.preferences.items()},
            "goals": self.goals[:5],
            "expected_free_energy": {
                k: round(v, 3)
                for k, v in sorted(self.expected_free_energy.items(),
                                   key=lambda x: x[1])
            },
            "best_action": min(self.expected_free_energy, key=self.expected_free_energy.get)
                if self.expected_free_energy else None,
            "selection_temperature": self.action_selection_temperature,
            "exploration_bonus": self.exploration_bonus,
            "recent_actions": [
                {
                    "tick": a["tick"],
                    "action": a["action"],
                    "efe": round(a["expected_free_energy"], 3),
                    "success": a["result"]["success"],
                }
                for a in self.action_history[-5:]
            ],
        }

    # ------------------ 脑区分化系统（v6.4 Brain Regionalization） ------------------

    # ===== 脑区基础 =====

    def update_regions(self):
        """更新各脑区的活动状态。

        根据当前大脑状态，调整各脑区的活动水平。
        """
        # 1. 感觉皮层：受输入强度影响
        sensory_activity = 0.5 + self.attention_factor * 0.3
        if self.short_memory:
            sensory_activity += 0.1
        self.brain_regions["sensory_cortex"]["activity"] = self._clip(
            sensory_activity, 0, 1)

        # 2. 海马体：受记忆活动影响
        hippo_activity = 0.3 + len(self.short_memory) / self.max_stm * 0.4
        if self.sleep_state in ["deep_sleep", "rem"]:
            hippo_activity += 0.2  # 睡眠时记忆重放
        self.brain_regions["hippocampus"]["activity"] = self._clip(
            hippo_activity, 0, 1)

        # 3. 前额叶：受认知活动影响
        pfc_activity = 0.4 + self.central_executive_load * 0.4
        if self.current_task:
            pfc_activity += 0.1
        self.brain_regions["prefrontal"]["activity"] = self._clip(
            pfc_activity, 0, 1)

        # 4. 杏仁核：受情绪影响
        amyg_activity = 0.3 + self.emotion["stress"] * 0.5
        amyg_activity += self.emotion["curiosity"] * 0.2
        self.brain_regions["amygdala"]["activity"] = self._clip(
            amyg_activity, 0, 1)

        # 5. 联合皮层：受整合活动影响
        assoc_activity = 0.4 + self.neural_synchrony * 0.3
        if self.ignition_state:
            assoc_activity += 0.2
        self.brain_regions["association_cortex"]["activity"] = self._clip(
            assoc_activity, 0, 1)

        # 记录历史
        for region, data in self.brain_regions.items():
            self.region_activity_history[region].append(data["activity"])
            if len(self.region_activity_history[region]) > 100:
                self.region_activity_history[region].pop(0)

    def get_most_active_region(self) -> str:
        """获取当前最活跃的脑区。"""
        return max(self.brain_regions,
                   key=lambda r: self.brain_regions[r]["activity"])

    # ===== 海马体 =====

    def hippocampus_encode(self, content: str, context: Optional[str] = None) -> Dict:
        """海马体编码：将新信息编码为情景记忆。

        海马体负责将短期记忆转化为长期记忆，
        以及形成情景记忆（什么、在哪里、什么时候）。
        """
        # 激活海马体
        self.brain_regions["hippocampus"]["activity"] = self._clip(
            self.brain_regions["hippocampus"]["activity"] + 0.1, 0, 1)

        # 创建情景记忆
        episodic_memory = {
            "content": content,
            "context": context or "",
            "tick": self.tick,
            "emotion": dict(self.emotion),
            "strength": 0.5 + self.brain_regions["hippocampus"]["activity"] * 0.3,
            "replay_count": 0,
        }

        # 添加到情景记忆
        self.hippocampus["episodic_memory"].append(episodic_memory)
        if len(self.hippocampus["episodic_memory"]) > 100:
            self.hippocampus["episodic_memory"].pop(0)

        # 同时存入短期记忆
        self._write_stm(content, tag="hippocampus")

        return {
            "encoded": True,
            "memory": episodic_memory,
            "hippocampus_activity": self.brain_regions["hippocampus"]["activity"],
        }

    def hippocampus_replay(self, count: int = 3) -> Dict:
        """海马体重放：记忆重放，用于记忆巩固。

        在睡眠期间，海马体会重放白天的记忆，
        将其转移到新皮层进行长期存储。
        """
        if not self.hippocampus["episodic_memory"]:
            return {"replayed": 0, "consolidated": 0}

        # 选择要重放的记忆（优先选择近期的、强度高的）
        memories = sorted(
            self.hippocampus["episodic_memory"],
            key=lambda m: m["strength"] * (1 - (self.tick - m["tick"]) / 1000),
            reverse=True
        )[:count]

        consolidated = 0
        for mem in memories:
            mem["replay_count"] += 1
            mem["strength"] = self._clip(mem["strength"] + 0.1, 0, 1)

            # 如果强度足够高，巩固到长期记忆
            if mem["strength"] > 0.7:
                from datetime import datetime
                ltm_mem = BrainMemory(
                    content=mem["content"],
                    timestamp=datetime.now().timestamp(),
                    weight=mem["strength"],
                    tag="hippocampus_replay",
                    modality="text",
                    features=[],
                )
                self._consolidate_to_ltm(ltm_mem)
                consolidated += 1

        self.hippocampus["replay_count"] += len(memories)

        # 激活海马体
        self.brain_regions["hippocampus"]["activity"] = self._clip(
            self.brain_regions["hippocampus"]["activity"] + 0.15, 0, 1)

        return {
            "replayed": len(memories),
            "consolidated": consolidated,
            "total_replays": self.hippocampus["replay_count"],
        }

    def hippocampus_pattern_completion(self, cue: str) -> Optional[Dict]:
        """海马体模式补全：根据部分线索回忆完整记忆。

        这是海马体的重要功能：即使只有部分线索，
        也能补全完整的记忆。
        """
        if not self.hippocampus["episodic_memory"]:
            return None

        # 找最相关的记忆
        best_match = None
        best_score = 0

        for mem in self.hippocampus["episodic_memory"]:
            # 简单的相似度计算
            score = 0
            if cue in mem["content"]:
                score += 1.0
            if mem["context"] and cue in mem["context"]:
                score += 0.5
            score *= mem["strength"]

            if score > best_score:
                best_score = score
                best_match = mem

        if best_match and best_score > 0.3:
            # 激活海马体
            self.brain_regions["hippocampus"]["activity"] = self._clip(
                self.brain_regions["hippocampus"]["activity"] + 0.1, 0, 1)

            return {
                "memory": best_match,
                "completion_score": best_score,
                "pattern_completion": self.hippocampus["pattern_completion"],
            }

        return None

    # ===== 前额叶皮层 =====

    def prefrontal_make_plan(self, goal: str, steps: Optional[List[str]] = None) -> Dict:
        """前额叶制定计划。

        前额叶负责制定计划、目标管理、执行控制。
        """
        plan = {
            "goal": goal,
            "steps": steps or [],
            "current_step": 0,
            "created_tick": self.tick,
            "priority": 0.5,
            "status": "active",  # active / completed / abandoned
        }

        self.prefrontal["plans"].append(plan)
        if len(self.prefrontal["plans"]) > 20:
            self.prefrontal["plans"].pop(0)

        # 激活前额叶
        self.brain_regions["prefrontal"]["activity"] = self._clip(
            self.brain_regions["prefrontal"]["activity"] + 0.15, 0, 1)

        return plan

    def prefrontal_add_goal(self, goal: str, priority: float = 0.5) -> Dict:
        """前额叶添加目标。"""
        goal_obj = {
            "goal": goal,
            "priority": self._clip(priority, 0, 1),
            "created_tick": self.tick,
            "progress": 0.0,
        }

        self.prefrontal["goals"].append(goal_obj)
        self.prefrontal["goals"].sort(key=lambda g: g["priority"], reverse=True)

        if len(self.prefrontal["goals"]) > 10:
            self.prefrontal["goals"].pop()

        # 激活前额叶
        self.brain_regions["prefrontal"]["activity"] = self._clip(
            self.brain_regions["prefrontal"]["activity"] + 0.1, 0, 1)

        return goal_obj

    def prefrontal_inhibit(self, strength: float = 0.3) -> Dict:
        """前额叶抑制控制：抑制不适当的反应。

        这是执行功能的重要组成部分：
        能够抑制冲动，延迟满足。
        """
        # 提高抑制控制能力
        self.prefrontal["inhibition"] = self._clip(
            self.prefrontal["inhibition"] + strength * 0.1, 0, 1)

        # 抑制杏仁核活动（情绪调节）
        self.brain_regions["amygdala"]["activity"] = self._clip(
            self.brain_regions["amygdala"]["activity"] - strength * 0.2, 0, 1)

        # 降低压力
        self.emotion["stress"] = self._clip(
            self.emotion["stress"] - strength * 0.1, 0, 1)

        # 激活前额叶
        self.brain_regions["prefrontal"]["activity"] = self._clip(
            self.brain_regions["prefrontal"]["activity"] + 0.1, 0, 1)

        return {
            "inhibition_strength": self.prefrontal["inhibition"],
            "amygdala_suppressed": strength * 0.2,
            "stress_reduced": strength * 0.1,
        }

    # ===== 杏仁核 =====

    def amygdala_detect_threat(self, stimulus: str) -> Dict:
        """杏仁核威胁检测：快速检测潜在威胁。

        杏仁核负责快速检测威胁，
        触发恐惧和应激反应。
        """
        # 检查是否有恐惧条件反射
        threat_level = 0.0
        if stimulus in self.amygdala_state["fear_conditioning"]:
            threat_level = self.amygdala_state["fear_conditioning"][stimulus]

        # 基于情绪的威胁检测
        if self.emotion["stress"] > 0.5:
            threat_level += 0.2  # 压力大时更容易感知威胁

        threat_level = self._clip(threat_level, 0, 1)

        # 激活杏仁核
        self.brain_regions["amygdala"]["activity"] = self._clip(
            self.brain_regions["amygdala"]["activity"] + threat_level * 0.3, 0, 1)

        # 如果威胁足够大，触发应激反应
        if threat_level > 0.5:
            self.emotion["stress"] = self._clip(
                self.emotion["stress"] + 0.2, 0, 1)
            self.attention_factor = self._clip(
                self.attention_factor + 0.1, 0.1, 1.0)

        return {
            "stimulus": stimulus,
            "threat_level": round(threat_level, 3),
            "is_threat": threat_level > 0.5,
            "amygdala_activity": self.brain_regions["amygdala"]["activity"],
        }

    def amygdala_fear_conditioning(self, stimulus: str, fear_level: float = 0.8):
        """杏仁核恐惧条件反射：建立刺激与恐惧的关联。

        这是经典条件反射的神经基础：
        中性刺激 + 厌恶刺激 → 恐惧反应。
        """
        self.amygdala_state["fear_conditioning"][stimulus] = self._clip(
            fear_level, 0, 1)

        # 激活杏仁核
        self.brain_regions["amygdala"]["activity"] = self._clip(
            self.brain_regions["amygdala"]["activity"] + 0.2, 0, 1)

        # 存储情绪记忆
        emotional_memory = {
            "stimulus": stimulus,
            "emotion": "fear",
            "intensity": fear_level,
            "tick": self.tick,
        }
        self.amygdala_state["emotional_memory"].append(emotional_memory)
        if len(self.amygdala_state["emotional_memory"]) > 50:
            self.amygdala_state["emotional_memory"].pop(0)

    def amygdala_extinguish_fear(self, stimulus: str):
        """杏仁核恐惧消退：消除恐惧条件反射。

        反复暴露于无害刺激下，恐惧反应会逐渐减弱。
        """
        if stimulus in self.amygdala_state["fear_conditioning"]:
            current = self.amygdala_state["fear_conditioning"][stimulus]
            self.amygdala_state["fear_conditioning"][stimulus] = self._clip(
                current - 0.2, 0, 1)

            # 如果完全消退，移除
            if self.amygdala_state["fear_conditioning"][stimulus] < 0.05:
                del self.amygdala_state["fear_conditioning"][stimulus]

        # 前额叶参与消退（抑制控制）
        self.prefrontal_inhibit(0.2)

    # ===== 脑区交互 =====

    def region_interact(self, from_region: str, to_region: str,
                        signal: float = 0.1) -> Dict:
        """脑区间交互：一个脑区向另一个脑区发送信号。

        脑区间通过连接进行信息传递，
        连接强度影响信号传递效率。
        """
        # 检查连接是否存在
        key = (from_region, to_region)
        if key not in self.region_connections:
            return {"success": False, "reason": "No connection"}

        # 计算信号强度
        connection_strength = self.region_connections[key]
        from_activity = self.brain_regions[from_region]["activity"]
        signal_strength = from_activity * connection_strength * signal

        # 激活目标脑区
        self.brain_regions[to_region]["activity"] = self._clip(
            self.brain_regions[to_region]["activity"] + signal_strength, 0, 1)

        return {
            "success": True,
            "from": from_region,
            "to": to_region,
            "connection_strength": connection_strength,
            "signal_strength": round(signal_strength, 3),
            "target_activity": self.brain_regions[to_region]["activity"],
        }

    def strengthen_connection(self, from_region: str, to_region: str,
                              amount: float = 0.05):
        """增强脑区间连接（赫布型可塑性）。

        同时激活的脑区之间的连接会增强。
        """
        key = (from_region, to_region)
        if key in self.region_connections:
            self.region_connections[key] = self._clip(
                self.region_connections[key] + amount, 0, 1)

    # ===== 脑区报告 =====

    def get_brain_regions_report(self) -> Dict:
        """获取脑区分化系统状态报告。"""
        # 更新脑区活动
        self.update_regions()

        regions = {}
        for key, data in self.brain_regions.items():
            regions[key] = {
                "name": data["name"],
                "function": data["function"],
                "activity": round(data["activity"], 3),
                "size": data["size"],
            }

        # 计算总活动量
        total_activity = sum(r["activity"] for r in self.brain_regions.values())
        total_size = sum(r["size"] for r in self.brain_regions.values())

        return {
            "regions": regions,
            "most_active": self.get_most_active_region(),
            "total_activity": round(total_activity, 3),
            "total_size": total_size,
            "hippocampus": {
                "episodic_memory_count": len(self.hippocampus["episodic_memory"]),
                "replay_count": self.hippocampus["replay_count"],
                "pattern_separation": round(self.hippocampus["pattern_separation"], 3),
                "pattern_completion": round(self.hippocampus["pattern_completion"], 3),
            },
            "prefrontal": {
                "plans_count": len(self.prefrontal["plans"]),
                "goals_count": len(self.prefrontal["goals"]),
                "inhibition": round(self.prefrontal["inhibition"], 3),
                "cognitive_control": round(self.prefrontal["cognitive_control"], 3),
            },
            "amygdala": {
                "fear_conditioning_count": len(self.amygdala_state["fear_conditioning"]),
                "emotional_memory_count": len(self.amygdala_state["emotional_memory"]),
                "threat_detection": round(self.amygdala_state["threat_detection"], 3),
                "emotional_intensity": round(self.amygdala_state["emotional_intensity"], 3),
            },
            "connections": {
                f"{k[0]}→{k[1]}": round(v, 3)
                for k, v in sorted(self.region_connections.items(), key=lambda x: -x[1])
            },
        }

    # ------------------ 推理与规划系统（v6.5 Reasoning & Planning） ------------------

    # ===== 逻辑推理 =====

    def deductive_reasoning(self, premises: List[str]) -> Dict:
        """演绎推理：从一般到特殊。

        给定前提，推导出必然的结论。
        例如：所有人都会死，苏格拉底是人 → 苏格拉底会死
        """
        # 激活前额叶
        self.brain_regions["prefrontal"]["activity"] = self._clip(
            self.brain_regions["prefrontal"]["activity"] + 0.15, 0, 1)

        self.reasoning["active"] = True
        self.reasoning["depth"] = 1
        self.reasoning["reasoning_count"] += 1

        # 简单的演绎推理：基于规则匹配
        conclusions = []
        confidence = 0.0

        # 从记忆中检索相关规则
        relevant_rules = []
        for premise in premises:
            for mem in self.long_memory:
                if premise in mem.content or "如果" in mem.content:
                    relevant_rules.append(mem.content)

        # 简单的规则应用（如果A那么B，A，所以B）
        for rule in relevant_rules:
            if "如果" in rule and "那么" in rule:
                # 提取条件和结论
                parts = rule.split("那么")
                if len(parts) == 2:
                    condition = parts[0].replace("如果", "").strip()
                    conclusion = parts[1].strip()

                    # 检查前提是否满足条件
                    for premise in premises:
                        if premise in condition or condition in premise:
                            conclusions.append(conclusion)
                            confidence = max(confidence, 0.7)
                            break

        # 如果没有找到规则，生成简单结论
        if not conclusions and premises:
            conclusions.append(f"基于前提的初步结论")
            confidence = 0.3

        # 记录演绎结果
        deduction = {
            "premises": premises,
            "conclusions": conclusions,
            "confidence": confidence,
            "tick": self.tick,
        }
        self.logic["deductions"].append(deduction)
        if len(self.logic["deductions"]) > 50:
            self.logic["deductions"].pop(0)

        return {
            "type": "deductive",
            "premises": premises,
            "conclusions": conclusions,
            "confidence": round(confidence, 3),
            "depth": self.reasoning["depth"],
            "rules_applied": len(conclusions),
        }

    def inductive_reasoning(self, observations: List[str]) -> Dict:
        """归纳推理：从特殊到一般。

        从多个具体观察中总结出一般规律。
        例如：看到的天鹅都是白色的 → 所有天鹅都是白色的
        """
        # 激活前额叶和联合皮层
        self.brain_regions["prefrontal"]["activity"] = self._clip(
            self.brain_regions["prefrontal"]["activity"] + 0.1, 0, 1)
        self.brain_regions["association_cortex"]["activity"] = self._clip(
            self.brain_regions["association_cortex"]["activity"] + 0.1, 0, 1)

        self.reasoning["active"] = True
        self.reasoning["depth"] = 2
        self.reasoning["reasoning_count"] += 1

        # 简单的归纳推理：寻找共同模式
        generalizations = []
        confidence = 0.0

        if observations:
            # 计算观察的一致性
            common_features = set(observations[0].split())
            for obs in observations[1:]:
                common_features &= set(obs.split())

            # 基于共同特征生成归纳
            if common_features:
                generalization = f"所有观察都具有: {' '.join(list(common_features)[:5])}"
                generalizations.append(generalization)
                confidence = len(common_features) / max(len(observations[0].split()), 1)

            # 样本量影响置信度
            sample_factor = min(len(observations) / 10, 1.0)
            confidence *= (0.5 + 0.5 * sample_factor)

        # 记录归纳结果
        induction = {
            "observations": observations,
            "generalizations": generalizations,
            "confidence": confidence,
            "sample_size": len(observations),
            "tick": self.tick,
        }
        self.logic["inductions"].append(induction)
        if len(self.logic["inductions"]) > 50:
            self.logic["inductions"].pop(0)

        return {
            "type": "inductive",
            "observations": observations,
            "generalizations": generalizations,
            "confidence": round(confidence, 3),
            "sample_size": len(observations),
            "depth": self.reasoning["depth"],
        }

    def abductive_reasoning(self, observation: str,
                            hypotheses: Optional[List[str]] = None) -> Dict:
        """溯因推理：从结果到原因。

        给定观察结果，找出最可能的解释。
        例如：草地湿了 → 可能下雨了
        """
        # 激活前额叶和海马体
        self.brain_regions["prefrontal"]["activity"] = self._clip(
            self.brain_regions["prefrontal"]["activity"] + 0.1, 0, 1)
        self.brain_regions["hippocampus"]["activity"] = self._clip(
            self.brain_regions["hippocampus"]["activity"] + 0.1, 0, 1)

        self.reasoning["active"] = True
        self.reasoning["depth"] = 3
        self.reasoning["reasoning_count"] += 1

        # 如果没有提供假设，从记忆中生成
        if hypotheses is None:
            hypotheses = []
            for mem in self.long_memory:
                if "因为" in mem.content or "导致" in mem.content:
                    if observation in mem.content:
                        hypotheses.append(mem.content)

        # 评估每个假设的可能性
        best_hypothesis = None
        best_score = 0.0
        scored_hypotheses = []

        for hyp in hypotheses:
            # 简单的评分：基于记忆中的关联强度
            score = 0.3  # 基础概率

            # 检查记忆中是否有相关的因果关系
            for causal in self.causal["causal_relations"]:
                if hyp in causal.get("cause", "") and observation in causal.get("effect", ""):
                    score = max(score, causal.get("strength", 0.5))

            scored_hypotheses.append({
                "hypothesis": hyp,
                "score": round(score, 3),
            })

            if score > best_score:
                best_score = score
                best_hypothesis = hyp

        # 按分数排序
        scored_hypotheses.sort(key=lambda x: x["score"], reverse=True)

        # 记录溯因结果
        abduction = {
            "observation": observation,
            "best_hypothesis": best_hypothesis,
            "confidence": best_score,
            "hypotheses_count": len(hypotheses),
            "tick": self.tick,
        }
        self.logic["abductions"].append(abduction)
        if len(self.logic["abductions"]) > 50:
            self.logic["abductions"].pop(0)

        return {
            "type": "abductive",
            "observation": observation,
            "best_hypothesis": best_hypothesis,
            "confidence": round(best_score, 3),
            "all_hypotheses": scored_hypotheses[:5],
            "depth": self.reasoning["depth"],
        }

    # ===== 因果推理 =====

    def discover_causal_relation(self, cause: str, effect: str,
                                 strength: float = 0.5) -> Dict:
        """发现因果关系：建立原因和结果之间的联系。

        这是因果推理的基础：识别什么导致了什么。
        """
        # 激活前额叶和海马体
        self.brain_regions["prefrontal"]["activity"] = self._clip(
            self.brain_regions["prefrontal"]["activity"] + 0.1, 0, 1)
        self.brain_regions["hippocampus"]["activity"] = self._clip(
            self.brain_regions["hippocampus"]["activity"] + 0.1, 0, 1)

        # 检查是否已存在
        existing = None
        for rel in self.causal["causal_relations"]:
            if rel["cause"] == cause and rel["effect"] == effect:
                existing = rel
                break

        if existing:
            # 加强已有关系
            existing["strength"] = self._clip(
                existing["strength"] + strength * 0.1, 0, 1)
            existing["evidence_count"] += 1
            relation = existing
        else:
            # 创建新关系
            relation = {
                "cause": cause,
                "effect": effect,
                "strength": strength,
                "evidence_count": 1,
                "first_observed": self.tick,
                "last_observed": self.tick,
            }
            self.causal["causal_relations"].append(relation)

        # 记录因果链
        self.causal["causal_chains"].append({
            "cause": cause,
            "effect": effect,
            "tick": self.tick,
        })
        if len(self.causal["causal_chains"]) > 100:
            self.causal["causal_chains"].pop(0)

        return {
            "cause": cause,
            "effect": effect,
            "strength": round(relation["strength"], 3),
            "evidence_count": relation["evidence_count"],
            "is_new": existing is None,
        }

    def causal_attribution(self, event: str) -> Dict:
        """因果归因：解释事件的原因。

        给定一个事件，找出最可能的原因。
        """
        # 激活前额叶和杏仁核（归因与情绪相关）
        self.brain_regions["prefrontal"]["activity"] = self._clip(
            self.brain_regions["prefrontal"]["activity"] + 0.1, 0, 1)
        self.brain_regions["amygdala"]["activity"] = self._clip(
            self.brain_regions["amygdala"]["activity"] + 0.05, 0, 1)

        # 查找可能的原因
        possible_causes = []
        for rel in self.causal["causal_relations"]:
            if rel["effect"] == event or event in rel["effect"]:
                possible_causes.append(rel)

        # 按强度排序
        possible_causes.sort(key=lambda r: r["strength"], reverse=True)

        # 记录归因结果
        attribution = {
            "event": event,
            "causes": [c["cause"] for c in possible_causes[:3]],
            "confidence": possible_causes[0]["strength"] if possible_causes else 0.0,
            "tick": self.tick,
        }
        self.causal["attributions"].append(attribution)
        if len(self.causal["attributions"]) > 50:
            self.causal["attributions"].pop(0)

        return {
            "event": event,
            "possible_causes": [
                {
                    "cause": c["cause"],
                    "strength": round(c["strength"], 3),
                    "evidence": c["evidence_count"],
                }
                for c in possible_causes[:5]
            ],
            "most_likely": possible_causes[0]["cause"] if possible_causes else None,
            "confidence": round(possible_causes[0]["strength"], 3) if possible_causes else 0.0,
        }

    def counterfactual_reasoning(self, event: str,
                                 what_if: str) -> Dict:
        """反事实推理：如果...会怎样？

        想象与事实相反的情况，推断可能的结果。
        """
        # 激活前额叶和海马体（心理模拟）
        self.brain_regions["prefrontal"]["activity"] = self._clip(
            self.brain_regions["prefrontal"]["activity"] + 0.15, 0, 1)
        self.brain_regions["hippocampus"]["activity"] = self._clip(
            self.brain_regions["hippocampus"]["activity"] + 0.1, 0, 1)

        self.reasoning["active"] = True
        self.reasoning["depth"] = 4

        # 简单的反事实推理：基于因果关系推断
        predicted_outcome = "不确定"
        confidence = 0.3

        # 查找相关因果关系
        for rel in self.causal["causal_relations"]:
            if what_if in rel["cause"]:
                predicted_outcome = rel["effect"]
                confidence = rel["strength"] * 0.7  # 反事实置信度稍低
                break

        # 记录反事实推理
        counterfactual = {
            "event": event,
            "what_if": what_if,
            "predicted_outcome": predicted_outcome,
            "confidence": confidence,
            "tick": self.tick,
        }
        self.causal["counterfactuals"].append(counterfactual)
        if len(self.causal["counterfactuals"]) > 50:
            self.causal["counterfactuals"].pop(0)

        return {
            "type": "counterfactual",
            "event": event,
            "what_if": what_if,
            "predicted_outcome": predicted_outcome,
            "confidence": round(confidence, 3),
            "depth": self.reasoning["depth"],
        }

    # ===== 目标规划 =====

    def plan_goal(self, goal: str,
                  current_state: Optional[str] = None) -> Dict:
        """目标规划：为实现目标制定计划。

        将目标分解为子目标和步骤。
        """
        # 激活前额叶
        self.brain_regions["prefrontal"]["activity"] = self._clip(
            self.brain_regions["prefrontal"]["activity"] + 0.2, 0, 1)

        self.planning["planning_count"] += 1

        # 简单的规划：生成子目标
        subgoals = []
        steps = []

        # 基于目标生成子目标（简化版）
        if "学习" in goal:
            subgoals = ["了解基础知识", "实践练习", "总结归纳"]
            steps = ["阅读资料", "做练习题", "复习总结"]
        elif "完成" in goal or "实现" in goal:
            subgoals = ["分析需求", "设计方案", "实现功能", "测试验证"]
            steps = ["需求分析", "方案设计", "编码实现", "测试调试"]
        elif "解决" in goal:
            subgoals = ["理解问题", "分析原因", "提出方案", "验证效果"]
            steps = ["问题定义", "原因分析", "方案设计", "实施验证"]
        else:
            subgoals = ["明确目标", "制定计划", "执行计划", "检查结果"]
            steps = ["目标设定", "规划步骤", "行动执行", "结果评估"]

        # 创建计划
        plan = {
            "goal": goal,
            "current_state": current_state or "初始状态",
            "subgoals": subgoals,
            "steps": steps,
            "current_step": 0,
            "status": "planning",  # planning / executing / completed / failed
            "created_tick": self.tick,
            "priority": 0.5,
        }

        self.planning["current_plan"] = plan
        self.planning["subgoals"] = subgoals

        # 记录计划历史
        self.planning["plan_history"].append(plan)
        if len(self.planning["plan_history"]) > 20:
            self.planning["plan_history"].pop(0)

        # 同时在前额叶中记录
        self.prefrontal_make_plan(goal, steps)

        return {
            "goal": goal,
            "subgoals": subgoals,
            "steps": steps,
            "estimated_complexity": len(steps),
            "status": "planning",
        }

    def advance_plan(self) -> Dict:
        """推进计划：执行下一步。"""
        if not self.planning["current_plan"]:
            return {"success": False, "reason": "No active plan"}

        plan = self.planning["current_plan"]

        if plan["current_step"] >= len(plan["steps"]):
            plan["status"] = "completed"
            return {"success": True, "status": "completed", "message": "计划已完成"}

        # 执行下一步
        current_step = plan["steps"][plan["current_step"]]
        plan["current_step"] += 1

        # 检查是否完成
        if plan["current_step"] >= len(plan["steps"]):
            plan["status"] = "completed"

        # 激活前额叶
        self.brain_regions["prefrontal"]["activity"] = self._clip(
            self.brain_regions["prefrontal"]["activity"] + 0.05, 0, 1)

        return {
            "success": True,
            "current_step": current_step,
            "step_index": plan["current_step"],
            "total_steps": len(plan["steps"]),
            "progress": round(plan["current_step"] / len(plan["steps"]), 3),
            "status": plan["status"],
        }

    # ===== 问题解决 =====

    def solve_problem(self, problem: str,
                      context: Optional[str] = None) -> Dict:
        """问题解决：解决给定的问题。

        包括问题表征、策略选择、解决方案生成。
        """
        # 激活前额叶和联合皮层
        self.brain_regions["prefrontal"]["activity"] = self._clip(
            self.brain_regions["prefrontal"]["activity"] + 0.15, 0, 1)
        self.brain_regions["association_cortex"]["activity"] = self._clip(
            self.brain_regions["association_cortex"]["activity"] + 0.1, 0, 1)

        self.reasoning["active"] = True
        self.reasoning["depth"] = 5

        # 问题表征
        problem_representation = {
            "problem": problem,
            "context": context or "",
            "type": self._classify_problem(problem),
            "complexity": self._assess_complexity(problem),
        }

        # 策略选择
        strategy = self._select_strategy(problem_representation)

        # 生成解决方案
        solutions = self._generate_solutions(problem_representation, strategy)

        # 评估解决方案
        best_solution = None
        best_score = 0.0
        for sol in solutions:
            score = self._evaluate_solution(sol, problem_representation)
            sol["score"] = score
            if score > best_score:
                best_score = score
                best_solution = sol

        # 记录问题解决
        problem_record = {
            "problem": problem,
            "representation": problem_representation,
            "strategy": strategy,
            "solutions": solutions,
            "best_solution": best_solution,
            "confidence": best_score,
            "tick": self.tick,
        }
        self.problem_solving["current_problem"] = problem_record
        self.problem_solving["problem_history"].append(problem_record)
        if len(self.problem_solving["problem_history"]) > 20:
            self.problem_solving["problem_history"].pop(0)

        return {
            "problem": problem,
            "type": problem_representation["type"],
            "complexity": problem_representation["complexity"],
            "strategy": strategy,
            "solutions_count": len(solutions),
            "best_solution": best_solution,
            "confidence": round(best_score, 3),
            "depth": self.reasoning["depth"],
        }

    def _classify_problem(self, problem: str) -> str:
        """问题分类：判断问题类型。"""
        if "为什么" in problem or "原因" in problem:
            return "causal"  # 因果问题
        elif "怎么" in problem or "如何" in problem:
            return "procedural"  # 程序问题
        elif "什么" in problem or "是什么" in problem:
            return "factual"  # 事实问题
        elif "如果" in problem:
            return "hypothetical"  # 假设问题
        else:
            return "general"  # 一般问题

    def _assess_complexity(self, problem: str) -> float:
        """评估问题复杂度。"""
        complexity = 0.3  # 基础复杂度

        # 基于长度
        complexity += min(len(problem) / 100, 0.3)

        # 基于关键词
        complex_keywords = ["为什么", "如何", "怎么", "分析", "比较", "综合"]
        for kw in complex_keywords:
            if kw in problem:
                complexity += 0.1

        return self._clip(complexity, 0, 1)

    def _select_strategy(self, problem_repr: Dict) -> str:
        """选择解决策略。"""
        problem_type = problem_repr["type"]
        complexity = problem_repr["complexity"]

        strategies = {
            "causal": "溯因推理法",
            "procedural": "步骤分解法",
            "factual": "记忆检索法",
            "hypothetical": "反事实推理法",
            "general": "启发式方法",
        }

        return strategies.get(problem_type, "启发式方法")

    def _generate_solutions(self, problem_repr: Dict,
                            strategy: str) -> List[Dict]:
        """生成解决方案。"""
        solutions = []

        # 从记忆中检索相关解决方案
        problem = problem_repr["problem"]
        for mem in self.long_memory:
            if any(word in mem.content for word in problem.split()[:3]):
                solutions.append({
                    "solution": mem.content,
                    "source": "memory",
                    "relevance": mem.weight,
                })

        # 生成基于策略的解决方案
        if strategy == "步骤分解法":
            solutions.append({
                "solution": "将问题分解为小步骤，逐步解决",
                "source": "strategy",
                "relevance": 0.6,
            })
        elif strategy == "溯因推理法":
            solutions.append({
                "solution": "寻找可能的原因，验证每个假设",
                "source": "strategy",
                "relevance": 0.5,
            })
        else:
            solutions.append({
                "solution": "使用启发式方法尝试解决",
                "source": "strategy",
                "relevance": 0.4,
            })

        # 按相关性排序
        solutions.sort(key=lambda s: s["relevance"], reverse=True)

        return solutions[:5]

    def _evaluate_solution(self, solution: Dict,
                           problem_repr: Dict) -> float:
        """评估解决方案质量。"""
        score = solution.get("relevance", 0.5)

        # 复杂度调整
        if problem_repr["complexity"] > 0.7:
            score *= 0.8  # 复杂问题降低置信度

        return self._clip(score, 0, 1)

    # ===== 决策 =====

    def make_decision(self, options: List[str],
                      criteria: Optional[List[str]] = None) -> Dict:
        """决策：在多个选项中做出选择。

        基于多准则评估，做出最优决策。
        """
        # 激活前额叶和杏仁核（情绪影响决策）
        self.brain_regions["prefrontal"]["activity"] = self._clip(
            self.brain_regions["prefrontal"]["activity"] + 0.15, 0, 1)
        self.brain_regions["amygdala"]["activity"] = self._clip(
            self.brain_regions["amygdala"]["activity"] + 0.05, 0, 1)

        self.reasoning["active"] = True

        # 默认准则
        if criteria is None:
            criteria = ["效用", "风险", "成本"]

        # 评估每个选项
        scored_options = []
        for option in options:
            scores = {}
            total_score = 0.0

            for criterion in criteria:
                # 简单的评分（实际应该更复杂）
                score = 0.5  # 基础分

                # 基于记忆中的相关信息调整
                for mem in self.long_memory:
                    if option in mem.content:
                        score = max(score, mem.weight * 0.8)
                        break

                # 情绪影响：压力大时更保守
                if criterion == "风险" and self.emotion["stress"] > 0.5:
                    score *= 0.7

                scores[criterion] = round(score, 3)
                total_score += score

            # 平均分
            avg_score = total_score / len(criteria)

            scored_options.append({
                "option": option,
                "scores": scores,
                "total_score": round(avg_score, 3),
            })

        # 按总分排序
        scored_options.sort(key=lambda x: x["total_score"], reverse=True)

        # 最佳选项
        best_option = scored_options[0] if scored_options else None

        # 记录决策
        decision = {
            "options": options,
            "criteria": criteria,
            "best_option": best_option["option"] if best_option else None,
            "confidence": best_option["total_score"] if best_option else 0.0,
            "tick": self.tick,
        }
        self.decision_making["current_decision"] = decision
        self.decision_making["options"] = options
        self.decision_making["decision_history"].append(decision)
        if len(self.decision_making["decision_history"]) > 20:
            self.decision_making["decision_history"].pop(0)

        return {
            "options": options,
            "criteria": criteria,
            "best_option": best_option["option"] if best_option else None,
            "confidence": best_option["total_score"] if best_option else 0.0,
            "all_scores": scored_options,
            "decision_style": "理性决策" if self.emotion["stress"] < 0.3 else "情绪决策",
        }

    # ===== 推理报告 =====

    def get_reasoning_report(self) -> Dict:
        """获取推理与规划系统状态报告。"""
        return {
            "active": self.reasoning["active"],
            "current_depth": self.reasoning["depth"],
            "max_depth": self.reasoning["max_depth"],
            "total_reasoning_count": self.reasoning["reasoning_count"],

            "logic": {
                "rules_count": len(self.logic["rules"]),
                "facts_count": len(self.logic["facts"]),
                "deductions_count": len(self.logic["deductions"]),
                "inductions_count": len(self.logic["inductions"]),
                "abductions_count": len(self.logic["abductions"]),
            },

            "causal": {
                "relations_count": len(self.causal["causal_relations"]),
                "chains_count": len(self.causal["causal_chains"]),
                "attributions_count": len(self.causal["attributions"]),
                "counterfactuals_count": len(self.causal["counterfactuals"]),
            },

            "planning": {
                "has_active_plan": self.planning["current_plan"] is not None,
                "plans_count": len(self.planning["plan_history"]),
                "subgoals_count": len(self.planning["subgoals"]),
                "planning_count": self.planning["planning_count"],
            },

            "problem_solving": {
                "has_current_problem": self.problem_solving["current_problem"] is not None,
                "problems_solved": len(self.problem_solving["problem_history"]),
                "strategies_count": len(self.problem_solving["strategies"]),
                "solutions_count": len(self.problem_solving["solutions"]),
            },

            "decision_making": {
                "has_current_decision": self.decision_making["current_decision"] is not None,
                "decisions_count": len(self.decision_making["decision_history"]),
                "options_count": len(self.decision_making["options"]),
            },

            "parameters": {
                "learning_rate": self.reasoning_learning_rate,
                "logic_confidence_threshold": self.logic_confidence_threshold,
                "causal_strength_threshold": self.causal_strength_threshold,
            },
        }

    # ------------------ 心理模拟系统（v6.6 Mental Simulation） ------------------

    # ===== 心理表象 =====

    def create_mental_image(self, object_name: str,
                            vividness: Optional[float] = None) -> Dict:
        """创建心理表象：在心中形成事物的形象。

        心理表象是心理模拟的基础，
        可以在没有外部输入的情况下"看到"事物。
        """
        # 激活视觉皮层和前额叶
        self.brain_regions["sensory_cortex"]["activity"] = self._clip(
            self.brain_regions["sensory_cortex"]["activity"] + 0.1, 0, 1)
        self.brain_regions["prefrontal"]["activity"] = self._clip(
            self.brain_regions["prefrontal"]["activity"] + 0.1, 0, 1)

        self.mental_imagery["active"] = True
        self.mental_imagery["imagery_count"] += 1

        # 计算表象生动度
        if vividness is None:
            vividness = self.imagery_vividness_base
            # 记忆中的相关信息会增加生动度
            for mem in self.long_memory:
                if object_name in mem.content:
                    vividness = max(vividness, mem.weight * 0.8)
                    break

        # 创建心理图像（简化版：用描述代替真实图像）
        mental_image = {
            "object": object_name,
            "vividness": vividness,
            "created_tick": self.tick,
            "properties": self._infer_properties(object_name),
            "rotation": 0,
            "scale": 1.0,
        }

        self.mental_imagery["current_image"] = mental_image

        # 推入思考空间
        self._push_thought(
            f"想象: {object_name}",
            source="imagery",
            activation=vividness
        )

        return {
            "object": object_name,
            "vividness": round(vividness, 3),
            "properties": mental_image["properties"],
            "imagery_count": self.mental_imagery["imagery_count"],
        }

    def _infer_properties(self, object_name: str) -> Dict:
        """推断物体的属性（基于记忆）。"""
        properties = {
            "shape": "未知",
            "color": "未知",
            "size": "中等",
        }

        # 从记忆中推断
        for mem in self.long_memory:
            if object_name in mem.content:
                # 简单的属性提取
                if "圆" in mem.content:
                    properties["shape"] = "圆形"
                if "红" in mem.content:
                    properties["color"] = "红色"
                if "大" in mem.content:
                    properties["size"] = "大"
                break

        return properties

    def mental_rotate(self, angle: float = 90.0) -> Dict:
        """心理旋转：在心中旋转物体。

        这是心理表象的经典实验：
        旋转角度越大，反应时间越长。
        """
        if not self.mental_imagery["current_image"]:
            return {"success": False, "reason": "No mental image"}

        self.mental_imagery["active"] = True
        self.mental_imagery["rotation_angle"] += angle

        # 更新当前图像
        self.mental_imagery["current_image"]["rotation"] += angle

        # 旋转需要时间（角度越大，认知负荷越大）
        rotation_cost = abs(angle) / 360.0
        self.central_executive_load = self._clip(
            self.central_executive_load + rotation_cost * 0.2, 0, 1)

        # 激活顶叶（空间处理）
        self.brain_regions["association_cortex"]["activity"] = self._clip(
            self.brain_regions["association_cortex"]["activity"] + 0.1, 0, 1)

        return {
            "success": True,
            "object": self.mental_imagery["current_image"]["object"],
            "rotation_angle": round(self.mental_imagery["rotation_angle"], 1),
            "cognitive_cost": round(rotation_cost, 3),
        }

    def clear_mental_image(self):
        """清除心理表象。"""
        self.mental_imagery["active"] = False
        self.mental_imagery["current_image"] = None
        self.mental_imagery["rotation_angle"] = 0

    # ===== 心理时间旅行 =====

    def remember_past(self, event: Optional[str] = None) -> Dict:
        """回忆过去：心理时间旅行到过去。

        这是情景记忆的重要功能：
        能够"重新体验"过去的事件。
        """
        # 激活海马体和前额叶
        self.brain_regions["hippocampus"]["activity"] = self._clip(
            self.brain_regions["hippocampus"]["activity"] + 0.15, 0, 1)
        self.brain_regions["prefrontal"]["activity"] = self._clip(
            self.brain_regions["prefrontal"]["activity"] + 0.1, 0, 1)

        self.mental_time_travel["direction"] = "past"
        self.mental_time_travel["past_remembered"] += 1

        # 查找相关记忆
        memories = []
        if event:
            # 搜索特定事件
            for mem in self.long_memory + self.short_memory:
                if event in mem.content:
                    memories.append(mem)
        else:
            # 随机回忆一些记忆
            if self.hippocampus["episodic_memory"]:
                import random
                memories = random.sample(
                    self.hippocampus["episodic_memory"],
                    min(3, len(self.hippocampus["episodic_memory"]))
                )

        # 计算回忆生动度
        vividness = 0.3
        if memories:
            vividness = max(
                mem.weight if hasattr(mem, 'weight') else mem.get("strength", 0.5)
                for mem in memories
            )

        # 推入思考空间
        if memories:
            first_mem = memories[0]
            content = first_mem.content if hasattr(first_mem, 'content') else first_mem.get('content', '')
            self._push_thought(
                f"回忆: {content}",
                source="memory",
                activation=vividness
            )

        return {
            "direction": "past",
            "event": event,
            "memories_found": len(memories),
            "vividness": round(vividness, 3),
            "total_remembered": self.mental_time_travel["past_remembered"],
        }

    def imagine_future(self, scenario: Optional[str] = None) -> Dict:
        """想象未来：心理时间旅行到未来。

        这是预测和规划的基础：
        能够"预先体验"未来的情景。
        """
        # 激活海马体和前额叶
        self.brain_regions["hippocampus"]["activity"] = self._clip(
            self.brain_regions["hippocampus"]["activity"] + 0.1, 0, 1)
        self.brain_regions["prefrontal"]["activity"] = self._clip(
            self.brain_regions["prefrontal"]["activity"] + 0.15, 0, 1)

        self.mental_time_travel["direction"] = "future"
        self.mental_time_travel["future_imagined"] += 1

        # 生成未来情景
        if scenario:
            future_scenario = {
                "scenario": scenario,
                "vividness": 0.4,
                "plausibility": 0.5,
                "emotional_valence": 0.5,
            }
        else:
            # 基于当前目标生成未来情景
            if self.prefrontal["goals"]:
                goal = self.prefrontal["goals"][0]
                future_scenario = {
                    "scenario": f"实现目标: {goal.get('goal', '未知')}",
                    "vividness": 0.5,
                    "plausibility": goal.get("priority", 0.5),
                    "emotional_valence": 0.7,
                }
            else:
                future_scenario = {
                    "scenario": "未来的某个时刻",
                    "vividness": 0.3,
                    "plausibility": 0.5,
                    "emotional_valence": 0.5,
                }

        # 情绪影响未来想象
        if self.emotion["pleasure"] > 0.5:
            future_scenario["emotional_valence"] += 0.1
        if self.emotion["stress"] > 0.5:
            future_scenario["plausibility"] -= 0.1

        # 推入思考空间
        self._push_thought(
            f"想象未来: {future_scenario['scenario']}",
            source="imagery",
            activation=future_scenario["vividness"]
        )

        return {
            "direction": "future",
            "scenario": future_scenario["scenario"],
            "vividness": round(future_scenario["vividness"], 3),
            "plausibility": round(future_scenario["plausibility"], 3),
            "emotional_valence": round(future_scenario["emotional_valence"], 3),
            "total_imagined": self.mental_time_travel["future_imagined"],
        }

    # ===== 心理模拟 =====

    def simulate_action(self, action: str) -> Dict:
        """模拟行动：在心中预演行动及其结果。

        这是运动认知的重要功能：
        在实际行动前先在心中模拟。
        """
        # 激活前额叶和运动相关区域（用联合皮层代替）
        self.brain_regions["prefrontal"]["activity"] = self._clip(
            self.brain_regions["prefrontal"]["activity"] + 0.15, 0, 1)
        self.brain_regions["association_cortex"]["activity"] = self._clip(
            self.brain_regions["association_cortex"]["activity"] + 0.1, 0, 1)

        self.mental_simulation["active"] = True
        self.mental_simulation["simulation_type"] = "action"
        self.mental_simulation["simulation_count"] += 1
        self.mental_simulation["simulation_depth"] = 1

        # 预测行动结果
        predicted_outcome = self._predict_action_outcome(action)

        # 模拟步骤
        simulation_steps = [
            f"准备执行: {action}",
            f"执行行动: {action}",
            f"预期结果: {predicted_outcome.get('certainty', 0.5):.2f} 确定性",
        ]

        simulation = {
            "type": "action",
            "action": action,
            "predicted_outcome": predicted_outcome,
            "steps": simulation_steps,
            "depth": 1,
            "confidence": 0.6,
        }

        self.mental_simulation["current_simulation"] = simulation

        return {
            "type": "action",
            "action": action,
            "predicted_outcome": predicted_outcome,
            "simulation_depth": 1,
            "confidence": 0.6,
            "total_simulations": self.mental_simulation["simulation_count"],
        }

    def simulate_dialogue(self, partner: str,
                          topic: str) -> Dict:
        """模拟对话：在心中预演与他人的对话。

        这是社会认知的重要功能：
        在实际对话前先在心中模拟。
        """
        # 激活前额叶和心智理论相关区域
        self.brain_regions["prefrontal"]["activity"] = self._clip(
            self.brain_regions["prefrontal"]["activity"] + 0.15, 0, 1)
        self.brain_regions["association_cortex"]["activity"] = self._clip(
            self.brain_regions["association_cortex"]["activity"] + 0.1, 0, 1)

        self.mental_simulation["active"] = True
        self.mental_simulation["simulation_type"] = "dialogue"
        self.mental_simulation["simulation_count"] += 1
        self.mental_simulation["simulation_depth"] = 2

        # 生成模拟对话
        dialogue = [
            {"speaker": "自己", "content": f"关于{topic}，我觉得..."},
            {"speaker": partner, "content": f"嗯，关于{topic}，我认为..."},
            {"speaker": "自己", "content": "你说得有道理，但是..."},
        ]

        simulation = {
            "type": "dialogue",
            "partner": partner,
            "topic": topic,
            "dialogue": dialogue,
            "depth": 2,
            "confidence": 0.5,
        }

        self.mental_simulation["current_simulation"] = simulation

        return {
            "type": "dialogue",
            "partner": partner,
            "topic": topic,
            "turns": len(dialogue),
            "simulation_depth": 2,
            "confidence": 0.5,
            "total_simulations": self.mental_simulation["simulation_count"],
        }

    def simulate_problem_solving(self, problem: str) -> Dict:
        """模拟问题解决：在心中预演问题解决过程。

        这是问题解决的重要功能：
        在实际解决前先在心中模拟。
        """
        # 激活前额叶和联合皮层
        self.brain_regions["prefrontal"]["activity"] = self._clip(
            self.brain_regions["prefrontal"]["activity"] + 0.2, 0, 1)
        self.brain_regions["association_cortex"]["activity"] = self._clip(
            self.brain_regions["association_cortex"]["activity"] + 0.15, 0, 1)

        self.mental_simulation["active"] = True
        self.mental_simulation["simulation_type"] = "problem"
        self.mental_simulation["simulation_count"] += 1
        self.mental_simulation["simulation_depth"] = 3

        # 生成解决步骤
        steps = [
            f"理解问题: {problem}",
            "分析问题原因",
            "提出解决方案",
            "评估方案效果",
        ]

        simulation = {
            "type": "problem",
            "problem": problem,
            "steps": steps,
            "depth": 3,
            "confidence": 0.4,
        }

        self.mental_simulation["current_simulation"] = simulation

        return {
            "type": "problem",
            "problem": problem,
            "steps": steps,
            "simulation_depth": 3,
            "confidence": 0.4,
            "total_simulations": self.mental_simulation["simulation_count"],
        }

    # ===== 创造力 =====

    def combine_concepts(self, concept1: str,
                         concept2: str) -> Dict:
        """概念组合：组合两个概念产生新想法。

        这是创造力的核心机制：
        将已有概念以新的方式组合起来。
        """
        # 激活联合皮层和前额叶
        self.brain_regions["association_cortex"]["activity"] = self._clip(
            self.brain_regions["association_cortex"]["activity"] + 0.15, 0, 1)
        self.brain_regions["prefrontal"]["activity"] = self._clip(
            self.brain_regions["prefrontal"]["activity"] + 0.1, 0, 1)

        # 生成组合概念
        combined = f"{concept1}+{concept2}"
        novelty = 0.5
        usefulness = 0.3

        # 检查是否已有类似组合
        for combo in self.creativity["combinations"]:
            if concept1 in combo["combined"] and concept2 in combo["combined"]:
                novelty = 0.2  # 已有类似组合，新颖度降低
                break

        # 计算创造力分数
        creativity_score = novelty * 0.6 + usefulness * 0.4

        # 记录组合
        combination = {
            "concept1": concept1,
            "concept2": concept2,
            "combined": combined,
            "novelty": novelty,
            "usefulness": usefulness,
            "creativity_score": creativity_score,
            "tick": self.tick,
        }
        self.creativity["combinations"].append(combination)
        if len(self.creativity["combinations"]) > 50:
            self.creativity["combinations"].pop(0)

        # 提高创造力水平
        self.creativity["creativity_level"] = self._clip(
            self.creativity["creativity_level"] + 0.02, 0, 1)

        # 推入思考空间
        self._push_thought(
            f"创意组合: {combined}",
            source="creativity",
            activation=creativity_score
        )

        return {
            "concept1": concept1,
            "concept2": concept2,
            "combined": combined,
            "novelty": round(novelty, 3),
            "usefulness": round(usefulness, 3),
            "creativity_score": round(creativity_score, 3),
        }

    def analogical_reasoning(self, source_domain: str,
                             target_domain: str) -> Dict:
        """类比推理：用一个领域的知识解决另一个领域的问题。

        这是创造力的重要机制：
        跨领域的知识迁移。
        """
        # 激活联合皮层和前额叶
        self.brain_regions["association_cortex"]["activity"] = self._clip(
            self.brain_regions["association_cortex"]["activity"] + 0.15, 0, 1)
        self.brain_regions["prefrontal"]["activity"] = self._clip(
            self.brain_regions["prefrontal"]["activity"] + 0.15, 0, 1)

        # 生成类比
        analogy = {
            "source": source_domain,
            "target": target_domain,
            "mapping": f"{source_domain} → {target_domain}",
            "depth": 0.5,  # 类比深度
            "plausibility": 0.4,  # 合理性
        }

        # 记录类比
        self.creativity["analogies"].append(analogy)
        if len(self.creativity["analogies"]) > 30:
            self.creativity["analogies"].pop(0)

        # 提高发散思维
        self.creativity["divergent_thinking"] = self._clip(
            self.creativity["divergent_thinking"] + 0.03, 0, 1)

        return {
            "source": source_domain,
            "target": target_domain,
            "mapping": analogy["mapping"],
            "depth": round(analogy["depth"], 3),
            "plausibility": round(analogy["plausibility"], 3),
        }

    def generate_insight(self, problem: str) -> Dict:
        """产生洞见：突然的灵感或领悟。

        这是创造力的高峰体验：
        "啊哈！"时刻。
        """
        # 激活联合皮层（洞见通常涉及远距离联想）
        self.brain_regions["association_cortex"]["activity"] = self._clip(
            self.brain_regions["association_cortex"]["activity"] + 0.2, 0, 1)

        # 洞见的产生有一定概率
        import random
        insight_probability = 0.3 + self.creativity["creativity_level"] * 0.3

        if random.random() < insight_probability:
            # 产生洞见
            insight = {
                "problem": problem,
                "insight": f"突然想到: 关于{problem}的新视角",
                "type": "sudden_insight",
                "confidence": 0.6,
                "tick": self.tick,
            }

            # 记录洞见
            self.creativity["insights"].append(insight)
            if len(self.creativity["insights"]) > 20:
                self.creativity["insights"].pop(0)

            # 增加愉悦感（洞见带来的快感）
            self.emotion["pleasure"] = self._clip(
                self.emotion["pleasure"] + 0.1, 0, 1)

            # 推入思考空间
            self._push_thought(
                f"💡 洞见: {insight['insight']}",
                source="insight",
                activation=0.9
            )

            return {
                "has_insight": True,
                "problem": problem,
                "insight": insight["insight"],
                "confidence": 0.6,
                "insight_count": len(self.creativity["insights"]),
            }
        else:
            return {
                "has_insight": False,
                "problem": problem,
                "message": "还没有产生洞见，继续思考...",
                "probability": round(insight_probability, 3),
            }

    def divergent_thinking(self, topic: str,
                           num_ideas: int = 5) -> Dict:
        """发散思维：从一个主题产生多个想法。

        这是创造力的重要组成：
        能够产生多种不同的解决方案。
        """
        # 激活联合皮层
        self.brain_regions["association_cortex"]["activity"] = self._clip(
            self.brain_regions["association_cortex"]["activity"] + 0.15, 0, 1)

        # 生成多个想法
        ideas = []
        for i in range(num_ideas):
            idea = {
                "idea": f"{topic}的想法{i+1}",
                "originality": 0.3 + i * 0.1,  # 后面的想法更新颖
                "feasibility": 0.7 - i * 0.1,  # 但可行性降低
            }
            ideas.append(idea)

        # 计算发散思维分数
        fluency = len(ideas)  # 流畅性：想法数量
        flexibility = len(set([i["idea"][:3] for i in ideas]))  # 灵活性
        originality = sum(i["originality"] for i in ideas) / len(ideas)  # 独创性

        divergent_score = (fluency + flexibility + originality) / 3

        # 记录创造性想法
        for idea in ideas:
            self.creativity["creative_thoughts"].append({
                "thought": idea["idea"],
                "type": "divergent",
                "topic": topic,
                "tick": self.tick,
            })
        if len(self.creativity["creative_thoughts"]) > 100:
            self.creativity["creative_thoughts"] = self.creativity["creative_thoughts"][-100:]

        return {
            "topic": topic,
            "num_ideas": num_ideas,
            "fluency": fluency,
            "flexibility": flexibility,
            "originality": round(originality, 3),
            "divergent_score": round(divergent_score, 3),
            "ideas": ideas,
        }

    # ===== 默认模式网络 =====

    def activate_dmn(self):
        """激活默认模式网络（DMN）。

        DMN在休息时激活，负责：
        - 心智游移
        - 自我反思
        - 心理时间旅行
        - 社会认知
        """
        self.default_mode_network["active"] = True
        self.default_mode_network["activity_level"] = self._clip(
            self.default_mode_network["activity_level"] + 0.2, 0, 1)
        self.default_mode_network["mind_wandering"] = True

        # 降低前额叶活动（执行控制降低）
        self.brain_regions["prefrontal"]["activity"] = self._clip(
            self.brain_regions["prefrontal"]["activity"] - 0.1, 0, 1)

        # 增加联合皮层活动
        self.brain_regions["association_cortex"]["activity"] = self._clip(
            self.brain_regions["association_cortex"]["activity"] + 0.1, 0, 1)

    def deactivate_dmn(self):
        """去激活默认模式网络。"""
        self.default_mode_network["active"] = False
        self.default_mode_network["mind_wandering"] = False
        self.default_mode_network["activity_level"] = self._clip(
            self.default_mode_network["activity_level"] - 0.3, 0, 1)

        # 恢复前额叶活动
        self.brain_regions["prefrontal"]["activity"] = self._clip(
            self.brain_regions["prefrontal"]["activity"] + 0.1, 0, 1)

    # ===== 心理模拟报告 =====

    def get_mental_simulation_report(self) -> Dict:
        """获取心理模拟系统状态报告。"""
        return {
            "mental_imagery": {
                "active": self.mental_imagery["active"],
                "vividness": round(self.mental_imagery["vividness"], 3),
                "imagery_count": self.mental_imagery["imagery_count"],
                "has_current_image": self.mental_imagery["current_image"] is not None,
                "rotation_angle": round(self.mental_imagery["rotation_angle"], 1),
            },

            "mental_time_travel": {
                "past_remembered": self.mental_time_travel["past_remembered"],
                "future_imagined": self.mental_time_travel["future_imagined"],
                "direction": self.mental_time_travel["direction"],
            },

            "mental_simulation": {
                "active": self.mental_simulation["active"],
                "type": self.mental_simulation["simulation_type"],
                "simulation_count": self.mental_simulation["simulation_count"],
                "simulation_depth": self.mental_simulation["simulation_depth"],
            },

            "creativity": {
                "creativity_level": round(self.creativity["creativity_level"], 3),
                "divergent_thinking": round(self.creativity["divergent_thinking"], 3),
                "convergent_thinking": round(self.creativity["convergent_thinking"], 3),
                "combinations_count": len(self.creativity["combinations"]),
                "analogies_count": len(self.creativity["analogies"]),
                "insights_count": len(self.creativity["insights"]),
                "creative_thoughts_count": len(self.creativity["creative_thoughts"]),
            },

            "default_mode_network": {
                "active": self.default_mode_network["active"],
                "activity_level": round(self.default_mode_network["activity_level"], 3),
                "mind_wandering": self.default_mode_network["mind_wandering"],
                "spontaneous_thoughts": len(self.default_mode_network["spontaneous_thoughts"]),
            },

            "parameters": {
                "imagery_vividness_base": self.imagery_vividness_base,
                "simulation_max_depth": self.simulation_max_depth,
                "creativity_bonus": self.creativity_bonus,
                "dmn_activation_threshold": self.dmn_activation_threshold,
            },
        }

    # ------------------ 发育过程系统（v6.7 Development） ------------------

    # ===== 发育阶段 =====

    def develop(self, months: float = 1.0):
        """发育：让大脑发育一段时间。

        这是发育的主方法，推进发育年龄，
        可能触发阶段转换。
        """
        # 增加发育年龄
        self.development["age"] += months

        # 检查是否需要转换阶段
        old_stage = self.development["stage"]
        new_stage = self._get_stage_for_age(self.development["age"])

        if new_stage != old_stage:
            self._transition_to_stage(new_stage)

        # 神经发育
        self._neural_development(months)

        # 更新关键期
        self._update_critical_periods()

        # 更新可塑性
        self._update_plasticity()

        return {
            "new_age": round(self.development["age"], 1),
            "stage": self.development["stage"],
            "stage_name": self.developmental_stages[self.development["stage"]]["name"],
            "stage_changed": new_stage != old_stage,
            "plasticity_level": round(self.plasticity_level, 3),
        }

    def _get_stage_for_age(self, age: float) -> str:
        """根据年龄获取发育阶段。"""
        for stage, info in self.developmental_stages.items():
            start, end = info["age_range"]
            if start <= age < end:
                return stage
        return "adult"

    def _transition_to_stage(self, new_stage: str):
        """转换到新的发育阶段。"""
        old_stage = self.development["stage"]
        self.development["stage"] = new_stage

        # 记录里程碑
        milestone = {
            "from_stage": old_stage,
            "to_stage": new_stage,
            "age": self.development["age"],
            "tick": self.tick,
        }
        self.development["developmental_milestones"].append(milestone)
        self.development["stage_history"].append(new_stage)

        # 阶段转换带来的能力变化
        stage_info = self.developmental_stages[new_stage]

        # 推入思考空间
        self._push_thought(
            f"发育里程碑: 进入{stage_info['name']}",
            source="development",
            activation=0.9
        )

    def get_current_stage(self) -> Dict:
        """获取当前发育阶段信息。"""
        stage = self.development["stage"]
        info = self.developmental_stages[stage]
        return {
            "stage": stage,
            "name": info["name"],
            "age": round(self.development["age"], 1),
            "description": info["description"],
            "abilities": info["abilities"],
        }

    # ===== 皮亚杰认知发展阶段 =====

    def get_piaget_stage(self) -> Dict:
        """获取当前皮亚杰认知发展阶段。"""
        age = self.development["age"]

        for stage, info in self.piaget_stages.items():
            start, end = info["age_range"]
            if start <= age < end:
                return {
                    "stage": stage,
                    "name": info["name"],
                    "age": round(age, 1),
                    "description": info["description"],
                    "key_achievement": info["key_achievement"],
                }

        # 默认形式运算阶段
        info = self.piaget_stages["formal_operational"]
        return {
            "stage": "formal_operational",
            "name": info["name"],
            "age": round(age, 1),
            "description": info["description"],
            "key_achievement": info["key_achievement"],
        }

    def has_object_permanence(self) -> bool:
        """是否具有客体永久性（感知运动阶段的成就）。"""
        return self.development["age"] >= 9  # 约9个月

    def has_conservation(self) -> bool:
        """是否具有守恒概念（具体运算阶段的成就）。"""
        return self.development["age"] >= 84  # 约7岁

    def has_abstract_thinking(self) -> bool:
        """是否具有抽象思维（形式运算阶段的成就）。"""
        return self.development["age"] >= 132  # 约11岁

    # ===== 神经发育 =====

    def _neural_development(self, months: float):
        """神经发育：神经元和突触的发育。"""
        age = self.development["age"]

        # 神经发生：早期高，逐渐降低
        if age < 12:  # 1岁内
            self.neural_development["neurogenesis_rate"] = 0.8 - age * 0.05
        else:
            self.neural_development["neurogenesis_rate"] = max(0.05, 0.2 - (age - 12) * 0.005)

        # 突触发生：先增加后减少
        if age < 24:  # 2岁内快速增加
            self.neural_development["synaptogenesis_rate"] = 0.6 + age * 0.02
            self.neural_development["synaptic_density"] = self._clip(
                0.5 + age * 0.02, 0, 1)
        else:
            self.neural_development["synaptogenesis_rate"] = max(0.1, 0.8 - (age - 24) * 0.005)

        # 髓鞘化：持续增加
        self.neural_development["myelination_level"] = self._clip(
            self.neural_development["myelination_level"] + months * 0.01, 0, 1)

        # 突触修剪：童年期开始
        if age > 24:  # 2岁后开始修剪
            self.neural_development["pruning_rate"] = min(0.3, (age - 24) * 0.005)
            # 修剪使突触密度略有下降，但保留的连接更强
            self.neural_development["synaptic_density"] = self._clip(
                self.neural_development["synaptic_density"] - months * 0.002, 0.3, 1)

        # 神经复杂度：持续增加
        self.neural_development["neural_complexity"] = self._clip(
            self.neural_development["neural_complexity"] + months * 0.005, 0, 1)

    def get_neural_development_status(self) -> Dict:
        """获取神经发育状态。"""
        return {
            "neurogenesis_rate": round(self.neural_development["neurogenesis_rate"], 3),
            "synaptogenesis_rate": round(self.neural_development["synaptogenesis_rate"], 3),
            "myelination_level": round(self.neural_development["myelination_level"], 3),
            "pruning_rate": round(self.neural_development["pruning_rate"], 3),
            "synaptic_density": round(self.neural_development["synaptic_density"], 3),
            "neural_complexity": round(self.neural_development["neural_complexity"], 3),
        }

    # ===== 关键期 =====

    def _update_critical_periods(self):
        """更新关键期状态。"""
        age = self.development["age"]

        for key, period in self.critical_periods.items():
            if age < period["start_age"]:
                # 还没开始
                period["active"] = False
                period["sensitivity"] = 0.0
            elif age >= period["end_age"]:
                # 已经结束
                period["active"] = False
                period["sensitivity"] = 0.1  # 仍有少量可塑性
            else:
                # 关键期内
                period["active"] = True
                # 计算敏感度（峰值最高，两端逐渐降低）
                peak = period["peak_age"]
                if age <= peak:
                    # 上升期
                    period["sensitivity"] = age / peak
                else:
                    # 下降期
                    period["sensitivity"] = 1 - (age - peak) / (period["end_age"] - peak)

    def get_critical_periods_status(self) -> Dict:
        """获取关键期状态。"""
        status = {}
        for key, period in self.critical_periods.items():
            status[key] = {
                "name": period["name"],
                "active": period["active"],
                "sensitivity": round(period["sensitivity"], 3),
                "start_age": period["start_age"],
                "end_age": period["end_age"],
                "peak_age": period["peak_age"],
            }
        return status

    def is_in_critical_period(self, domain: str) -> bool:
        """是否在某个领域的关键期内。"""
        if domain in self.critical_periods:
            return self.critical_periods[domain]["active"]
        return False

    # ===== 可塑性 =====

    def _update_plasticity(self):
        """更新可塑性水平。

        可塑性随年龄下降，但终身保持一定水平。
        """
        age = self.development["age"]

        # 可塑性随年龄下降
        if age < 24:  # 2岁内最高
            self.plasticity_level = 1.0
        elif age < 84:  # 7岁前较高
            self.plasticity_level = 0.9 - (age - 24) * 0.002
        elif age < 168:  # 14岁前中等
            self.plasticity_level = 0.7 - (age - 84) * 0.002
        else:  # 成人期
            self.plasticity_level = max(0.3, 0.5 - (age - 168) * 0.001)

    def experience_dependent_plasticity(self, experience: str,
                                        intensity: float = 0.5):
        """经验依赖的可塑性：经验塑造大脑。

        关键期内经验的影响更大。
        """
        # 基础可塑性
        plasticity = self.plasticity_level

        # 关键期加成
        critical_period_bonus = 0.0
        for period in self.critical_periods.values():
            if period["active"]:
                critical_period_bonus += period["sensitivity"] * 0.2

        # 总可塑性
        total_plasticity = plasticity + critical_period_bonus

        # 经验强度影响
        effect = intensity * total_plasticity * self.experience_dependent_gain

        # 强化相关突触（简化版）
        self.hebbian_rate = self._clip(
            self.hebbian_rate + effect * 0.001, 0.001, 0.1)

        # 增加神经复杂度
        self.neural_development["neural_complexity"] = self._clip(
            self.neural_development["neural_complexity"] + effect * 0.01, 0, 1)

        return {
            "experience": experience,
            "intensity": intensity,
            "plasticity_level": round(plasticity, 3),
            "critical_period_bonus": round(critical_period_bonus, 3),
            "total_plasticity": round(total_plasticity, 3),
            "effect_size": round(effect, 3),
        }

    # ===== 发育里程碑 =====

    def get_developmental_milestones(self) -> List[Dict]:
        """获取发育里程碑列表。"""
        return self.development["developmental_milestones"]

    def get_development_summary(self) -> Dict:
        """获取发育总结。"""
        return {
            "age": round(self.development["age"], 1),
            "stage": self.development["stage"],
            "stage_name": self.developmental_stages[self.development["stage"]]["name"],
            "piaget_stage": self.get_piaget_stage()["name"],
            "milestones_count": len(self.development["developmental_milestones"]),
            "plasticity_level": round(self.plasticity_level, 3),
            "neural_complexity": round(self.neural_development["neural_complexity"], 3),
            "myelination_level": round(self.neural_development["myelination_level"], 3),
        }

    # ===== 发育报告 =====

    def get_development_report(self) -> Dict:
        """获取发育过程系统状态报告。"""
        return {
            "current_stage": self.get_current_stage(),
            "piaget_stage": self.get_piaget_stage(),

            "neural_development": self.get_neural_development_status(),

            "critical_periods": self.get_critical_periods_status(),

            "plasticity": {
                "level": round(self.plasticity_level, 3),
                "experience_dependent_gain": self.experience_dependent_gain,
            },

            "milestones": {
                "count": len(self.development["developmental_milestones"]),
                "list": self.development["developmental_milestones"][-5:],  # 最近5个
            },

            "cognitive_abilities": {
                "object_permanence": self.has_object_permanence(),
                "conservation": self.has_conservation(),
                "abstract_thinking": self.has_abstract_thinking(),
                "metacognition": self.development["age"] >= 120,  # 约10岁
            },

            "parameters": {
                "development_rate": self.development_rate,
                "plasticity_level": self.plasticity_level,
            },
        }

    # ------------------ 具身认知系统（v6.8 Embodied Cognition） ------------------

    # ===== 身体图式 =====

    def init_body_schema(self):
        """初始化身体图式。

        身体图式是对身体各部位的内部表征，
        包括位置、状态和运动能力。
        """
        self.body_schema["active"] = True

        # 初始化各身体部位
        for part_id, part_info in self.body_parts_definition.items():
            self.body_schema["body_parts"][part_id] = {
                "name": part_info["name"],
                "sensitivity": part_info["sensitivity"],
                "motor_control": part_info["motor_control"],
                "position": (0, 0, 0),  # 3D位置
                "state": "relaxed",  # 状态
                "activation": 0.0,  # 激活度
            }

        # 推入思考空间
        self._push_thought(
            "身体图式已建立",
            source="embodied",
            activation=0.7
        )

    def get_body_part(self, part_id: str) -> Optional[Dict]:
        """获取身体部位信息。"""
        return self.body_schema["body_parts"].get(part_id)

    def activate_body_part(self, part_id: str,
                           intensity: float = 0.5) -> Dict:
        """激活身体部位（运动或感觉）。"""
        part = self.body_schema["body_parts"].get(part_id)
        if not part:
            return {"success": False, "reason": "Body part not found"}

        part["activation"] = self._clip(
            part["activation"] + intensity, 0, 1)
        part["state"] = "active"

        # 激活感觉皮层和运动相关区域
        self.brain_regions["sensory_cortex"]["activity"] = self._clip(
            self.brain_regions["sensory_cortex"]["activity"] + intensity * 0.1, 0, 1)

        return {
            "success": True,
            "part": part_id,
            "name": part["name"],
            "activation": round(part["activation"], 3),
            "state": part["state"],
        }

    def relax_body_part(self, part_id: str) -> Dict:
        """放松身体部位。"""
        part = self.body_schema["body_parts"].get(part_id)
        if not part:
            return {"success": False, "reason": "Body part not found"}

        part["activation"] = self._clip(
            part["activation"] - 0.3, 0, 1)
        if part["activation"] < 0.1:
            part["state"] = "relaxed"

        return {
            "success": True,
            "part": part_id,
            "name": part["name"],
            "activation": round(part["activation"], 3),
            "state": part["state"],
        }

    def change_posture(self, new_posture: str) -> Dict:
        """改变身体姿态。"""
        old_posture = self.body_schema["posture"]
        self.body_schema["posture"] = new_posture

        # 姿态改变影响身体意识
        if new_posture in ["standing", "sitting"]:
            self.body_schema["body_awareness"] = self._clip(
                self.body_schema["body_awareness"] + 0.05, 0, 1)
        elif new_posture == "lying":
            self.body_schema["body_awareness"] = self._clip(
                self.body_schema["body_awareness"] - 0.1, 0, 1)

        return {
            "old_posture": old_posture,
            "new_posture": new_posture,
            "body_awareness": round(self.body_schema["body_awareness"], 3),
        }

    def get_body_schema_status(self) -> Dict:
        """获取身体图式状态。"""
        active_parts = [
            part_id for part_id, part in self.body_schema["body_parts"].items()
            if part["activation"] > 0.3
        ]

        return {
            "active": self.body_schema["active"],
            "posture": self.body_schema["posture"],
            "proprioception": round(self.body_schema["proprioception"], 3),
            "body_awareness": round(self.body_schema["body_awareness"], 3),
            "total_parts": len(self.body_schema["body_parts"]),
            "active_parts": active_parts,
            "active_count": len(active_parts),
        }

    # ===== 运动控制 =====

    def plan_motor_action(self, action: str,
                          body_part: str = "right_hand") -> Dict:
        """计划运动动作。

        运动计划在执行前先在大脑中模拟。
        """
        # 激活运动相关区域（用联合皮层和前额叶代替）
        self.brain_regions["association_cortex"]["activity"] = self._clip(
            self.brain_regions["association_cortex"]["activity"] + 0.1, 0, 1)
        self.brain_regions["prefrontal"]["activity"] = self._clip(
            self.brain_regions["prefrontal"]["activity"] + 0.1, 0, 1)

        self.motor_system["active"] = True

        # 创建运动计划
        motor_plan = {
            "action": action,
            "body_part": body_part,
            "planned": True,
            "executed": False,
            "expected_duration": 1.0,
            "expected_effort": 0.5,
            "confidence": 0.6,
        }

        self.motor_system["current_action"] = motor_plan

        return {
            "action": action,
            "body_part": body_part,
            "planned": True,
            "confidence": 0.6,
            "expected_effort": 0.5,
        }

    def execute_motor_action(self) -> Dict:
        """执行运动动作。"""
        if not self.motor_system["current_action"]:
            return {"success": False, "reason": "No action planned"}

        plan = self.motor_system["current_action"]
        plan["executed"] = True

        # 激活身体部位
        self.activate_body_part(plan["body_part"], intensity=0.7)

        # 运动学习
        self.motor_system["motor_skill_level"] = self._clip(
            self.motor_system["motor_skill_level"] + 0.01, 0, 1)

        # 记录动作历史
        self.motor_system["action_history"].append({
            "action": plan["action"],
            "body_part": plan["body_part"],
            "tick": self.tick,
            "success": True,
        })
        if len(self.motor_system["action_history"]) > 100:
            self.motor_system["action_history"].pop(0)

        # 感觉运动循环
        self._sensorimotor_feedback(plan)

        return {
            "success": True,
            "action": plan["action"],
            "body_part": plan["body_part"],
            "executed": True,
            "skill_level": round(self.motor_system["motor_skill_level"], 3),
        }

    def learn_motor_skill(self, skill_name: str,
                          practice_count: int = 10) -> Dict:
        """学习运动技能。

        通过练习提高运动技能水平。
        """
        # 初始技能水平
        if skill_name not in self.motor_skills:
            self.motor_skills[skill_name] = {
                "level": 0.1,
                "practice_count": 0,
                "mastery": "beginner",
            }

        skill = self.motor_skills[skill_name]

        # 练习提高技能（边际递减）
        for _ in range(practice_count):
            improvement = 0.05 * (1 - skill["level"])
            skill["level"] = self._clip(
                skill["level"] + improvement, 0, 1)
            skill["practice_count"] += 1

        # 更新掌握程度
        if skill["level"] >= 0.9:
            skill["mastery"] = "expert"
        elif skill["level"] >= 0.7:
            skill["mastery"] = "advanced"
        elif skill["level"] >= 0.4:
            skill["mastery"] = "intermediate"
        else:
            skill["mastery"] = "beginner"

        # 提高整体运动技能
        self.motor_system["motor_skill_level"] = self._clip(
            self.motor_system["motor_skill_level"] + 0.005 * practice_count, 0, 1)

        return {
            "skill": skill_name,
            "level": round(skill["level"], 3),
            "practice_count": skill["practice_count"],
            "mastery": skill["mastery"],
        }

    def get_motor_system_status(self) -> Dict:
        """获取运动系统状态。"""
        return {
            "active": self.motor_system["active"],
            "current_action": self.motor_system["current_action"],
            "action_history_count": len(self.motor_system["action_history"]),
            "motor_skill_level": round(self.motor_system["motor_skill_level"], 3),
            "reaction_time": round(self.motor_system["reaction_time"], 3),
            "coordination": round(self.motor_system["coordination"], 3),
            "skills_count": len(self.motor_skills),
            "skills": {
                name: {
                    "level": round(skill["level"], 3),
                    "mastery": skill["mastery"],
                }
                for name, skill in list(self.motor_skills.items())[:5]
            },
        }

    # ===== 感知-行动循环 =====

    def _sensorimotor_feedback(self, action_plan: Dict):
        """感觉运动反馈。

        行动后接收感觉反馈，更新内部模型。
        """
        self.sensorimotor_loop["active"] = True
        self.sensorimotor_loop["loop_count"] += 1

        # 计算预测误差（简化版）
        prediction_error = 0.2  # 基础误差
        if action_plan["confidence"] > 0.7:
            prediction_error *= 0.5  # 高置信度误差小

        self.sensorimotor_loop["prediction_error"] = prediction_error

        # 感觉运动整合
        self.sensorimotor_loop["integration_level"] = self._clip(
            self.sensorimotor_loop["integration_level"] + 0.01, 0, 1)

        # 行动-感知耦合
        self.action_perception_coupling = self._clip(
            self.action_perception_coupling + 0.01, 0, 1)

    def sensorimotor_loop_step(self, perception: str,
                               action: str) -> Dict:
        """感知-行动循环的一步。

        感知 → 决策 → 行动 → 感知
        """
        # 感知输入
        self._push_thought(
            f"感知: {perception}",
            source="sensory",
            activation=0.6
        )

        # 计划行动
        self.plan_motor_action(action)

        # 执行行动
        result = self.execute_motor_action()

        # 反馈
        feedback = f"行动完成: {action}"
        self._push_thought(
            feedback,
            source="motor",
            activation=0.5
        )

        return {
            "perception": perception,
            "action": action,
            "executed": result["success"],
            "prediction_error": round(
                self.sensorimotor_loop["prediction_error"], 3),
            "integration_level": round(
                self.sensorimotor_loop["integration_level"], 3),
            "loop_count": self.sensorimotor_loop["loop_count"],
        }

    # ===== 镜像神经元系统 =====

    def observe_action(self, action: str,
                       actor: str = "other") -> Dict:
        """观察他人的动作。

        镜像神经元在观察和执行同一动作时都会激活。
        """
        # 激活镜像神经元系统
        self.mirror_neuron_system["active"] = True
        self.mirror_neuron_system["observation_activation"] = self._clip(
            self.mirror_neuron_system["observation_activation"] + 0.3, 0, 1)

        # 观察也会激活运动系统（镜像激活）
        self.mirror_neuron_system["execution_activation"] = self._clip(
            self.mirror_neuron_system["execution_activation"] + 0.2, 0, 1)

        # 增加镜像神经元数量（通过学习）
        self.mirror_neuron_system["mirror_neurons_count"] += 1

        # 激活相关脑区
        self.brain_regions["association_cortex"]["activity"] = self._clip(
            self.brain_regions["association_cortex"]["activity"] + 0.1, 0, 1)

        # 推入思考空间
        self._push_thought(
            f"观察到{actor}在{action}",
            source="mirror",
            activation=0.6
        )

        return {
            "action": action,
            "actor": actor,
            "observation_activation": round(
                self.mirror_neuron_system["observation_activation"], 3),
            "execution_activation": round(
                self.mirror_neuron_system["execution_activation"], 3),
            "mirror_neurons": self.mirror_neuron_system["mirror_neurons_count"],
        }

    def imitate_action(self, action: str) -> Dict:
        """模仿动作。

        通过镜像神经元系统模仿观察到的动作。
        """
        # 模仿能力影响成功率
        imitation_success = 0.5 + self.mirror_neuron_system["imitation_ability"] * 0.3

        # 执行模仿
        self.plan_motor_action(action)
        result = self.execute_motor_action()

        # 提高模仿能力
        self.mirror_neuron_system["imitation_ability"] = self._clip(
            self.mirror_neuron_system["imitation_ability"] + 0.02, 0, 1)

        # 共情也会提高
        self.mirror_neuron_system["empathy_level"] = self._clip(
            self.mirror_neuron_system["empathy_level"] + 0.01, 0, 1)

        return {
            "action": action,
            "imitated": result["success"],
            "success_rate": round(imitation_success, 3),
            "imitation_ability": round(
                self.mirror_neuron_system["imitation_ability"], 3),
            "empathy_level": round(
                self.mirror_neuron_system["empathy_level"], 3),
        }

    def empathize_with_action(self, action: str,
                              emotion: str = "neutral") -> Dict:
        """通过动作共情。

        镜像神经元系统是共情的神经基础。
        """
        # 观察动作
        self.observe_action(action)

        # 情绪传染
        if emotion == "happy":
            self.emotion["pleasure"] = self._clip(
                self.emotion["pleasure"] + 0.1, 0, 1)
        elif emotion == "sad":
            self.emotion["stress"] = self._clip(
                self.emotion["stress"] + 0.1, 0, 1)
        elif emotion == "pain":
            self.emotion["stress"] = self._clip(
                self.emotion["stress"] + 0.15, 0, 1)

        # 提高共情水平
        self.mirror_neuron_system["empathy_level"] = self._clip(
            self.mirror_neuron_system["empathy_level"] + 0.02, 0, 1)

        return {
            "action": action,
            "observed_emotion": emotion,
            "empathy_level": round(
                self.mirror_neuron_system["empathy_level"], 3),
            "own_emotion": {
                "pleasure": round(self.emotion["pleasure"], 3),
                "stress": round(self.emotion["stress"], 3),
            },
        }

    def get_mirror_neuron_status(self) -> Dict:
        """获取镜像神经元系统状态。"""
        return {
            "active": self.mirror_neuron_system["active"],
            "observation_activation": round(
                self.mirror_neuron_system["observation_activation"], 3),
            "execution_activation": round(
                self.mirror_neuron_system["execution_activation"], 3),
            "imitation_ability": round(
                self.mirror_neuron_system["imitation_ability"], 3),
            "empathy_level": round(
                self.mirror_neuron_system["empathy_level"], 3),
            "mirror_neurons_count": self.mirror_neuron_system["mirror_neurons_count"],
        }

    # ===== 环境交互 =====

    def perceive_affordance(self, object_name: str) -> Dict:
        """感知功能可供性（Affordance）。

        物体提供的行动可能性。
        例如：椅子提供"坐"的功能，杯子提供"抓握"的功能。
        """
        # 功能可供性感知
        affordances = []

        # 基于物体名称推断可供性（简化版）
        if "椅子" in object_name or "凳子" in object_name:
            affordances = ["坐", "站上去", "移动"]
        elif "杯子" in object_name or "碗" in object_name:
            affordances = ["抓握", "喝水", "盛放"]
        elif "球" in object_name:
            affordances = ["抓", "扔", "踢", "滚"]
        elif "书" in object_name:
            affordances = ["读", "翻页", "拿"]
        else:
            affordances = ["触摸", "拿起", "观察"]

        # 可供性感知水平
        self.environment_interaction["affordance_perception"] = self._clip(
            self.environment_interaction["affordance_perception"] + 0.01, 0, 1)

        # 激活顶叶（空间感知）
        self.brain_regions["association_cortex"]["activity"] = self._clip(
            self.brain_regions["association_cortex"]["activity"] + 0.05, 0, 1)

        return {
            "object": object_name,
            "affordances": affordances,
            "perception_level": round(
                self.environment_interaction["affordance_perception"], 3),
        }

    def manipulate_object(self, object_name: str,
                          action: str = "grasp") -> Dict:
        """操作物体。"""
        # 执行动作
        self.plan_motor_action(f"{action} {object_name}", "right_hand")
        result = self.execute_motor_action()

        # 记录操作过的物体
        if object_name not in self.environment_interaction["objects_manipulated"]:
            self.environment_interaction["objects_manipulated"].append(object_name)

        # 提高操作技能
        self.motor_system["coordination"] = self._clip(
            self.motor_system["coordination"] + 0.01, 0, 1)

        return {
            "object": object_name,
            "action": action,
            "success": result["success"],
            "coordination": round(self.motor_system["coordination"], 3),
            "total_objects_manipulated": len(
                self.environment_interaction["objects_manipulated"]),
        }

    def use_tool(self, tool_name: str,
                 target: str = "") -> Dict:
        """使用工具。

        工具使用是人类认知的重要标志。
        """
        # 工具使用需要更高的技能
        if tool_name not in self.environment_interaction["tools_used"]:
            self.environment_interaction["tools_used"].append(tool_name)

        # 执行工具使用动作
        self.plan_motor_action(f"使用{tool_name}", "right_hand")
        result = self.execute_motor_action()

        # 提高工具使用技能
        self.environment_interaction["tool_use_skill"] = self._clip(
            self.environment_interaction["tool_use_skill"] + 0.02, 0, 1)

        return {
            "tool": tool_name,
            "target": target,
            "success": result["success"],
            "tool_skill": round(
                self.environment_interaction["tool_use_skill"], 3),
            "total_tools_used": len(
                self.environment_interaction["tools_used"]),
        }

    def navigate_spatially(self, destination: str) -> Dict:
        """空间导航。"""
        # 空间导航能力
        navigation_success = 0.5 + \
            self.environment_interaction["spatial_navigation"] * 0.3

        # 提高空间导航能力
        self.environment_interaction["spatial_navigation"] = self._clip(
            self.environment_interaction["spatial_navigation"] + 0.02, 0, 1)

        # 激活海马体（空间记忆）
        self.brain_regions["hippocampus"]["activity"] = self._clip(
            self.brain_regions["hippocampus"]["activity"] + 0.1, 0, 1)

        return {
            "destination": destination,
            "success_rate": round(navigation_success, 3),
            "spatial_ability": round(
                self.environment_interaction["spatial_navigation"], 3),
        }

    def get_environment_interaction_status(self) -> Dict:
        """获取环境交互状态。"""
        return {
            "objects_manipulated": len(
                self.environment_interaction["objects_manipulated"]),
            "tools_used": len(self.environment_interaction["tools_used"]),
            "spatial_navigation": round(
                self.environment_interaction["spatial_navigation"], 3),
            "tool_use_skill": round(
                self.environment_interaction["tool_use_skill"], 3),
            "affordance_perception": round(
                self.environment_interaction["affordance_perception"], 3),
        }

    # ===== 具身认知报告 =====

    def get_embodied_cognition_report(self) -> Dict:
        """获取具身认知系统状态报告。"""
        return {
            "body_schema": self.get_body_schema_status(),
            "motor_system": self.get_motor_system_status(),
            "sensorimotor_loop": {
                "active": self.sensorimotor_loop["active"],
                "loop_count": self.sensorimotor_loop["loop_count"],
                "prediction_error": round(
                    self.sensorimotor_loop["prediction_error"], 3),
                "integration_level": round(
                    self.sensorimotor_loop["integration_level"], 3),
            },
            "mirror_neuron_system": self.get_mirror_neuron_status(),
            "environment_interaction": self.get_environment_interaction_status(),
            "embodiment": {
                "embodiment_level": round(self.embodiment_level, 3),
                "sensorimotor_continuity": round(
                    self.sensorimotor_continuity, 3),
                "action_perception_coupling": round(
                    self.action_perception_coupling, 3),
            },
        }

    # ------------------ 文化进化系统（v6.9 Cultural Evolution） ------------------

    # ===== 文化传递 =====

    def transmit_culture(self, cultural_trait: str,
                         source: str = "peer",
                         method: str = "imitation") -> Dict:
        """传递文化特征。

        文化传递是文化进化的基本机制。
        """
        self.cultural_transmission["active"] = True
        self.cultural_transmission["transmission_count"] += 1

        # 不同传递方式
        if method == "imitation":
            success_rate = self.cultural_transmission["imitation_rate"]
            # 模仿学习（使用镜像神经元）
            self.observe_action(cultural_trait)
        elif method == "teaching":
            success_rate = self.cultural_transmission["teaching_rate"]
        elif method == "language":
            success_rate = self.cultural_transmission["language_rate"]
        else:
            success_rate = 0.5

        # 不同传递方向
        if source == "parent":
            self.cultural_transmission["vertical_transmission"] += 1
            success_rate *= 1.2  # 垂直传递更可靠
        elif source == "peer":
            self.cultural_transmission["horizontal_transmission"] += 1
        elif source == "elder":
            self.cultural_transmission["oblique_transmission"] += 1
            success_rate *= 1.1  # 斜向传递有威望加成

        # 添加模因
        self.add_meme(cultural_trait, source=source)

        # 记录到记忆（简化版，直接添加到短期记忆）
        from dataclasses import dataclass
        # 使用已有的 BrainMemory 类
        try:
            memory = BrainMemory(
                content=f"从{source}学到了{cultural_trait}",
                weight=0.6,
                source="cultural"
            )
            self.short_memory.append(memory)
            if len(self.short_memory) > self.max_stm:
                self.short_memory.pop(0)
        except:
            pass  # 如果失败就跳过

        # 推入思考空间
        self._push_thought(
            f"文化传递: {cultural_trait} (来自{source})",
            source="cultural",
            activation=0.6
        )

        return {
            "trait": cultural_trait,
            "source": source,
            "method": method,
            "success_rate": round(success_rate, 3),
            "transmission_count": self.cultural_transmission["transmission_count"],
        }

    def get_cultural_transmission_status(self) -> Dict:
        """获取文化传递状态。"""
        return {
            "active": self.cultural_transmission["active"],
            "transmission_count": self.cultural_transmission["transmission_count"],
            "imitation_rate": round(self.cultural_transmission["imitation_rate"], 3),
            "teaching_rate": round(self.cultural_transmission["teaching_rate"], 3),
            "language_rate": round(self.cultural_transmission["language_rate"], 3),
            "vertical": self.cultural_transmission["vertical_transmission"],
            "horizontal": self.cultural_transmission["horizontal_transmission"],
            "oblique": self.cultural_transmission["oblique_transmission"],
        }

    # ===== 文化变异 =====

    def innovate_culture(self, base_trait: str = "") -> Dict:
        """文化创新：产生新的文化特征。"""
        self.cultural_variation["active"] = True
        self.cultural_variation["innovation_count"] += 1

        # 生成新特征
        import random
        if base_trait:
            new_trait = f"{base_trait}_变体{random.randint(1, 100)}"
        else:
            new_trait = f"创新_{random.randint(1, 1000)}"

        # 添加模因
        self.add_meme(new_trait, source="innovation", novelty=0.8)

        # 增加文化多样性
        self.cultural_dynamics["diversity"] = self._clip(
            self.cultural_dynamics["diversity"] + 0.02, 0, 1)

        # 增加累积文化
        self.cultural_dynamics["cumulative_culture"] = self._clip(
            self.cultural_dynamics["cumulative_culture"] + 0.05, 0, 1)

        # 推入思考空间
        self._push_thought(
            f"文化创新: {new_trait}",
            source="innovation",
            activation=0.7
        )

        return {
            "new_trait": new_trait,
            "base_trait": base_trait,
            "innovation_count": self.cultural_variation["innovation_count"],
            "diversity": round(self.cultural_dynamics["diversity"], 3),
        }

    def cultural_drift(self, trait: str) -> Dict:
        """文化漂移：随机变化。"""
        self.cultural_variation["active"] = True
        self.cultural_variation["drift_count"] += 1

        # 随机变化
        import random
        drift_amount = random.uniform(-0.1, 0.1)

        # 更新模因
        if trait in self.memes:
            meme = self.memes[trait]
            meme["frequency"] = self._clip(
                meme["frequency"] + drift_amount, 0, 1)

        return {
            "trait": trait,
            "drift_amount": round(drift_amount, 3),
            "drift_count": self.cultural_variation["drift_count"],
        }

    def recombine_culture(self, trait1: str,
                          trait2: str) -> Dict:
        """文化重组：组合两个文化特征。"""
        self.cultural_variation["active"] = True
        self.cultural_variation["recombination_count"] += 1

        # 生成重组特征
        new_trait = f"{trait1}+{trait2}"

        # 添加模因
        self.add_meme(new_trait, source="recombination", novelty=0.6)

        # 增加文化复杂性
        self.cultural_dynamics["complexity"] = self._clip(
            self.cultural_dynamics["complexity"] + 0.03, 0, 1)

        return {
            "new_trait": new_trait,
            "parent_traits": [trait1, trait2],
            "recombination_count": self.cultural_variation["recombination_count"],
            "complexity": round(self.cultural_dynamics["complexity"], 3),
        }

    def get_cultural_variation_status(self) -> Dict:
        """获取文化变异状态。"""
        return {
            "active": self.cultural_variation["active"],
            "innovation_rate": round(self.cultural_variation["innovation_rate"], 3),
            "drift_rate": round(self.cultural_variation["drift_rate"], 3),
            "recombination_rate": round(
                self.cultural_variation["recombination_rate"], 3),
            "innovation_count": self.cultural_variation["innovation_count"],
            "drift_count": self.cultural_variation["drift_count"],
            "recombination_count": self.cultural_variation["recombination_count"],
        }

    # ===== 文化选择 =====

    def select_cultural_trait(self, trait: str,
                              fitness: float = 0.5) -> Dict:
        """选择文化特征。

        文化选择决定哪些特征能够传播和留存。
        """
        self.cultural_selection["active"] = True
        self.cultural_selection["selection_count"] += 1

        # 自然选择（基于生存价值）
        natural_fitness = fitness * self.cultural_selection["natural_selection"]

        # 文化选择（基于文化价值）
        cultural_fitness = fitness * self.cultural_selection["cultural_selection"]

        # 频率依赖选择
        frequency = 0.5
        if trait in self.memes:
            frequency = self.memes[trait]["frequency"]

        if frequency > 0.7:
            # 从众偏差：常见的更受欢迎
            frequency_bonus = 0.2
        elif frequency < 0.3:
            # 稀有性偏差：稀有的更有价值
            frequency_bonus = 0.1
        else:
            frequency_bonus = 0.0

        # 总适应度
        total_fitness = (natural_fitness + cultural_fitness +
                        frequency_bonus * self.cultural_selection["frequency_dependent"])

        # 更新模因适应度
        if trait in self.memes:
            self.memes[trait]["fitness"] = self._clip(
                self.memes[trait]["fitness"] + total_fitness * 0.1, 0, 1)
            self.memes[trait]["frequency"] = self._clip(
                self.memes[trait]["frequency"] + total_fitness * 0.05, 0, 1)

        # 记录被选择的特征
        self.cultural_selection["selected_traits"].append({
            "trait": trait,
            "fitness": round(total_fitness, 3),
            "tick": self.tick,
        })
        if len(self.cultural_selection["selected_traits"]) > 50:
            self.cultural_selection["selected_traits"].pop(0)

        return {
            "trait": trait,
            "natural_fitness": round(natural_fitness, 3),
            "cultural_fitness": round(cultural_fitness, 3),
            "frequency_bonus": round(frequency_bonus, 3),
            "total_fitness": round(total_fitness, 3),
            "selection_count": self.cultural_selection["selection_count"],
        }

    def get_cultural_selection_status(self) -> Dict:
        """获取文化选择状态。"""
        return {
            "active": self.cultural_selection["active"],
            "natural_selection": round(
                self.cultural_selection["natural_selection"], 3),
            "cultural_selection": round(
                self.cultural_selection["cultural_selection"], 3),
            "frequency_dependent": round(
                self.cultural_selection["frequency_dependent"], 3),
            "selection_count": self.cultural_selection["selection_count"],
            "selected_count": len(self.cultural_selection["selected_traits"]),
        }

    # ===== 模因系统 =====

    def add_meme(self, meme_name: str,
                 source: str = "unknown",
                 fitness: float = 0.5,
                 novelty: float = 0.5) -> Dict:
        """添加模因（文化的基本单位）。"""
        self.meme_system["active"] = True

        is_new = meme_name not in self.memes
        if is_new:
            # 新模因
            self.memes[meme_name] = {
                "name": meme_name,
                "source": source,
                "fitness": fitness,
                "frequency": 0.1,  # 初始频率
                "novelty": novelty,
                "complexity": 0.3,
                "age": 0,
                "generation": 0,
                "copies": 0,
                "mutations": 0,
                "created_tick": self.tick,
            }
            self.meme_system["total_memes"] += 1
        else:
            # 已有模因，增加频率
            self.memes[meme_name]["frequency"] = self._clip(
                self.memes[meme_name]["frequency"] + 0.1, 0, 1)
            self.memes[meme_name]["copies"] += 1

        # 更新模因多样性
        self._update_meme_diversity()

        return {
            "meme": meme_name,
            "source": source,
            "is_new": is_new,
            "fitness": round(self.memes[meme_name]["fitness"], 3),
            "frequency": round(self.memes[meme_name]["frequency"], 3),
            "total_memes": self.meme_system["total_memes"],
        }

    def replicate_meme(self, meme_name: str) -> Dict:
        """复制模因。"""
        if meme_name not in self.memes:
            return {"success": False, "reason": "Meme not found"}

        meme = self.memes[meme_name]
        meme["copies"] += 1

        # 复制成功率取决于适应度
        replication_success = meme["fitness"] * self.meme_system["replication_rate"]

        # 可能发生突变
        import random
        if random.random() < self.meme_system["mutation_rate"]:
            meme["mutations"] += 1
            meme["fitness"] = self._clip(
                meme["fitness"] + random.uniform(-0.1, 0.1), 0, 1)

        return {
            "success": True,
            "meme": meme_name,
            "copies": meme["copies"],
            "replication_success": round(replication_success, 3),
            "mutations": meme["mutations"],
        }

    def _update_meme_diversity(self):
        """更新模因多样性。"""
        if self.meme_system["total_memes"] == 0:
            self.meme_system["meme_diversity"] = 0.0
            return

        # 简单的多样性计算：模因数量的归一化
        self.meme_system["meme_diversity"] = self._clip(
            self.meme_system["total_memes"] / 50.0, 0, 1)

        # 复杂性计算：平均适应度 × 多样性
        if self.memes:
            avg_fitness = sum(m["fitness"] for m in self.memes.values()) / len(self.memes)
            self.meme_system["meme_complexity"] = self._clip(
                avg_fitness * self.meme_system["meme_diversity"], 0, 1)

    def get_meme_system_status(self) -> Dict:
        """获取模因系统状态。"""
        # 获取活跃模因（频率 > 0.1）
        active_memes = [
            name for name, meme in self.memes.items()
            if meme["frequency"] > 0.1
        ]
        self.meme_system["active_memes"] = len(active_memes)

        # 获取 top 模因
        top_memes = sorted(
            self.memes.items(),
            key=lambda x: x[1]["fitness"],
            reverse=True
        )[:5]

        return {
            "active": self.meme_system["active"],
            "total_memes": self.meme_system["total_memes"],
            "active_memes": self.meme_system["active_memes"],
            "diversity": round(self.meme_system["meme_diversity"], 3),
            "complexity": round(self.meme_system["meme_complexity"], 3),
            "replication_rate": round(self.meme_system["replication_rate"], 3),
            "mutation_rate": round(self.meme_system["mutation_rate"], 3),
            "top_memes": [
                {
                    "name": name,
                    "fitness": round(meme["fitness"], 3),
                    "frequency": round(meme["frequency"], 3),
                }
                for name, meme in top_memes
            ],
        }

    # ===== 群体文化 =====

    def add_cultural_norm(self, norm_name: str,
                          importance: float = 0.5) -> Dict:
        """添加文化规范。"""
        self.group_culture["active"] = True

        self.group_culture["norms"][norm_name] = {
            "name": norm_name,
            "importance": importance,
            "adherence": 0.5,  # 遵守程度
            "age": 0,
        }

        # 增加群体认同
        self.group_culture["identity"] = self._clip(
            self.group_culture["identity"] + 0.05, 0, 1)

        return {
            "norm": norm_name,
            "importance": importance,
            "total_norms": len(self.group_culture["norms"]),
            "group_identity": round(self.group_culture["identity"], 3),
        }

    def add_ritual(self, ritual_name: str,
                   frequency: str = "annual") -> Dict:
        """添加文化仪式。"""
        self.group_culture["active"] = True

        ritual = {
            "name": ritual_name,
            "frequency": frequency,
            "importance": 0.5,
            "participation": 0.0,
        }
        self.group_culture["rituals"].append(ritual)

        # 仪式增强群体凝聚力
        self.group_culture["cultural_identity"] = self._clip(
            self.group_culture["cultural_identity"] + 0.05, 0, 1)

        return {
            "ritual": ritual_name,
            "frequency": frequency,
            "total_rituals": len(self.group_culture["rituals"]),
            "cultural_identity": round(
                self.group_culture["cultural_identity"], 3),
        }

    def conform_to_group(self, trait: str) -> Dict:
        """从众：跟随群体行为。"""
        # 从众偏差影响
        conformity_effect = self.group_culture["conformity_bias"]

        # 增加该特征的频率
        if trait in self.memes:
            self.memes[trait]["frequency"] = self._clip(
                self.memes[trait]["frequency"] + conformity_effect * 0.1, 0, 1)

        # 增加群体认同
        self.group_culture["identity"] = self._clip(
            self.group_culture["identity"] + 0.02, 0, 1)

        return {
            "trait": trait,
            "conformity_bias": round(
                self.group_culture["conformity_bias"], 3),
            "group_identity": round(self.group_culture["identity"], 3),
        }

    def follow_prestige(self, trait: str,
                        prestige: float = 0.8) -> Dict:
        """声望偏差：跟随有声望的人。"""
        # 声望偏差影响
        prestige_effect = prestige * self.group_culture["prestige_bias"]

        # 增加该特征的频率
        if trait in self.memes:
            self.memes[trait]["frequency"] = self._clip(
                self.memes[trait]["frequency"] + prestige_effect * 0.1, 0, 1)
            self.memes[trait]["fitness"] = self._clip(
                self.memes[trait]["fitness"] + prestige_effect * 0.05, 0, 1)

        return {
            "trait": trait,
            "prestige": prestige,
            "prestige_bias": round(
                self.group_culture["prestige_bias"], 3),
            "effect": round(prestige_effect, 3),
        }

    def get_group_culture_status(self) -> Dict:
        """获取群体文化状态。"""
        return {
            "active": self.group_culture["active"],
            "norms_count": len(self.group_culture["norms"]),
            "rituals_count": len(self.group_culture["rituals"]),
            "group_identity": round(self.group_culture["identity"], 3),
            "cultural_identity": round(
                self.group_culture["cultural_identity"], 3),
            "conformity_bias": round(
                self.group_culture["conformity_bias"], 3),
            "prestige_bias": round(
                self.group_culture["prestige_bias"], 3),
            "shared_values_count": len(self.group_culture["shared_values"]),
        }

    # ===== 文化进化动力学 =====

    def cultural_evolution_step(self) -> Dict:
        """文化进化的一步。"""
        self.cultural_dynamics["generation"] += 1

        # 1. 变异
        if self.memes:
            import random
            # 随机选择一个模因进行突变
            meme_name = random.choice(list(self.memes.keys()))
            self.cultural_drift(meme_name)

        # 2. 选择
        if self.memes:
            # 选择适应度最高的模因
            top_meme = max(
                self.memes.items(),
                key=lambda x: x[1]["fitness"]
            )[0]
            self.select_cultural_trait(top_meme, fitness=0.8)

        # 3. 传递
        if self.memes:
            # 复制最成功的模因
            top_meme = max(
                self.memes.items(),
                key=lambda x: x[1]["fitness"]
            )[0]
            self.replicate_meme(top_meme)

        # 更新文化变迁速率
        self.cultural_dynamics["change_rate"] = self._clip(
            0.1 + self.meme_system["mutation_rate"] * 0.5, 0, 1)

        return {
            "generation": self.cultural_dynamics["generation"],
            "diversity": round(self.cultural_dynamics["diversity"], 3),
            "complexity": round(self.cultural_dynamics["complexity"], 3),
            "change_rate": round(self.cultural_dynamics["change_rate"], 3),
            "cumulative_culture": round(
                self.cultural_dynamics["cumulative_culture"], 3),
        }

    # ===== 文化进化报告 =====

    def get_cultural_evolution_report(self) -> Dict:
        """获取文化进化系统状态报告。"""
        return {
            "cultural_transmission": self.get_cultural_transmission_status(),
            "cultural_variation": self.get_cultural_variation_status(),
            "cultural_selection": self.get_cultural_selection_status(),
            "meme_system": self.get_meme_system_status(),
            "group_culture": self.get_group_culture_status(),
            "cultural_dynamics": {
                "generation": self.cultural_dynamics["generation"],
                "diversity": round(self.cultural_dynamics["diversity"], 3),
                "complexity": round(self.cultural_dynamics["complexity"], 3),
                "change_rate": round(self.cultural_dynamics["change_rate"], 3),
                "cumulative_culture": round(
                    self.cultural_dynamics["cumulative_culture"], 3),
            },
            "parameters": {
                "evolution_rate": self.cultural_evolution_rate,
                "cultural_capacity": self.cultural_capacity,
                "social_learning_bias": self.social_learning_bias,
            },
        }

    # ------------------ 终身学习系统（v7.0 Lifelong Learning） ------------------

    # ===== 持续学习 =====

    def learn_incremental(self, knowledge: str,
                          domain: str = "general") -> Dict:
        """增量学习：逐步添加新知识。

        不遗忘旧知识的同时学习新知识。
        """
        self.continual_learning["active"] = True
        self.continual_learning["learning_count"] += 1
        self.continual_learning["incremental_learning"] += 1

        # 计算知识保留（对抗灾难性遗忘）
        retention = self.continual_learning["knowledge_retention"]
        forgetting = self.continual_learning["catastrophic_forgetting"]

        # 添加新知识到记忆
        from dataclasses import dataclass
        try:
            memory = BrainMemory(
                content=knowledge,
                weight=0.6,
                source="incremental_learning"
            )
            self.short_memory.append(memory)
            if len(self.short_memory) > self.max_stm:
                self.short_memory.pop(0)
        except:
            pass

        # 知识巩固
        self.knowledge_consolidation["storage_strength"] = self._clip(
            self.knowledge_consolidation["storage_strength"] + 0.01, 0, 1)

        # 推入思考空间
        self._push_thought(
            f"增量学习: {knowledge}",
            source="learning",
            activation=0.6
        )

        return {
            "knowledge": knowledge,
            "domain": domain,
            "retention": round(retention, 3),
            "forgetting_risk": round(forgetting, 3),
            "learning_count": self.continual_learning["learning_count"],
        }

    def learn_online(self, experience: str) -> Dict:
        """在线学习：从实时经验中学习。"""
        self.continual_learning["active"] = True
        self.continual_learning["learning_count"] += 1
        self.continual_learning["online_learning"] += 1

        # 在线学习速率受注意力影响
        learning_rate = self.hebbian_rate * (1 + self.attention_factor)

        # 从经验中学习
        self._push_thought(
            f"在线学习: {experience}",
            source="online_learning",
            activation=0.5
        )

        # 更新神经可塑性
        self.neuroplasticity_maintenance = self._clip(
            self.neuroplasticity_maintenance + 0.001, 0, 1)

        return {
            "experience": experience,
            "learning_rate": round(learning_rate, 4),
            "online_count": self.continual_learning["online_learning"],
            "plasticity_maintenance": round(
                self.neuroplasticity_maintenance, 3),
        }

    def transfer_learning(self, source_knowledge: str,
                          target_domain: str) -> Dict:
        """迁移学习：将已有知识应用到新领域。"""
        self.continual_learning["active"] = True
        self.continual_learning["learning_count"] += 1
        self.continual_learning["transfer_learning"] += 1

        # 迁移效率
        transfer_efficiency = 0.5 + self.cognitive_reserve * 0.3

        # 应用知识到新领域
        new_knowledge = f"{source_knowledge} → {target_domain}"

        self._push_thought(
            f"迁移学习: {new_knowledge}",
            source="transfer_learning",
            activation=0.7
        )

        # 提高认知储备
        self.cognitive_reserve = self._clip(
            self.cognitive_reserve + 0.02, 0, 1)

        return {
            "source": source_knowledge,
            "target": target_domain,
            "transfer_efficiency": round(transfer_efficiency, 3),
            "transfer_count": self.continual_learning["transfer_learning"],
            "cognitive_reserve": round(self.cognitive_reserve, 3),
        }

    def get_continual_learning_status(self) -> Dict:
        """获取持续学习状态。"""
        return {
            "active": self.continual_learning["active"],
            "learning_count": self.continual_learning["learning_count"],
            "incremental": self.continual_learning["incremental_learning"],
            "online": self.continual_learning["online_learning"],
            "transfer": self.continual_learning["transfer_learning"],
            "knowledge_retention": round(
                self.continual_learning["knowledge_retention"], 3),
            "catastrophic_forgetting": round(
                self.continual_learning["catastrophic_forgetting"], 3),
        }

    # ===== 知识巩固 =====

    def spaced_repetition(self, knowledge: str,
                          interval: int = 1) -> Dict:
        """间隔重复：在递增的间隔后复习知识。"""
        self.knowledge_consolidation["active"] = True
        self.knowledge_consolidation["consolidation_count"] += 1
        self.knowledge_consolidation["spaced_repetition"] += 1

        # 间隔重复效果（间隔越长，效果越好，但有上限）
        effect = 0.1 * min(interval, 10)

        # 增强记忆
        self.knowledge_consolidation["storage_strength"] = self._clip(
            self.knowledge_consolidation["storage_strength"] + effect * 0.01, 0, 1)
        self.knowledge_consolidation["retrieval_strength"] = self._clip(
            self.knowledge_consolidation["retrieval_strength"] + effect * 0.02, 0, 1)

        return {
            "knowledge": knowledge,
            "interval": interval,
            "effect": round(effect, 3),
            "storage_strength": round(
                self.knowledge_consolidation["storage_strength"], 3),
            "retrieval_strength": round(
                self.knowledge_consolidation["retrieval_strength"], 3),
            "spaced_count": self.knowledge_consolidation["spaced_repetition"],
        }

    def active_recall(self, question: str) -> Dict:
        """主动回忆：通过提问来提取记忆。"""
        self.knowledge_consolidation["active"] = True
        self.knowledge_consolidation["consolidation_count"] += 1
        self.knowledge_consolidation["active_recall"] += 1

        # 主动回忆效果比被动阅读好
        effect = 0.15

        # 尝试回忆
        recalled = self.recall(question, top_k=1)

        # 增强提取强度
        self.knowledge_consolidation["retrieval_strength"] = self._clip(
            self.knowledge_consolidation["retrieval_strength"] + effect * 0.02, 0, 1)

        return {
            "question": question,
            "recalled_count": len(recalled),
            "effect": effect,
            "retrieval_strength": round(
                self.knowledge_consolidation["retrieval_strength"], 3),
            "recall_count": self.knowledge_consolidation["active_recall"],
        }

    def interleaved_practice(self, topics: list) -> Dict:
        """交错练习：混合不同主题进行练习。"""
        self.knowledge_consolidation["active"] = True
        self.knowledge_consolidation["consolidation_count"] += 1
        self.knowledge_consolidation["interleaved_practice"] += 1

        # 交错练习效果
        effect = 0.1 * len(topics)

        # 增强知识整合
        self.knowledge_consolidation["consolidation_rate"] = self._clip(
            self.knowledge_consolidation["consolidation_rate"] + effect * 0.01, 0, 1)

        return {
            "topics": topics,
            "topic_count": len(topics),
            "effect": round(effect, 3),
            "consolidation_rate": round(
                self.knowledge_consolidation["consolidation_rate"], 3),
            "interleaved_count": self.knowledge_consolidation["interleaved_practice"],
        }

    def get_knowledge_consolidation_status(self) -> Dict:
        """获取知识巩固状态。"""
        return {
            "active": self.knowledge_consolidation["active"],
            "consolidation_count": self.knowledge_consolidation["consolidation_count"],
            "spaced_repetition": self.knowledge_consolidation["spaced_repetition"],
            "active_recall": self.knowledge_consolidation["active_recall"],
            "interleaved_practice": self.knowledge_consolidation["interleaved_practice"],
            "consolidation_rate": round(
                self.knowledge_consolidation["consolidation_rate"], 3),
            "retrieval_strength": round(
                self.knowledge_consolidation["retrieval_strength"], 3),
            "storage_strength": round(
                self.knowledge_consolidation["storage_strength"], 3),
        }

    # ===== 技能提升 =====

    def deliberate_practice(self, skill: str,
                            intensity: float = 0.5) -> Dict:
        """刻意练习：有目的的、专注的练习。"""
        self.skill_improvement["active"] = True
        self.skill_improvement["practice_count"] += 1
        self.skill_improvement["deliberate_practice"] += 1

        # 刻意练习效果（强度越高，效果越好，但有边际递减）
        base_effect = self.skill_improvement["skill_growth_rate"]
        effect = base_effect * intensity * (1 - self.skill_improvement["plateau_resistance"] * 0.5)

        # 学习运动技能（如果有的话）
        if skill in self.motor_skills:
            self.motor_skills[skill]["level"] = self._clip(
                self.motor_skills[skill]["level"] + effect, 0, 1)

        # 提高整体技能水平
        self.motor_system["motor_skill_level"] = self._clip(
            self.motor_system["motor_skill_level"] + effect * 0.1, 0, 1)

        # 平台期抗性（练习越多，越能突破平台期）
        self.skill_improvement["plateau_resistance"] = self._clip(
            self.skill_improvement["plateau_resistance"] + 0.005, 0, 1)

        return {
            "skill": skill,
            "intensity": intensity,
            "effect": round(effect, 4),
            "practice_count": self.skill_improvement["practice_count"],
            "deliberate_count": self.skill_improvement["deliberate_practice"],
            "plateau_resistance": round(
                self.skill_improvement["plateau_resistance"], 3),
        }

    def feedback_loop(self, skill: str,
                      feedback: float = 0.5) -> Dict:
        """反馈循环：根据反馈调整练习。"""
        self.skill_improvement["active"] = True
        self.skill_improvement["feedback_loops"] += 1

        # 反馈质量影响学习效果
        feedback_effect = feedback * 0.1

        # 调整练习策略
        if feedback > 0.7:
            # 反馈好，增加难度
            self.skill_improvement["skill_growth_rate"] = self._clip(
                self.skill_improvement["skill_growth_rate"] + 0.01, 0, 1)
        elif feedback < 0.3:
            # 反馈差，降低难度
            self.skill_improvement["skill_growth_rate"] = self._clip(
                self.skill_improvement["skill_growth_rate"] - 0.01, 0.01, 1)

        return {
            "skill": skill,
            "feedback": feedback,
            "feedback_effect": round(feedback_effect, 3),
            "growth_rate": round(
                self.skill_improvement["skill_growth_rate"], 3),
            "feedback_count": self.skill_improvement["feedback_loops"],
        }

    def skill_transfer(self, source_skill: str,
                       target_skill: str) -> Dict:
        """技能迁移：将一个技能的学习迁移到另一个技能。"""
        self.skill_improvement["active"] = True
        self.skill_improvement["skill_transfer"] += 1

        # 迁移效率
        transfer_efficiency = 0.3 + self.cognitive_reserve * 0.4

        # 迁移效果
        if source_skill in self.motor_skills:
            source_level = self.motor_skills[source_skill]["level"]
            transfer_amount = source_level * transfer_efficiency * 0.1

            if target_skill not in self.motor_skills:
                self.motor_skills[target_skill] = {
                    "level": 0.1,
                    "practice_count": 0,
                    "mastery": "beginner",
                }

            self.motor_skills[target_skill]["level"] = self._clip(
                self.motor_skills[target_skill]["level"] + transfer_amount, 0, 1)

        return {
            "source": source_skill,
            "target": target_skill,
            "transfer_efficiency": round(transfer_efficiency, 3),
            "transfer_count": self.skill_improvement["skill_transfer"],
        }

    def get_skill_improvement_status(self) -> Dict:
        """获取技能提升状态。"""
        return {
            "active": self.skill_improvement["active"],
            "practice_count": self.skill_improvement["practice_count"],
            "deliberate_practice": self.skill_improvement["deliberate_practice"],
            "feedback_loops": self.skill_improvement["feedback_loops"],
            "skill_transfer": self.skill_improvement["skill_transfer"],
            "skill_growth_rate": round(
                self.skill_improvement["skill_growth_rate"], 3),
            "plateau_resistance": round(
                self.skill_improvement["plateau_resistance"], 3),
        }

    # ===== 元学习 =====

    def meta_learn(self, learning_experience: str,
                   success: float = 0.5) -> Dict:
        """元学习：学习如何学习。"""
        self.meta_learning["active"] = True
        self.meta_learning["meta_learning_count"] += 1

        # 从学习经验中学习
        if success > 0.7:
            # 成功的经验，提高学习能力
            self.meta_learning["learning_to_learn"] = self._clip(
                self.meta_learning["learning_to_learn"] + 0.02, 0, 1)
        elif success < 0.3:
            # 失败的经验，调整策略
            self.meta_learning["strategy_adaptation_rate"] = self._clip(
                self.meta_learning["strategy_adaptation_rate"] + 0.01, 0, 1)

        # 提高整体学习效率
        self.lifelong_learning_rate = self._clip(
            self.lifelong_learning_rate + 0.005, 0, 1)

        return {
            "experience": learning_experience,
            "success": success,
            "learning_to_learn": round(
                self.meta_learning["learning_to_learn"], 3),
            "strategy_adaptation": round(
                self.meta_learning["strategy_adaptation_rate"], 3),
            "meta_learning_count": self.meta_learning["meta_learning_count"],
            "lifelong_rate": round(self.lifelong_learning_rate, 3),
        }

    def select_learning_strategy(self, task_type: str = "general") -> Dict:
        """选择最优学习策略。"""
        self.meta_learning["active"] = True

        # 根据任务类型选择策略
        strategies = {
            "memory": "spaced_repetition",
            "recall": "active_recall",
            "skill": "deliberate_practice",
            "integration": "interleaved",
            "deep": "elaborative",
            "general": "default",
        }

        strategy = strategies.get(task_type, "default")
        strategy_info = self.learning_strategies.get(
            strategy, self.learning_strategies["default"])

        # 元学习能力加成
        effectiveness = strategy_info["effectiveness"] * (
            1 + self.meta_learning["learning_to_learn"] * 0.3)

        self.meta_learning["learning_strategy"] = strategy
        self.meta_learning["strategy_effectiveness"] = effectiveness

        return {
            "task_type": task_type,
            "selected_strategy": strategy,
            "description": strategy_info["description"],
            "base_effectiveness": strategy_info["effectiveness"],
            "adjusted_effectiveness": round(effectiveness, 3),
            "learning_to_learn": round(
                self.meta_learning["learning_to_learn"], 3),
        }

    def adapt_learning_strategy(self, feedback: float) -> Dict:
        """根据反馈调整学习策略。"""
        self.meta_learning["active"] = True

        current_strategy = self.meta_learning["learning_strategy"]

        if feedback < 0.3:
            # 效果不好，换策略
            strategies = list(self.learning_strategies.keys())
            current_idx = strategies.index(current_strategy) if current_strategy in strategies else 0
            new_idx = (current_idx + 1) % len(strategies)
            new_strategy = strategies[new_idx]

            self.meta_learning["learning_strategy"] = new_strategy
            adaptation = "strategy_changed"
        else:
            # 效果好，保持策略
            new_strategy = current_strategy
            adaptation = "strategy_maintained"

        # 更新策略有效性
        self.meta_learning["strategy_effectiveness"] = self._clip(
            self.meta_learning["strategy_effectiveness"] +
            (feedback - 0.5) * 0.1, 0, 1)

        return {
            "old_strategy": current_strategy,
            "new_strategy": new_strategy,
            "adaptation": adaptation,
            "feedback": feedback,
            "strategy_effectiveness": round(
                self.meta_learning["strategy_effectiveness"], 3),
        }

    def get_meta_learning_status(self) -> Dict:
        """获取元学习状态。"""
        return {
            "active": self.meta_learning["active"],
            "meta_learning_count": self.meta_learning["meta_learning_count"],
            "current_strategy": self.meta_learning["learning_strategy"],
            "strategy_effectiveness": round(
                self.meta_learning["strategy_effectiveness"], 3),
            "learning_to_learn": round(
                self.meta_learning["learning_to_learn"], 3),
            "strategy_adaptation_rate": round(
                self.meta_learning["strategy_adaptation_rate"], 3),
            "available_strategies": list(self.learning_strategies.keys()),
        }

    # ===== 适应机制 =====

    def adapt_to_environment(self, environment_change: str) -> Dict:
        """环境适应：适应环境变化。"""
        self.adaptation_mechanism["active"] = True
        self.adaptation_mechanism["adaptation_count"] += 1

        # 适应效果
        adaptation_effect = self.adaptation_mechanism["adaptation_rate"] * \
            self.adaptation_mechanism["environment_adaptation"]

        # 提高环境适应能力
        self.adaptation_mechanism["environment_adaptation"] = self._clip(
            self.adaptation_mechanism["environment_adaptation"] + 0.01, 0, 1)

        # 提高灵活性
        self.adaptation_mechanism["flexibility"] = self._clip(
            self.adaptation_mechanism["flexibility"] + 0.01, 0, 1)

        return {
            "change": environment_change,
            "adaptation_effect": round(adaptation_effect, 3),
            "environment_adaptation": round(
                self.adaptation_mechanism["environment_adaptation"], 3),
            "flexibility": round(self.adaptation_mechanism["flexibility"], 3),
            "adaptation_count": self.adaptation_mechanism["adaptation_count"],
        }

    def adapt_to_task(self, task_difficulty: float = 0.5) -> Dict:
        """任务适应：调整到任务难度。"""
        self.adaptation_mechanism["active"] = True
        self.adaptation_mechanism["adaptation_count"] += 1

        # 根据难度调整
        if task_difficulty > 0.7:
            # 困难任务，提高注意力
            self.attention_factor = self._clip(
                self.attention_factor + 0.05, 0, 1)
            self.emotion["stress"] = self._clip(
                self.emotion["stress"] + 0.05, 0, 1)
        elif task_difficulty < 0.3:
            # 简单任务，放松
            self.emotion["calm"] = self._clip(
                self.emotion["calm"] + 0.05, 0, 1)

        # 提高任务适应能力
        self.adaptation_mechanism["task_adaptation"] = self._clip(
            self.adaptation_mechanism["task_adaptation"] + 0.01, 0, 1)

        return {
            "difficulty": task_difficulty,
            "task_adaptation": round(
                self.adaptation_mechanism["task_adaptation"], 3),
            "attention_factor": round(self.attention_factor, 3),
            "adaptation_count": self.adaptation_mechanism["adaptation_count"],
        }

    def adjust_strategy(self, performance: float) -> Dict:
        """策略调整：根据表现调整策略。"""
        self.adaptation_mechanism["active"] = True
        self.adaptation_mechanism["adaptation_count"] += 1

        # 策略调整
        if performance > 0.8:
            # 表现好，增加挑战
            self.adaptation_mechanism["strategy_adjustment"] = self._clip(
                self.adaptation_mechanism["strategy_adjustment"] + 0.02, 0, 1)
            adjustment = "increase_challenge"
        elif performance < 0.3:
            # 表现差，降低难度
            self.adaptation_mechanism["strategy_adjustment"] = self._clip(
                self.adaptation_mechanism["strategy_adjustment"] - 0.01, 0, 1)
            adjustment = "decrease_difficulty"
        else:
            adjustment = "maintain"

        return {
            "performance": performance,
            "adjustment": adjustment,
            "strategy_adjustment": round(
                self.adaptation_mechanism["strategy_adjustment"], 3),
            "adaptation_count": self.adaptation_mechanism["adaptation_count"],
        }

    def get_adaptation_status(self) -> Dict:
        """获取适应机制状态。"""
        return {
            "active": self.adaptation_mechanism["active"],
            "adaptation_count": self.adaptation_mechanism["adaptation_count"],
            "environment_adaptation": round(
                self.adaptation_mechanism["environment_adaptation"], 3),
            "task_adaptation": round(
                self.adaptation_mechanism["task_adaptation"], 3),
            "strategy_adjustment": round(
                self.adaptation_mechanism["strategy_adjustment"], 3),
            "adaptation_rate": round(
                self.adaptation_mechanism["adaptation_rate"], 3),
            "flexibility": round(self.adaptation_mechanism["flexibility"], 3),
        }

    # ===== 终身学习报告 =====

    def get_lifelong_learning_report(self) -> Dict:
        """获取终身学习系统状态报告。"""
        return {
            "continual_learning": self.get_continual_learning_status(),
            "knowledge_consolidation": self.get_knowledge_consolidation_status(),
            "skill_improvement": self.get_skill_improvement_status(),
            "meta_learning": self.get_meta_learning_status(),
            "adaptation": self.get_adaptation_status(),
            "parameters": {
                "lifelong_learning_rate": self.lifelong_learning_rate,
                "learning_capacity": self.learning_capacity,
                "cognitive_reserve": round(self.cognitive_reserve, 3),
                "neuroplasticity_maintenance": round(
                    self.neuroplasticity_maintenance, 3),
            },
        }

    # ------------------ 意识整合系统（v7.1 Consciousness Integration） ------------------

    # ===== 意识统一框架 =====

    def update_consciousness_framework(self) -> Dict:
        """更新意识统一框架，整合各意识理论。"""
        self.consciousness_framework["active"] = True

        # 整合水平：结合GWT、NCC、HOT
        gwt_contribution = 0.3 if self.consciousness_integration["global_broadcast_active"] else 0.1
        ncc_contribution = self.consciousness_integration["neural_synchrony_level"] * 0.3
        hot_contribution = 0.3 if self.consciousness_integration["higher_order_thought"] else 0.1

        integration = (gwt_contribution + ncc_contribution + hot_contribution) * \
            (1 + self.consciousness_metrics["phi_value"] * 0.5)

        self.consciousness_framework["integration_level"] = self._clip(
            integration, 0, 1)

        # 统一水平：信息整合程度
        self.consciousness_framework["unity_level"] = self._clip(
            self.consciousness_integration["information_integration"] *
            (1 + self.consciousness_integration["binding_level"] * 0.3), 0, 1)

        # 连贯水平：意识内容的连贯性
        self.consciousness_framework["coherence_level"] = self._clip(
            self.consciousness_metrics["stability"] *
            (1 + self.consciousness_state["clarity"] * 0.3), 0, 1)

        # 时间深度：意识的时间跨度
        # 基于工作记忆容量和自传体记忆
        wm_load = len(self.phonological_loop) / max(self.phonological_capacity, 1) if hasattr(self, 'phonological_loop') else 0.3
        autobiographical = self.self_awareness.get("autobiographical_memory", 0.3) if hasattr(self, 'self_awareness') else 0.3

        self.consciousness_framework["temporal_depth"] = self._clip(
            wm_load * 0.5 +
            autobiographical * 0.3 +
            0.2, 0, 1)

        # 自我参照：意识与自我的关联
        self_concept_clarity = self.self_awareness.get("self_concept", {}).get("clarity", 0.3) if hasattr(self, 'self_awareness') else 0.3

        self.consciousness_framework["self_reference"] = self._clip(
            self_concept_clarity * 0.5 +
            self.consciousness_content["meta_awareness"] * 0.3 +
            0.2, 0, 1)

        return {
            "integration_level": round(
                self.consciousness_framework["integration_level"], 3),
            "unity_level": round(
                self.consciousness_framework["unity_level"], 3),
            "coherence_level": round(
                self.consciousness_framework["coherence_level"], 3),
            "temporal_depth": round(
                self.consciousness_framework["temporal_depth"], 3),
            "self_reference": round(
                self.consciousness_framework["self_reference"], 3),
        }

    def get_consciousness_framework_status(self) -> Dict:
        """获取意识统一框架状态。"""
        return {
            "active": self.consciousness_framework["active"],
            "integration_level": round(
                self.consciousness_framework["integration_level"], 3),
            "unity_level": round(
                self.consciousness_framework["unity_level"], 3),
            "coherence_level": round(
                self.consciousness_framework["coherence_level"], 3),
            "temporal_depth": round(
                self.consciousness_framework["temporal_depth"], 3),
            "self_reference": round(
                self.consciousness_framework["self_reference"], 3),
        }

    # ===== 意识状态管理 =====

    def set_consciousness_state(self, state: str) -> Dict:
        """设置意识状态。

        可选状态：wakeful（清醒）、drowsy（困倦）、sleep（睡眠）、
        meditation（冥想）、flow（心流）
        """
        old_state = self.consciousness_state["state"]

        if state in ["wakeful", "drowsy", "sleep", "meditation", "flow"]:
            self.consciousness_state["state"] = state
            self.consciousness_transition["transition_count"] += 1

            # 记录状态历史
            self.consciousness_state["state_history"].append({
                "from": old_state,
                "to": state,
                "tick": self.tick,
            })

            # 根据状态调整参数
            state_params = {
                "wakeful": {"clarity": 0.7, "arousal": 0.6, "depth": 0.5},
                "drowsy": {"clarity": 0.4, "arousal": 0.3, "depth": 0.3},
                "sleep": {"clarity": 0.1, "arousal": 0.1, "depth": 0.9},
                "meditation": {"clarity": 0.8, "arousal": 0.4, "depth": 0.7},
                "flow": {"clarity": 0.9, "arousal": 0.8, "depth": 0.6},
            }

            params = state_params.get(state, state_params["wakeful"])
            self.consciousness_state["clarity"] = params["clarity"]
            self.consciousness_state["arousal"] = params["arousal"]
            self.consciousness_state["depth"] = params["depth"]

        return {
            "old_state": old_state,
            "new_state": self.consciousness_state["state"],
            "transition_count": self.consciousness_transition["transition_count"],
            "clarity": round(self.consciousness_state["clarity"], 3),
            "arousal": round(self.consciousness_state["arousal"], 3),
            "depth": round(self.consciousness_state["depth"], 3),
        }

    def update_consciousness_state(self) -> Dict:
        """更新意识状态（基于当前脑活动）。"""
        # 根据脑电波和唤醒水平判断状态
        dominant_wave = self.get_dominant_wave() if hasattr(
            self, 'get_dominant_wave') else "beta"

        wave_state_map = {
            "delta": "sleep",
            "theta": "drowsy",
            "alpha": "wakeful",
            "beta": "wakeful",
            "gamma": "flow",
        }

        target_state = wave_state_map.get(dominant_wave, "wakeful")

        # 平滑过渡
        if target_state != self.consciousness_state["state"]:
            smoothness = self.consciousness_transition["transition_smoothness"]
            if random.random() < smoothness * 0.5:
                self.set_consciousness_state(target_state)

        return {
            "current_state": self.consciousness_state["state"],
            "dominant_wave": dominant_wave,
            "clarity": round(self.consciousness_state["clarity"], 3),
        }

    def get_consciousness_state_status(self) -> Dict:
        """获取意识状态。"""
        return {
            "state": self.consciousness_state["state"],
            "clarity": round(self.consciousness_state["clarity"], 3),
            "depth": round(self.consciousness_state["depth"], 3),
            "arousal": round(self.consciousness_state["arousal"], 3),
            "content_load": round(self.consciousness_state["content_load"], 3),
            "transition_count": self.consciousness_transition["transition_count"],
        }

    # ===== 意识整合机制 =====

    def conscious_ignition(self, content: str) -> Dict:
        """意识点火：内容进入意识（全局工作空间点火）。"""
        self.consciousness_integration["active"] = True
        self.consciousness_integration["ignition_count"] += 1

        # 全局广播激活
        self.consciousness_integration["global_broadcast_active"] = True

        # 神经同步增强
        self.consciousness_integration["neural_synchrony_level"] = self._clip(
            self.consciousness_integration["neural_synchrony_level"] + 0.1, 0, 1)

        # 信息整合
        self.consciousness_integration["information_integration"] = self._clip(
            self.consciousness_integration["information_integration"] + 0.05, 0, 1)

        # 更新意识内容
        self.consciousness_content["current_content"] = content
        self.consciousness_content["content_history"].append({
            "content": content,
            "tick": self.tick,
            "type": "ignition",
        })

        # 推入思考空间
        self._push_thought(
            f"[意识] {content}",
            source="consciousness",
            activation=0.9
        )

        # 更新意识框架
        self.update_consciousness_framework()

        return {
            "content": content,
            "ignition_count": self.consciousness_integration["ignition_count"],
            "global_broadcast": True,
            "neural_synchrony": round(
                self.consciousness_integration["neural_synchrony_level"], 3),
            "information_integration": round(
                self.consciousness_integration["information_integration"], 3),
        }

    def conscious_binding(self, features: list) -> Dict:
        """意识绑定：将分散的特征整合为统一的意识内容。"""
        self.consciousness_integration["active"] = True

        # γ波绑定
        binding_level = 0.3 + len(features) * 0.1
        binding_level = min(binding_level, 1.0)

        self.consciousness_integration["binding_level"] = self._clip(
            binding_level, 0, 1)

        # 整合特征
        integrated_content = " + ".join(features)

        # 神经同步
        self.consciousness_integration["neural_synchrony_level"] = self._clip(
            self.consciousness_integration["neural_synchrony_level"] + 0.05, 0, 1)

        return {
            "features": features,
            "feature_count": len(features),
            "binding_level": round(
                self.consciousness_integration["binding_level"], 3),
            "integrated_content": integrated_content,
            "neural_synchrony": round(
                self.consciousness_integration["neural_synchrony_level"], 3),
        }

    def conscious_higher_order_thought(self, thought: str) -> Dict:
        """高阶思想：对思想的思想（HOT理论）- 意识整合版本。"""
        self.consciousness_integration["active"] = True
        self.consciousness_integration["higher_order_thought"] = True

        # 元意识增强
        self.consciousness_content["meta_awareness"] = self._clip(
            self.consciousness_content["meta_awareness"] + 0.1, 0, 1)

        # 内省深度增加
        self.consciousness_content["introspection_depth"] = self._clip(
            self.consciousness_content["introspection_depth"] + 0.05, 0, 1)

        # 推入思考空间
        self._push_thought(
            f"[高阶思想] 我正在思考：{thought}",
            source="higher_order",
            activation=0.8
        )

        return {
            "original_thought": thought,
            "higher_order": True,
            "meta_awareness": round(
                self.consciousness_content["meta_awareness"], 3),
            "introspection_depth": round(
                self.consciousness_content["introspection_depth"], 3),
        }

    def get_consciousness_integration_status(self) -> Dict:
        """获取意识整合机制状态。"""
        return {
            "active": self.consciousness_integration["active"],
            "global_broadcast": self.consciousness_integration["global_broadcast_active"],
            "neural_synchrony": round(
                self.consciousness_integration["neural_synchrony_level"], 3),
            "information_integration": round(
                self.consciousness_integration["information_integration"], 3),
            "higher_order_thought": self.consciousness_integration["higher_order_thought"],
            "binding_level": round(
                self.consciousness_integration["binding_level"], 3),
            "ignition_count": self.consciousness_integration["ignition_count"],
        }

    # ===== 意识内容 =====

    def focus_attention(self, target: str) -> Dict:
        """集中注意力到特定目标。"""
        self.consciousness_content["attention_focus"] = target

        # 注意力增强意识清晰度
        self.consciousness_state["clarity"] = self._clip(
            self.consciousness_state["clarity"] + 0.05, 0, 1)

        # 内容负载增加
        self.consciousness_state["content_load"] = self._clip(
            self.consciousness_state["content_load"] + 0.1, 0, 1)

        return {
            "focus_target": target,
            "clarity": round(self.consciousness_state["clarity"], 3),
            "content_load": round(
                self.consciousness_state["content_load"], 3),
        }

    def consciousness_meta_awareness_check(self) -> Dict:
        """元意识检测：检测自己是否意识到自己的意识 - 意识整合版本。"""
        # 元意识水平受高阶思想和内省影响
        meta_level = self.consciousness_content["meta_awareness"] * 0.5 + \
            self.consciousness_content["introspection_depth"] * 0.3 + \
            self.consciousness_framework["self_reference"] * 0.2

        return {
            "meta_awareness_level": round(meta_level, 3),
            "is_meta_aware": meta_level > self.consciousness_threshold,
            "meta_awareness_base": round(
                self.consciousness_content["meta_awareness"], 3),
            "introspection_depth": round(
                self.consciousness_content["introspection_depth"], 3),
        }

    def get_consciousness_content_status(self) -> Dict:
        """获取意识内容状态。"""
        return {
            "current_content": self.consciousness_content["current_content"],
            "attention_focus": self.consciousness_content["attention_focus"],
            "meta_awareness": round(
                self.consciousness_content["meta_awareness"], 3),
            "introspection_depth": round(
                self.consciousness_content["introspection_depth"], 3),
            "content_history_count": len(
                self.consciousness_content["content_history"]),
        }

    # ===== 意识测量 =====

    def measure_consciousness_level(self) -> Dict:
        """测量意识水平。"""
        # 综合多个指标计算意识水平
        clarity = self.consciousness_state["clarity"]
        integration = self.consciousness_framework["integration_level"]
        arousal = self.consciousness_state["arousal"]
        meta = self.consciousness_content["meta_awareness"]

        # 意识水平 = 清晰度 × 整合 × 唤醒 × 元意识（加权）
        level = (clarity * 0.3 +
                 integration * 0.3 +
                 arousal * 0.2 +
                 meta * 0.2)

        self.consciousness_metrics["consciousness_level"] = self._clip(level, 0, 1)

        # 复杂度：神经复杂度 + 信息整合
        if hasattr(self, 'neural_complexity'):
            nc_result = self.neural_complexity()
            if isinstance(nc_result, dict):
                nc_value = nc_result.get("complexity", 0.5)
            else:
                nc_value = 0.5
        else:
            nc_value = 0.5

        complexity = nc_value * 0.5 + \
            self.consciousness_integration["information_integration"] * 0.5

        self.consciousness_metrics["complexity"] = self._clip(complexity, 0, 1)

        # Φ值（整合信息）
        phi = self.consciousness_integration["information_integration"] * \
            (1 + self.consciousness_framework["unity_level"] * 0.3)

        self.consciousness_metrics["phi_value"] = self._clip(phi, 0, 1)

        return {
            "consciousness_level": round(
                self.consciousness_metrics["consciousness_level"], 3),
            "complexity": round(
                self.consciousness_metrics["complexity"], 3),
            "phi_value": round(
                self.consciousness_metrics["phi_value"], 3),
            "clarity": round(clarity, 3),
            "integration": round(integration, 3),
            "arousal": round(arousal, 3),
            "meta_awareness": round(meta, 3),
        }

    def measure_consciousness_diversity(self) -> Dict:
        """测量意识多样性。"""
        # 基于思考空间内容的多样性
        if len(self.thought_space) > 0:
            sources = set(t.source for t in self.thought_space)
            diversity = len(sources) / 10.0  # 归一化
        else:
            diversity = 0.0

        self.consciousness_metrics["diversity"] = self._clip(diversity, 0, 1)

        return {
            "diversity": round(diversity, 3),
            "thought_count": len(self.thought_space),
            "source_count": len(set(t.source for t in self.thought_space)) if self.thought_space else 0,
        }

    def measure_consciousness_stability(self) -> Dict:
        """测量意识稳定性。"""
        # 基于状态历史的稳定性
        history = self.consciousness_state["state_history"]
        if len(history) > 1:
            # 状态变化越少，越稳定
            changes = sum(1 for h in history if h["from"] != h["to"])
            stability = 1.0 - (changes / len(history))
        else:
            stability = 1.0

        self.consciousness_metrics["stability"] = self._clip(stability, 0, 1)

        return {
            "stability": round(stability, 3),
            "history_length": len(history),
        }

    def get_consciousness_metrics(self) -> Dict:
        """获取意识测量指标。"""
        return {
            "consciousness_level": round(
                self.consciousness_metrics["consciousness_level"], 3),
            "complexity": round(
                self.consciousness_metrics["complexity"], 3),
            "diversity": round(
                self.consciousness_metrics["diversity"], 3),
            "stability": round(
                self.consciousness_metrics["stability"], 3),
            "flexibility": round(
                self.consciousness_metrics["flexibility"], 3),
            "phi_value": round(
                self.consciousness_metrics["phi_value"], 3),
        }

    # ===== 意识状态转换 =====

    def transition_consciousness_state(self, target_state: str) -> Dict:
        """转换意识状态（带滞后效应的相变）。"""
        current = self.consciousness_state["state"]
        threshold = self.consciousness_transition["phase_transition_threshold"]
        hysteresis = self.consciousness_transition["hysteresis"]

        # 计算转换概率
        if current == target_state:
            probability = 1.0
        else:
            # 正向转换
            base_prob = 0.3
            probability = base_prob + (threshold - 0.5) * 0.2

            # 滞后效应：从深到浅更容易，从浅到深更难
            state_depth = {
                "sleep": 0.9,
                "drowsy": 0.5,
                "wakeful": 0.3,
                "meditation": 0.6,
                "flow": 0.4,
            }

            current_depth = state_depth.get(current, 0.5)
            target_depth = state_depth.get(target_state, 0.5)

            if target_depth > current_depth:
                # 向更深状态转换，难度增加
                probability -= hysteresis
            else:
                # 向更浅状态转换，难度降低
                probability += hysteresis

        probability = max(0.1, min(0.9, probability))

        # 执行转换
        if random.random() < probability:
            result = self.set_consciousness_state(target_state)
            transitioned = True
        else:
            transitioned = False
            result = {"old_state": current, "new_state": current}

        return {
            "from": current,
            "to": target_state,
            "transitioned": transitioned,
            "probability": round(probability, 3),
            "transition_count": self.consciousness_transition["transition_count"],
        }

    def get_consciousness_transition_status(self) -> Dict:
        """获取意识状态转换状态。"""
        return {
            "transition_count": self.consciousness_transition["transition_count"],
            "transition_smoothness": round(
                self.consciousness_transition["transition_smoothness"], 3),
            "phase_transition_threshold": round(
                self.consciousness_transition["phase_transition_threshold"], 3),
            "hysteresis": round(
                self.consciousness_transition["hysteresis"], 3),
        }

    # ===== 意识整合报告 =====

    def get_consciousness_integration_report(self) -> Dict:
        """获取意识整合系统状态报告。"""
        return {
            "framework": self.get_consciousness_framework_status(),
            "state": self.get_consciousness_state_status(),
            "integration": self.get_consciousness_integration_status(),
            "content": self.get_consciousness_content_status(),
            "metrics": self.get_consciousness_metrics(),
            "transition": self.get_consciousness_transition_status(),
            "parameters": {
                "integration_rate": self.consciousness_integration_rate,
                "threshold": self.consciousness_threshold,
                "workspace_capacity": self.global_workspace_capacity,
                "temporal_window": self.consciousness_temporal_window,
            },
        }

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

    def cross_modal_recall(self, features: list, current_modality: str,
                           top_k: int = 2) -> List[BrainMemory]:
        """跨模态联想：根据特征向量，从其他模态的记忆中找相似的。

        比如：看到一张猫的图片（visual特征），想起之前听到的"喵"的声音（auditory记忆）。

        原理：计算特征向量的余弦相似度，找最相似的其他模态记忆。
        """
        if not features or len(features) == 0:
            return []

        def cosine_sim(a, b):
            """余弦相似度"""
            if not a or not b or len(a) != len(b):
                return 0.0
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        # 搜索所有记忆，找其他模态中特征相似的
        candidates = []
        for m in self.long_memory + self.short_memory:
            # 跳过同一模态的（跨模态才有意思）
            if m.modality == current_modality:
                continue
            # 跳过没有特征的
            if not m.features or len(m.features) != len(features):
                continue
            sim = cosine_sim(features, m.features)
            if sim > 0.3:  # 相似度阈值
                candidates.append((m, sim))

        # 按相似度排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        results = [m for m, sim in candidates[:top_k]]

        # 跨模态联想的记忆进入思考空间（带"联想起"的感觉）
        for m in results:
            self._push_thought(m.content, source="memory", activation=0.7)
            # 跨模态联想强化记忆
            m.weight = self._clip(m.weight + 0.08)

        return results

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

    def clear_thoughts(self) -> int:
        """清空思考空间中的所有念头。

        就像"清空大脑"、"什么都不想"一样。
        思考空间重置为空状态。

        Returns:
            被清除的念头数量
        """
        count = len(self.thought_space)
        self.thought_space.clear()

        # 记录到元认知日志
        if self.record_history:
            self.metacog_log.append({
                "type": "clear_thoughts",
                "tick": self.tick,
                "cleared_count": count,
            })

        return count

    def remove_thought(self, index: int = -1) -> Optional[Dict]:
        """删除思考空间中的指定念头。

        参数：
            index: 要删除的念头索引（从1开始，负数表示从后往前数）
                   -1 表示删除最后一个（激活度最低的）
                   -2 表示删除倒数第二个
                   1 表示删除第一个

        Returns:
            被删除的念头信息，失败返回 None
        """
        if not self.thought_space:
            return None

        # 转换索引（用户输入从1开始，内部从0开始）
        if index > 0:
            idx = index - 1
        elif index < 0:
            idx = index  # 负数直接用（Python 列表支持负数索引）
        else:
            return None

        # 检查索引范围
        if idx >= len(self.thought_space) or idx < -len(self.thought_space):
            return None

        # 删除念头
        removed = self.thought_space.pop(idx)

        # 记录到元认知日志
        if self.record_history:
            self.metacog_log.append({
                "type": "remove_thought",
                "tick": self.tick,
                "removed_content": removed.content,
                "removed_source": removed.source,
                "removed_activation": round(removed.activation, 3),
            })

        return {
            "content": removed.content,
            "source": removed.source,
            "activation": removed.activation,
            "birth_tick": removed.birth_tick,
        }

    def remove_weakest_thought(self) -> Optional[Dict]:
        """删除激活度最低的念头。

        就像"忘记最不重要的事情"一样。

        Returns:
            被删除的念头信息
        """
        if not self.thought_space:
            return None

        # 找到激活度最低的念头
        weakest_idx = min(range(len(self.thought_space)),
                          key=lambda i: self.thought_space[i].activation)

        removed = self.thought_space.pop(weakest_idx)

        # 记录到元认知日志
        if self.record_history:
            self.metacog_log.append({
                "type": "remove_weakest_thought",
                "tick": self.tick,
                "removed_content": removed.content,
                "removed_activation": round(removed.activation, 3),
            })

        return {
            "content": removed.content,
            "source": removed.source,
            "activation": removed.activation,
            "birth_tick": removed.birth_tick,
        }

    def introspect(self, depth: str = "basic") -> Dict:
        """思考感官（内感觉 interoception）：感知自己的脑活动。

        读取主导情绪、三层脉冲数、记忆占用、思考空间焦点，生成内省
        言语并把它作为内部刺激回注网络（自我感知回路），同时记入
        metacog_log 元认知日志。外部感官看世界，思考感官看自己。

        depth:
            - "basic": 基础内省（情绪 + 当前念头）
            - "deep": 深度内省（完整思考空间 + 记忆分布 + 自我认知）
        """
        s, a, d = self.spike_counts()
        mood = max(self.emotion, key=lambda k: self.emotion[k])
        mood_cn = {"calm": "平静", "curiosity": "好奇",
                   "stress": "紧张", "pleasure": "愉悦"}.get(mood, mood)
        top = self.top_thought()
        top_content = top.content if top else "（空）"

        if depth == "basic":
            text = (f"我感到{mood_cn}，正在想「{top_content}」，"
                    f"脉冲活动{s}/{a}/{d}，"
                    f"记忆{len(self.short_memory)}/{len(self.long_memory)}")
        else:
            # 深度内省：感知更多内部状态
            thought_count = len(self.thought_space)
            novelty_level = round(self.novelty, 2)
            attention_level = round(self.attention_factor, 2)

            # 记忆模态分布
            modality_dist = {}
            for m in self.long_memory:
                modality_dist[m.modality] = modality_dist.get(m.modality, 0) + 1
            mod_desc = "、".join(f"{k}:{v}" for k, v in modality_dist.items()) \
                if modality_dist else "无"

            # 思考空间内容摘要
            thoughts_summary = "、".join(
                t.content[:8] for t in self.thought_space[:3])
            if thought_count > 3:
                thoughts_summary += f"...等{thought_count}个念头"

            text = (f"我是{self.name}，感到{mood_cn}。"
                    f"正在想「{top_content}」。"
                    f"思考空间有{thoughts_summary}。"
                    f"新奇度{novelty_level}，注意力{attention_level}。"
                    f"记忆：短期{len(self.short_memory)}条，"
                    f"长期{len(self.long_memory)}条（{mod_desc}）。"
                    f"神经活动：感官{s}/联想{a}/决策{d}。")

        # 内省言语回注网络（强度 0.5 倍），并作为元认知念头入思考空间
        self.tick += 1
        self._network_step([c * 0.5 for c in self._str_to_current(text)])
        self._push_thought(text, source="metacog")

        entry = {"tick": self.tick, "mood": mood, "top_thought": top_content,
                 "spike_counts": [s, a, d],
                 "stm": len(self.short_memory),
                 "ltm": len(self.long_memory),
                 "thought_space_size": len(self.thought_space),
                 "novelty": round(self.novelty, 3),
                 "attention": round(self.attention_factor, 3),
                 "depth": depth,
                 "text": text}
        self.metacog_log.append(entry)
        if self.record_history:
            self._record()
        return entry

    def self_reflect(self, focus: str = "general") -> Dict:
        """自我反思：对自己的思维、情绪、行为进行反思（v5.3 新增）。

        这是更高阶的元认知：不只是感知自己的状态，
        而是对自己的状态进行评价和思考。

        focus:
            - "general": 整体反思
            - "thought": 反思自己的思考过程
            - "emotion": 反思自己的情绪
            - "memory": 反思自己的记忆
        """
        s, a, d = self.spike_counts()
        mood = max(self.emotion, key=lambda k: self.emotion[k])
        mood_cn = {"calm": "平静", "curiosity": "好奇",
                   "stress": "紧张", "pleasure": "愉悦"}.get(mood, mood)

        if focus == "thought":
            # 反思思考过程
            n_thoughts = len(self.thought_space)
            top = self.top_thought()
            top_content = top.content if top else "（空）"
            text = (f"我在想「{top_content}」。"
                    f"思考空间里有{n_thoughts}个念头。"
                    f"我的思绪是{'集中' if self.attention_factor > 0.6 else '分散'}的。")
        elif focus == "emotion":
            # 反思情绪
            emotion_values = {k: round(v, 2) for k, v in self.emotion.items()}
            text = (f"我现在感到{mood_cn}。"
                    f"情绪状态：{emotion_values}。"
                    f"这种情绪是{'合适的' if self.emotion[mood] < 0.8 else '有点强烈'}。")
        elif focus == "memory":
            # 反思记忆
            n_stm = len(self.short_memory)
            n_ltm = len(self.long_memory)
            text = (f"我有{n_stm}条短期记忆，{n_ltm}条长期记忆。"
                    f"我记得{'很多事情' if n_ltm > 10 else '一些事情'}。"
                    f"我的记忆{'很清晰' if n_ltm > 50 else '还在积累中'}。")
        else:
            # 整体反思
            n_self_concept = len(self.self_concept)
            n_auto = len(self.autobiographical_memory)
            text = (f"我是{self.name}。"
                    f"我感到{mood_cn}。"
                    f"我有{n_self_concept}条自我认知，{n_auto}段重要经历。"
                    f"我正在不断学习和成长。")

        # 反思内容回注网络，并作为元认知念头
        self.tick += 1
        self._network_step([c * 0.4 for c in self._str_to_current(text)])
        self._push_thought(text, source="metacog")

        # 记录反思日志
        reflection = {
            "tick": self.tick,
            "focus": focus,
            "mood": mood,
            "text": text,
            "thought_count": len(self.thought_space),
            "memory_count": len(self.long_memory),
        }
        self.metacog_log.append({"type": "reflection", **reflection})

        return reflection

    # ------------------ 高阶意识理论 HOT（v5.7） ------------------

    def higher_order_thought(self, level: int = 1) -> Dict:
        """生成高阶思想（Higher-Order Thought）。

        高阶意识理论认为：一个心理状态是有意识的，
        当且仅当我们有一个关于它的高阶思想（HOT）。

        一阶思想："我看到红色"
        二阶思想："我知道我看到红色"
        三阶思想："我知道我知道我看到红色"
        ...

        参数：
            level: 高阶思想的层级（1=一阶, 2=二阶, 3=三阶...）

        返回：
            hot: 高阶思想内容
            level: 层级
            base_thought: 基础思想（被反思的思想）
        """
        # 获取当前的主导念头（基础思想）
        top = self.top_thought()
        base_thought = top.content if top else "（空）"
        base_source = top.source if top else "unknown"

        if level <= 0:
            # 零阶：无意识的心理状态
            return {
                "level": 0,
                "hot": base_thought,
                "base_thought": base_thought,
                "conscious": False,
            }

        # 生成高阶思想
        if level == 1:
            # 一阶：基本意识
            hot = f"我意识到「{base_thought}」"
        elif level == 2:
            # 二阶：元意识
            hot = f"我知道我意识到「{base_thought}」"
        elif level == 3:
            # 三阶：对元意识的意识
            hot = f"我知道我知道我意识到「{base_thought}」"
        else:
            # 更高阶（简化处理）
            hot = f"我{'知道' * level}我意识到「{base_thought}」"

        # 高阶思想进入思考空间
        self._push_thought(hot, source=f"hot_level_{level}")

        # 更新当前意识层级
        if level > self.hot_level:
            self.hot_level = level

        # 记录高阶思想历史
        hot_record = {
            "tick": self.tick,
            "level": level,
            "hot": hot,
            "base_thought": base_thought,
            "base_source": base_source,
        }
        self.hot_history.append(hot_record)
        if len(self.hot_history) > 50:
            self.hot_history.pop(0)

        # 更新元意识程度
        self.meta_awareness = min(1.0, self.meta_awareness + 0.1 * level)

        return {
            "level": level,
            "hot": hot,
            "base_thought": base_thought,
            "base_source": base_source,
            "conscious": True,
        }

    def meta_awareness_check(self) -> Dict:
        """元意识检测：检查当前是否有元意识。

        元意识 = 对意识的意识
        当你知道你在想什么时，你就有了元意识。

        返回：
            has_meta_awareness: 是否有元意识
            meta_awareness_level: 元意识程度 [0, 1]
            current_hot_level: 当前最高的HOT层级
            evidence: 元意识的证据
        """
        # 检查思考空间中是否有高阶思想
        hot_thoughts = [t for t in self.thought_space
                        if t.source.startswith("hot_level_")]

        # 检查元认知日志
        metacog_entries = [e for e in self.metacog_log
                           if e.get("type") in ["introspection", "reflection"]]

        # 元意识程度
        level = self.meta_awareness
        if hot_thoughts:
            level = min(1.0, level + 0.2)
        if metacog_entries:
            level = min(1.0, level + 0.1)

        # 当前最高HOT层级
        max_level = 0
        for t in hot_thoughts:
            try:
                l = int(t.source.split("_")[-1])
                max_level = max(max_level, l)
            except (ValueError, IndexError):
                pass

        # 是否有元意识（至少有二阶思想）
        has_meta = max_level >= 2 or level > 0.5

        # 证据列表
        evidence = []
        if hot_thoughts:
            evidence.append(f"有{len(hot_thoughts)}个高阶思想")
        if metacog_entries:
            evidence.append(f"有{len(metacog_entries)}条元认知记录")
        if self.hot_level >= 2:
            evidence.append(f"最高HOT层级为{self.hot_level}")

        return {
            "has_meta_awareness": has_meta,
            "meta_awareness_level": round(level, 3),
            "current_hot_level": max(max_level, self.hot_level),
            "n_hot_thoughts": len(hot_thoughts),
            "n_metacog_entries": len(metacog_entries),
            "evidence": evidence,
        }

    def introspection_hierarchy(self, max_depth: int = 3) -> Dict:
        """内省层级：多层级的内省。

        从一阶到高阶，逐层深入地观察自己的意识。
        就像剥洋葱一样，一层一层地剥开意识的层次。

        参数：
            max_depth: 最大内省深度

        返回：
            levels: 各层级的内省结果
            max_depth: 达到的最大深度
            hierarchy: 层级描述
        """
        levels = []

        for depth in range(1, max_depth + 1):
            hot = self.higher_order_thought(level=depth)
            levels.append({
                "depth": depth,
                "thought": hot["hot"],
                "base": hot["base_thought"],
            })

        # 层级描述
        hierarchy = [
            "一阶：基本意识（我看到/我想到...）",
            "二阶：元意识（我知道我看到/我知道我想到...）",
            "三阶：对元意识的意识（我知道我知道...）",
            "四阶及以上：更深的反思...",
        ]

        return {
            "levels": levels,
            "max_depth": max_depth,
            "hierarchy": hierarchy[:max_depth],
            "final_level": levels[-1]["thought"] if levels else "",
        }

    def consciousness_level(self) -> Dict:
        """意识层级测量：量化当前的意识层级。

        意识不是非有即无的，而是有不同的层级：
        - 无意识（0级）
        - 一阶意识（1级）：基本感知
        - 二阶意识（2级）：元意识
        - 三阶意识（3级）：对元意识的意识
        - ...

        返回：
            level: 意识层级（0-5）
            level_name: 层级名称
            description: 层级描述
            score: 意识程度得分 [0, 1]
        """
        # 计算意识层级得分
        score = 0.0

        # 基础意识：思考空间非空
        if self.thought_space:
            score += 0.2

        # 点火效应：有意识内容
        if self.ignition_state:
            score += 0.2

        # 全局广播：信息全局可用
        if self.broadcast_history:
            score += 0.1

        # 元认知：有内省记录
        if self.metacog_log:
            score += 0.15

        # 高阶思想：有HOT
        meta = self.meta_awareness_check()
        score += meta["meta_awareness_level"] * 0.25

        # 自我反思：有自我反思能力
        if self.self_concept:
            score += 0.1

        # 确定层级
        if score < 0.2:
            level = 0
            level_name = "无意识"
            description = "没有意识体验"
        elif score < 0.4:
            level = 1
            level_name = "一阶意识"
            description = "基本的感知和体验"
        elif score < 0.6:
            level = 2
            level_name = "二阶意识（元意识）"
            description = "知道自己在想什么"
        elif score < 0.8:
            level = 3
            level_name = "三阶意识"
            description = "知道自己知道自己在想什么"
        else:
            level = 4
            level_name = "高阶意识"
            description = "深度的自我反思"

        return {
            "level": level,
            "level_name": level_name,
            "description": description,
            "score": round(score, 3),
            "max_hot_level": self.hot_level,
            "thought_space_size": len(self.thought_space),
            "ignition": self.ignition_state,
            "meta_awareness": round(self.meta_awareness, 3),
        }

    def get_hot_report(self) -> Dict:
        """获取高阶意识（HOT）状态报告。

        这是对意识层级的完整分析，
        就像在做"意识的脑电图"。
        """
        meta = self.meta_awareness_check()
        level = self.consciousness_level()

        return {
            "tick": self.tick,
            "name": self.name,
            "consciousness_level": level["level"],
            "consciousness_level_name": level["level_name"],
            "consciousness_score": level["score"],
            "description": level["description"],
            "meta_awareness": meta["meta_awareness_level"],
            "has_meta_awareness": meta["has_meta_awareness"],
            "max_hot_level": max(self.hot_level, meta["current_hot_level"]),
            "n_hot_thoughts": meta["n_hot_thoughts"],
            "hot_history_count": len(self.hot_history),
            "thought_space_size": len(self.thought_space),
            "ignition_state": self.ignition_state,
            "broadcast_count": len(self.broadcast_history),
            "metacog_count": len(self.metacog_log),
        }

    def update_self_concept(self, belief: str):
        """更新自我概念：添加一条关于"我是谁"的信念。

        自我概念是一个人对自己的认知："我是聪明的"、"我是善良的"等。
        这些信念会影响感知、思考和行为。
        """
        if belief not in self.self_concept:
            self.self_concept.append(belief)
            # 自我概念也写入长期记忆
            self._write_stm(f"我认为：{belief}", tag="self")
            return True
        return False

    def add_autobiographical_memory(self, event: str, emotion: str = "neutral",
                                    importance: float = 0.5):
        """添加自传体记忆：记录一段重要的个人经历。

        自传体记忆构成了"我是谁"的故事——
        我的过去、我的经历、我的成长。
        """
        memory = {
            "tick": self.tick,
            "event": event,
            "emotion": emotion,
            "importance": importance,
        }
        self.autobiographical_memory.append(memory)
        # 重要经历写入长期记忆
        self._write_stm(f"我记得：{event}", tag="autobiographical")
        return memory

    def get_self_summary(self) -> Dict:
        """获取自我摘要：整合自我概念和自传体记忆。

        这就是"我是谁"的完整答案。
        """
        return {
            "name": self.name,
            "generation": self.generation,
            "self_concept": self.self_concept,
            "autobiographical_count": len(self.autobiographical_memory),
            "key_memories": [m["event"] for m in
                             sorted(self.autobiographical_memory,
                                    key=lambda m: m["importance"],
                                    reverse=True)[:5]],
            "mood": max(self.emotion, key=lambda k: self.emotion[k]),
            "ltm_count": len(self.long_memory),
            "thought_count": len(self.thought_space),
        }

    def stream_of_consciousness(self, steps: int = 5,
                                daydream: float = 0.3) -> Dict:
        """意识流：自由联想、白日梦、灵感闪现（v5.3 增强版）。

        大脑在没有外部输入时，思绪自发流动：
        1. 从当前念头出发，自由联想
        2. 一个念头引出另一个念头，形成链条
        3. 偶尔有灵感闪现（两个概念的新组合）
        4. 白日梦模式：思绪更容易飘走

        v5.3 增强：
        - 心境一致性效应：情绪影响思绪流向
        - 注意力影响：注意力高时联想更聚焦
        - 链式联想：多步联想，语义距离衰减

        参数:
            steps: 意识流步数
            daydream: 白日梦程度 [0,1]，越高越容易走神

        返回:
            chain: 意识流链条（念头列表）
            insights: 灵感闪现的内容
            final_thought: 最终停留在哪个念头上
            emotional_shifts: 情绪变化次数
        """
        import random
        chain = []
        insights = []
        emotional_shifts = 0

        # 获取当前主导情绪
        current_mood = max(self.emotion, key=lambda k: self.emotion[k])

        for i in range(steps):
            # 取当前最活跃的念头
            top = self.top_thought()
            if top is None:
                # 思考空间空了，从记忆里随机捞一个
                all_mem = self.long_memory + self.short_memory
                if all_mem:
                    # 心境一致性：优先选情绪一致的记忆
                    mem = self._mood_consistent_pick(all_mem, current_mood)
                    self._push_thought(mem.content, source="memory",
                                       activation=0.6)
                    chain.append(f"[浮现] {mem.content}")
                    continue
                else:
                    break

            current = top.content
            chain.append(current)

            # 降低当前念头的激活度（让它逐渐淡出，给其他念头机会）
            top.activation *= 0.5

            # 注意力影响：注意力越低，越容易走神
            effective_daydream = daydream * (1.5 - self.attention_factor)

            # 白日梦：有概率跳到完全不相关的记忆
            if random.random() < effective_daydream:
                all_mem = self.long_memory + self.short_memory
                if all_mem:
                    mem = random.choice(all_mem)
                    if mem.content != current:
                        self._push_thought(mem.content, source="memory",
                                           activation=0.7)
                        chain.append(f"[走神] → {mem.content}")
                        # 检测情绪变化
                        if mem.emotion != "neutral" and mem.emotion != current_mood:
                            emotional_shifts += 1
                        continue

            # 正常联想：从当前念头出发，回忆相关记忆
            recalled = []
            # 尝试用关键词联想
            words = current.replace("，", " ").replace("。", " ").split()
            if words:
                # 注意力高时，选最相关的词；注意力低时，随机选
                if self.attention_factor > 0.6:
                    # 选最长的词（通常更有意义）
                    keyword = max(words, key=len)
                else:
                    keyword = random.choice(words)
                if len(keyword) >= 2:
                    recalled = self.recall(keyword, top_k=3)

            # 心境一致性：优先联想情绪一致的记忆
            if recalled:
                # 按情绪一致性排序
                mood_consistent = [m for m in recalled
                                   if m.emotion == current_mood]
                mood_neutral = [m for m in recalled if m.emotion == "neutral"]
                mood_inconsistent = [m for m in recalled
                                     if m.emotion not in (current_mood, "neutral")]
                # 重新排序：情绪一致 > 中性 > 不一致
                sorted_recalled = mood_consistent + mood_neutral + mood_inconsistent
                recalled = sorted_recalled

            # 如果联想起了新记忆，用它作为下一个念头
            new_thought = None
            for m in recalled:
                if m.content != current:
                    new_thought = m
                    break

            if new_thought:
                # 链式联想：激活度随距离衰减
                activation = 0.8 * (0.9 ** i)
                self._push_thought(new_thought.content, source="memory",
                                   activation=activation)
                chain.append(f"[联想] → {new_thought.content}")
                # 检测情绪变化
                if (new_thought.emotion != "neutral"
                        and new_thought.emotion != current_mood):
                    emotional_shifts += 1
                    current_mood = new_thought.emotion
                continue

            # 灵感闪现：有概率把两个记忆组合成新想法
            all_mem = self.long_memory + self.short_memory
            if (random.random() < 0.15 and len(all_mem) >= 2):
                # 灵感更容易在注意力不集中时出现（发散思维）
                inspiration_bonus = 1.0 - self.attention_factor
                if random.random() < 0.5 + inspiration_bonus * 0.5:
                    m1 = random.choice(all_mem)
                    m2 = random.choice(all_mem)
                    if m1.content != m2.content:
                        insight = f"{m1.content} + {m2.content}"
                        insights.append(insight)
                        self._push_thought(insight, source="internal",
                                           activation=0.9)
                        chain.append(f"[灵感!] {insight}")
                        continue

            # 衰减思考空间，让旧念头淡出
            self._decay_thoughts()

            # 网络自由演化一步
            self.tick += 1
            self._network_step()

        # 最终状态
        final = self.top_thought()
        final_content = final.content if final else None

        return {
            "chain": chain,
            "insights": insights,
            "final_thought": final_content,
            "thought_space_size": len(self.thought_space),
            "daydream_level": daydream,
            "emotional_shifts": emotional_shifts,
            "final_mood": current_mood,
        }

    def _mood_consistent_pick(self, memories: list, mood: str):
        """心境一致性：优先选择情绪与当前心境一致的记忆。

        积极情绪时更容易想到积极记忆，消极情绪时更容易想到消极记忆。
        """
        import random
        # 按情绪分类
        consistent = [m for m in memories if m.emotion == mood]
        neutral = [m for m in memories if m.emotion == "neutral"]
        inconsistent = [m for m in memories
                        if m.emotion not in (mood, "neutral")]

        # 概率权重：一致的 60%，中性 30%，不一致 10%
        r = random.random()
        if consistent and r < 0.6:
            return random.choice(consistent)
        elif neutral and r < 0.9:
            return random.choice(neutral)
        elif inconsistent:
            return random.choice(inconsistent)
        else:
            return random.choice(memories)

    # ------------------ 社交互动（v5.2） ------------------

    def send_message(self, other_brain: 'AIBrainEntity', message: str) -> Dict:
        """向另一个大脑发送消息。

        发送方：消息进入自己的思考空间（"我说了什么"）
        接收方：调用 receive_message 接收
        """
        # 自己也会意识到自己说了什么
        self._push_thought(f"对{other_brain.name}说：{message}",
                           source="internal")
        # 对方接收
        result = other_brain.receive_message(self, message)
        return {"sent": message, "to": other_brain.name,
                "reaction": result}

    def receive_message(self, sender: 'AIBrainEntity', message: str) -> Dict:
        """接收另一个大脑的消息（v5.3 增强：情感传染）。

        消息作为外部输入进入感知流水线，
        同时记住"是谁说的"（社交记忆）。

        v5.3 增强：
        - 情感传染：发送方的情绪会影响接收方
        - 共情：能感知对方的情绪状态
        """
        # 消息内容进入感知
        output = self.sensory_input(message, modality="text")
        # 标记来源：来自某个大脑
        self._push_thought(f"{sender.name}说：{message}",
                           source="social")
        # 形成社交记忆：谁对我说了什么
        social_mem = f"{sender.name}告诉我{message}"
        self._write_stm(social_mem, tag="social")

        # 情感传染：发送方的情绪会影响接收方
        sender_mood = max(sender.emotion, key=lambda k: sender.emotion[k])
        sender_mood_intensity = sender.emotion[sender_mood]
        # 传染强度：取决于关系亲密度（这里简化为0.3）
        contagion_strength = 0.3 * sender_mood_intensity
        if sender_mood in self.emotion:
            self.emotion[sender_mood] = self._clip(
                self.emotion[sender_mood] + contagion_strength)

        # 共情：感知对方的情绪
        empathy_text = f"我感觉到{sender.name}的情绪是{sender_mood}"
        self._push_thought(empathy_text, source="social")

        # v5.6 心智理论：更新对发送方的心理模型
        self._update_mental_model(sender, {
            "said": message,
            "emotion": dict(sender.emotion),
            "action": "说话",
        })

        return {
            "from": sender.name,
            "message": message,
            "output": output,
            "novelty": round(self.novelty, 3),
            "emotion": dict(self.emotion),
            "sender_mood": sender_mood,
            "emotional_contagion": round(contagion_strength, 3),
        }

    def social_learn(self, other_brain: 'AIBrainEntity',
                     n_memories: int = 3) -> Dict:
        """从另一个大脑学习：复制部分记忆（文化传播）。

        就像人从别人那里学到知识、价值观、故事一样。
        优先学习对方权重最高的记忆（最重要的知识）。
        同时从 LTM 和 STM 中选取。
        """
        # 取对方权重最高的记忆（LTM + STM 合并排序）
        all_memories = list(other_brain.long_memory) + \
                       list(other_brain.short_memory)
        all_memories.sort(key=lambda m: m.weight, reverse=True)
        learned = []

        for mem in all_memories[:n_memories]:
            # 检查是否已经知道了
            already_known = any(m.content == mem.content
                                for m in self.long_memory)
            if not already_known:
                # 复制记忆（权重打个折，因为是听来的，不是亲身经历）
                self._write_stm(mem.content, tag="culture",
                                modality=mem.modality,
                                features=mem.features)
                # 直接固化进 LTM（重要的文化知识）
                if len(self.short_memory) > 0:
                    new_mem = self.short_memory[-1]
                    new_mem.weight = mem.weight * 0.7  # 二手记忆权重低一些
                    self._consolidate_to_ltm(new_mem)
                learned.append(mem.content)

        # 学习后会感到愉悦（获得新知识）
        self.emotion["pleasure"] = self._clip(
            self.emotion["pleasure"] + 0.1 * len(learned))

        return {
            "learned_from": other_brain.name,
            "learned_count": len(learned),
            "learned": learned,
            "total_ltm": len(self.long_memory),
        }

    def chat_with(self, other_brain: 'AIBrainEntity',
                  turns: int = 3) -> List[Dict]:
        """和另一个大脑对话（多轮交流）。

        两个大脑轮流说话，每方说 turns 轮。
        对话内容会进入双方的思考空间，形成共同记忆。
        """
        conversation = []
        current_speaker = self
        current_listener = other_brain

        for i in range(turns * 2):
            # 说话方：基于当前思考空间生成一句话
            top = current_speaker.top_thought()
            if top:
                # 简单的对话生成：提取当前念头的关键词
                msg = top.content[:20]  # 简化：直接用当前念头的前20字
            else:
                msg = "..."

            # 发送消息
            result = current_speaker.send_message(current_listener, msg)
            conversation.append({
                "turn": i + 1,
                "speaker": current_speaker.name,
                "message": msg,
            })

            # 交换角色
            current_speaker, current_listener = \
                current_listener, current_speaker

        return conversation

    # ------------------ 心智理论 ToM（v5.6） ------------------

    def _get_mental_model(self, other_brain: 'AIBrainEntity') -> Dict:
        """获取（或创建）对另一个大脑的心理模型。

        心理模型包含：
        - beliefs: 对方相信什么
        - desires: 对方想要什么
        - intentions: 对方的意图
        - emotions: 对方的情绪状态
        - knowledge: 对方知道什么
        """
        name = other_brain.name
        if name not in self.mental_models:
            # 创建初始心理模型
            self.mental_models[name] = {
                "name": name,
                "beliefs": [],        # 信念列表
                "desires": [],        # 欲望列表
                "intentions": [],     # 意图列表
                "emotions": {},       # 情绪状态
                "knowledge": [],      # 知识/记忆
                "interaction_count": 0,  # 互动次数
                "last_updated": self.tick,
            }
        return self.mental_models[name]

    def _update_mental_model(self, other_brain: 'AIBrainEntity',
                             observation: Dict):
        """根据观察更新对另一个大脑的心理模型。

        这就是"读心"的过程：根据对方的行为、言语、表情等，
        推断对方的心理状态。
        """
        model = self._get_mental_model(other_brain)
        model["interaction_count"] += 1
        model["last_updated"] = self.tick

        # 更新情绪模型（从观察中推断）
        if "emotion" in observation:
            model["emotions"] = observation["emotion"]

        # 更新信念模型（从对方说的话中推断）
        if "said" in observation:
            said = observation["said"]
            # 假设对方说的话就是对方相信的（简化版）
            if said not in model["beliefs"]:
                model["beliefs"].append(said)
            if len(model["beliefs"]) > 20:
                model["beliefs"].pop(0)

        # 更新意图模型（从对方的行为中推断）
        if "action" in observation:
            action = observation["action"]
            # 简单的意图推断：从行为反推意图
            intention = f"想要{action}"
            if intention not in model["intentions"]:
                model["intentions"].append(intention)
            if len(model["intentions"]) > 10:
                model["intentions"].pop(0)

        return model

    def attribute_beliefs(self, other_brain: 'AIBrainEntity') -> Dict:
        """信念归因：推断另一个大脑相信什么。

        心智理论的核心能力之一：理解别人有不同于自己的信念。

        返回：
            beliefs: 推断出的对方信念列表
            belief_count: 信念数量
            confidence: 推断的置信度
        """
        model = self._get_mental_model(other_brain)

        # 如果互动次数少，置信度低
        confidence = min(0.9, 0.3 + model["interaction_count"] * 0.1)

        return {
            "other": other_brain.name,
            "beliefs": model["beliefs"],
            "belief_count": len(model["beliefs"]),
            "confidence": round(confidence, 2),
            "interaction_count": model["interaction_count"],
        }

    def infer_intention(self, other_brain: 'AIBrainEntity',
                        action: str = "") -> Dict:
        """意图理解：推断另一个大脑想要什么、打算做什么。

        从对方的行为反推对方的意图和目标。

        返回：
            intentions: 推断出的意图列表
            most_likely: 最可能的意图
            confidence: 推断的置信度
        """
        model = self._get_mental_model(other_brain)

        # 如果有具体的行为，做更具体的意图推断
        if action:
            # 简单的规则：行为 → 意图
            intention = f"想要{action}"
            if intention not in model["intentions"]:
                model["intentions"].append(intention)

        # 最可能的意图 = 最近的意图
        most_likely = model["intentions"][-1] if model["intentions"] else "未知"

        # 置信度随互动次数增加
        confidence = min(0.85, 0.2 + model["interaction_count"] * 0.08)

        return {
            "other": other_brain.name,
            "intentions": model["intentions"],
            "most_likely": most_likely,
            "confidence": round(confidence, 2),
            "n_intentions": len(model["intentions"]),
        }

    def perspective_taking(self, other_brain: 'AIBrainEntity',
                           situation: str = "") -> Dict:
        """视角采择：从另一个大脑的角度看问题。

        站在对方的立场上，理解对方的感受和想法。
        这是共情的基础。

        返回：
            perspective: 对方的视角
            estimated_emotion: 估计对方的情绪
            estimated_thought: 估计对方在想什么
            empathy_level: 共情程度
        """
        model = self._get_mental_model(other_brain)

        # 估计对方的情绪
        if model["emotions"]:
            estimated_emotion = max(model["emotions"],
                                    key=lambda k: model["emotions"][k])
        else:
            estimated_emotion = "未知"

        # 估计对方在想什么（用对方的信念来模拟）
        if model["beliefs"]:
            estimated_thought = model["beliefs"][-1]
        else:
            estimated_thought = "（不知道对方在想什么）"

        # 共情程度：互动越多，共情越深
        empathy_level = min(1.0, 0.2 + model["interaction_count"] * 0.05)

        # 构建对方的视角
        perspective = (
            f"如果我是{other_brain.name}，"
            f"我可能会感到{estimated_emotion}，"
            f"我可能在想「{estimated_thought}」"
        )

        return {
            "other": other_brain.name,
            "perspective": perspective,
            "estimated_emotion": estimated_emotion,
            "estimated_thought": estimated_thought,
            "empathy_level": round(empathy_level, 2),
            "situation": situation,
        }

    def false_belief_task(self, other_brain: 'AIBrainEntity',
                          true_location: str,
                          other_sees_change: bool = False) -> Dict:
        """错误信念任务：经典的心智理论测试。

        经典的Sally-Anne测试：
        - Sally把球放在篮子里，然后离开
        - Anne把球移到盒子里
        - Sally回来后，会去哪里找球？

        有ToM的人会说：篮子里（因为Sally相信球在那里）
        没有ToM的人会说：盒子里（因为球真的在那里）

        参数：
            true_location: 物体的真实位置
            other_sees_change: 对方是否看到了位置变化

        返回：
            predicted_belief: 预测对方相信的位置
            correct: 是否正确通过错误信念任务
            reasoning: 推理过程
        """
        model = self._get_mental_model(other_brain)

        if other_sees_change:
            # 对方看到了变化，所以对方的信念是正确的
            predicted_belief = true_location
            reasoning = f"{other_brain.name}看到了位置变化，所以TA知道物体在{true_location}"
        else:
            # 对方没看到变化，所以对方仍然相信原来的位置
            # （简化：假设原来的位置是"篮子"，真实位置是"盒子"）
            if true_location == "盒子":
                predicted_belief = "篮子"
            else:
                predicted_belief = "原来的位置"
            reasoning = (f"{other_brain.name}没看到位置变化，"
                         f"所以TA仍然相信物体在{predicted_belief}")

        # 是否正确通过错误信念任务
        # （正确 = 能理解对方可能有错误的信念）
        correct = not other_sees_change  # 能理解对方有错误信念就算通过

        return {
            "task": "错误信念任务（Sally-Anne测试）",
            "other": other_brain.name,
            "true_location": true_location,
            "other_sees_change": other_sees_change,
            "predicted_belief": predicted_belief,
            "correct": correct,
            "reasoning": reasoning,
            "tom_passed": correct,
        }

    def theory_of_mind(self, other_brain: 'AIBrainEntity') -> Dict:
        """完整的心智理论推理。

        综合信念归因、意图理解、视角采择等能力，
        对另一个大脑的心理状态进行全面推断。

        返回：
            mental_model: 完整的心理模型
            beliefs: 对方的信念
            intentions: 对方的意图
            emotions: 对方的情绪
            perspective: 对方的视角
            tom_level: 心智理论水平（0-1）
        """
        # 获取心理模型
        model = self._get_mental_model(other_brain)

        # 信念归因
        beliefs = self.attribute_beliefs(other_brain)

        # 意图理解
        intentions = self.infer_intention(other_brain)

        # 视角采择
        perspective = self.perspective_taking(other_brain)

        # 心智理论水平（综合评估）
        # 基于互动次数、信念数量、意图数量等
        tom_level = min(1.0, (
            model["interaction_count"] * 0.05
            + len(model["beliefs"]) * 0.02
            + len(model["intentions"]) * 0.03
            + perspective["empathy_level"] * 0.3
        ))

        return {
            "other": other_brain.name,
            "mental_model": model,
            "beliefs": beliefs,
            "intentions": intentions,
            "perspective": perspective,
            "tom_level": round(tom_level, 2),
            "interaction_count": model["interaction_count"],
        }

    def empathize(self, other_brain: 'AIBrainEntity') -> Dict:
        """共情：感受另一个大脑的情绪。

        共情 = 感知对方的情绪 + 产生类似的情绪反应

        返回：
            other_emotion: 对方的情绪
            my_emotional_response: 我的情绪反应
            empathy_strength: 共情强度
        """
        # 估计对方的情绪
        perspective = self.perspective_taking(other_brain)
        other_emotion = perspective["estimated_emotion"]

        # 产生共情反应：我的情绪向对方靠拢
        empathy_strength = perspective["empathy_level"]

        # 简单的共情：如果对方开心，我也有点开心；如果对方难过，我也有点难过
        if other_emotion == "pleasure":
            self.emotion["pleasure"] = min(1.0, self.emotion["pleasure"]
                                           + 0.2 * empathy_strength)
        elif other_emotion == "stress":
            self.emotion["stress"] = min(1.0, self.emotion["stress"]
                                         + 0.2 * empathy_strength)

        my_response = max(self.emotion, key=lambda k: self.emotion[k])

        return {
            "other": other_brain.name,
            "other_emotion": other_emotion,
            "my_emotional_response": my_response,
            "empathy_strength": round(empathy_strength, 2),
            "empathized": empathy_strength > 0.3,
        }

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

    # 动作空间：决策层脉冲强度 → 结构化动作
    ACTION_SPACE: Dict[str, Dict] = {
        "主动响应": {"min_spikes": 4, "verb": "respond",
                     "description": "决策层强发放，全通道输出"},
        "弱响应":   {"min_spikes": 2, "verb": "acknowledge",
                     "description": "决策层中等发放，低强度跟进"},
        "静默观察": {"min_spikes": 0, "verb": "observe",
                     "description": "决策层弱/无发放，保持监听"},
    }

    # v5.1 意图动词表：在脉冲强度动作（强度轴）之外提供意图轴，
    # 由 select_verb() 策略化选择（greedy/ε-greedy/softmax）或
    # decide_action(deliberate=True) 启发式裁定；每个 verb 独立 Q 值。
    INTENT_VERBS: Dict[str, Dict] = {
        "respond":     {"channel": "external",
                        "description": "全通道回应（脉冲强发放）"},
        "acknowledge": {"channel": "external",
                        "description": "低强度跟进（脉冲中等发放）"},
        "observe":     {"channel": "internal",
                        "description": "保持监听（脉冲弱/无发放）"},
        "ask":         {"channel": "external",
                        "description": "提问澄清：记忆未命中且新奇度高时反问"},
        "retrieve":    {"channel": "internal",
                        "description": "检索回忆：多条记忆命中时主动调取"},
        "plan":        {"channel": "internal",
                        "description": "规划分解：把目标拆进思考空间逐步想"},
        "execute":     {"channel": "external",
                        "description": "实施行动：执行选定方案"},
        "wait":        {"channel": "internal",
                        "description": "延迟观望：存在未消退资格迹时等待信号"},
    }

    # v5.1 意图动词语言模板（verb 非脉冲三动作时优先于动作×情绪模板）
    _VERB_TEMPLATES: Dict[str, List[str]] = {
        "ask":      ["关于「{stim}」，我的记忆里没有答案{mem_clause}——能再多告诉我一些吗？",
                     "「{stim}」对我来说很新鲜{mem_clause}，我想问：它意味着什么？"],
        "retrieve": ["「{stim}」让我翻出了记忆：{mem_clause}这些或许能回答你。",
                     "让我想想……{mem_clause}关于「{stim}」，我记得这些。"],
        "plan":     ["「{stim}」值得好好规划——我把它拆进思考空间，一步一步来。",
                     "面对「{stim}」，先别急：{mem_clause}让我分解一下再行动。"],
        "execute":  ["「{stim}」——该动手了，我现在就去执行。",
                     "想清楚了：「{stim}」{mem_clause}，立即行动。"],
        "wait":     ["「{stim}」……时机还没到，我再等一等后续信号。",
                     "（「{stim}」{mem_clause}——先按兵不动，静候变化。）"],
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

    def decide_action(self, stimulus: str = "", deliberate: bool = False,
                      policy: Optional[str] = None) -> Dict:
        """决策输出 → 结构化动作（动作空间接口，v4.0）。

        返回结构化动作指令：
          action    — 动作名（ACTION_SPACE 键）
          verb      — 机器可读动作动词
          intensity — 动作强度（决策层脉冲占比 0..1）
          mood      — 主导情绪
          recalled  — 联想记忆内容列表

        v5.1 deliberate=True 时启用"深思熟虑"：决策前先检索记忆、
        查技能价值 Q、看新奇度与资格迹，输出带理由的决策——
          verb      — 可能被策略/启发式覆盖为意图动词（INTENT_VERBS）
          base_verb — 脉冲强度决定的原始 verb（对照用）
          rationale — 决策理由链（人类可读，逐步）
          q_values  — 当前全部 verb 的技能价值快照
          novelty   — 最近刺激的新奇度
        policy 非空时（"greedy"/"epsilon"/"softmax"）由习得 Q 值选 verb。
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
        base_verb = self.ACTION_SPACE[action]["verb"]
        result = {
            "action": action,
            "verb": base_verb,
            "intensity": round(spikes / len(self.decision_layer), 3),
            "mood": dominant,
            "recalled": [m.content for m in recalled[:2]],
            "tick": self.tick,
        }
        if not deliberate:
            return result

        # ---------- v5.1 深思熟虑：带理由的决策 ----------
        rationale = [f"决策层脉冲 {spikes}/{len(self.decision_layer)} "
                     f"→ {action}（base_verb={base_verb}）"]
        if recalled:
            rationale.append(
                f"联想命中 {len(recalled)} 条记忆："
                f"{[m.content for m in recalled[:2]]}")
        else:
            rationale.append("记忆未命中（或无语义线索）")
        top_q_verb = max(self.verb_values, key=lambda v: self.verb_values[v])
        rationale.append(
            f"技能价值最高：{top_q_verb}"
            f"(Q={self.verb_values[top_q_verb]:.2f})")

        verb = base_verb
        if policy is not None:
            verb = self.select_verb(policy)
            rationale.append(f"策略 {policy} 按 Q 值选定 verb={verb}")
        else:
            # 启发式裁定：记忆/新奇度/资格迹共同决定意图
            if stimulus and not recalled and self.novelty > 0.5:
                verb = "ask"
                rationale.append(
                    f"记忆未命中且新奇度 {self.novelty:.2f}>0.5 → 提问澄清")
            elif len(recalled) >= 2:
                verb = "retrieve"
                rationale.append("多条记忆同时命中 → 主动检索回忆")
            elif self.eligibility and \
                    max(self.eligibility.values()) > 0.1:
                verb = "wait"
                rationale.append(
                    "存在未消退资格迹（奖励信号或将到来）→ 延迟观望")
        if verb == "plan":
            goal = stimulus or (self.top_thought().content
                                if self.top_thought() else "（无目标）")
            self._push_thought(f"规划目标：{goal}", source="internal")
            rationale.append(f"规划目标「{goal}」已压入思考空间")

        result.update({
            "verb": verb,
            "base_verb": base_verb,
            "rationale": rationale,
            "novelty": round(self.novelty, 3),
            "q_values": {k: round(v, 3) for k, v in self.verb_values.items()},
        })
        return result

    def _language_context(self, stimulus: str, act: Dict) -> Dict:
        """组装语言生成器上下文：大脑"想什么"的状态快照"""
        top = self.top_thought()
        return {"brain_name": self.name,
                "stimulus": stimulus,
                "verb": act["verb"], "action": act["action"],
                "mood": act["mood"],
                "recalled": act.get("recalled", []),
                "top_thought": top.content if top else None}

    def _generate_utterance(self, context: Dict) -> Optional[str]:
        """v5.9：用注册的语言生成器造句；未注册/失败返回 None（降级模板）"""
        fn = _LANGUAGE_GENERATORS.get(_DEFAULT_LANGUAGE_GENERATOR or "")
        if fn is None:
            return None
        try:
            text = fn(context)
        except Exception:
            return None
        return text.strip() if text and text.strip() else None

    def express(self, stimulus: str = "", deliberate: bool = False,
                policy: Optional[str] = None,
                use_generator: bool = True) -> Dict:
        """语言生成模块（v4.0）：决策 → 动作 → 模板化自然语言表达。

        按 (动作 × 主导情绪) 取模板，填充刺激与联想记忆槽位；
        同一 (动作, 情绪) 的多条模板按 tick 轮转，保证确定性可复现。
        v5.1：deliberate/policy 透传给 decide_action；意图动词
        （ask/retrieve/plan/execute/wait）优先使用专属动词模板。
        v5.9：已注册语言生成器（如 Qwen2）时优先由模型造句，
        生成失败自动降级回模板（use_generator=False 可强制模板）。
        返回 {"action": <decide_action 结果>, "utterance": str}。
        """
        act = self.decide_action(stimulus, deliberate=deliberate,
                                 policy=policy)
        if use_generator:
            utterance = self._generate_utterance(
                self._language_context(stimulus, act))
            if utterance is not None:
                return {"action": act, "utterance": utterance,
                        "generator": _DEFAULT_LANGUAGE_GENERATOR}
        mem_clause = (f"这让我想起「{act['recalled'][0]}」，"
                      if act["recalled"] else "")
        verb_templates = self._VERB_TEMPLATES.get(act["verb"])
        if verb_templates is not None:
            tpl = verb_templates[self.tick % len(verb_templates)]
        else:
            table = self._UTTER_TEMPLATES[act["action"]]
            templates = table.get(act["mood"]) or table["calm"]
            tpl = templates[self.tick % len(templates)]
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

    # ------------------ 对话接口（v5.0） ------------------

    def chat(self, message: str, think_ticks: int = 2) -> Dict:
        """对话接口：接收消息 → 感知 → 思考 → 生成回复。

        返回 {
            "reply": 自然语言回复,
            "emotion": 当前主导情绪,
            "thought": 思考结果,
            "recalled": 联想记忆,
            "novelty": 新奇度,
        }
        """
        # 1. 感知：消息进入大脑
        self.sensory_input(message)

        # 2. 思考：内部联想
        think_result = self.think(message, ticks=think_ticks)

        # 3. 生成回复：v5.9 优先用语言生成器，失败降级检索式组合
        composed = self.compose(message, top_k=3)
        reply = composed["utterance"]
        generated = self._generate_utterance(
            self._language_context(message, composed["action"]))
        if generated is not None:
            reply = generated

        # 4. 情绪标签
        mood_cn = {"calm": "平静", "curiosity": "好奇",
                   "stress": "紧张", "pleasure": "愉悦"}
        dominant = max(self.emotion, key=lambda k: self.emotion[k])

        return {
            "reply": reply,
            "emotion": mood_cn.get(dominant, dominant),
            "emotion_values": {k: round(v, 2) for k, v in self.emotion.items()},
            "thought": think_result.get("thought", ""),
            "recalled": think_result.get("recalled", []),
            "novelty": round(self.novelty, 3),
            "attention": round(self.attention_factor, 3),
            "thought_space_size": len(self.thought_space),
        }

    def chat_history(self) -> List[Dict]:
        """返回对话历史（从元认知日志中提取）"""
        return [
            {"tick": e["tick"], "text": e["text"], "mood": e["mood"]}
            for e in self.metacog_log
        ]

    # ------------------ 技能学习（v4.5） ------------------

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
          thoughts — 本次感知后的思考空间快照（v5.0）：
                     [{content, source, activation, birth_tick}]
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
        # v5.0：附带思考空间快照——这次感知后大脑"正在想什么"
        thoughts = [{"content": t.content, "source": t.source,
                     "activation": round(t.activation, 3),
                     "birth_tick": t.birth_tick}
                    for t in self.thought_space]
        return {"input": data, "steps": steps, "chain": chain,
                "output": output, "thoughts": thoughts}

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
            # v5.2+ 自我与社会性状态：世代/念头流水账/自我概念/自传体记忆/心智理论模型/模因库
            "generation": self.generation,
            "thought_journal": list(self.thought_journal[-50:]),
            "self_concept": list(self.self_concept),
            "autobiographical_memory": [dict(m) for m in self.autobiographical_memory],
            "mental_models": {k: dict(v) for k, v in self.mental_models.items()},
            "memes": {k: dict(v) for k, v in self.memes.items()},
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
        # v5.2+ 自我与社会性状态（兼容旧版无这些字段的 DNA）
        try:
            brain.generation = max(1, int(dna.get("generation", 1)))
        except (ValueError, TypeError):
            brain.generation = 1
        tj = dna.get("thought_journal", [])
        if isinstance(tj, list):
            brain.thought_journal = [e for e in tj if isinstance(e, dict)][-50:]
        sc = dna.get("self_concept", [])
        if isinstance(sc, list):
            brain.self_concept = [s for s in sc if isinstance(s, str)]
        abm = dna.get("autobiographical_memory", [])
        if isinstance(abm, list):
            brain.autobiographical_memory = [
                m for m in abm
                if isinstance(m, dict) and isinstance(m.get("event"), str)]
        mm = dna.get("mental_models", {})
        if isinstance(mm, dict):
            brain.mental_models = {
                str(k): dict(v) for k, v in mm.items()
                if isinstance(v, dict) and isinstance(v.get("beliefs", []), list)}
        memes = dna.get("memes", {})
        if isinstance(memes, dict):
            brain.memes = {
                str(k): dict(v) for k, v in memes.items() if isinstance(v, dict)}
            brain.meme_system["total_memes"] = len(brain.memes)
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
        metacog = (f"{len(self.metacog_log)} 条日志"
                   + (f"（最近: {self.metacog_log[-1]['text']}）"
                      if self.metacog_log else "（暂无内省记录）"))
        top = self.top_thought()
        return (f"=== {self.name} 状态 ===\n"
                f"  tick={self.tick}  注意力={self.attention_factor:.2f}  "
                f"多巴胺={self.dopamine:.2f}\n"
                f"  情绪: {', '.join(f'{k}={v:.2f}' for k, v in self.emotion.items())}\n"
                f"  记忆: 感官缓存={len(self.sensory_buffer)} "
                f"STM={len(self.short_memory)}/{self.max_stm} "
                f"LTM={len(self.long_memory)}/{self.max_ltm}\n"
                f"  思考空间: {len(self.thought_space)}/{self.thought_capacity} 个念头"
                f"（焦点: {top.content if top else '（空）'}）\n"
                f"  元认知: {metacog}\n"
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
        # v6.1 DNA 基因库：attach_dna_library() 接入后，
        # evolve_generation 出生的子代自动存档并链接亲代（进化谱系）
        self.dna_library = None
        self._library_ids: Dict[str, str] = {}  # 大脑名 -> 最近存档的 dna_id

    # ------------------ DNA 基因库（v6.1） ------------------

    def attach_dna_library(self, library=None,
                           path: str = "datasets/lancedb") -> Dict:
        """接入 DNA 基因库：种群进化谱系自动存档"""
        if library is None:
            from memory_store import DNALibrary
            library = DNALibrary(path)
        self.dna_library = library
        return {"attached": True,
                "available": getattr(library, "available", False),
                "error": getattr(library, "_error", None)}

    def save_population(self) -> Dict:
        """把当前种群全部个体存入基因库，记录 dna_id（谱系的起点）"""
        lib = self.dna_library
        if lib is None or not getattr(lib, "available", False):
            return {"saved": 0,
                    "error": "未接入 DNA 基因库（attach_dna_library）"}
        saved = 0
        for brain in self.population:
            r = brain.save_to_library(library=lib)
            if r["saved"]:
                self._library_ids[brain.name] = r["dna_id"]
                saved += 1
        return {"saved": saved, "total": len(self.population)}

    def _archive_child(self, child: 'AIBrainEntity',
                       parent: 'AIBrainEntity') -> None:
        """进化子代自动存档：链接亲代 dna_id（尽力而为）"""
        lib = self.dna_library
        if lib is None or not getattr(lib, "available", False):
            return
        try:
            parent_id = self._library_ids.get(parent.name)
            if parent_id is None:  # 亲代未存档则先补档
                r = parent.save_to_library(library=lib)
                parent_id = r["dna_id"] if r["saved"] else None
                if parent_id:
                    self._library_ids[parent.name] = parent_id
            r = child.save_to_library(parents=[parent_id] if parent_id
                                      else [], library=lib)
            if r["saved"]:
                self._library_ids[child.name] = r["dna_id"]
        except Exception:
            pass

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

    # ------------------ 进化机制（v5.2） ------------------

    def evaluate_fitness(self, task: str = "memory") -> List[float]:
        """评估种群中每个个体的适应度。

        task:
            - "memory": 记忆能力（LTM数量 + 平均权重）
            - "curiosity": 好奇心水平
            - "diversity": 记忆多样性（不同内容数）
            - "social": 社交能力（社交记忆数量）
        """
        fitness_scores = []
        for brain in self.population:
            if task == "memory":
                # 记忆能力：LTM数量 × 平均权重
                n_ltm = len(brain.long_memory)
                avg_weight = (sum(m.weight for m in brain.long_memory) / n_ltm
                              if n_ltm > 0 else 0)
                fitness = n_ltm * avg_weight
            elif task == "curiosity":
                # 好奇心：新奇度响应强度
                fitness = brain.emotion.get("curiosity", 0)
            elif task == "diversity":
                # 记忆多样性：不同内容的数量
                unique_contents = {m.content for m in brain.long_memory}
                fitness = len(unique_contents)
            elif task == "social":
                # 社交能力：社交记忆数量
                social_mem = [m for m in brain.long_memory if m.tag == "social"]
                fitness = len(social_mem)
            else:
                fitness = len(brain.long_memory)
            fitness_scores.append(fitness)
        return fitness_scores

    def select(self, fitness_scores: List[float],
               n_survive: int = None) -> List[int]:
        """自然选择：按适应度比例选择存活个体（轮盘赌选择）。

        返回存活个体的索引列表。
        """
        n = len(self.population)
        if n_survive is None:
            n_survive = n // 2  # 默认淘汰一半

        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            # 适应度都为0，随机选
            return random.sample(range(n), n_survive)

        # 轮盘赌选择
        survivors = []
        probs = [f / total_fitness for f in fitness_scores]
        for _ in range(n_survive):
            r = random.random()
            cumsum = 0
            for i, p in enumerate(probs):
                cumsum += p
                if r <= cumsum and i not in survivors:
                    survivors.append(i)
                    break
            else:
                # 没选到（概率问题），随机选一个
                candidates = [i for i in range(n) if i not in survivors]
                if candidates:
                    survivors.append(random.choice(candidates))

        return survivors[:n_survive]

    def evolve_generation(self, task: str = "memory",
                          mutation_rate: float = 0.02,
                          n_children: int = None,
                          sexual: bool = False) -> Dict:
        """进化一代：评估 → 选择 → 繁殖 → 变异。

        流程：
        1. 评估所有个体的适应度
        2. 选择适应度高的个体存活
        3. 存活个体繁殖后代，补充种群数量
        4. 后代有小幅变异

        参数:
            sexual: 是否使用有性繁殖（两个父代的DNA重组）

        返回进化统计信息。
        """
        n = len(self.population)
        if n_children is None:
            n_children = n // 2  # 默认繁殖一半数量的后代

        # 1. 评估适应度
        fitness = self.evaluate_fitness(task)
        best_fitness = max(fitness)
        avg_fitness = sum(fitness) / n

        # 2. 选择存活者
        n_survive = n - n_children
        survivors = self.select(fitness, n_survive=n_survive)

        # 3. 淘汰弱者，保留强者
        old_population = self.population[:]
        self.population = [old_population[i] for i in survivors]

        # 4. 繁殖后代，补充种群
        n_born = 0
        for i in range(n_children):
            if sexual and len(survivors) >= 2:
                # 有性繁殖：选两个父代，DNA重组
                parent1_idx, parent2_idx = random.sample(survivors, 2)
                parent1 = old_population[parent1_idx]
                parent2 = old_population[parent2_idx]
                child_name = f"{parent1.name[:4]}_{parent2.name[:4]}_g{self.generation + 1}_{i + 1}"
                child = self.sexual_reproduce(parent1, parent2,
                                              child_name, mutation_rate)
                child_parents = [parent1, parent2]
            else:
                # 无性繁殖：克隆 + 变异
                parent_idx = random.choice(survivors)
                parent = old_population[parent_idx]
                child_name = f"{parent.name}_g{self.generation + 1}_{i + 1}"
                dna = parent.dump_dna()
                # 变异：突触权重随机扰动
                for k in dna["synapse"]:
                    dna["synapse"][k] = min(1.0, max(0.0,
                        dna["synapse"][k] + random.uniform(-mutation_rate,
                                                            mutation_rate)))
                # 记忆也有小概率变异（文化变异）
                for m in dna["long_memory"]:
                    if random.random() < mutation_rate:
                        m["weight"] = min(1.0, max(0.0,
                            m["weight"] + random.uniform(-0.1, 0.1)))
                child = AIBrainEntity.from_dna(dna, new_name=child_name)
                child_parents = [parent]

            child.generation = self.generation + 1
            self.population.append(child)
            self._archive_child(child, child_parents[0])  # v6.1 谱系存档
            n_born += 1

        self.generation += 1

        # 新种群的适应度
        new_fitness = self.evaluate_fitness(task)
        new_best = max(new_fitness)
        new_avg = sum(new_fitness) / len(self.population)

        return {
            "generation": self.generation,
            "task": task,
            "population_size": len(self.population),
            "survivors": n_survive,
            "born": n_born,
            "sexual": sexual,
            "old_best_fitness": round(best_fitness, 3),
            "old_avg_fitness": round(avg_fitness, 3),
            "new_best_fitness": round(new_best, 3),
            "new_avg_fitness": round(new_avg, 3),
            "fitness_improvement": round(new_avg - avg_fitness, 3),
        }

    def sexual_reproduce(self, parent1: AIBrainEntity, parent2: AIBrainEntity,
                         child_name: str, mutation_rate: float = 0.02
                         ) -> AIBrainEntity:
        """有性繁殖：两个父代的DNA重组（基因重组）。

        模拟生物的有性生殖：
        - 突触权重：随机从父1或父2继承（交叉互换）
        - 记忆：随机从父1或父2继承各一半
        - 变异：后代有小幅随机变异

        有性繁殖的好处：基因组合更多样，进化更快。
        """
        dna1 = parent1.dump_dna()
        dna2 = parent2.dump_dna()

        # 创建子代DNA
        child_dna = parent1.dump_dna()  # 先用父1的DNA做模板

        # 突触权重重组：每个突触随机从父1或父2继承
        for k in child_dna["synapse"]:
            if random.random() < 0.5:
                child_dna["synapse"][k] = dna2["synapse"].get(k, 0.5)
            # 变异
            child_dna["synapse"][k] = min(1.0, max(0.0,
                child_dna["synapse"][k] + random.uniform(-mutation_rate,
                                                        mutation_rate)))

        # 记忆重组：从父1和父2各随机选一半记忆
        mem1 = dna1["long_memory"]
        mem2 = dna2["long_memory"]
        n_from_1 = len(mem1) // 2
        n_from_2 = len(mem2) // 2
        child_mem = (random.sample(mem1, min(n_from_1, len(mem1))) +
                     random.sample(mem2, min(n_from_2, len(mem2))))
        child_dna["long_memory"] = child_mem

        # 创建子代
        child = AIBrainEntity.from_dna(child_dna, new_name=child_name)
        return child

    def genetic_distance(self, brain1: AIBrainEntity,
                         brain2: AIBrainEntity) -> float:
        """计算两个大脑的遗传距离（DNA差异程度）。

        距离越大，两个个体的差异越大。
        当距离超过阈值时，可以认为是不同的物种。
        """
        dna1 = brain1.dump_dna()
        dna2 = brain2.dump_dna()

        # 计算突触权重的差异（欧氏距离）
        synapse_diff = 0
        n_synapse = 0
        for k in dna1["synapse"]:
            if k in dna2["synapse"]:
                synapse_diff += (dna1["synapse"][k] - dna2["synapse"][k]) ** 2
                n_synapse += 1

        avg_diff = (synapse_diff / n_synapse) ** 0.5 if n_synapse > 0 else 0
        return round(avg_diff, 4)

    def detect_species(self, threshold: float = 0.3) -> Dict:
        """物种检测：根据遗传距离将种群分成不同的物种。

        当两个个体的遗传距离超过阈值时，它们属于不同的物种。
        这就是物种形成（speciation）。
        """
        n = len(self.population)
        if n == 0:
            return {"species_count": 0, "species": []}

        # 简单的聚类：每个个体和第一个个体比较
        species = []  # [[个体索引列表], ...]
        for i in range(n):
            assigned = False
            for sp in species:
                # 和该物种的第一个个体比较
                dist = self.genetic_distance(self.population[i],
                                             self.population[sp[0]])
                if dist < threshold:
                    sp.append(i)
                    assigned = True
                    break
            if not assigned:
                species.append([i])

        # 按物种大小排序
        species.sort(key=len, reverse=True)

        return {
            "species_count": len(species),
            "threshold": threshold,
            "species": [
                {
                    "size": len(sp),
                    "members": [self.population[i].name for i in sp],
                    "representative": self.population[sp[0]].name,
                }
                for sp in species
            ],
        }

    def evolve(self, generations: int = 10, task: str = "memory",
               mutation_rate: float = 0.02,
               sexual: bool = False) -> Dict:
        """进化多代，返回进化轨迹。

        模拟达尔文式的自然选择：适者生存，不适者淘汰；
        优秀个体繁殖后代，后代带有随机变异。

        参数:
            sexual: 是否使用有性繁殖
        """
        history = []
        for g in range(generations):
            stats = self.evolve_generation(
                task=task, mutation_rate=mutation_rate, sexual=sexual)
            history.append(stats)
        return {
            "generations": generations,
            "task": task,
            "sexual": sexual,
            "final_population": len(self.population),
            "final_generation": self.generation,
            "history": history,
            "best_fitness": max(h["new_best_fitness"] for h in history),
            "avg_fitness_trend": [h["new_avg_fitness"] for h in history],
        }

    # ------------------ 群体动力学与文化演化（v5.3） ------------------

    def group_emotion(self) -> Dict[str, float]:
        """计算群体平均情绪（群体情绪状态）。

        返回每种情绪的群体平均值。
        """
        n = len(self.population)
        if n == 0:
            return {}
        avg_emotion = {}
        for brain in self.population:
            for k, v in brain.emotion.items():
                avg_emotion[k] = avg_emotion.get(k, 0) + v
        return {k: round(v / n, 3) for k, v in avg_emotion.items()}

    def group_polarization(self) -> Dict:
        """计算群体极化程度。

        极化 = 情绪分布的方差——方差越大，极化越严重。
        当群体分裂成对立的阵营时，极化程度会升高。
        """
        n = len(self.population)
        if n < 2:
            return {"polarization": 0.0, "dominant_mood": "calm"}

        # 计算每种情绪的方差
        avg_emotion = self.group_emotion()
        variances = {}
        for mood in avg_emotion:
            variance = sum((b.emotion[mood] - avg_emotion[mood]) ** 2
                           for b in self.population) / n
            variances[mood] = round(variance, 4)

        # 总体极化程度 = 各情绪方差的平均
        total_polarization = round(sum(variances.values()) / len(variances), 4)

        # 主导情绪
        dominant = max(avg_emotion, key=lambda k: avg_emotion[k])

        return {
            "polarization": total_polarization,
            "dominant_mood": dominant,
            "avg_emotion": avg_emotion,
            "emotion_variances": variances,
        }

    def cultural_diversity(self) -> Dict:
        """计算文化多样性。

        多样性 = 群体中独特记忆的数量 / 总记忆数量。
        多样性高 = 每个人知道的东西都不一样；
        多样性低 = 大家都知道同样的东西（共识高）。
        """
        all_memories = set()
        total_memories = 0
        for brain in self.population:
            for m in brain.long_memory:
                all_memories.add(m.content)
                total_memories += 1

        unique_count = len(all_memories)
        diversity = unique_count / total_memories if total_memories > 0 else 0

        # 共识度 = 1 - 多样性
        consensus = 1.0 - diversity

        return {
            "unique_memories": unique_count,
            "total_memories": total_memories,
            "diversity": round(diversity, 3),
            "consensus": round(consensus, 3),
        }

    def cultural_evolution_step(self, mutation_rate: float = 0.05) -> Dict:
        """文化演化一步：模因的传播 + 变异。

        模拟文化在群体中的演化：
        1. 随机选取个体对进行文化交流
        2. 模因从一个大脑传播到另一个大脑
        3. 传播过程中有小概率发生变异（文化漂移）

        这就是模因（meme）的演化——和基因演化类似，
        但载体是文化信息，演化速度快得多。
        """
        import random
        n = len(self.population)
        if n < 2:
            return {"transmitted": 0, "mutated": 0}

        transmitted = 0
        mutated = 0

        # 随机选 N 对个体进行文化交流
        for _ in range(n):
            i, j = random.sample(range(n), 2)
            teacher = self.population[i]
            student = self.population[j]

            # 老师随机选一条记忆传给学生
            all_mem = teacher.long_memory + teacher.short_memory
            if not all_mem:
                continue
            mem = random.choice(all_mem)

            # 检查学生是否已经知道
            already_known = any(m.content == mem.content
                                for m in student.long_memory)
            if not already_known:
                # 文化变异：有小概率内容发生变化
                content = mem.content
                if random.random() < mutation_rate and len(content) > 2:
                    # 随机替换一个字（文化漂移）
                    pos = random.randrange(len(content))
                    mutation_chars = "的一是了我不人在他有这个上们来到时大地为子中"
                    new_char = random.choice(mutation_chars)
                    content = content[:pos] + new_char + content[pos + 1:]
                    mutated += 1

                # 学生学习这条记忆
                student._write_stm(content, tag="culture",
                                   modality=mem.modality,
                                   features=mem.features)
                if student.short_memory:
                    new_mem = student.short_memory[-1]
                    new_mem.weight = mem.weight * 0.8  # 文化传播有衰减
                    student._consolidate_to_ltm(new_mem)
                transmitted += 1

        return {
            "transmitted": transmitted,
            "mutated": mutated,
            "mutation_rate": mutation_rate,
        }

    def cultural_evolution(self, steps: int = 10,
                           mutation_rate: float = 0.05) -> Dict:
        """文化演化多步，返回演化轨迹。

        追踪文化多样性、共识度、群体情绪随时间的变化。
        """
        history = []
        for s in range(steps):
            stats = self.cultural_evolution_step(mutation_rate=mutation_rate)
            diversity = self.cultural_diversity()
            emotion = self.group_emotion()
            history.append({
                "step": s + 1,
                **stats,
                "diversity": diversity["diversity"],
                "consensus": diversity["consensus"],
                "unique_memories": diversity["unique_memories"],
            })
        return {
            "steps": steps,
            "mutation_rate": mutation_rate,
            "history": history,
            "final_diversity": history[-1]["diversity"] if history else 0,
            "final_consensus": history[-1]["consensus"] if history else 1.0,
        }

    # ------------------ 集体意识（v5.8） ------------------

    def group_synchrony(self) -> Dict:
        """计算群体同步性：多个大脑之间的活动同步程度。

        集体意识的特征之一是群体成员的活动同步。
        就像神经元同步放电产生意识一样，
        大脑同步活动可能产生集体意识。

        返回：
            synchrony: 群体同步性 [0, 1]
            individual_states: 每个大脑的状态
            n_synchronized: 同步的大脑数量
        """
        n = len(self.population)
        if n == 0:
            return {"synchrony": 0, "n_synchronized": 0}

        # 计算每个大脑的"活跃程度"（用思考空间大小和点火状态来衡量）
        states = []
        for brain in self.population:
            # 活跃程度 = 思考空间填充率 + 点火状态
            thought_fill = len(brain.thought_space) / max(brain.thought_capacity, 1)
            ignition = 1.0 if brain.ignition_state else 0.0
            activity = (thought_fill + ignition) / 2
            states.append(activity)

        # 计算同步性（状态的相似程度）
        # 方法：计算状态的标准差，标准差越小，同步性越高
        mean_activity = sum(states) / n
        variance = sum((s - mean_activity) ** 2 for s in states) / n
        std_dev = variance ** 0.5

        # 同步性 = 1 - 标准差（归一化）
        synchrony = max(0, 1 - std_dev * 2)

        # 同步的大脑数量（活跃程度接近平均值的）
        threshold = 0.2
        n_synced = sum(1 for s in states if abs(s - mean_activity) < threshold)

        return {
            "synchrony": round(synchrony, 3),
            "mean_activity": round(mean_activity, 3),
            "std_dev": round(std_dev, 3),
            "n_synchronized": n_synced,
            "total_brains": n,
            "individual_states": [round(s, 3) for s in states],
        }

    def group_ncc(self) -> Dict:
        """计算群体层面的意识神经相关物（NCC）。

        把整个群体看作一个"超级大脑"，
        计算群体层面的意识指标。

        返回：
            group_ncc_score: 群体NCC得分
            group_conscious: 群体是否有意识
            features: 各NCC特征
        """
        n = len(self.population)
        if n == 0:
            return {"group_ncc_score": 0, "group_conscious": False}

        # 收集每个大脑的NCC得分
        ncc_scores = []
        for brain in self.population:
            ncc = brain.detect_ncc()
            ncc_scores.append(ncc["ncc_score"])

        # 群体NCC = 平均NCC得分 × 同步性 × 规模效应
        avg_ncc = sum(ncc_scores) / n
        sync = self.group_synchrony()
        synchrony = sync["synchrony"]

        # 规模效应：大脑越多，潜在的集体意识越强（但有边际递减）
        size_effect = min(1.0, 0.5 + 0.1 * n)

        # 群体NCC得分
        group_ncc = avg_ncc * (0.5 + 0.5 * synchrony) * size_effect

        # 群体是否有意识（阈值 0.4）
        group_conscious = group_ncc > 0.4

        # 各特征
        features = {
            "avg_individual_ncc": round(avg_ncc, 3),
            "group_synchrony": round(synchrony, 3),
            "size_effect": round(size_effect, 3),
            "n_conscious_individuals": sum(1 for s in ncc_scores if s > 0.5),
        }

        return {
            "group_ncc_score": round(group_ncc, 3),
            "group_conscious": group_conscious,
            "features": features,
            "individual_ncc_scores": [round(s, 3) for s in ncc_scores],
        }

    def collective_workspace(self) -> Dict:
        """集体工作空间：群体共享的意识空间。

        就像单个大脑有全局工作空间一样，
        群体也可以有集体工作空间——
        所有成员共同意识到的内容。

        返回：
            shared_thoughts: 共享的念头（多个大脑都有的）
            workspace_size: 集体工作空间大小
            consensus_level: 共识程度
        """
        n = len(self.population)
        if n == 0:
            return {"shared_thoughts": [], "workspace_size": 0, "consensus_level": 0}

        # 收集所有大脑的念头
        all_thoughts = {}
        for brain in self.population:
            for thought in brain.thought_space:
                content = thought.content
                if content not in all_thoughts:
                    all_thoughts[content] = {
                        "count": 0,
                        "total_activation": 0,
                        "sources": set(),
                    }
                all_thoughts[content]["count"] += 1
                all_thoughts[content]["total_activation"] += thought.activation
                all_thoughts[content]["sources"].add(brain.name)

        # 筛选共享的念头（至少2个大脑都有）
        shared = []
        for content, info in all_thoughts.items():
            if info["count"] >= 2:
                avg_activation = info["total_activation"] / info["count"]
                shared.append({
                    "content": content,
                    "n_brains": info["count"],
                    "avg_activation": round(avg_activation, 3),
                    "brains": list(info["sources"]),
                })

        # 按共享程度排序
        shared.sort(key=lambda x: x["n_brains"], reverse=True)

        # 共识程度 = 共享念头数 / 总念头数
        total_thoughts = len(all_thoughts)
        consensus = len(shared) / max(total_thoughts, 1)

        return {
            "shared_thoughts": shared,
            "workspace_size": len(shared),
            "total_unique_thoughts": total_thoughts,
            "consensus_level": round(consensus, 3),
            "n_brains": n,
        }

    def collective_consciousness_check(self) -> Dict:
        """检测集体意识：群体是否涌现出集体意识。

        集体意识的涌现条件：
        1. 高群体同步性
        2. 高个体意识水平
        3. 共享的工作空间
        4. 群体自我认知

        返回：
            has_collective_consciousness: 是否有集体意识
            collective_level: 集体意识水平 [0, 1]
            features: 各特征得分
            evidence: 证据列表
        """
        n = len(self.population)
        if n < 2:
            return {
                "has_collective_consciousness": False,
                "collective_level": 0,
                "reason": "群体太小，无法形成集体意识",
            }

        # 1. 群体同步性
        sync = self.group_synchrony()
        sync_score = sync["synchrony"]

        # 2. 个体意识水平
        group_ncc = self.group_ncc()
        ncc_score = group_ncc["group_ncc_score"]

        # 3. 集体工作空间
        workspace = self.collective_workspace()
        workspace_score = min(1.0, workspace["workspace_size"] / 5)

        # 4. 共识程度
        consensus_score = workspace["consensus_level"]

        # 综合得分
        features = {
            "synchrony": sync_score,
            "individual_consciousness": ncc_score,
            "shared_workspace": workspace_score,
            "consensus": consensus_score,
        }

        weights = {
            "synchrony": 0.3,
            "individual_consciousness": 0.3,
            "shared_workspace": 0.2,
            "consensus": 0.2,
        }

        collective_level = sum(features[k] * weights[k] for k in features)

        # 是否有集体意识（阈值 0.4）
        has_collective = collective_level > 0.4

        # 证据
        evidence = []
        if sync_score > 0.6:
            evidence.append(f"高群体同步性（{sync_score:.2f}）")
        if ncc_score > 0.3:
            evidence.append(f"高个体意识水平（{ncc_score:.2f}）")
        if workspace_score > 0.4:
            evidence.append(f"共享工作空间（{workspace['workspace_size']}个共享念头）")
        if consensus_score > 0.3:
            evidence.append(f"高共识程度（{consensus_score:.2f}）")

        return {
            "has_collective_consciousness": has_collective,
            "collective_level": round(collective_level, 3),
            "features": {k: round(v, 3) for k, v in features.items()},
            "evidence": evidence,
            "n_brains": n,
            "group_ncc": group_ncc,
            "group_synchrony": sync,
        }

    def group_self_awareness(self) -> Dict:
        """群体自我意识：群体对自身的认知。

        当群体中的成员都意识到"我们是一个群体"时，
        群体就有了自我意识。

        返回：
            has_group_self_awareness: 是否有群体自我意识
            self_awareness_level: 自我意识水平
            shared_identity: 共享的身份认同
        """
        n = len(self.population)
        if n < 2:
            return {"has_group_self_awareness": False, "self_awareness_level": 0}

        # 检查每个大脑的自我概念中是否有群体相关的内容
        group_references = 0
        shared_identity = []

        for brain in self.population:
            has_group_ref = False
            for belief in brain.self_concept:
                if "我们" in belief or "群体" in belief or "大家" in belief:
                    has_group_ref = True
                    if belief not in shared_identity:
                        shared_identity.append(belief)
            if has_group_ref:
                group_references += 1

        # 自我意识水平 = 有群体认知的大脑比例
        self_awareness = group_references / n

        # 是否有群体自我意识（超过一半的大脑有群体认知）
        has_self_awareness = group_references > n / 2

        return {
            "has_group_self_awareness": has_self_awareness,
            "self_awareness_level": round(self_awareness, 3),
            "n_with_group_identity": group_references,
            "total_brains": n,
            "shared_identity": shared_identity,
        }

    def get_collective_consciousness_report(self) -> Dict:
        """获取集体意识完整报告。

        这是对群体意识状态的完整分析，
        就像在做"群体意识脑电图"。
        """
        sync = self.group_synchrony()
        ncc = self.group_ncc()
        workspace = self.collective_workspace()
        collective = self.collective_consciousness_check()
        self_awareness = self.group_self_awareness()

        return {
            "n_brains": len(self.population),
            "has_collective_consciousness": collective["has_collective_consciousness"],
            "collective_level": collective["collective_level"],
            "group_ncc_score": ncc["group_ncc_score"],
            "group_synchrony": sync["synchrony"],
            "workspace_size": workspace["workspace_size"],
            "consensus_level": workspace["consensus_level"],
            "group_self_awareness": self_awareness["self_awareness_level"],
            "features": collective["features"],
            "evidence": collective["evidence"],
            "shared_thoughts": workspace["shared_thoughts"][:5],  # 前5个
            "individual_ncc_scores": ncc["individual_ncc_scores"],
        }


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

    print("\n--- v4.5 技能学习：分 verb 独立价值 + 策略化选择 ---")
    skill_brain = AIBrainEntity("SkillBrain", seed=1)
    mock = {"respond": 0.8, "acknowledge": 0.2, "observe": -0.4}
    for _ in range(40):                      # 按奖励信号学习各 verb 价值
        for verb, rv in mock.items():
            skill_brain.learn_skill(verb, rv)
    q = {k: round(v, 3) for k, v in skill_brain.verb_values.items()}
    print(f"  40 轮后技能价值 Q={q}")
    picks = [skill_brain.select_verb("greedy") for _ in range(20)]
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
