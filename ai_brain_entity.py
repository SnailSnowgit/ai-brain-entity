# -*- coding: utf-8 -*-
"""
AI大脑实体（类脑架构 Python 完整可运行代码）v3.0
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
  - 群体智能（BrainSwarm：文化传递 / 广播）
  - 脉冲活动记录（供可视化/实验分析）
"""

import time
import random
import json
import math
import hashlib
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional


# ===================== 多模态编码（可选真实模型，自动降级） =====================

_CLIP_CACHE = None      # (model, processor)
_WHISPER_CACHE = None   # model


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


def encode_image(path: str, dim: int = 512) -> List[float]:
    """图像 → embedding。优先使用 CLIP（需 pip install transformers pillow torch），
    不可用时自动降级为伪 embedding。"""
    global _CLIP_CACHE
    try:
        if _CLIP_CACHE is None:
            from transformers import CLIPModel, CLIPProcessor
            _CLIP_CACHE = (
                CLIPModel.from_pretrained("openai/clip-vit-base-patch32"),
                CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"),
            )
        from PIL import Image
        model, processor = _CLIP_CACHE
        inputs = processor(images=Image.open(path), return_tensors="pt")
        feats = model.get_image_features(**inputs)
        return feats[0].detach().cpu().numpy().tolist()
    except Exception:
        return _file_pseudo_embedding(path, dim)


def encode_audio(path: str, dim: int = 512) -> List[float]:
    """音频 → embedding。优先使用 Whisper 编码器（需 pip install openai-whisper），
    不可用时自动降级为伪 embedding。"""
    global _WHISPER_CACHE
    try:
        if _WHISPER_CACHE is None:
            import whisper
            _WHISPER_CACHE = whisper.load_model("base")
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
    tag: str         # 记忆标签：sensory / emotion / event / culture


# ===================== 大脑实体 =====================

class AIBrainEntity:
    def __init__(self, brain_name: str, seed: Optional[int] = None,
                 record_history: bool = False):
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

        # 2. 记忆系统（三级结构）
        self.sensory_buffer: List[str] = []   # 瞬时感官缓存
        self.short_memory: List[BrainMemory] = []
        self.long_memory: List[BrainMemory] = []
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

    # ------------------ 多模态接口（v3.0） ------------------

    def perceive_image(self, path: str, label: str = "") -> str:
        """感知一张图片：CLIP embedding（或降级伪 embedding）→ 感官层"""
        vec = encode_image(path)
        tag = label or f"<image:{os.path.basename(path)}>"
        return self.sensory_input_vector(vec, label=tag)

    def perceive_audio(self, path: str, label: str = "") -> str:
        """感知一段音频：Whisper encoder embedding（或降级伪 embedding）→ 感官层"""
        vec = encode_audio(path)
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
        """多模态接口：接收图像/音频 embedding 等任意数值向量"""
        currents = self._normalize_vector(vec, len(self.sense_layer))
        content = label or f"<vector[{len(vec)}]>"
        return self._perceive(content, currents, tag="sensory")

    def _perceive(self, content: str, input_currents: List[float], tag: str) -> str:
        """统一的感知-认知流水线"""
        self.tick += 1
        self.sensory_buffer.append(content)
        if len(self.sensory_buffer) > 8:
            self.sensory_buffer.pop(0)

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

    def recall(self, keyword: str, top_k: int = 3,
               exclude_latest: bool = True) -> List[BrainMemory]:
        """按关键词联想回忆（LTM 优先，按权重排序），回忆会强化记忆（再巩固）。

        exclude_latest=True 时排除最近一次写入的 STM，避免"回声记忆"——
        即刚听到的内容立即被当作联想结果返回。
        """
        stm = self.short_memory[:-1] if exclude_latest else self.short_memory
        hits = [m for m in self.long_memory + stm if keyword in m.content]
        hits.sort(key=lambda m: m.weight, reverse=True)
        # 去重（同一内容可能在 LTM 与 STM 各有一份）
        seen, unique = set(), []
        for m in hits:
            if m.content not in seen:
                seen.add(m.content)
                unique.append(m)
        for m in unique[:top_k]:
            m.weight = self._clip(m.weight + 0.05)  # 回忆强化
        return unique[:top_k]

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
        """序列化大脑状态（记忆 + 突触 + 情绪），可用于保存或克隆"""
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
        }

    @classmethod
    def from_dna(cls, dna: dict, new_name: Optional[str] = None) -> "AIBrainEntity":
        """从 DNA 克隆一个继承记忆与突触的新实体"""
        brain = cls(new_name or (dna["name"] + "_clone"))
        brain.tick = dna["tick"]
        brain.synapse = {}
        for k, w in dna["synapse"].items():
            a, b = k.split(",")
            brain.synapse[(int(a), int(b))] = w
        # 循环突触（v2.1+ DNA；兼容旧版无此字段的 DNA）
        for k, w in dna.get("recurrent_synapse", {}).items():
            a, b = k.split(",")
            brain.recurrent_synapse[(int(a), int(b))] = w
        brain._recurrent_out = {}
        brain._recurrent_in = {}
        for (pre, post) in brain.recurrent_synapse:
            brain._recurrent_out.setdefault(pre, []).append(post)
            brain._recurrent_in.setdefault(post, []).append(pre)
        brain.short_memory = [BrainMemory(**m) for m in dna["short_memory"]]
        brain.long_memory = [BrainMemory(**m) for m in dna["long_memory"]]
        brain.emotion = dict(dna["emotion"])
        brain.attention_factor = dna["attention_factor"]
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

    def reproduce(self, parent_idx: int, child_name: str,
                  mutation: float = 0.02) -> AIBrainEntity:
        """有性繁衍简化版：克隆父代 DNA，突触权重加小幅变异，
        子代加入种群（模拟代际演化）。"""
        parent = self.population[parent_idx]
        dna = parent.dump_dna()
        for k in dna["synapse"]:
            dna["synapse"][k] = min(1.0, max(0.0,
                dna["synapse"][k] + random.uniform(-mutation, mutation)))
        child = AIBrainEntity.from_dna(dna, new_name=child_name)
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
    dna_path = "brain_dna.json"
    brain.save_dna(dna_path)
    clone = AIBrainEntity.load_dna(dna_path, new_name="Brain-02")
    print(f"  克隆体 {clone.name} 继承了 LTM={len(clone.long_memory)} 条长期记忆")
    print(f"  克隆体回忆: {[m.content for m in clone.recall('记忆')]}")
