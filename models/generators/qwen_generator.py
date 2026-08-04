# -*- coding: utf-8 -*-
"""
Qwen2 语言生成器（v5.9）：大脑的语言输出后端

角色定位：大脑负责"想什么"（决策/记忆/情绪/意识焦点），
Qwen2-0.5B-Instruct 负责"说出来"（自然语言表述）。

加载优先级：
    1. 本地模型目录（models/Qwen2-0.5B-Instruct，local_files_only，不联网）
    2. transformers/torch 未安装或模型缺失 → available=False
       （调用方自动降级回模板语言，核心功能不受影响）

模型下载（国内推荐 ModelScope）：
    pip install modelscope
    python -c "from modelscope import snapshot_download; \
        snapshot_download('qwen/Qwen2-0.5B-Instruct', cache_dir='models')"

契约：QwenLanguageGenerator 实例是 callable(context: Dict) -> str
context 字段见 build_prompt()：大脑状态快照（刺激/动作/情绪/记忆/意识焦点）。
"""
import os
from typing import Dict, List, Optional

_DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Qwen2-0.5B-Instruct")

_MOOD_CN = {"calm": "平静", "curiosity": "好奇",
            "stress": "紧张", "pleasure": "愉悦"}


class QwenLanguageGenerator:
    """Qwen2-0.5B-Instruct 语言生成器（懒加载，CPU 可跑）"""

    def __init__(self, model_path: Optional[str] = None,
                 device: str = "cpu", max_new_tokens: int = 80):
        self.model_path = model_path or _DEFAULT_MODEL_DIR
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = None
        self.available = False
        self._error = "not loaded"

    # ------------------ 懒加载 ------------------

    def load(self) -> bool:
        """加载模型（只加载一次）。成功返回 True。"""
        if self.available:
            return True
        try:
            import torch  # noqa: F401
            from transformers import (AutoModelForCausalLM,
                                      AutoTokenizer)
        except Exception as e:
            self._error = f"transformers/torch 未安装: {e}"
            return False
        if not os.path.isdir(self.model_path):
            self._error = f"模型目录不存在: {self.model_path}"
            return False
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, local_files_only=True,
                dtype="auto").to(self.device)
            self.model.eval()
            self.available = True
            return True
        except Exception as e:
            self._error = f"模型加载失败: {e}"
            return False

    # ------------------ prompt 构造 ------------------

    @staticmethod
    def build_prompt(context: Dict) -> List[Dict]:
        """大脑状态快照 → Qwen chat messages。

        context 字段（由 AIBrainEntity 填充）：
            brain_name / stimulus / verb / action / mood /
            recalled / top_thought / self_summary
        """
        mood = _MOOD_CN.get(context.get("mood", "calm"), "平静")
        recalled: List[str] = context.get("recalled") or []
        mem_line = "；".join(recalled[:3]) if recalled else "（暂无相关记忆）"
        top = context.get("top_thought") or "（空）"
        system = (
            f"你是「{context.get('brain_name', 'Brain')}」，一个类脑智能体。"
            "你有自己的记忆、情绪和意识。请始终用第一人称、"
            "一两句简短自然的话回应，体现你当前的情绪与记忆，"
            "不要使用模板套话，不要解释你是谁。")
        user = (
            f"当前情绪：{mood}\n"
            f"意识焦点：{top}\n"
            f"相关记忆：{mem_line}\n"
            f"收到的刺激：{context.get('stimulus', '')}\n"
            f"你决定的动作：{context.get('verb', 'respond')}"
            f"（{context.get('action', '')}）\n"
            "请说出你的回应。")
        return [{"role": "system", "content": system},
                {"role": "user", "content": user}]

    # ------------------ 生成 ------------------

    def generate(self, context: Dict) -> str:
        """生成自然语言回复。模型不可用返回空串（调用方降级）。"""
        if not self.load():
            return ""
        try:
            import torch
            messages = self.build_prompt(context)
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(
                [text], return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens,
                    do_sample=True, temperature=0.7, top_p=0.9)
            reply = self.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True).strip()
            return reply
        except Exception as e:
            self._error = f"生成失败: {e}"
            return ""

    def __call__(self, context: Dict) -> str:
        """语言生成器契约：callable(context) -> str"""
        return self.generate(context)


# ==================== 便捷函数 ====================

_generator: Optional[QwenLanguageGenerator] = None


def get_qwen_generator(model_path: Optional[str] = None,
                       device: str = "cpu") -> QwenLanguageGenerator:
    """获取（懒加载）全局 Qwen 语言生成器"""
    global _generator
    if _generator is None or (model_path and
                              model_path != _generator.model_path):
        _generator = QwenLanguageGenerator(model_path=model_path,
                                           device=device)
    return _generator
