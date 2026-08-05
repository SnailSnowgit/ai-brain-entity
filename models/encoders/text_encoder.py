# -*- coding: utf-8 -*-
"""
文本语义编码器（v6.2）

为文本记忆生成真正的语义 embedding，打通 recall_semantic 的语义检索：
相似含义（"火焰"~"燃烧"）不再是字面哈希相近才命中。

降级链（与多模态编码器同风格，模型不可用绝不崩溃）：
    1. sentence-transformers 本地模型（models/ 下或显式 model_path）
    2. 返回 None → 调用方自动回退零依赖哈希向量（memory_store.text_to_vector）

统一接口：
    encode(text) -> List[float] | None     # None 表示不可用，调用方兜底
    available -> bool
    info()    -> {"available", "model", "dim", "error"}

模型获取（任选其一）：
    - 环境变量 AI_BRAIN_TEXT_MODEL 指向本地模型目录
    - models/bge-small-zh-v1.5/（默认中文小模型，~100MB）
    - attach_text_encoder(model_path="...") 显式指定
下载示例（需网络）：
    python -c "from sentence_transformers import SentenceTransformer; \
        SentenceTransformer('BAAI/bge-small-zh-v1.5').save('models/bge-small-zh-v1.5')"
"""
import os
from typing import Dict, List, Optional

_DEFAULT_MODEL_DIRS = (
    "models/bge-small-zh-v1.5",
    "models/text-embedding",
    "models/paraphrase-multilingual-MiniLM-L12-v2",
)


class TextSemanticEncoder:
    """sentence-transformers 文本语义编码器（可选加载，缺失自动降级）。"""

    def __init__(self, model_path: Optional[str] = None,
                 device: str = "cpu", dim: int = 512):
        self.dim = dim
        self.model = None
        self.model_name: Optional[str] = None
        self.error: Optional[str] = None

        path = (model_path
                or os.environ.get("AI_BRAIN_TEXT_MODEL")
                or next((d for d in _DEFAULT_MODEL_DIRS if os.path.isdir(d)),
                        None))
        if path is None:
            self.error = "未找到本地文本模型（models/bge-small-zh-v1.5 等）"
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            self.error = "sentence-transformers 未安装"
            return
        try:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 只用本地模型
            self.model = SentenceTransformer(path, device=device)
            self.model_name = os.path.basename(os.path.abspath(path))
        except Exception as e:  # 模型损坏/缺文件等
            self.error = f"模型加载失败: {e}"
            self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None

    def encode(self, text: str) -> Optional[List[float]]:
        """编码文本为 dim 维 L2 归一化向量；不可用时返回 None。"""
        if not self.available or not isinstance(text, str) or not text:
            return None
        try:
            vec = self.model.encode(text, normalize_embeddings=True)
            vec = [float(v) for v in vec]
        except Exception:
            return None
        # 维度对齐：截断或补零到 self.dim，再归一化
        if len(vec) > self.dim:
            vec = vec[:self.dim]
        elif len(vec) < self.dim:
            vec = vec + [0.0] * (self.dim - len(vec))
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def info(self) -> Dict:
        return {"available": self.available,
                "model": self.model_name,
                "dim": self.dim,
                "error": self.error}


def create_text_encoder(model_path: Optional[str] = None,
                        device: str = "cpu",
                        dim: int = 512) -> TextSemanticEncoder:
    """工厂函数：任何失败都返回不可用编码器，绝不抛异常。"""
    try:
        return TextSemanticEncoder(model_path=model_path, device=device, dim=dim)
    except Exception as e:
        enc = TextSemanticEncoder.__new__(TextSemanticEncoder)
        enc.dim = dim
        enc.model = None
        enc.model_name = None
        enc.error = f"初始化失败: {e}"
        return enc
