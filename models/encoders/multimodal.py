# -*- coding: utf-8 -*-
"""
多模态编码器集合（v5.0）

听觉：whisper 语音识别 → 文本
视觉：CLIP/BLIP 图像理解 → 文本描述 / 特征向量

所有编码器统一接口：encode(input) -> {"text": str, "features": list, "meta": dict}
- text：语义文本（用于进入大脑语言系统）
- features：数值特征（用于直接注入脉冲网络）
- meta：元信息（时长、尺寸、置信度等）

编码器可选加载：模型不可用时自动降级，不会导致大脑崩溃。
支持跨进程调用：当前Python没有模型时，自动调用系统Python 3.8的模型。
"""
import os
import sys
import json
import subprocess
from typing import Dict, List, Optional


def _find_system_python() -> Optional[str]:
    """查找系统Python 3.8（有whisper/transformers的那个）"""
    # 优先尝试 py -3.8
    try:
        result = subprocess.run(
            ["py", "-3.8", "-c", "import sys; print(sys.executable)"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            py_path = result.stdout.strip()
            # 检查是否有whisper
            result2 = subprocess.run(
                ["py", "-3.8", "-c", "import whisper; print('ok')"],
                capture_output=True, text=True, timeout=10
            )
            if result2.returncode == 0:
                return py_path
    except Exception:
        pass
    return None


def _remote_encode(mode: str, path: str) -> Optional[Dict]:
    """通过subprocess调用系统Python进行编码"""
    system_py = _find_system_python()
    if not system_py:
        return None

    service_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "multimodal_service.py")
    if not os.path.exists(service_script):
        return None

    try:
        result = subprocess.run(
            [system_py, service_script, mode, path],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            return json.loads(result.stdout.strip())
    except Exception:
        pass
    return None


# ==================== 听觉编码器（Whisper） ====================

class AudioEncoder:
    """语音 → 文本 编码器（基于 OpenAI Whisper）

    优先级：本地whisper > 远程系统Python > 降级占位
    """

    def __init__(self, model_size: str = "base", device: str = "cpu"):
        self.model_size = model_size
        self.device = device
        self.model = None
        self.available = False
        self.use_remote = False
        self._error = "not loaded"
        self._load()

    def _load(self):
        """尝试加载 whisper 模型，失败则尝试远程，再失败则标记不可用"""
        # 1. 尝试本地加载
        try:
            import whisper
            self.model = whisper.load_model(self.model_size, device=self.device)
            self.available = True
            self.use_remote = False
            return
        except Exception as e:
            self._error = f"local: {str(e)}"

        # 2. 尝试远程（系统Python 3.8）
        remote_result = _remote_encode("audio", __file__)  # 用当前文件测试
        if remote_result is not None:
            self.available = True
            self.use_remote = True
            return

        # 3. 都不行
        self.model = None
        self.available = False

    def encode(self, audio_path: str) -> Dict:
        """
        音频文件 → 识别文本 + 元信息

        返回:
            text: 识别出的文本
            language: 检测到的语言
            duration: 音频时长（秒）
            features: 音频特征（MFCC-like 简化版，用于脉冲注入）
        """
        # 远程模式
        if self.use_remote:
            result = _remote_encode("audio", audio_path)
            if result is not None and "error" not in result:
                # 确保有features字段
                if "features" not in result:
                    result["features"] = [0.0] * 16
                return result
            # 远程失败，降级
            filename = os.path.basename(audio_path)
            return {
                "text": f"[音频: {filename}]",
                "language": "unknown",
                "duration": 0,
                "features": [0.0] * 16,
                "meta": {"encoder": "fallback", "error": str(result.get("error", "remote failed")) if result else "remote failed"},
            }

        if not self.available:
            # 降级：返回文件名作为占位文本
            filename = os.path.basename(audio_path)
            return {
                "text": f"[音频: {filename}]",
                "language": "unknown",
                "duration": 0,
                "features": [0.0] * 16,
                "meta": {"encoder": "fallback", "error": self._error},
            }

        try:
            result = self.model.transcribe(audio_path)
            text = result.get("text", "").strip()
            lang = result.get("language", "unknown")
            duration = result.get("duration", 0)
        except Exception as e:
            # ffmpeg 不可用时的降级：用 wave 模块读取 wav 文件
            try:
                import wave
                import numpy as np
                with wave.open(audio_path, 'rb') as wf:
                    frames = wf.readframes(wf.getnframes())
                    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    sr = wf.getframerate()
                    duration = len(audio) / sr
                # 直接用 numpy 数组调用 whisper
                result = self.model.transcribe(audio, fp16=False)
                text = result.get("text", "").strip()
                lang = result.get("language", "unknown")
            except Exception as e2:
                # 完全降级
                filename = os.path.basename(audio_path)
                return {
                    "text": f"[音频: {filename}]",
                    "language": "unknown",
                    "duration": 0,
                    "features": [0.0] * 16,
                    "meta": {"encoder": f"whisper-{self.model_size}",
                             "error": str(e2)},
                }

        # 构造 16 维音频特征（用于注入感官层）
        # 用词数和音节数粗略模拟音频能量分布
        words = text.split()
        features = [0.0] * 16
        if words:
            for i, w in enumerate(words[:16]):
                features[i] = min(1.0, len(w) / 10.0)

        return {
            "text": text,
            "language": lang,
            "duration": duration,
            "features": features,
            "meta": {
                "encoder": f"whisper-{self.model_size}",
                "segments": len(result.get("segments", [])),
            },
        }


# ==================== 视觉编码器（CLIP / BLIP） ====================

class VisionEncoder:
    """图像 → 文本描述 + 特征向量 编码器

    支持两种模式：
    1. caption：生成图像描述文本（BLIP / GIT 等）
    2. features：提取图像特征向量（CLIP 等）

    优先级：本地BLIP > 远程系统Python > PIL降级
    """

    def __init__(self, model_name: str = "auto", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self.available = False
        self.mode = "none"  # caption / features / both / fallback / remote
        self.use_remote = False
        self._error = "not loaded"
        self._load()

    def _load(self):
        """尝试加载视觉模型，失败则尝试远程，再失败则降级"""
        # 0. 检查是否强制降级（环境变量控制）
        if os.environ.get("MULTIMODAL_OFFLINE", "0") == "1":
            try:
                from PIL import Image
                self.available = True
                self.mode = "fallback"
                return
            except ImportError:
                self.available = False
                return

        # 1. 检查是否有本地缓存的 BLIP 模型
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        has_blip = False
        blip_model_dir = None
        if os.path.exists(cache_dir):
            for f in os.listdir(cache_dir):
                if "blip-image-captioning" in f.lower():
                    # 检查是否有模型权重文件
                    model_path = os.path.join(cache_dir, f, "snapshots")
                    if os.path.exists(model_path):
                        for snap in os.listdir(model_path):
                            snap_path = os.path.join(model_path, snap)
                            if os.path.exists(os.path.join(snap_path, "pytorch_model.bin")) or \
                               os.path.exists(os.path.join(snap_path, "model.safetensors")):
                                has_blip = True
                                blip_model_dir = snap_path
                                break
                    break

        # 2. 尝试本地加载
        try:
            if has_blip or self.model_name != "auto":
                from transformers import BlipProcessor, BlipForConditionalGeneration
                # 尝试加载 BLIP 图像描述模型
                self.processor = BlipProcessor.from_pretrained(
                    "Salesforce/blip-image-captioning-base",
                    local_files_only=has_blip  # 有缓存时只用本地，不联网
                )
                self.model = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-base",
                    local_files_only=has_blip
                ).to(self.device)
                self.available = True
                self.mode = "caption"
                return
        except Exception as e:
            self._error = f"local: {str(e)}"

        # 3. 尝试远程（系统Python 3.8）
        remote_result = _remote_encode("image", __file__)
        if remote_result is not None:
            self.available = True
            self.use_remote = True
            self.mode = "remote"
            return

        # 3. 降级：用 PIL 做简单特征提取
        try:
            from PIL import Image
            self.available = True
            self.mode = "fallback"
        except ImportError:
            self.available = False

    def encode(self, image_path: str) -> Dict:
        """
        图像文件 → 描述文本 + 特征向量

        返回:
            text: 图像描述文本
            features: 图像特征向量（16维，用于注入感官层）
            meta：元信息（尺寸、模式等）
        """
        # 远程模式
        if self.use_remote:
            result = _remote_encode("image", image_path)
            if result is not None and "error" not in result:
                if "features" not in result:
                    result["features"] = [0.0] * 16
                return result
            # 远程失败，降级到PIL
            self.use_remote = False
            self.mode = "fallback"

        if not self.available:
            filename = os.path.basename(image_path)
            return {
                "text": f"[图像: {filename}]",
                "features": [0.0] * 16,
                "meta": {"encoder": "unavailable", "error": self._error},
            }

        if self.mode == "caption":
            try:
                from PIL import Image
                raw_image = Image.open(image_path).convert("RGB")
                inputs = self.processor(raw_image, return_tensors="pt").to(self.device)
                out = self.model.generate(**inputs, max_length=50)
                caption = self.processor.decode(out[0], skip_special_tokens=True)

                # 构造 16 维特征（从图像像素统计）
                features = self._image_to_features(raw_image)

                return {
                    "text": caption,
                    "features": features,
                    "meta": {
                        "encoder": "blip-base",
                        "mode": "caption",
                        "size": raw_image.size,
                    },
                }
            except Exception as e:
                return {
                    "text": f"[图像识别失败: {e}]",
                    "features": [0.0] * 16,
                    "meta": {"encoder": "blip-base", "error": str(e)},
                }
        else:
            # fallback 模式：只用 PIL 提取简单特征
            try:
                from PIL import Image
                img = Image.open(image_path).convert("RGB")
                features = self._image_to_features(img)
                filename = os.path.basename(image_path)
                return {
                    "text": f"[图像: {filename}, {img.size[0]}x{img.size[1]}]",
                    "features": features,
                    "meta": {"encoder": "pil-fallback", "size": img.size},
                }
            except Exception as e:
                return {
                    "text": f"[图像读取失败: {e}]",
                    "features": [0.0] * 16,
                    "meta": {"encoder": "fallback", "error": str(e)},
                }

    def _image_to_features(self, img, dims: int = 16) -> List[float]:
        """图像 → 16 维归一化特征向量（颜色直方图简化版）"""
        from PIL import Image
        img_small = img.resize((4, 4))  # 4x4 = 16 像素
        pixels = list(img_small.getdata())
        # 取每个像素的亮度作为特征
        features = []
        for r, g, b in pixels[:dims]:
            brightness = (r + g + b) / 3 / 255.0
            features.append(brightness)
        # 补齐到 dims 维
        while len(features) < dims:
            features.append(0.0)
        return features[:dims]


# ==================== 便捷函数 ====================

_audio_encoder = None
_vision_encoder = None


def get_audio_encoder(model_size: str = "base") -> AudioEncoder:
    """获取（懒加载）全局音频编码器"""
    global _audio_encoder
    if _audio_encoder is None:
        _audio_encoder = AudioEncoder(model_size=model_size)
    return _audio_encoder


def get_vision_encoder(model_name: str = "auto") -> VisionEncoder:
    """获取（懒加载）全局视觉编码器"""
    global _vision_encoder
    if _vision_encoder is None:
        _vision_encoder = VisionEncoder(model_name=model_name)
    return _vision_encoder


def encode_audio(path: str) -> Dict:
    """便捷函数：编码音频"""
    return get_audio_encoder().encode(path)


def encode_image(path: str) -> Dict:
    """便捷函数：编码图像"""
    return get_vision_encoder().encode(path)
