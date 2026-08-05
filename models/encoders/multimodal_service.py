# -*- coding: utf-8 -*-
"""
多模态编码器服务（用系统Python 3.8运行，有whisper/transformers）

用法：
    py -3.8 multimodal_service.py <mode> <input_path>
    mode: audio / image / text

输出：JSON格式的编码结果
"""
import sys
import json
import os


def encode_audio(audio_path: str) -> dict:
    """编码音频，返回结果字典"""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from multimodal import AudioEncoder
    encoder = AudioEncoder(model_size="base")
    return encoder.encode(audio_path)


def encode_image(image_path: str) -> dict:
    """编码图像，返回结果字典"""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from multimodal import VisionEncoder
    # 优先使用本地缓存的 BLIP 模型
    encoder = VisionEncoder(model_name="auto")
    return encoder.encode(image_path)


def encode_text(text: str) -> dict:
    """编码文本，返回结果字典"""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from multimodal import LanguageEncoder
    encoder = LanguageEncoder(model_name="Qwen/Qwen2-0.5B")
    return encoder.encode(text)


def generate_text(arg: str) -> dict:
    """生成文本，返回结果字典
    arg 格式: max_length|temperature|prompt
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from multimodal import LanguageEncoder
    encoder = LanguageEncoder(model_name="Qwen/Qwen2-0.5B")

    # 解析参数
    parts = arg.split("|", 2)
    if len(parts) == 3:
        max_length = int(parts[0])
        temperature = float(parts[1])
        prompt = parts[2]
    else:
        max_length = 100
        temperature = 0.7
        prompt = arg

    return encoder.generate(prompt, max_length=max_length, temperature=temperature)


def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: multimodal_service.py <mode> <path>"}))
        sys.exit(1)

    mode = sys.argv[1]
    path = sys.argv[2]

    # 设置镜像源（如果需要）
    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    try:
        if mode == "audio":
            result = encode_audio(path)
        elif mode == "image":
            result = encode_image(path)
        elif mode == "text":
            result = encode_text(path)
        elif mode == "generate":
            result = generate_text(path)
        else:
            result = {"error": f"Unknown mode: {mode}"}
    except Exception as e:
        result = {"error": str(e)}

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
