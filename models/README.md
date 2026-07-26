# models/ — 自定义多模态模型目录

存放自定义多模态编码模型的本地权重与配置。配合 `ai_brain_entity.py` 的
可插拔编码器接口使用（v3.1）。

## 目录约定

```
models/
├── README.md                # 本文件
├── my-clip-finetuned/       # 自定义微调 CLIP（HF 格式目录）
│   ├── config.json
│   ├── preprocessor_config.json
│   └── model.safetensors    # 权重文件（已被 .gitignore 忽略）
├── my-whisper/              # 自定义 Whisper 模型目录
└── encoders/                # 纯 Python 自定义编码器（无需深度学习框架）
    └── my_encoder.py
```

## 使用方式

**1) 本地微调模型目录（transformers / whisper 格式）**

```python
from ai_brain_entity import set_clip_model, set_whisper_model

set_clip_model("models/my-clip-finetuned")   # 指向本地目录而非 HF 名称
set_whisper_model("models/my-whisper")
```

**2) 纯 Python 自定义编码器（零依赖，推荐入门）**

```python
from ai_brain_entity import register_image_encoder
from models.encoders.my_encoder import encode

register_image_encoder(encode, name="mine")
```

**3) 验证接入状态**

```python
from ai_brain_entity import list_encoders
print(list_encoders())
```

## 注意

- 模型权重文件（`*.pt` / `*.bin` / `*.safetensors` / `*.onnx` 等）
  已在 `.gitignore` 中忽略，不会进入版本库。
- 编码器契约：`callable(path: str) -> 数值序列`，长度任意，
  进入感官层前会自动重采样到 16 维。
- 未安装深度学习依赖时无需放置任何模型，通路会自动降级为
  确定性伪 embedding，核心功能不受影响。
