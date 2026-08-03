# -*- coding: utf-8 -*-
"""persona_maid —— 女仆人格数据集加载器 + 大脑实体表达桥接（零第三方依赖）。

数据集 data/persona_maid.json 结构：
  meta        — 名称 / 版本 / 许可证
  persona     — 人格档案（小铃：口癖、职责、禁忌）
  utterances  — 话术矩阵 [mood][verb] -> List[str]
                mood ∈ {calm, curiosity, stress, pleasure}
                verb ∈ {respond, acknowledge, observe}
                （维度与 AIBrainEntity.decide_action 的输出一一对应）
  feedback    — 受夸 praised / 受责 scolded 反应
  greetings   — 问候语
  scenes      — 多轮情景对话样例

用法：
    from persona_maid import load_persona, utterance, maid_express
    persona = load_persona()
    line = utterance(persona, "curiosity", "respond")
    out  = maid_express(brain, "主人回来啦")   # brain: AIBrainEntity
"""

import json
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent / "data" / "persona_maid.json"

MOODS = ("calm", "curiosity", "stress", "pleasure")
VERBS = ("respond", "acknowledge", "observe")
FEEDBACK_KINDS = ("praised", "scolded")


def load_persona(path=None):
    """加载并校验女仆人格数据集，返回 dict。"""
    p = Path(path) if path else DEFAULT_PATH
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    _validate(data)
    return data


def _validate(data):
    for key in ("persona", "utterances", "feedback", "greetings", "scenes"):
        if key not in data:
            raise ValueError(f"persona_maid.json 缺少键: {key}")
    utt = data["utterances"]
    for mood in MOODS:
        if mood not in utt:
            raise ValueError(f"utterances 缺少情绪: {mood}")
        for verb in VERBS:
            lines = utt[mood].get(verb)
            if not isinstance(lines, list) or not lines:
                raise ValueError(f"utterances[{mood}][{verb}] 为空")
            if any(not isinstance(s, str) or not s.strip() for s in lines):
                raise ValueError(f"utterances[{mood}][{verb}] 含空条目")
    for kind in FEEDBACK_KINDS:
        if not data["feedback"].get(kind):
            raise ValueError(f"feedback 缺少: {kind}")


def utterance(persona, mood, verb, seed=0):
    """按 (情绪, 动作动词) 确定性选取一条话术。seed 通常为 brain.tick。"""
    if mood not in MOODS:
        mood = "calm"
    if verb not in VERBS:
        verb = "observe"
    lines = persona["utterances"][mood][verb]
    return lines[seed % len(lines)]


def maid_feedback(persona, kind, seed=0):
    """受夸/受责反应。kind ∈ {praised, scolded}。"""
    if kind not in FEEDBACK_KINDS:
        raise ValueError(f"未知反馈类型: {kind}")
    lines = persona["feedback"][kind]
    return lines[seed % len(lines)]


def maid_greeting(persona, seed=0):
    lines = persona["greetings"]
    return lines[seed % len(lines)]


def maid_express(brain, stimulus="", persona=None):
    """女仆风格表达桥接：decide_action 的 (mood × verb) → 人格话术。

    复用大脑实体的决策输出；若联想起记忆，追加一句女仆式引用。
    返回 {"action": <decide_action 结果>, "utterance": str}。
    """
    if persona is None:
        persona = load_persona()
    act = brain.decide_action(stimulus)
    line = utterance(persona, act["mood"], act["verb"], seed=act["tick"])
    if act["recalled"]:
        line += f"（小铃想起了「{act['recalled'][0]}」……）"
    return {"action": act, "utterance": line}
