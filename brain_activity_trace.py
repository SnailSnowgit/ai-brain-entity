# -*- coding: utf-8 -*-
"""
大脑活动追踪导出器：录制一段真实的"感知-学习"过程，
导出每个网络步的完整神经状态（膜电位/脉冲/情绪/注意力/记忆），
供大脑活动可视化 Widget 回放。纯本地，无任何外部 API。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ai_brain_entity import AIBrainEntity

BASE = Path(__file__).parent

STIMULI = [
    "火焰是危险的",
    "火焰是危险的",          # 重复：观察 STDP 强化后的通路放大
    "记忆是智慧的基石",
    "神经元脉冲正在传递信号",
    "记忆是智慧的基石",      # 重复：观察联想回忆
    "今天外面的天气很好",
]


def main() -> None:
    brain = AIBrainEntity("LiveBrain", seed=42)

    # 预训练：让突触产生强弱分化，画面更有结构
    for s in ("火焰是危险的", "记忆是智慧的基石"):
        for _ in range(12):
            brain.sensory_input(s)

    frames = []
    all_neurons = brain.sense_layer + brain.assoc_layer + brain.decision_layer

    orig_step = brain._network_step
    def wrapped(external=None):
        orig_step(external)
        frames.append({
            "pot": [round(n.potential, 3) for n in all_neurons],
            "spk": [[n.id for n in layer if n.spike] for layer in
                    (brain.sense_layer, brain.assoc_layer, brain.decision_layer)],
            "tick": brain.tick,
            "emo": {k: round(v, 3) for k, v in brain.emotion.items()},
            "att": round(brain.attention_factor, 3),
            "dopa": round(brain.dopamine, 3),
            "stm": len(brain.short_memory),
            "ltm": len(brain.long_memory),
        })
    brain._network_step = wrapped

    def backfill(start: int) -> None:
        """tick 结束后，把该 tick 所有帧的 tick 级状态回填为更新后的值。
        _perceive/free_run 中 emo/att/dopa/stm/ltm 在全部网络步之后才更新，
        wrapped() 录帧时这些量还是上一 tick 的旧值（时序错位），必须回填。"""
        for fr in frames[start:]:
            fr["emo"] = {k: round(v, 3) for k, v in brain.emotion.items()}
            fr["att"] = round(brain.attention_factor, 3)
            fr["dopa"] = round(brain.dopamine, 3)
            fr["stm"] = len(brain.short_memory)
            fr["ltm"] = len(brain.long_memory)

    outputs = []
    for s in STIMULI:
        n_frames_before = len(frames)
        out = brain.sensory_input(s)
        backfill(n_frames_before)   # 回填本 tick 全部帧的 tick 级状态
        outputs.append({"tick": brain.tick, "stimulus": s,
                        "output": out, "frame": n_frames_before})
    for _ in range(4):              # 尾声：观察回响衰减（逐 tick 回填）
        n_frames_before = len(frames)
        brain.free_run(1)
        backfill(n_frames_before)
    brain._network_step = orig_step

    # 为每个 tick 生成可读的脉冲思考链（与 thought_chain 同构），
    # 供可视化面板随播放进度逐行高亮"大脑在想什么"
    NSTEPS = 1 + brain.settle_ticks   # 每 tick 的网络步数（刺激步 + 回响步）
    for o in outputs:
        f0 = o["frame"]
        lines = [f"1. 感官编码：{o['stimulus']!r} → 16 维输入电流"]
        for i in range(NSTEPS):
            fr = frames[f0 + i]
            ns, na, nd = (len(x) for x in fr["spk"])
            if i == 0:
                lines.append(f"2. 刺激步：感官层 {ns}/16 → 联想层 {na}/32"
                             f"（突触汇集） → 决策层 {nd}/8")
            else:
                silent = "（回声衰减，趋于静息）" if not (ns or na or nd) else ""
                lines.append(f"{i + 2}. 回响+{i}：感官 {ns} / 联想 {na} / "
                             f"决策 {nd}{silent}")
        lines.append(f"{NSTEPS + 2}. 决策输出：{o['output']}")
        o["chain"] = lines
        o["n_frames"] = NSTEPS

    # 邻接表（突触连接）：前神经元 -> [后神经元]
    def adj(out_dict):
        return {str(k): v for k, v in sorted(out_dict.items())}

    trace = {
        "meta": {
            "name": brain.name,
            "layers": [16, 32, 8],
            "stimuli": STIMULI,
            "outputs": outputs,
            "n_frames": len(frames),
        },
        "edges": {
            "ff": adj(brain._ff_out),
            "rec": adj(brain._recurrent_out),
        },
        "frames": frames,
    }

    DATA = BASE / "datasets"
    DATA.mkdir(exist_ok=True)
    out_path = DATA / "brain_activity_trace.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, separators=(",", ":"))
    print(f"frames={len(frames)}  outputs={len(outputs)}  -> {out_path}")
    print(f"size={out_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
