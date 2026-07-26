# -*- coding: utf-8 -*-
"""
脉冲思考链（Spike CoT）场景数据导出器：用 thought_chain() 录制三种
典型场景下"感知 → 传导 → 回响 → 决策"的完整脉冲因果链，供
「脉冲思考链 · Spike CoT」可视化 Widget 播放。纯本地，无任何外部 API。

三种场景（同一刺激/对照设计）：
  1. cold    —— 冷脑首次接触：突触未强化，脉冲几乎无法传导
  2. trained —— 40 轮训练后的已学刺激：脉冲传遍三层并涌现决策
  3. novel   —— 同一训练脑对未见新刺激：传导弱、回响快速衰减
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ai_brain_entity import AIBrainEntity

BASE = Path(__file__).parent

STIM = "危险！森林起火了"
NOVEL = "今天天气晴朗"
TRAIN_ROUNDS = 40
SETTLE_TICKS = 5


def trace(brain: AIBrainEntity, stim: str) -> dict:
    """录制一次思考链：每步三层脉冲 id + 可读链 + 决策输出"""
    tc = brain.thought_chain(stim)
    return {
        "input": tc["input"],
        "steps": [{"sense": s, "assoc": a, "decision": d}
                  for s, a, d in tc["steps"]],
        "chain": tc["chain"],
        "output": tc["output"],
    }


def main() -> None:
    # 场景一：冷脑首次接触（突触未强化）
    cold = AIBrainEntity("cold", seed=42)
    cold.settle_ticks = SETTLE_TICKS
    s_cold = trace(cold, STIM)

    # 场景二/三：同一个脑，先训练再分别测已学刺激与新刺激
    trained = AIBrainEntity("trained", seed=42)
    trained.settle_ticks = SETTLE_TICKS
    for _ in range(TRAIN_ROUNDS):
        trained.sensory_input(STIM)
        trained.free_run(4)
    s_trained = trace(trained, STIM)
    s_novel = trace(trained, NOVEL)

    data = {
        "meta": {
            "layers": {"sense": 16, "assoc": 32, "decision": 8},
            "training": f"场景二/三的脑经 {TRAIN_ROUNDS} 轮「{STIM}」+ 自由回响训练（seed=42）",
            "source": "ai_brain_entity.py thought_chain() 真实运行输出",
        },
        "scenarios": [
            {"id": "cold", "label": "冷脑 · 首次接触",
             "desc": "突触未强化，刺激几乎无法传导", "data": s_cold},
            {"id": "trained", "label": "训练后 · 已学刺激",
             "desc": "40 轮训练后，脉冲沿突触传遍三层并触发决策", "data": s_trained},
            {"id": "novel", "label": "训练后 · 未见新刺激",
             "desc": "同一脑对新句子：传导弱、回响快速衰减", "data": s_novel},
        ],
    }

    DATA = BASE / "data"
    DATA.mkdir(exist_ok=True)
    out_path = DATA / "thought_chain_scenarios.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"scenarios={len(data['scenarios'])}  "
          f"steps={[len(s['data']['steps']) for s in data['scenarios']]}  "
          f"-> {out_path}")
    print(f"size={out_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
