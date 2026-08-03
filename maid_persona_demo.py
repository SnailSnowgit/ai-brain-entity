# -*- coding: utf-8 -*-
"""maid_persona_demo —— 女仆人格 × 大脑实体 集成演示。

场景：主人归来 → 新鲜事 → 夸奖 → 责骂，展示情绪驱动的女仆话术选择。
运行：python maid_persona_demo.py
"""

from ai_brain_entity import AIBrainEntity
from persona_maid import (load_persona, maid_express, maid_feedback,
                          maid_greeting)


def main():
    persona = load_persona()
    brain = AIBrainEntity("小铃")
    name = persona["persona"]["name"]
    print(f"=== 女仆人格演示：{name} ===\n")

    # 1. 问候
    print(maid_greeting(persona, seed=brain.tick), "\n")

    # 2. 不同刺激下的表达（先感知，再决策表达）
    for stim in ["主人回来啦", "窗外有一只没见过的鸟", "晚饭想吃什么"]:
        brain.sensory_input(stim)
        out = maid_express(brain, stim, persona)
        act = out["action"]
        print(f"[刺激] {stim}")
        print(f"  动作={act['action']} 情绪={act['mood']} 强度={act['intensity']}")
        print(f"  {name}: {out['utterance']}\n")

    # 3. 夸奖 → 奖励 → 愉悦
    brain.reward(0.8)
    brain.reward(0.8)
    brain.sensory_input("再来一杯茶")
    print("[主人夸奖了她] reward(0.8)×2")
    print(f"  {name}: {maid_feedback(persona, 'praised', seed=brain.tick)}")
    out = maid_express(brain, "再来一杯茶", persona)
    print(f"  情绪={out['action']['mood']}")
    print(f"  {name}: {out['utterance']}\n")

    # 4. 责骂 → 惩罚 → 紧张
    brain.reward(-0.6)
    brain.reward(-0.6)
    brain.sensory_input("把地板擦干净")
    print("[主人责骂了她] reward(-0.6)×2")
    print(f"  {name}: {maid_feedback(persona, 'scolded', seed=brain.tick)}")
    out = maid_express(brain, "把地板擦干净", persona)
    print(f"  情绪={out['action']['mood']}")
    print(f"  {name}: {out['utterance']}\n")

    print(f"=== tick={brain.tick} 完成 ===")


if __name__ == "__main__":
    main()
