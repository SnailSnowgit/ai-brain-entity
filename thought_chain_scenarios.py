# -*- coding: utf-8 -*-
"""
脉冲思考链（Spike CoT）场景数据导出器：用 thought_chain() 记录多种
典型场景下"感知 → 传导 → 回响 → 决策"的完整脉冲因果链，供
「脉冲思考链 · Spike CoT」可视化 Widget 播放。纯本地，无任何外部 API。

场景列表：
  基础学习场景（同一刺激对照）：
    1. cold    —— 冷脑首次接触：突触未强化，脉冲几乎无法传导
    2. trained —— 40 轮训练后的已学刺激：脉冲传遍三层并触发决策
    3. novel   —— 同一训练脑对新刺激：传导弱、回响快速衰减

  好奇驱动场景（v4.9）：
    4. curious_new  —— 全新刺激触发注意捕获：高新奇度→高好奇心→注意力放大
    5. curious_old  —— 熟悉刺激：低新奇度→好奇心回落→注意力收窄
    6. after_rpe    —— 大奖励预测误差后：意外感提升后续刺激的新奇度

  人格差异场景（v4.9.1）：
    7. high_sss —— 高寻求刺激者：新刺激引发更强好奇心与探索
    8. low_sss  —— 低寻求刺激者：新刺激反应温和，偏好熟悉

  习惯化场景（v4.9.1）：
    9.  hab_first —— 习惯化脑第1次见广告：新奇度高
    10. hab_fifth —— 同一脑第5次见：新奇度因习惯化显著衰减
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ai_brain_entity import AIBrainEntity

BASE = Path(__file__).parent

STIM = "危险！森林起火了"
NOVEL = "今天天气晴朗"
TRAIN_ROUNDS = 40
SETTLE_TICKS = 5


def trace(brain: AIBrainEntity, stim: str) -> dict:
    """记录一次思考链：每步三层脉冲 id + 可读链 + 决策输出 + 内部状态快照"""
    # 记录感知前的状态
    pre_state = {
        "novelty": round(brain.novelty, 3),
        "curiosity": round(brain.emotion["curiosity"], 3),
        "attention": round(brain.attention_factor, 3),
        "dopamine": round(brain.dopamine, 3),
        "value_estimate": round(brain.value_estimate, 3),
        "epsilon": round(brain.effective_epsilon(), 3),
    }
    tc = brain.thought_chain(stim)
    # 感知后的状态
    post_state = {
        "novelty": round(brain.novelty, 3),
        "curiosity": round(brain.emotion["curiosity"], 3),
        "attention": round(brain.attention_factor, 3),
        "dopamine": round(brain.dopamine, 3),
        "value_estimate": round(brain.value_estimate, 3),
        "epsilon": round(brain.effective_epsilon(), 3),
    }
    return {
        "input": tc["input"],
        "steps": [{"sense": s, "assoc": a, "decision": d}
                  for s, a, d in tc["steps"]],
        "chain": tc["chain"],
        "output": tc["output"],
        "state": {"before": pre_state, "after": post_state},
    }


def main() -> None:
    scenarios = []

    # ========== 基础学习场景 ==========

    # 1. 冷脑首次接触
    cold = AIBrainEntity("cold", seed=42)
    cold.settle_ticks = SETTLE_TICKS
    scenarios.append({
        "id": "cold", "label": "冷脑 · 首次接触",
        "group": "基础学习",
        "desc": "突触未强化，刺激几乎无法传导到联想层和决策层",
        "data": trace(cold, STIM),
    })

    # 2/3. 训练后的脑
    trained = AIBrainEntity("trained", seed=42)
    trained.settle_ticks = SETTLE_TICKS
    for _ in range(TRAIN_ROUNDS):
        trained.sensory_input(STIM)
        trained.free_run(4)
    scenarios.append({
        "id": "trained", "label": "训练后 · 已学刺激",
        "group": "基础学习",
        "desc": "40轮训练后，脉冲沿强化突触传遍三层并触发决策",
        "data": trace(trained, STIM),
    })
    scenarios.append({
        "id": "novel", "label": "训练后 · 新刺激",
        "group": "基础学习",
        "desc": "同一脑对未训练过的新句子：传导弱、回响快速衰减",
        "data": trace(trained, NOVEL),
    })

    # ========== 好奇驱动场景 ==========

    # 4. 全新刺激触发注意捕获
    curious_brain = AIBrainEntity("curious", seed=42, habituation_rate=0.0)
    curious_brain.settle_ticks = SETTLE_TICKS
    # 先训练一些无关内容建立基础活动
    for _ in range(10):
        curious_brain.sensory_input("日常背景噪音")
        curious_brain.free_run(2)
    scenarios.append({
        "id": "curious_new", "label": "好奇 · 全新刺激",
        "group": "好奇驱动",
        "desc": "全新刺激：高新奇度→好奇心上升→注意力因子放大→更多脉冲注入",
        "data": trace(curious_brain, "从未见过的紫色极光在天空舞动"),
    })

    # 5. 熟悉刺激（好奇心回落）
    familiar_brain = AIBrainEntity("familiar", seed=42, habituation_rate=0.0)
    familiar_brain.settle_ticks = SETTLE_TICKS
    for _ in range(30):
        familiar_brain.sensory_input("每天走过的那条小路")
        familiar_brain.free_run(2)
    scenarios.append({
        "id": "curious_old", "label": "好奇 · 熟悉刺激",
        "group": "好奇驱动",
        "desc": "烂熟于心的刺激：新奇度归零→好奇心回落→注意力收窄→脉冲减少",
        "data": trace(familiar_brain, "每天走过的那条小路"),
    })

    # 6. 大RPE后的意外感
    rpe_brain = AIBrainEntity("rpe", seed=42)
    rpe_brain.settle_ticks = SETTLE_TICKS
    for _ in range(15):
        rpe_brain.sensory_input("按下按钮得到小奖励")
        rpe_brain.reward_td(0.3)
        rpe_brain.free_run(2)
    # 突然给一个大负奖励（意外！）
    rpe_brain.sensory_input("按下按钮得到小奖励")
    rpe_brain.reward_td(-0.8)  # 大负RPE
    scenarios.append({
        "id": "after_rpe", "label": "好奇 · 意外之后",
        "group": "好奇驱动",
        "desc": "刚经历大预测误差(|RPE|≈1.1)后，下一刺激的新奇度被意外感抬高",
        "data": trace(rpe_brain, "按钮旁边出现了一个新开关"),
    })

    # ========== 人格差异场景 ==========
    # 两个脑先经相同基础训练建立突触通路，再面对同一新刺激
    # （冷脑都无脉冲，人格差异需在有基础活动的脑上才能看出脉冲数差异）

    # 7. 高寻求刺激者
    explorer = AIBrainEntity("explorer", seed=42,
                             sensation_seeking=0.9, habituation_rate=0.0)
    explorer.settle_ticks = SETTLE_TICKS
    for _ in range(20):
        explorer.sensory_input("日常散步看到的风景")
        explorer.free_run(2)
    scenarios.append({
        "id": "high_sss", "label": "人格 · 探险家(高SSS)",
        "group": "人格差异",
        "desc": "高寻求刺激者：新刺激引发更强好奇心(×2.0增益)和更高探索率(ε=0.22)",
        "data": trace(explorer, "神秘的地下洞穴入口"),
    })

    # 8. 低寻求刺激者
    homebody = AIBrainEntity("homebody", seed=42,
                             sensation_seeking=0.1, habituation_rate=0.0)
    homebody.settle_ticks = SETTLE_TICKS
    for _ in range(20):
        homebody.sensory_input("日常散步看到的风景")
        homebody.free_run(2)
    scenarios.append({
        "id": "low_sss", "label": "人格 · 保守者(低SSS)",
        "group": "人格差异",
        "desc": "低寻求刺激者：同一新刺激反应温和(×0.5增益)，更偏好熟悉",
        "data": trace(homebody, "神秘的地下洞穴入口"),
    })

    # ========== 习惯化场景 ==========

    # 9/10. 习惯化：第1次 vs 第5次
    hab_brain = AIBrainEntity("habituating", seed=42,
                              habituation_rate=0.4, sensation_seeking=0.5)
    hab_brain.settle_ticks = SETTLE_TICKS
    # 先建立基础突触活动
    for _ in range(15):
        hab_brain.sensory_input("看电视节目")
        hab_brain.free_run(2)
    ad_stim = "限时特惠！买一送一！"
    # 第1次
    scenarios.append({
        "id": "hab_first", "label": "习惯化 · 第1次",
        "group": "习惯化",
        "desc": "第一次看到广告：新奇度0.70，好奇心上升，注意力放大",
        "data": trace(hab_brain, ad_stim),
    })
    # 第2-4次（不记录，只累积习惯化）
    for _ in range(3):
        hab_brain.sensory_input(ad_stim)
        hab_brain.free_run(1)
    # 第5次
    scenarios.append({
        "id": "hab_fifth", "label": "习惯化 · 第5次",
        "group": "习惯化",
        "desc": "第五次看到同一广告：新奇度因习惯化(1/(1+4×0.4)≈0.38)和记忆命中双重衰减",
        "data": trace(hab_brain, ad_stim),
    })

    # ========== 组装输出 ==========

    data = {
        "meta": {
            "layers": {"sense": 16, "assoc": 32, "decision": 8},
            "settle_ticks": SETTLE_TICKS,
            "groups": {
                "基础学习": "冷脑vs训练脑vs新刺激的突触传导对比",
                "好奇驱动": "新奇度→好奇心→注意捕获→探索率的完整通路",
                "人格差异": "寻求刺激(SSS)人格参数对同一刺激的不同反应",
                "习惯化": "反复暴露同一刺激后新奇度的指数衰减",
            },
            "source": "ai_brain_entity.py thought_chain() 真实运行输出",
            "version": "v4.9.1",
        },
        "scenarios": scenarios,
    }

    DATA = BASE / "data"
    DATA.mkdir(exist_ok=True)
    out_path = DATA / "thought_chain_scenarios.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"scenarios={len(scenarios)}  groups={len(data['meta']['groups'])}")
    for s in scenarios:
        n_steps = len(s["data"]["steps"])
        nov = s["data"]["state"]["after"]["novelty"]
        att = s["data"]["state"]["after"]["attention"]
        print(f"  [{s['group']}] {s['id']:14s} steps={n_steps}  "
              f"novelty={nov:.2f} attention={att:.2f}")
    print(f"-> {out_path}")
    print(f"size={out_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
