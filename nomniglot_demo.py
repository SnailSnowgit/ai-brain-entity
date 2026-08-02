# -*- coding: utf-8 -*-
"""N-Omniglot 真实神经形态数据驱动 AI 大脑实体（v5.1 演示）。

两部分：
1) 少样本基准：k-shot 最近原型分类（k=1..5），验证导出数据可区分；
2) 大脑驱动：真实笔画事件向量经可学习投影进入 AIBrainEntity 感官层，
   学习期带奖励，测试期无标签感知，报告脉冲/突触/记忆/新奇度变化。

运行: python nomniglot_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ai_brain_entity import AIBrainEntity
from nomniglot import (classes_of, few_shot_split, load,
                       nearest_prototype_accuracy, sample_to_vector, stats)


def main():
    data = load()
    samples, meta = data["samples"], data["meta"]
    print("=" * 60)
    print("N-Omniglot 真实事件数据 × AIBrainEntity")
    print("=" * 60)
    print(f"数据: {meta['alphabet']} 字符集, {meta['num_classes']} 类 × "
          f"{meta['samples_per_class']} 样本, {meta['frames_per_sample']} 帧 × "
          f"{meta['grid']}×{meta['grid']} 网格")
    st = stats(samples)
    print(f"样本事件数: min={st['events_min']} mean={st['events_mean']} "
          f"max={st['events_max']}, 平均时长 {st['duration_s_mean']}s")

    # ---- 1) 少样本基准 ----
    print("\n[1/2] k-shot 最近原型基准（13 类，随机=%.1f%%）"
          % (100 / len(classes_of(samples))))
    for k in (1, 2, 3, 5):
        sup, qry = few_shot_split(samples, k, seed=7)
        acc = nearest_prototype_accuracy(sup, qry)
        print(f"  {k}-shot: {acc * 100:.1f}%  (support={len(sup)}, query={len(qry)})")

    # ---- 2) 大脑驱动 ----
    print("\n[2/2] 真实事件向量驱动 AIBrainEntity（可学习投影通路）")
    brain = AIBrainEntity("nomniglot_brain", seed=42)
    support, query = few_shot_split(samples, 5, seed=7)

    syn0 = brain.synapse_mean()
    for s in support:
        vec = sample_to_vector(s)
        brain.sensory_input_vector(vec, label=f"拉丁字符{s['class']:02d}")
        brain.reward(0.6)  # 学习期奖励调制
    sp = brain.spike_counts()
    print(f"  学习 {len(support)} 个样本后: 突触均值 {syn0:.3f} -> "
          f"{brain.synapse_mean():.3f}, 脉冲计数 感/联/决 = {sp}")

    # 新奇度：未见过的内容首次出现新奇度高，重复后回落
    probe = sample_to_vector(query[0])
    nov = []
    for rank in range(4):
        brain.sensory_input_vector(probe, label="未见字符Z")
        nov.append(brain.novelty)
    print(f"  未见内容重复 4 次, 新奇度: "
          + " -> ".join(f"{v:.2f}" for v in nov))

    # 测试期无标签感知
    hits_before = len(brain.long_memory)
    for s in query[:20]:
        brain.sensory_input_vector(sample_to_vector(s))
    print(f"  测试期感知 20 个无标签样本: LTM 条目 {hits_before} -> "
          f"{len(brain.long_memory)}, tick = {brain.tick}")
    out = brain.express("拉丁字符")
    print(f"  表达: {out['utterance'][:80]}")
    print("\n完成。")


if __name__ == "__main__":
    main()
