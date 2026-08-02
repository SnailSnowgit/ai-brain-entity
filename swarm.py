# -*- coding: utf-8 -*-
"""
群体文化传递实验层（基于 ai_brain_entity.BrainSwarm）v3.0
==============================================================
ai_brain_entity.py 内置的 BrainSwarm 提供基础群体能力（文化传递 culture_round、
广播 broadcast、繁衍 reproduce）。本模块在其之上提供可量化的文化演化实验工具：

  - transmit()      ：定向文化传递（供体 -> 受体），带保真度控制——
                      记忆权重按 fidelity 打折，内容以 (1-fidelity) 概率
                      发生单字变异（文化漂移 / meme 变异）。
  - meme_trace()    ：追踪某文化主题在群体中的分布（序列相似度匹配）。
  - generation_chain()：文化沿世代链逐代传递，可选 rehearse（温习）——
                      温习使文化记忆重新固化进 LTM；不温习的文化随权重
                      逐代衰减跌破固化阈值而灭绝（文化消亡对照）。
  - cultural_similarity()：两个实体 LTM 内容的文化重合度（Jaccard）。

依赖：原生 Python + ai_brain_entity.py（v3.0 核心）。
"""

import time
import random
import difflib
from typing import List, Dict, Optional, Tuple

from ai_brain_entity import AIBrainEntity, BrainMemory, BrainSwarm

# 文化变异用的替换字符池
_MUTATION_CHARS = ("的一是了我不人在他有这个上们来到时大地为子中你说生国年"
                   "着就那和要她出也得里后自以会家可下而过天去能对小多然于"
                   "心学么之都好看起发当没成只如事把还用第样道想作种开")


# ------------------ 定向文化传递 ------------------

def _mutate(content: str, mutation_rate: float, rng: random.Random) -> str:
    """文化漂移：以 mutation_rate 概率随机替换一个字符"""
    if not content or rng.random() >= mutation_rate:
        return content
    pos = rng.randrange(len(content))
    ch = _MUTATION_CHARS[rng.randrange(len(_MUTATION_CHARS))]
    return content[:pos] + ch + content[pos + 1:]


def transmit(donor: AIBrainEntity, receiver: AIBrainEntity,
             top_k: int = 5, fidelity: float = 0.9,
             seed: Optional[int] = None) -> List[Tuple[str, float]]:
    """定向文化传递：donor 经 DNA 快照把 top_k 条最强长期记忆教给 receiver。

    fidelity: 传递保真度——记忆权重按 fidelity 打折，内容以 (1-fidelity)
              概率发生单字变异（文化漂移）。
    高权重文化记忆直接固化进受体 LTM；弱记忆进入 STM 参与容量竞争。
    返回 [(传递后的内容, 权重), ...]。
    """
    rng = random.Random(seed)
    dna = donor.dump_dna()
    ltm = sorted(dna["long_memory"], key=lambda m: m["weight"],
                 reverse=True)[:top_k]
    transmitted: List[Tuple[str, float]] = []
    for m in ltm:
        content = _mutate(m["content"], 1.0 - fidelity, rng)
        weight = AIBrainEntity._clip(m["weight"] * fidelity)
        mem = BrainMemory(content=content, timestamp=time.time(),
                          weight=weight, tag="culture")
        if weight >= receiver.stm_consolidate_threshold:
            receiver._consolidate_to_ltm(mem)
        else:
            receiver.short_memory.append(mem)
        transmitted.append((content, weight))

    # STM 溢出：按核心规则竞争（强者固化、弱者遗忘）
    while len(receiver.short_memory) > receiver.max_stm:
        weakest = min(receiver.short_memory, key=lambda m: m.weight)
        receiver.short_memory.remove(weakest)
        if weakest.weight >= receiver.stm_consolidate_threshold:
            receiver._consolidate_to_ltm(weakest)
    return transmitted


# ------------------ 群体分析 ------------------

def cultural_similarity(a: AIBrainEntity, b: AIBrainEntity) -> float:
    """两个实体长期记忆内容的文化重合度（Jaccard）"""
    sa = {m.content for m in a.long_memory}
    sb = {m.content for m in b.long_memory}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def meme_trace(brains: List[AIBrainEntity],
               keyword: str) -> List[Tuple[str, float, float]]:
    """追踪某文化主题在群体中的分布。
    返回 [(实体名, 最佳匹配度, 记忆权重), ...]，匹配度用序列相似度。
    """
    trace = []
    for b in brains:
        best_ratio, best_w = 0.0, 0.0
        for m in b.long_memory:
            r = difflib.SequenceMatcher(None, keyword, m.content).ratio()
            if r > best_ratio:
                best_ratio, best_w = r, m.weight
        trace.append((b.name, best_ratio, best_w))
    return trace


# ------------------ 世代链实验 ------------------

def _generation_report(brain: AIBrainEntity, gen: int,
                       memes: List[str]) -> Dict:
    matched, mutated = [], 0
    for meme in memes:
        best, best_r = None, 0.0
        for m in brain.long_memory:
            r = difflib.SequenceMatcher(None, meme, m.content).ratio()
            if r > best_r:
                best_r, best = r, m
        if best is not None and best_r >= 0.6:
            matched.append((best.content, round(best.weight, 3),
                            round(best_r, 2)))
            if best.content != meme:
                mutated += 1
    return {
        "generation": gen,
        "survived": len(matched),
        "mutated": mutated,
        "avg_weight": round(sum(w for _, w, _ in matched) / len(matched), 3)
                      if matched else 0.0,
        "memories": matched,
    }


def generation_chain(swarm: BrainSwarm, memes: List[str],
                     teach_times: int = 25, fidelity: float = 0.85,
                     seed: int = 1, rehearse_times: int = 3) -> List[Dict]:
    """文化世代传递实验：第 0 代学会全部 memes，然后沿种群逐代传递。

    rehearse_times: 受体被教后温习所学内容的次数。温习让文化记忆重新
    经感知通路强化并固化进 LTM——不温习的文化随权重逐代衰减
    （×fidelity/代）跌破固化阈值而灭绝（文化消亡对照）。

    返回每代的文化存活报告列表。
    """
    population = swarm.population
    # 第 0 代：学习全部文化记忆
    for meme in memes:
        for _ in range(teach_times):
            population[0].sensory_input(meme)

    report = [_generation_report(population[0], 0, memes)]
    for g in range(1, len(population)):
        got = transmit(population[g - 1], population[g],
                       top_k=len(memes), fidelity=fidelity, seed=seed + g)
        # 温习：受体反复感知所学内容，触发 STM->LTM 固化
        for content, _ in got:
            for _ in range(rehearse_times):
                population[g].sensory_input(content)
        report.append(_generation_report(population[g], g, memes))
    return report


# ===================== 共识涌现相变扫描（v4.1） =====================

def consensus_phase_scan(sizes=(3, 5, 8, 12),
                         topologies=("fully_connected", "ring",
                                     "small_world", "random"),
                         meme: str = "钻木可以取火",
                         teach_times: int = 25,
                         max_rounds: int = 60,
                         threshold: float = 0.9,
                         p: float = 0.3,
                         seed: int = 1) -> List[Dict]:
    """共识涌现的相变条件扫描：种群规模 × 连接拓扑 → 收敛速度。

    对每个 (规模, 拓扑) 组合：构建种群 → 0 号个体学会模因 → 设定拓扑 →
    测量模因覆盖率达到 threshold 所需的水平传播轮数（consensus_convergence）。

    返回每格 {"size", "topology", "rounds", "converged", "mean_degree",
             "coverage"} 的列表，可直接制表/绘图。
    """
    grid = []
    for size in sizes:
        names = [f"N{i}" for i in range(size)]
        for topo in topologies:
            swarm = BrainSwarm(names, seed=seed)
            for _ in range(teach_times):
                swarm.population[0].sensory_input(meme)
            adj = swarm.set_topology(topo, p=p, seed=seed)
            mean_degree = (sum(len(v) for v in adj.values()) / size
                           if size else 0.0)
            conv = swarm.consensus_convergence(
                meme, max_rounds=max_rounds, threshold=threshold,
                direction="horizontal")
            grid.append({
                "size": size,
                "topology": topo,
                "rounds": conv["rounds"],
                "converged": conv["converged"],
                "mean_degree": round(mean_degree, 2),
                "coverage": conv["coverage"],
            })
    return grid


# ===================== 演示 =====================

if __name__ == "__main__":
    swarm = BrainSwarm(["长者", "学徒", "孩童"], seed=42)
    memes = ["钻木可以取火", "结绳可以记事", "观测星象定节气"]

    print("--- 第 0 代学习文化知识 ---")
    for meme in memes:
        for _ in range(30):
            swarm.population[0].sensory_input(meme)
        print(f"  长者学会: {meme}")
    print(f"  长者 LTM={len(swarm.population[0].long_memory)} 条")

    print("\n--- 文化传递：长者 -> 学徒 -> 孩童（保真度 0.85）---")
    for g in (1, 2):
        got = transmit(swarm.population[g - 1], swarm.population[g],
                       top_k=3, fidelity=0.85, seed=g)
        d, r = swarm.population[g - 1].name, swarm.population[g].name
        for content, w in got:
            mark = "" if content in memes else "（变异）"
            print(f"  {d} -> {r}: 「{content}」 w={w:.2f} {mark}")

    print("\n--- 各代对「钻木可以取火」的记忆 ---")
    for name, ratio, w in meme_trace(swarm.population, "钻木可以取火"):
        print(f"  {name}: 匹配度={ratio:.2f} 权重={w:.2f}")

    print("\n--- 文化相似度 ---")
    pop = swarm.population
    for i in range(3):
        for j in range(i + 1, 3):
            sim = cultural_similarity(pop[i], pop[j])
            print(f"  {pop[i].name} <-> {pop[j].name}: {sim:.2f}")

    print("\n--- 世代链实验（温习 vs 不温习）---")
    for label, rehearse in (("传递+温习", 3), ("仅传递", 0)):
        s = BrainSwarm(["G0", "G1", "G2", "G3"], seed=7)
        rep = generation_chain(s, memes, teach_times=25, fidelity=0.85,
                               seed=1, rehearse_times=rehearse)
        survived = [r["survived"] for r in rep]
        print(f"  {label}: 各代存活={survived}")

    print("\n--- v4.1 共识涌现相变扫描：规模 × 拓扑 → 收敛轮数 ---")
    grid = consensus_phase_scan(sizes=(4, 8, 12), seed=1)
    header = f"  {'拓扑':<18}{'度@N=4':>8}" + "".join(
        f"{s:>8}" for s in (4, 8, 12))
    print(header)
    for topo in ("fully_connected", "ring", "small_world", "random"):
        row = [g for g in grid if g["topology"] == topo]
        cells = []
        for g in row:
            cells.append(f"{g['rounds'] if g['converged'] else '>60':>8}")
        print(f"  {topo:<18}{row[0]['mean_degree']:>8}" + "".join(cells))

    print("\n--- v4.2 拓扑自适应：共同演化 vs 静态拓扑（N=12 环形）---")
    for label, adaptive in (("静态环形", False), ("共同演化环形", True)):
        s = BrainSwarm([f"C{i}" for i in range(12)], seed=1)
        for _ in range(25):
            s.population[0].sensory_input("钻木可以取火")
        s.set_topology("ring")
        if adaptive:
            res = s.coevolve_consensus("钻木可以取火", max_rounds=60,
                                       threshold=0.9, rewire_prob=0.5)
            r = res["rounds"] if res["converged"] else ">60"
            print(f"  {label}: 收敛轮数={r}  "
                  f"边生灭 新生{res['born_total']}/重连{res['rewired_total']}  "
                  f"末态平均度={res['final_degree']}（初始 2.0）")
        else:
            res = s.consensus_convergence("钻木可以取火", max_rounds=60,
                                          threshold=0.9)
            r = res["rounds"] if res["converged"] else ">60"
            print(f"  {label}: 收敛轮数={r}  "
                  f"覆盖率 {res['coverage'][:3]}...{res['coverage'][-1]}")

    print("\n--- v4.2 重连概率 φ 的三区动力学（N=12 共同演化环形）---")
    for phi, note in ((0.2, "传播主导：结构平静，共识最快"),
                      (0.5, "平衡区：传播与结构适应协同"),
                      (0.8, "结构主导：边 churn 剧烈，共识被拖慢")):
        s = BrainSwarm([f"C{i}" for i in range(12)], seed=1)
        for _ in range(25):
            s.population[0].sensory_input("钻木可以取火")
        s.set_topology("ring")
        res = s.coevolve_consensus("钻木可以取火", max_rounds=60,
                                   threshold=0.9, rewire_prob=phi)
        r = res["rounds"] if res["converged"] else ">60"
        print(f"  φ={phi}: 收敛={r} 轮  "
              f"模仿{res['copied_total']} 重连{res['rewired_total']} "
              f"新生{res['born_total']}  | {note}")

    print("\n--- v4.3 多模因竞争：垄断 vs 极化（N=12 环形，两阵营各半）---")
    rivals = ["钻木可以取火", "燧石可以取火"]
    for phi in (0.2, 0.85):
        s = BrainSwarm([f"M{i}" for i in range(12)], seed=1)
        for k in range(12):  # 前半持钻木，后半持燧石
            meme = rivals[0] if k < 6 else rivals[1]
            s.population[k].long_memory.append(BrainMemory(
                content=meme, timestamp=time.time(), weight=1.0,
                tag="culture"))
        s.set_topology("ring")
        res = s.competition_dynamics(rivals, max_rounds=60,
                                     dominance=0.9, rewire_prob=phi)
        if res["converged"]:
            outcome = f"垄断：「{res['winner']}」{res['rounds']} 轮统一全网"
        else:
            f_ = res["final"]
            outcome = (f"极化共存：两阵营 {f_[rivals[0]]:.2f}/"
                       f"{f_[rivals[1]]:.2f} 各自封闭")
        print(f"  φ={phi}: {outcome}  "
              f"(转化{res['converted_total']} 重连{res['rewired_total']})")
