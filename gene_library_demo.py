# -*- coding: utf-8 -*-
"""
DNA 基因库管理演示

功能：
- 创建多个不同性格的大脑
- 将 DNA 存入基因库
- 相似度搜索
- 进化选择
- 谱系追踪
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from ai_brain_entity import AIBrainEntity

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))
from gene_library import GeneLanceDB


def print_section(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print()


def create_brain_with_personality(name: str, sensation_seeking: float,
                                  habituation_rate: float) -> AIBrainEntity:
    """创建具有特定人格的大脑"""
    brain = AIBrainEntity(name)
    brain.sensation_seeking = sensation_seeking
    brain.habituation_rate = habituation_rate
    return brain


def main():
    print_section("🧬 DNA 基因库演示")

    # 初始化基因库
    lib = GeneLanceDB("datasets/gene_library")
    print(f"✅ 基因库初始化完成")
    print(f"   当前基因数: {lib.get_gene_count()}")

    # 1. 创建多个不同性格的大脑
    print_section("1️⃣ 创建不同性格的大脑")

    personalities = [
        ("探险家", 0.9, 0.2),   # 高寻求刺激，低习惯化
        ("保守者", 0.1, 0.5),   # 低寻求刺激，高习惯化
        ("平衡型", 0.5, 0.3),   # 平衡型
        ("学者型", 0.3, 0.4),   # 中等寻求刺激，较高习惯化
        ("冒险家", 0.8, 0.1),   # 很高寻求刺激，很低习惯化
    ]

    gene_ids = []
    for name, sss, hab in personalities:
        print(f"  创建大脑: {name} (SSS={sss}, Hab={hab})")
        brain = create_brain_with_personality(name, sss, hab)

        # 让大脑学习一些东西
        for i in range(5):
            brain.sensory_input(f"{name}的记忆{i}")
            brain.think()

        # 计算适应度（记忆数量作为适应度指标）
        fitness = len(brain.long_memory) + len(brain.short_memory) * 0.5

        # 存入基因库
        dna = brain.dump_dna()
        gene_id = lib.add_gene(dna, fitness=fitness, generation=1)
        gene_ids.append(gene_id)
        print(f"    ✅ 存入基因库: {gene_id} (适应度={fitness:.1f})")

    print()
    print(f"  共创建 {len(gene_ids)} 个基因")

    # 2. 列出所有基因
    print_section("2️⃣ 基因库列表")
    genes = lib.list_genes(limit=10)
    for g in genes:
        print(f"  [{g['gene_id']}] {g['name']:8s} "
              f"适应度={g['fitness']:.1f} "
              f"世代={g['generation']} "
              f"记忆={g['memory_count']} "
              f"存活={g['tick']}tick")

    # 3. 相似度搜索
    print_section("3️⃣ 相似度搜索")

    # 用探险家的 DNA 搜索
    explorer_dna = lib.get_gene(gene_ids[0])
    print(f"  查询: 探险家 ({gene_ids[0]})")
    print()

    similar = lib.search_similar(explorer_dna, limit=5)
    print(f"  最相似的 5 个基因:")
    for i, s in enumerate(similar):
        print(f"    {i+1}. [{s['gene_id']}] {s['name']:8s} "
              f"相似度={s['score']:.3f} "
              f"适应度={s['fitness']:.1f}")

    # 4. 进化选择
    print_section("4️⃣ 进化选择（按适应度）")

    top_genes = lib.select_by_fitness(top_k=3)
    print(f"  适应度最高的 3 个基因:")
    for i, g in enumerate(top_genes):
        print(f"    {i+1}. [{g['gene_id']}] {g['name']:8s} "
              f"适应度={g['fitness']:.1f} "
              f"世代={g['generation']}")

    # 5. 基因重组（进化下一代）
    print_section("5️⃣ 基因重组（进化下一代）")

    # 选择前两名进行重组
    parent1 = top_genes[0]
    parent2 = top_genes[1]
    print(f"  父代 1: {parent1['name']} ({parent1['gene_id']})")
    print(f"  父代 2: {parent2['name']} ({parent2['gene_id']})")

    # 创建子代（简单的基因重组：人格参数取平均）
    dna1 = parent1['dna']
    dna2 = parent2['dna']

    child_dna = dict(dna1)  # 复制父代1的DNA
    child_dna['name'] = f"子代-{parent1['name'][:2]}{parent2['name'][:2]}"

    # 人格参数重组（取平均 + 少量变异）
    import random
    sss1 = dna1['personality']['sensation_seeking']
    sss2 = dna2['personality']['sensation_seeking']
    hab1 = dna1['personality']['habituation_rate']
    hab2 = dna2['personality']['habituation_rate']

    child_sss = (sss1 + sss2) / 2 + random.uniform(-0.1, 0.1)
    child_hab = (hab1 + hab2) / 2 + random.uniform(-0.1, 0.1)

    child_sss = max(0.0, min(1.0, child_sss))
    child_hab = max(0.0, min(1.0, child_hab))

    child_dna['personality']['sensation_seeking'] = child_sss
    child_dna['personality']['habituation_rate'] = child_hab

    print()
    print(f"  子代: {child_dna['name']}")
    print(f"    寻求刺激: {child_sss:.2f} (父代: {sss1:.2f} + {sss2:.2f})")
    print(f"    习惯化速率: {child_hab:.2f} (父代: {hab1:.2f} + {hab2:.2f})")

    # 存入基因库
    child_id = lib.add_gene(
        child_dna,
        fitness=0.0,  # 新子代适应度待评估
        generation=2,
        parent_ids=[parent1['gene_id'], parent2['gene_id']]
    )
    print(f"    ✅ 存入基因库: {child_id} (第2代)")

    # 6. 谱系追踪
    print_section("6️⃣ 谱系追踪")

    genealogy = lib.get_genealogy(child_id, max_depth=3)
    print(f"  子代 {child_id} 的祖先谱系:")
    for g in genealogy:
        indent = "  " * g['depth']
        print(f"    {indent}第{g['generation']}代: "
              f"[{g['gene_id']}] {g['name']} "
              f"(适应度={g['fitness']:.1f})")

    # 7. 最终统计
    print_section("7️⃣ 基因库统计")
    print(f"  基因总数: {lib.get_gene_count()}")
    print(f"  第1代: {len([g for g in genes if g['generation'] == 1])} 个")
    print(f"  第2代: {lib.get_gene_count() - len(genes)} 个")
    print()
    print("  所有基因:")
    all_genes = lib.list_genes(limit=20)
    for g in all_genes:
        print(f"    [{g['gene_id']}] {g['name']:12s} "
              f"世代={g['generation']} "
              f"适应度={g['fitness']:.1f}")

    print_section("✅ 演示完成")
    print("  基因库数据保存在: datasets/gene_library/")
    print()


if __name__ == "__main__":
    main()
