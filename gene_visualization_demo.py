# -*- coding: utf-8 -*-
"""
基因库 3D 可视化演示

生成多种可视化图表：
1. 按世代着色的 3D 散点图
2. 按适应度着色的 3D 散点图
3. 按人格着色的 3D 散点图
4. 进化轨迹图
5. 谱系追踪图
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))

import matplotlib
matplotlib.use('Agg')  # 非交互式后端

from gene_library import GeneLanceDB
from gene_visualizer import GeneVisualizer3D, create_demo_gene_library


def print_section(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print()


def main():
    print_section("🧬 基因库 3D 可视化演示")

    # 创建演示基因库
    print("创建演示基因库...")
    lib = create_demo_gene_library()
    print(f"✅ 基因总数: {lib.get_gene_count()}")

    # 列出所有基因
    print()
    print("基因列表:")
    genes = lib.list_genes(limit=50)
    for g in genes:
        print(f"  [{g['gene_id']}] {g['name']:10s} "
              f"世代={g['generation']} "
              f"适应度={g['fitness']:.1f}")

    # 初始化可视化器
    print_section("初始化可视化器")
    vis = GeneVisualizer3D(lib)
    vis.load_genes()

    # PCA 降维
    print_section("PCA 降维")
    vis.pca_fit(n_components=3)
    print()
    print("  主成分解释方差:")
    print(f"    PC1: {vis.explained_variance_ratio[0]:.1%}")
    print(f"    PC2: {vis.explained_variance_ratio[1]:.1%}")
    print(f"    PC3: {vis.explained_variance_ratio[2]:.1%}")
    print(f"    累计: {sum(vis.explained_variance_ratio):.1%}")

    # 创建输出目录
    os.makedirs("figures", exist_ok=True)

    # 1. 按世代着色
    print_section("1️⃣ 3D 散点图（按世代着色）")
    print("  生成图表...")
    vis.plot_3d_scatter(
        color_by="generation",
        title="基因库 3D 可视化（按世代着色）",
        save_path="figures/gene_library_3d_generation.png",
        show=False,
    )
    print("  ✅ 已保存: figures/gene_library_3d_generation.png")

    # 2. 按适应度着色
    print_section("2️⃣ 3D 散点图（按适应度着色）")
    print("  生成图表...")
    vis.plot_3d_scatter(
        color_by="fitness",
        title="基因库 3D 可视化（按适应度着色）",
        save_path="figures/gene_library_3d_fitness.png",
        show=False,
    )
    print("  ✅ 已保存: figures/gene_library_3d_fitness.png")

    # 3. 按寻求刺激着色
    print_section("3️⃣ 3D 散点图（按寻求刺激着色）")
    print("  生成图表...")
    vis.plot_3d_scatter(
        color_by="sensation_seeking",
        title="基因库 3D 可视化（按寻求刺激人格着色）",
        save_path="figures/gene_library_3d_personality.png",
        show=False,
    )
    print("  ✅ 已保存: figures/gene_library_3d_personality.png")

    # 4. 进化轨迹
    print_section("4️⃣ 进化轨迹图")
    print("  生成图表...")
    vis.plot_evolution_trajectory(
        title="基因进化轨迹（按世代分组）",
        save_path="figures/gene_evolution_trajectory.png",
        show=False,
    )
    print("  ✅ 已保存: figures/gene_evolution_trajectory.png")

    # 5. 谱系追踪
    print_section("5️⃣ 谱系追踪图")
    print("  查找第3代基因...")
    gen3_genes = [g for g in genes if g["generation"] == 3]
    if gen3_genes:
        target_gene = gen3_genes[0]
        print(f"  追踪基因: {target_gene['name']} ({target_gene['gene_id']})")
        print("  生成图表...")
        vis.plot_genealogy(
            target_gene["gene_id"],
            max_depth=3,
            title=f"基因谱系追踪 - {target_gene['name']}",
            save_path="figures/gene_genealogy.png",
            show=False,
        )
        print("  ✅ 已保存: figures/gene_genealogy.png")

    # 总结
    print_section("✅ 演示完成")
    print("  生成的图表:")
    print("    1. gene_library_3d_generation.png  (按世代着色)")
    print("    2. gene_library_3d_fitness.png     (按适应度着色)")
    print("    3. gene_library_3d_personality.png (按人格着色)")
    print("    4. gene_evolution_trajectory.png   (进化轨迹)")
    print("    5. gene_genealogy.png              (谱系追踪)")
    print()
    print("  所有图表保存在: figures/ 目录")
    print()


if __name__ == "__main__":
    main()
