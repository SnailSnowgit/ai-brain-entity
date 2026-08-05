# -*- coding: utf-8 -*-
"""
基因库 3D 可视化模块

功能：
- PCA 降维到 3D
- 3D 散点图可视化
- 按世代/适应度/人格着色
- 谱系连线
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Optional, Tuple

# 中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

sys.path.insert(0, os.path.dirname(__file__))
from gene_library import GeneLanceDB


class GeneVisualizer3D:
    """基因库 3D 可视化"""

    def __init__(self, gene_library: GeneLanceDB):
        """初始化可视化器

        Args:
            gene_library: 基因库实例
        """
        self.lib = gene_library
        self.genes = []
        self.vectors = []
        self.pca_components = None
        self.pca_mean = None

    def load_genes(self):
        """加载所有基因数据"""
        all_genes = self.lib.list_genes(limit=1000)
        self.genes = []
        self.vectors = []

        for g in all_genes:
            gene_id = g["gene_id"]
            # 获取完整 DNA 和向量
            dna = self.lib.get_gene(gene_id)
            if dna:
                vector = self.lib._dna_to_vector(dna)
                self.genes.append({
                    "gene_id": gene_id,
                    "name": g["name"],
                    "fitness": g["fitness"],
                    "generation": g["generation"],
                    "memory_count": g["memory_count"],
                    "tick": g["tick"],
                    "vector": vector,
                    "dna": dna,
                })
                self.vectors.append(vector)

        self.vectors = np.array(self.vectors)
        print(f"✅ 加载 {len(self.genes)} 个基因")

    def pca_fit(self, n_components: int = 3):
        """PCA 降维

        Args:
            n_components: 降维后的维度
        """
        if len(self.vectors) == 0:
            print("⚠️ 没有基因数据")
            return

        # 中心化
        self.pca_mean = np.mean(self.vectors, axis=0)
        centered = self.vectors - self.pca_mean

        # SVD 分解
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)

        # 取前 n_components 个主成分
        self.pca_components = Vt[:n_components].T

        # 计算解释方差比例
        explained_var = (S ** 2) / (len(self.vectors) - 1)
        total_var = np.sum(explained_var)
        self.explained_variance_ratio = explained_var[:n_components] / total_var

        print(f"✅ PCA 降维完成")
        print(f"   解释方差比例: {self.explained_variance_ratio}")
        print(f"   累计解释方差: {np.sum(self.explained_variance_ratio):.3f}")

    def pca_transform(self, vectors: np.ndarray) -> np.ndarray:
        """将向量转换到 PCA 空间

        Args:
            vectors: 原始向量

        Returns:
            降维后的向量
        """
        if self.pca_components is None:
            raise ValueError("请先调用 pca_fit()")

        centered = vectors - self.pca_mean
        return centered @ self.pca_components

    def get_3d_positions(self) -> np.ndarray:
        """获取所有基因的 3D 坐标

        Returns:
            3D 坐标数组 (n_genes, 3)
        """
        if self.pca_components is None:
            self.pca_fit(n_components=3)

        return self.pca_transform(self.vectors)

    def plot_3d_scatter(self,
                        color_by: str = "generation",
                        title: str = "基因库 3D 可视化",
                        save_path: Optional[str] = None,
                        show: bool = True):
        """绘制 3D 散点图

        Args:
            color_by: 着色方式 ('generation', 'fitness', 'sensation_seeking')
            title: 图表标题
            save_path: 保存路径（可选）
            show: 是否显示图表
        """
        if len(self.genes) == 0:
            print("⚠️ 没有基因数据")
            return

        positions = self.get_3d_positions()

        # 准备颜色数据
        if color_by == "generation":
            colors = [g["generation"] for g in self.genes]
            cmap = "viridis"
            cbar_label = "世代"
        elif color_by == "fitness":
            colors = [g["fitness"] for g in self.genes]
            cmap = "RdYlGn"
            cbar_label = "适应度"
        elif color_by == "sensation_seeking":
            colors = [g["dna"]["personality"]["sensation_seeking"]
                      for g in self.genes]
            cmap = "coolwarm"
            cbar_label = "寻求刺激"
        else:
            colors = "blue"
            cmap = None
            cbar_label = ""

        # 创建图形
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制散点
        scatter = ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            c=colors,
            cmap=cmap,
            s=100,
            alpha=0.8,
            edgecolors='black',
            linewidths=0.5,
        )

        # 添加标签
        for i, gene in enumerate(self.genes):
            ax.text(
                positions[i, 0],
                positions[i, 1],
                positions[i, 2],
                gene["name"],
                fontsize=8,
                ha='center',
                va='bottom',
            )

        # 设置轴标签
        ax.set_xlabel(f'PC1 ({self.explained_variance_ratio[0]:.1%})')
        ax.set_ylabel(f'PC2 ({self.explained_variance_ratio[1]:.1%})')
        ax.set_zlabel(f'PC3 ({self.explained_variance_ratio[2]:.1%})')

        # 设置标题
        ax.set_title(title, fontsize=14, pad=20)

        # 添加颜色条
        if cmap:
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
            cbar.set_label(cbar_label)

        # 设置视角
        ax.view_init(elev=25, azim=45)

        # 保存
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ 图表已保存: {save_path}")

        # 显示
        if show:
            plt.show()

        return fig, ax

    def plot_genealogy(self,
                       gene_id: str,
                       max_depth: int = 5,
                       title: str = "基因谱系追踪",
                       save_path: Optional[str] = None,
                       show: bool = True):
        """绘制基因谱系（3D）

        Args:
            gene_id: 起始基因ID
            max_depth: 最大追溯深度
            title: 图表标题
            save_path: 保存路径
            show: 是否显示
        """
        if len(self.genes) == 0:
            print("⚠️ 没有基因数据")
            return

        positions = self.get_3d_positions()

        # 获取谱系
        genealogy = self.lib.get_genealogy(gene_id, max_depth)
        if not genealogy:
            print("⚠️ 没有谱系数据")
            return

        # 创建基因ID到索引的映射
        gene_id_to_idx = {g["gene_id"]: i for i, g in enumerate(self.genes)}

        # 创建图形
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制所有基因（灰色）
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            c='lightgray',
            s=50,
            alpha=0.3,
        )

        # 绘制谱系基因（彩色）
        genealogy_ids = [g["gene_id"] for g in genealogy]
        genealogy_indices = [gene_id_to_idx[gid] for gid in genealogy_ids
                             if gid in gene_id_to_idx]

        if genealogy_indices:
            genealogy_positions = positions[genealogy_indices]

            # 按世代着色
            generations = [genealogy[i]["generation"]
                           for i in range(len(genealogy_indices))]

            scatter = ax.scatter(
                genealogy_positions[:, 0],
                genealogy_positions[:, 1],
                genealogy_positions[:, 2],
                c=generations,
                cmap="viridis",
                s=200,
                alpha=0.9,
                edgecolors='black',
                linewidths=1,
                marker='D',
            )

            # 添加标签
            for i, idx in enumerate(genealogy_indices):
                gene = self.genes[idx]
                ax.text(
                    positions[idx, 0],
                    positions[idx, 1],
                    positions[idx, 2],
                    f"{gene['name']}\n(第{gene['generation']}代)",
                    fontsize=9,
                    ha='center',
                    va='bottom',
                    fontweight='bold',
                )

            # 绘制谱系连线
            for i in range(len(genealogy_indices) - 1):
                idx1 = genealogy_indices[i]
                idx2 = genealogy_indices[i + 1]
                ax.plot(
                    [positions[idx1, 0], positions[idx2, 0]],
                    [positions[idx1, 1], positions[idx2, 1]],
                    [positions[idx1, 2], positions[idx2, 2]],
                    'r-',
                    linewidth=2,
                    alpha=0.8,
                )

            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
            cbar.set_label('世代')

        # 设置轴标签
        ax.set_xlabel(f'PC1 ({self.explained_variance_ratio[0]:.1%})')
        ax.set_ylabel(f'PC2 ({self.explained_variance_ratio[1]:.1%})')
        ax.set_zlabel(f'PC3 ({self.explained_variance_ratio[2]:.1%})')

        ax.set_title(title, fontsize=14, pad=20)
        ax.view_init(elev=25, azim=45)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ 图表已保存: {save_path}")

        if show:
            plt.show()

        return fig, ax

    def plot_evolution_trajectory(self,
                                  generations: List[int] = None,
                                  title: str = "进化轨迹",
                                  save_path: Optional[str] = None,
                                  show: bool = True):
        """绘制进化轨迹（按世代连接）

        Args:
            generations: 要显示的世代列表（None 显示所有）
            title: 图表标题
            save_path: 保存路径
            show: 是否显示
        """
        if len(self.genes) == 0:
            print("⚠️ 没有基因数据")
            return

        positions = self.get_3d_positions()

        # 按世代分组
        genes_by_gen = {}
        for i, gene in enumerate(self.genes):
            gen = gene["generation"]
            if gen not in genes_by_gen:
                genes_by_gen[gen] = []
            genes_by_gen[gen].append(i)

        # 创建图形
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制每个世代
        sorted_gens = sorted(genes_by_gen.keys())
        if generations:
            sorted_gens = [g for g in sorted_gens if g in generations]

        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_gens)))

        for i, gen in enumerate(sorted_gens):
            indices = genes_by_gen[gen]
            gen_positions = positions[indices]

            ax.scatter(
                gen_positions[:, 0],
                gen_positions[:, 1],
                gen_positions[:, 2],
                c=[colors[i]],
                s=150,
                alpha=0.8,
                edgecolors='black',
                linewidths=0.5,
                label=f'第{gen}代',
            )

            # 添加标签
            for idx in indices:
                gene = self.genes[idx]
                ax.text(
                    positions[idx, 0],
                    positions[idx, 1],
                    positions[idx, 2],
                    gene["name"],
                    fontsize=7,
                    ha='center',
                    va='bottom',
                )

        # 设置轴标签
        ax.set_xlabel(f'PC1 ({self.explained_variance_ratio[0]:.1%})')
        ax.set_ylabel(f'PC2 ({self.explained_variance_ratio[1]:.1%})')
        ax.set_zlabel(f'PC3 ({self.explained_variance_ratio[2]:.1%})')

        ax.set_title(title, fontsize=14, pad=20)
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        ax.view_init(elev=25, azim=45)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ 图表已保存: {save_path}")

        if show:
            plt.show()

        return fig, ax


def create_demo_gene_library() -> GeneLanceDB:
    """创建演示用基因库"""
    from ai_brain_entity import AIBrainEntity
    import random

    lib = GeneLanceDB("data/vis_demo_gene_library")

    # 清空现有数据
    # （直接创建新库即可）

    # 创建不同世代的基因
    personalities = [
        ("探险家", 0.9, 0.2),
        ("保守者", 0.1, 0.5),
        ("平衡型", 0.5, 0.3),
        ("学者型", 0.3, 0.4),
        ("冒险家", 0.8, 0.1),
        ("沉思者", 0.2, 0.6),
        ("社交家", 0.7, 0.3),
        ("谨慎型", 0.15, 0.55),
    ]

    # 第1代
    gen1_ids = []
    for name, sss, hab in personalities[:5]:
        brain = AIBrainEntity(name)
        brain.sensation_seeking = sss
        brain.habituation_rate = hab

        for i in range(3):
            brain.sensory_input(f"{name}的记忆{i}")
            brain.think()

        fitness = len(brain.long_memory) + random.uniform(0, 5)
        dna = brain.dump_dna()
        gene_id = lib.add_gene(dna, fitness=fitness, generation=1)
        gen1_ids.append(gene_id)

    # 第2代（重组）
    gen2_ids = []
    for i in range(4):
        p1 = random.choice(gen1_ids)
        p2 = random.choice([g for g in gen1_ids if g != p1])

        dna1 = lib.get_gene(p1)
        dna2 = lib.get_gene(p2)

        child_dna = dict(dna1)
        child_dna['name'] = f"G2-{i+1}"

        sss1 = dna1['personality']['sensation_seeking']
        sss2 = dna2['personality']['sensation_seeking']
        hab1 = dna1['personality']['habituation_rate']
        hab2 = dna2['personality']['habituation_rate']

        child_sss = (sss1 + sss2) / 2 + random.uniform(-0.15, 0.15)
        child_hab = (hab1 + hab2) / 2 + random.uniform(-0.1, 0.1)

        child_sss = max(0.0, min(1.0, child_sss))
        child_hab = max(0.0, min(1.0, child_hab))

        child_dna['personality']['sensation_seeking'] = child_sss
        child_dna['personality']['habituation_rate'] = child_hab

        fitness = random.uniform(3, 12)
        gene_id = lib.add_gene(
            child_dna,
            fitness=fitness,
            generation=2,
            parent_ids=[p1, p2]
        )
        gen2_ids.append(gene_id)

    # 第3代（继续进化）
    for i in range(3):
        p1 = random.choice(gen2_ids)
        p2 = random.choice([g for g in gen2_ids if g != p1])

        dna1 = lib.get_gene(p1)
        dna2 = lib.get_gene(p2)

        child_dna = dict(dna1)
        child_dna['name'] = f"G3-{i+1}"

        sss1 = dna1['personality']['sensation_seeking']
        sss2 = dna2['personality']['sensation_seeking']
        hab1 = dna1['personality']['habituation_rate']
        hab2 = dna2['personality']['habituation_rate']

        child_sss = (sss1 + sss2) / 2 + random.uniform(-0.1, 0.1)
        child_hab = (hab1 + hab2) / 2 + random.uniform(-0.08, 0.08)

        child_sss = max(0.0, min(1.0, child_sss))
        child_hab = max(0.0, min(1.0, child_hab))

        child_dna['personality']['sensation_seeking'] = child_sss
        child_dna['personality']['habituation_rate'] = child_hab

        fitness = random.uniform(5, 15)
        lib.add_gene(
            child_dna,
            fitness=fitness,
            generation=3,
            parent_ids=[p1, p2]
        )

    return lib


if __name__ == "__main__":
    # 设置 matplotlib 后端
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端

    print("🧬 基因库 3D 可视化测试")
    print()

    # 创建演示基因库
    print("创建演示基因库...")
    lib = create_demo_gene_library()
    print(f"基因总数: {lib.get_gene_count()}")
    print()

    # 初始化可视化器
    print("初始化可视化器...")
    vis = GeneVisualizer3D(lib)
    vis.load_genes()
    print()

    # PCA 降维
    print("PCA 降维...")
    vis.pca_fit(n_components=3)
    print()

    # 创建输出目录
    os.makedirs("figures", exist_ok=True)

    # 1. 按世代着色
    print("生成 3D 散点图（按世代着色）...")
    vis.plot_3d_scatter(
        color_by="generation",
        title="基因库 3D 可视化（按世代着色）",
        save_path="figures/gene_library_3d_generation.png",
        show=False,
    )

    # 2. 按适应度着色
    print("生成 3D 散点图（按适应度着色）...")
    vis.plot_3d_scatter(
        color_by="fitness",
        title="基因库 3D 可视化（按适应度着色）",
        save_path="figures/gene_library_3d_fitness.png",
        show=False,
    )

    # 3. 按寻求刺激着色
    print("生成 3D 散点图（按寻求刺激着色）...")
    vis.plot_3d_scatter(
        color_by="sensation_seeking",
        title="基因库 3D 可视化（按寻求刺激着色）",
        save_path="figures/gene_library_3d_sss.png",
        show=False,
    )

    # 4. 进化轨迹
    print("生成进化轨迹图...")
    vis.plot_evolution_trajectory(
        title="基因进化轨迹",
        save_path="figures/gene_evolution_trajectory.png",
        show=False,
    )

    print()
    print("✅ 所有图表已生成到 figures/ 目录")
