# -*- coding: utf-8 -*-
"""
DNA 基因库 - 基于 LanceDB 的大脑基因存储与检索

功能：
- 存储大脑 DNA
- 按特征向量相似度检索
- 进化选择（按适应度筛选）
- 基因重组
- 谱系追踪
"""
import os
import json
import time
import uuid
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict

try:
    import lancedb
    import pyarrow as pa
    LANCEDB_AVAILABLE = True
except ImportError:
    LANCEDB_AVAILABLE = False


@dataclass
class GeneEntry:
    """基因库条目"""
    gene_id: str           # 基因唯一ID
    name: str              # 大脑名称
    dna: str               # DNA JSON 字符串
    vector: List[float]    # 特征向量（用于相似度检索）
    fitness: float         # 适应度
    generation: int        # 世代
    parent_ids: List[str]  # 父代基因ID
    created_at: float      # 创建时间戳
    personality: Dict[str, float]  # 人格参数
    memory_count: int      # 记忆数量
    tick: int              # 存活时间步


class GeneLanceDB:
    """基于 LanceDB 的 DNA 基因库"""

    def __init__(self, db_path: str = "data/gene_library"):
        """初始化基因库

        Args:
            db_path: 数据库路径
        """
        self.db_path = db_path
        self.db = None
        self.table = None
        self.vector_dim = 16  # 特征向量维度

        if LANCEDB_AVAILABLE:
            self._init_db()
        else:
            print("⚠️ LanceDB 不可用，使用内存模式")
            self._memory_store = {}

    def _init_db(self):
        """初始化 LanceDB"""
        os.makedirs(self.db_path, exist_ok=True)
        self.db = lancedb.connect(self.db_path)

        # 定义 schema
        schema = pa.schema([
            pa.field("gene_id", pa.string()),
            pa.field("name", pa.string()),
            pa.field("dna", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), self.vector_dim)),
            pa.field("fitness", pa.float32()),
            pa.field("generation", pa.int32()),
            pa.field("parent_ids", pa.string()),  # JSON 字符串
            pa.field("created_at", pa.float64()),
            pa.field("personality", pa.string()),  # JSON 字符串
            pa.field("memory_count", pa.int32()),
            pa.field("tick", pa.int64()),
        ])

        # 打开或创建表
        try:
            self.table = self.db.open_table("genes")
        except Exception:
            self.table = self.db.create_table("genes", schema=schema)

    def _dna_to_vector(self, dna: dict) -> List[float]:
        """将 DNA 转换为特征向量

        特征维度：
        0: sensation_seeking (寻求刺激)
        1: habituation_rate (习惯化速率)
        2: attention_factor (注意力因子)
        3: calm (平静情绪)
        4: curiosity (好奇情绪)
        5: stress (压力情绪)
        6: joy (喜悦情绪)
        7: sadness (悲伤情绪)
        8: anger (愤怒情绪)
        9: fear (恐惧情绪)
        10: memory_count (记忆数量)
        11: tick_norm (存活时间归一化)
        12: synapse_strength (平均突触强度)
        13: thought_count (思考空间念头数)
        14: novelty_avg (平均新奇度)
        15: fitness (适应度)
        """
        vector = [0.0] * self.vector_dim

        # 人格参数
        personality = dna.get("personality", {})
        vector[0] = personality.get("sensation_seeking", 0.5)
        vector[1] = personality.get("habituation_rate", 0.3)

        # 注意力
        vector[2] = dna.get("attention_factor", 0.6)

        # 情绪
        emotion = dna.get("emotion", {})
        vector[3] = emotion.get("calm", 0.8)
        vector[4] = emotion.get("curiosity", 0.3)
        vector[5] = emotion.get("stress", 0.0)
        vector[6] = emotion.get("joy", 0.0)
        vector[7] = emotion.get("sadness", 0.0)
        vector[8] = emotion.get("anger", 0.0)
        vector[9] = emotion.get("fear", 0.0)

        # 记忆数量（归一化）
        long_mem = dna.get("long_memory", [])
        short_mem = dna.get("short_memory", [])
        vector[10] = min(len(long_mem) / 100.0, 1.0)

        # 存活时间（归一化）
        tick = dna.get("tick", 0)
        vector[11] = min(tick / 1000.0, 1.0)

        # 平均突触强度
        synapse = dna.get("synapse", {})
        if synapse:
            avg_strength = sum(synapse.values()) / len(synapse)
            vector[12] = min(avg_strength, 1.0)

        # 思考空间念头数
        thought_space = dna.get("thought_space", [])
        vector[13] = min(len(thought_space) / 9.0, 1.0)

        # 平均新奇度（估算）
        vector[14] = 0.5  # 默认值

        # 适应度（默认）
        vector[15] = 0.0

        return vector

    def add_gene(self, dna: dict, fitness: float = 0.0,
                 generation: int = 0, parent_ids: List[str] = None) -> str:
        """添加基因到库中

        Args:
            dna: 大脑 DNA
            fitness: 适应度
            generation: 世代
            parent_ids: 父代基因ID列表

        Returns:
            基因ID
        """
        gene_id = str(uuid.uuid4())[:8]
        name = dna.get("name", "unknown")
        vector = self._dna_to_vector(dna)
        vector[15] = fitness  # 最后一维是适应度

        personality = dna.get("personality", {})
        long_mem = dna.get("long_memory", [])
        short_mem = dna.get("short_memory", [])
        memory_count = len(long_mem) + len(short_mem)
        tick = dna.get("tick", 0)

        entry = {
            "gene_id": gene_id,
            "name": name,
            "dna": json.dumps(dna, ensure_ascii=False),
            "vector": vector,
            "fitness": fitness,
            "generation": generation,
            "parent_ids": json.dumps(parent_ids or []),
            "created_at": time.time(),
            "personality": json.dumps(personality),
            "memory_count": memory_count,
            "tick": tick,
        }

        if self.table is not None:
            self.table.add([entry])
        else:
            self._memory_store[gene_id] = entry

        return gene_id

    def get_gene(self, gene_id: str) -> Optional[dict]:
        """获取基因 DNA

        Args:
            gene_id: 基因ID

        Returns:
            DNA 字典
        """
        if self.table is not None:
            result = self.table.search().where(
                f"gene_id = '{gene_id}'"
            ).limit(1).to_list()
            if result:
                return json.loads(result[0]["dna"])
            return None
        else:
            entry = self._memory_store.get(gene_id)
            if entry:
                return json.loads(entry["dna"])
            return None

    def search_similar(self, dna: dict, limit: int = 5) -> List[dict]:
        """搜索相似基因

        Args:
            dna: 查询 DNA
            limit: 返回数量

        Returns:
            相似基因列表 [{gene_id, name, fitness, score, dna}]
        """
        vector = self._dna_to_vector(dna)

        if self.table is not None:
            results = self.table.search(vector).limit(limit).to_list()
            return [
                {
                    "gene_id": r["gene_id"],
                    "name": r["name"],
                    "fitness": r["fitness"],
                    "score": r.get("_distance", 0),
                    "dna": json.loads(r["dna"]),
                }
                for r in results
            ]
        else:
            # 内存模式：简单余弦相似度
            results = []
            for gid, entry in self._memory_store.items():
                score = self._cosine_similarity(vector, entry["vector"])
                results.append({
                    "gene_id": gid,
                    "name": entry["name"],
                    "fitness": entry["fitness"],
                    "score": score,
                    "dna": json.loads(entry["dna"]),
                })
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:limit]

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """余弦相似度"""
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = sum(a * a for a in v1) ** 0.5
        norm2 = sum(b * b for b in v2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def select_by_fitness(self, top_k: int = 10) -> List[dict]:
        """按适应度选择最优基因

        Args:
            top_k: 返回前K个

        Returns:
            基因列表
        """
        if self.table is not None:
            results = self.table.search().where(
                "fitness >= 0"
            ).limit(top_k).to_list()
            # 按适应度排序
            results.sort(key=lambda x: x["fitness"], reverse=True)
            return [
                {
                    "gene_id": r["gene_id"],
                    "name": r["name"],
                    "fitness": r["fitness"],
                    "generation": r["generation"],
                    "dna": json.loads(r["dna"]),
                }
                for r in results[:top_k]
            ]
        else:
            entries = sorted(
                self._memory_store.values(),
                key=lambda x: x["fitness"],
                reverse=True
            )
            return [
                {
                    "gene_id": e["gene_id"],
                    "name": e["name"],
                    "fitness": e["fitness"],
                    "generation": e["generation"],
                    "dna": json.loads(e["dna"]),
                }
                for e in entries[:top_k]
            ]

    def get_gene_count(self) -> int:
        """获取基因总数"""
        if self.table is not None:
            return self.table.count_rows()
        else:
            return len(self._memory_store)

    def list_genes(self, limit: int = 20) -> List[dict]:
        """列出所有基因

        Args:
            limit: 返回数量

        Returns:
            基因列表
        """
        if self.table is not None:
            results = self.table.search().limit(limit).to_list()
            return [
                {
                    "gene_id": r["gene_id"],
                    "name": r["name"],
                    "fitness": r["fitness"],
                    "generation": r["generation"],
                    "memory_count": r["memory_count"],
                    "tick": r["tick"],
                }
                for r in results
            ]
        else:
            return [
                {
                    "gene_id": e["gene_id"],
                    "name": e["name"],
                    "fitness": e["fitness"],
                    "generation": e["generation"],
                    "memory_count": e["memory_count"],
                    "tick": e["tick"],
                }
                for e in list(self._memory_store.values())[:limit]
            ]

    def delete_gene(self, gene_id: str) -> bool:
        """删除基因

        Args:
            gene_id: 基因ID

        Returns:
            是否成功
        """
        if self.table is not None:
            self.table.delete(f"gene_id = '{gene_id}'")
            return True
        else:
            if gene_id in self._memory_store:
                del self._memory_store[gene_id]
                return True
            return False

    def update_fitness(self, gene_id: str, fitness: float) -> bool:
        """更新基因适应度

        Args:
            gene_id: 基因ID
            fitness: 新的适应度

        Returns:
            是否成功
        """
        if self.table is not None:
            # LanceDB 不直接支持更新，需要删除后重新插入
            gene = self.get_gene(gene_id)
            if gene:
                # 获取完整条目
                result = self.table.search().where(
                    f"gene_id = '{gene_id}'"
                ).limit(1).to_list()
                if result:
                    entry = result[0]
                    entry["fitness"] = fitness
                    # 更新向量中的适应度
                    vector = list(entry["vector"])
                    vector[15] = fitness
                    entry["vector"] = vector
                    # 删除旧的，插入新的
                    self.delete_gene(gene_id)
                    self.table.add([entry])
                    return True
            return False
        else:
            if gene_id in self._memory_store:
                self._memory_store[gene_id]["fitness"] = fitness
                # 更新向量
                vector = list(self._memory_store[gene_id]["vector"])
                vector[15] = fitness
                self._memory_store[gene_id]["vector"] = vector
                return True
            return False

    def get_genealogy(self, gene_id: str, max_depth: int = 5) -> List[dict]:
        """获取基因谱系（祖先追踪）

        Args:
            gene_id: 基因ID
            max_depth: 最大追溯深度

        Returns:
            谱系列表
        """
        genealogy = []
        current_id = gene_id
        depth = 0

        while current_id and depth < max_depth:
            dna = self.get_gene(current_id)
            if not dna:
                break

            # 获取条目信息
            if self.table is not None:
                result = self.table.search().where(
                    f"gene_id = '{current_id}'"
                ).limit(1).to_list()
                if result:
                    entry = result[0]
                    genealogy.append({
                        "gene_id": current_id,
                        "name": entry["name"],
                        "generation": entry["generation"],
                        "fitness": entry["fitness"],
                        "depth": depth,
                    })
                    parent_ids = json.loads(entry["parent_ids"])
            else:
                entry = self._memory_store.get(current_id)
                if entry:
                    genealogy.append({
                        "gene_id": current_id,
                        "name": entry["name"],
                        "generation": entry["generation"],
                        "fitness": entry["fitness"],
                        "depth": depth,
                    })
                    parent_ids = json.loads(entry["parent_ids"])
                else:
                    break

            # 追溯第一个父代
            if parent_ids:
                current_id = parent_ids[0]
            else:
                break

            depth += 1

        return genealogy


# 便捷函数
def get_gene_library(path: str = "data/gene_library") -> GeneLanceDB:
    """获取基因库实例"""
    return GeneLanceDB(path)


if __name__ == "__main__":
    # 测试
    print("🧬 基因库测试")
    print()

    lib = GeneLanceDB("datasets/test_gene_library")

    # 创建测试 DNA
    test_dna = {
        "name": "TestBrain",
        "tick": 100,
        "synapse": {"0,1": 0.5, "1,2": 0.3},
        "recurrent_synapse": {},
        "short_memory": [],
        "long_memory": [{"content": "test", "tag": "test", "weight": 0.5}],
        "emotion": {"calm": 0.8, "curiosity": 0.3, "stress": 0.0},
        "attention_factor": 0.6,
        "personality": {
            "sensation_seeking": 0.5,
            "habituation_rate": 0.3,
        },
        "exposure_count": {},
        "thought_space": [],
        "metacog_log": [],
    }

    # 添加基因
    gene_id = lib.add_gene(test_dna, fitness=0.8, generation=1)
    print(f"✅ 添加基因: {gene_id}")

    # 基因数量
    print(f"📊 基因总数: {lib.get_gene_count()}")

    # 列出基因
    genes = lib.list_genes()
    print(f"📋 基因列表: {len(genes)} 个")
    for g in genes:
        print(f"  - {g['gene_id']}: {g['name']} (fitness={g['fitness']})")

    # 相似搜索
    similar = lib.search_similar(test_dna, limit=3)
    print(f"🔍 相似基因: {len(similar)} 个")
    for s in similar:
        print(f"  - {s['gene_id']}: {s['name']} (score={s['score']:.3f})")

    # 适应度选择
    top = lib.select_by_fitness(top_k=3)
    print(f"🏆 最优基因: {len(top)} 个")
    for t in top:
        print(f"  - {t['gene_id']}: {t['name']} (fitness={t['fitness']})")

    print()
    print("✅ 测试完成！")
