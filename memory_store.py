# -*- coding: utf-8 -*-
"""
LanceDB 记忆存储后端（v6.0）

把大脑的长期记忆从"内存列表（上限 500 条）"扩展为"本地向量库（磁盘，
容量无限）"：content + features(512 维) + weight + tag + modality +
timestamp 全部持久化，支持向量近邻检索（语义回忆）。

降级策略（与 CLIP/Whisper/Qwen 同一风格）：
    lancedb 未安装 → LanceMemoryStore.available=False，
    AIBrainEntity 行为完全不变（纯内存 + 关键词 recall）。

安装：
    pip install lancedb        # 纯 wheel（Rust 内核），无需编译

数据位置：data/lancedb/（已被 .gitignore 忽略则不会入库）。
"""
import os
import time
from typing import Dict, List, Optional

_STORE_DIM = 512


def _escape(s: str) -> str:
    """SQL 字符串字面量转义（单引号加倍）"""
    return s.replace("'", "''")


def text_to_vector(text: str, dim: int = _STORE_DIM) -> List[float]:
    """零依赖确定性文本向量：字符 bigram 哈希 + L2 归一化。

    用于文本记忆的语义检索。局限：字面相近的文本才相似（"火焰"~"火星"），
    真正的语义相似（"火焰"~"燃烧"）需要记忆携带 CLIP/Qwen 等真实
    embedding（features 字段），本函数只是无外部模型时的兜底通路。
    """
    vec = [0.0] * dim
    tokens = ([text[i:i + 2] for i in range(max(len(text) - 1, 1))]
              if len(text) >= 2 else [text])
    for tok in tokens:
        vec[hash(tok) % dim] += 1.0
        vec[hash(tok[::-1]) % dim] += 0.5  # 反向 bigram 补充位置信息
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


class LanceMemoryStore:
    """LTM 向量持久化后端。

    用法：
        store = LanceMemoryStore("data/lancedb")
        if store.available:
            store.add(mem, brain_name="Brain-01")
            rows = store.search_vector(features, top_k=3)
    """

    def __init__(self, path: str = "data/lancedb",
                 table: str = "long_memory", dim: int = _STORE_DIM):
        self.path = path
        self.table_name = table
        self.dim = dim
        self.available = False
        self._error = "not loaded"
        self._db = None
        self._tbl = None
        try:
            import lancedb
        except Exception as e:
            self._error = f"lancedb 未安装: {e}"
            return
        try:
            os.makedirs(path, exist_ok=True)
            self._db = lancedb.connect(path)
            if table in self._db.table_names():
                self._tbl = self._db.open_table(table)
            self.available = True
        except Exception as e:
            self._error = f"LanceDB 打开失败: {e}"

    # ------------------ 内部 ------------------

    def _row(self, mem, brain_name: str) -> Dict:
        vec = list(getattr(mem, "features", []) or [])[:self.dim]
        if not vec:  # 文本记忆无 features → 哈希向量兜底
            vec = text_to_vector(mem.content, self.dim)
        vec = vec + [0.0] * (self.dim - len(vec))
        return {"content": mem.content, "vector": vec,
                "weight": float(mem.weight),
                "tag": getattr(mem, "tag", "") or "",
                "modality": getattr(mem, "modality", "text"),
                "brain": brain_name,
                "timestamp": float(getattr(mem, "timestamp", time.time()))}

    def _ensure_table(self, first_row: Dict):
        if self._tbl is None:
            self._tbl = self._db.create_table(self.table_name, [first_row])
            return True
        return False

    def _where_one(self, content: str, brain_name: str) -> str:
        return (f"content = '{_escape(content)}' AND "
                f"brain = '{_escape(brain_name)}'")

    def _scan_rows(self) -> List[Dict]:
        """全表扫描（Arrow → dict 列表）"""
        return self._tbl.to_arrow().to_pylist()

    # ------------------ 写入 ------------------

    def add(self, mem, brain_name: str = "") -> bool:
        """写入一条记忆（同 brain+content 先删后写，相当于 upsert）"""
        if not self.available:
            return False
        row = self._row(mem, brain_name)
        if self._ensure_table(row):
            return True
        try:
            self._tbl.delete(self._where_one(mem.content, brain_name))
            self._tbl.add([row])
            return True
        except Exception as e:
            self._error = f"写入失败: {e}"
            return False

    def update_weight(self, content: str, weight: float,
                      brain_name: str = "") -> bool:
        """更新权重（再巩固/强化同步）"""
        if not self.available or self._tbl is None:
            return False
        try:
            self._tbl.update(where=self._where_one(content, brain_name),
                             values={"weight": float(weight)})
            return True
        except Exception as e:
            self._error = f"权重更新失败: {e}"
            return False

    # ------------------ 检索 ------------------

    def search_vector(self, vec: List[float], top_k: int = 3,
                      brain_name: Optional[str] = None) -> List[Dict]:
        """向量近邻检索：返回 [{content, weight, tag, modality, distance}]"""
        if not self.available or self._tbl is None:
            return []
        vec = list(vec)[:self.dim] + [0.0] * max(0, self.dim - len(vec))
        try:
            q = self._tbl.search(vec[:self.dim]).limit(top_k)
            if brain_name:
                q = q.where(f"brain = '{_escape(brain_name)}'")
            rows = q.to_list()
        except Exception as e:
            self._error = f"检索失败: {e}"
            return []
        return [{"content": r["content"], "weight": r["weight"],
                 "tag": r["tag"], "modality": r["modality"],
                 "distance": round(float(r.get("_distance", 0.0)), 4)}
                for r in rows]

    def search_text(self, keyword: str, top_k: int = 3) -> List[Dict]:
        """关键词检索（字面匹配扫描，规模小时的简单通路）"""
        if not self.available or self._tbl is None:
            return []
        try:
            rows = [r for r in self._scan_rows()
                    if keyword in r["content"]]
        except Exception as e:
            self._error = f"检索失败: {e}"
            return []
        rows.sort(key=lambda r: -r["weight"])
        return [{"content": r["content"], "weight": r["weight"],
                 "tag": r["tag"], "modality": r["modality"]}
                for r in rows[:top_k]]

    # ------------------ 维护 ------------------

    def decay(self, factor: float = 0.995,
              forget_threshold: float = 0.05) -> int:
        """全部记忆权重 ×factor，低于阈值删除（与大脑衰减节律同步）。
        返回被删除的条数。"""
        if not self.available or self._tbl is None:
            return 0
        try:
            rows = self._scan_rows()
            for r in rows:
                self._tbl.update(
                    where=(f"content = '{_escape(r['content'])}' AND "
                           f"brain = '{_escape(r['brain'])}'"),
                    values={"weight": r["weight"] * factor})
            before = self._tbl.count_rows()
            self._tbl.delete(f"weight < {forget_threshold}")
            return before - self._tbl.count_rows()
        except Exception as e:
            self._error = f"衰减同步失败: {e}"
            return 0

    def count(self) -> int:
        if not self.available or self._tbl is None:
            return 0
        return self._tbl.count_rows()

    def info(self) -> Dict:
        return {"available": self.available, "path": self.path,
                "table": self.table_name, "rows": self.count(),
                "error": None if self.available else self._error}
