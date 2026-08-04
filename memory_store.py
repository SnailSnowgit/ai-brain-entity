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
import json
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
        self._vtbl = None  # 版本表（v6.1 懒建）
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

    def _scan_rows(self, tbl=None) -> List[Dict]:
        """全表扫描（Arrow → dict 列表）"""
        return (tbl or self._tbl).to_arrow().to_pylist()

    # ------------------ 写入 ------------------

    def add(self, mem, brain_name: str = "") -> bool:
        """写入一条记忆（同 brain+content 先删后写，相当于 upsert）"""
        if not self.available:
            return False
        row = self._row(mem, brain_name)
        if self._ensure_table(row):
            self._log_version(mem.content, brain_name,
                              float(mem.weight), "add")
            return True
        try:
            self._tbl.delete(self._where_one(mem.content, brain_name))
            self._tbl.add([row])
            self._log_version(mem.content, brain_name,
                              float(mem.weight), "add")
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
            self._log_version(content, brain_name, float(weight),
                              "reinforce")
            return True
        except Exception as e:
            self._error = f"权重更新失败: {e}"
            return False

    # ------------------ 检索 ------------------

    def search_vector(self, vec: List[float], top_k: int = 3,
                      brain_name: Optional[str] = None,
                      modality: Optional[str] = None,
                      exclude_modality: Optional[str] = None) -> List[Dict]:
        """向量近邻检索：返回 [{content, weight, tag, modality, distance}]

        modality="visual" 只查该模态；exclude_modality="text" 排除该模态
        （跨模态联想：看到图像 → 只从非视觉记忆里联想）。
        """
        if not self.available or self._tbl is None:
            return []
        vec = list(vec)[:self.dim] + [0.0] * max(0, self.dim - len(vec))
        try:
            q = self._tbl.search(vec[:self.dim]).limit(top_k)
            clauses = []
            if brain_name:
                clauses.append(f"brain = '{_escape(brain_name)}'")
            if modality:
                clauses.append(f"modality = '{_escape(modality)}'")
            if exclude_modality:
                clauses.append(f"modality != '{_escape(exclude_modality)}'")
            if clauses:
                q = q.where(" AND ".join(clauses))
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
                new_w = r["weight"] * factor
                self._tbl.update(
                    where=(f"content = '{_escape(r['content'])}' AND "
                           f"brain = '{_escape(r['brain'])}'"),
                    values={"weight": new_w})
                self._log_version(r["content"], r["brain"], new_w, "decay")
            before = self._tbl.count_rows()
            self._tbl.delete(f"weight < {forget_threshold}")
            return before - self._tbl.count_rows()
        except Exception as e:
            self._error = f"衰减同步失败: {e}"
            return 0

    # ------------------ 记忆版本控制（v6.1） ------------------

    def _versions_table(self, first_row: Optional[Dict] = None):
        """版本表：每条记忆的修改历史（懒建）"""
        if self._vtbl is None:
            if "memory_versions" in self._db.table_names():
                self._vtbl = self._db.open_table("memory_versions")
            elif first_row is not None:
                self._vtbl = self._db.create_table("memory_versions",
                                                   [first_row])
        return self._vtbl

    def _log_version(self, content: str, brain_name: str,
                     weight: float, reason: str) -> None:
        """记录一次记忆变更（尽力而为）。reason: add/reinforce/decay"""
        try:
            existed = self._versions_table() is not None
            row = {"content": content, "brain": brain_name,
                   "version": 1, "weight": float(weight),
                   "reason": reason, "timestamp": time.time()}
            tbl = self._versions_table(
                first_row=None if existed else row)
            if tbl is None:
                return
            if existed:  # 首行已在建表时写入，只追加后续版本
                hist = [r for r in self._scan_rows(tbl)
                        if r["content"] == content
                        and r["brain"] == brain_name]
                row["version"] = max((r["version"] for r in hist),
                                     default=0) + 1
                tbl.add([row])
        except Exception:
            pass

    def memory_history(self, content: str,
                       brain_name: str = "") -> List[Dict]:
        """一条记忆的完整修改历史（按版本号升序）：
        [{version, weight, reason, timestamp}]——记忆的演化轨迹。"""
        if not self.available:
            return []
        tbl = self._versions_table()
        if tbl is None:
            return []
        hist = [r for r in self._scan_rows(tbl)
                if r["content"] == content and r["brain"] == brain_name]
        hist.sort(key=lambda r: r["version"])
        return [{"version": r["version"], "weight": r["weight"],
                 "reason": r["reason"], "timestamp": r["timestamp"]}
                for r in hist]

    def recall_version(self, content: str, version: int = -1,
                       brain_name: str = "") -> Optional[Dict]:
        """"回忆"过去的版本：version=-1 最新、-2 上一个，或指定版本号"""
        hist = self.memory_history(content, brain_name)
        if not hist:
            return None
        if version < 0:
            idx = max(len(hist) + version, 0)
            return hist[idx]
        for h in hist:
            if h["version"] == version:
                return h
        return None

    def count(self) -> int:
        if not self.available or self._tbl is None:
            return 0
        return self._tbl.count_rows()

    def info(self) -> Dict:
        return {"available": self.available, "path": self.path,
                "table": self.table_name, "rows": self.count(),
                "error": None if self.available else self._error}


# ==================== DNA 基因库（v6.1） ====================

class DNALibrary:
    """DNA 基因库：存储多个大脑的完整 DNA（dump_dna 快照）。

    能力：
        - save/get：DNA 存取（JSON 整体入库存 content 列之外的 dna 字段）
        - search：按人格参数（sensation_seeking/habituation_rate）、
          世代、名字检索
        - lineage：进化谱系追踪（沿 parents 链回溯祖先）

    lancedb 未安装时 available=False，调用方降级。
    """

    def __init__(self, path: str = "data/lancedb",
                 table: str = "dna_library"):
        self.path = path
        self.table_name = table
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

    @staticmethod
    def _meta(dna: dict, generation: int,
              parents: List[str]) -> Dict:
        personality = dna.get("personality", {}) or {}
        return {
            "name": dna.get("name", "?"),
            "generation": int(generation),
            "sensation_seeking": float(
                personality.get("sensation_seeking", 0.5)),
            "habituation_rate": float(
                personality.get("habituation_rate", 0.2)),
            "n_ltm": len(dna.get("long_memory", [])),
            "parents": json.dumps(parents or [], ensure_ascii=False),
        }

    def _scan(self) -> List[Dict]:
        if self._tbl is None:
            return []
        return self._tbl.to_arrow().to_pylist()

    # ------------------ 存取 ------------------

    def save(self, dna: dict, generation: int = 1,
             parents: Optional[List[str]] = None) -> str:
        """保存一份 DNA，返回 dna_id。parents 为亲代 dna_id 列表（谱系）"""
        if not self.available:
            return ""
        parents = parents or []
        meta = self._meta(dna, generation, parents)
        dna_id = f"{meta['name']}_g{generation}_{int(time.time() * 1000)}"
        row = {"dna_id": dna_id, **meta,
               "dna": json.dumps(dna, ensure_ascii=False, default=str),
               "created_at": time.time()}
        if self._tbl is None:
            self._tbl = self._db.create_table(self.table_name, [row])
        else:
            self._tbl.add([row])
        return dna_id

    def get(self, dna_id: str) -> Optional[dict]:
        """取回完整 DNA（可直接 AIBrainEntity.from_dna(dna)）"""
        for r in self._scan():
            if r["dna_id"] == dna_id:
                return json.loads(r["dna"])
        return None

    def meta_of(self, dna_id: str) -> Optional[Dict]:
        """取元信息（不含 DNA 本体）"""
        for r in self._scan():
            if r["dna_id"] == dna_id:
                return {k: r[k] for k in
                        ("dna_id", "name", "generation",
                         "sensation_seeking", "habituation_rate",
                         "n_ltm", "parents", "created_at")}
        return None

    # ------------------ 搜索 ------------------

    def search(self, name_contains: Optional[str] = None,
               min_generation: Optional[int] = None,
               sensation_seeking: Optional[tuple] = None,
               habituation_rate: Optional[tuple] = None) -> List[Dict]:
        """按人格/参数搜索（范围参数为 (min, max) 元组）。
        返回元信息列表（不含 DNA 本体）。"""
        out = []
        for r in self._scan():
            if name_contains and name_contains not in r["name"]:
                continue
            if min_generation is not None and \
                    r["generation"] < min_generation:
                continue
            if sensation_seeking and not (
                    sensation_seeking[0] <= r["sensation_seeking"]
                    <= sensation_seeking[1]):
                continue
            if habituation_rate and not (
                    habituation_rate[0] <= r["habituation_rate"]
                    <= habituation_rate[1]):
                continue
            out.append(self.meta_of(r["dna_id"]))
        out.sort(key=lambda m: -m["created_at"])
        return out

    # ------------------ 谱系 ------------------

    def lineage(self, dna_id: str) -> List[Dict]:
        """进化谱系追踪：从该个体沿 parents 链回溯到始祖。
        返回 [始祖, ..., 亲代]（不含自身）。"""
        chain, seen, current = [], set(), dna_id
        while current and current not in seen:
            seen.add(current)
            meta = self.meta_of(current)
            if meta is None:
                break
            parents = json.loads(meta["parents"] or "[]")
            if not parents:
                break
            current = parents[0]  # 主谱系取第一亲代
            parent_meta = self.meta_of(current)
            if parent_meta:
                chain.append(parent_meta)
        chain.reverse()
        return chain

    def count(self) -> int:
        if not self.available or self._tbl is None:
            return 0
        return self._tbl.count_rows()

    def info(self) -> Dict:
        return {"available": self.available, "path": self.path,
                "table": self.table_name, "rows": self.count(),
                "error": None if self.available else self._error}
