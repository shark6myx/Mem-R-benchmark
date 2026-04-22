# Leiden + Community Evolution History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace non-deterministic Louvain with Leiden community detection, and add `evolution_history` tracking to `CommunitySummary` so community membership changes are recorded across each rebuild.

**Architecture:** Two independent changes to `memory_layer.py`. Task 1 swaps the algorithm inside `_run_community_detection()` — the surrounding pipeline (`rebuild_communities`, graph construction, retrieval) is unchanged. Task 2 adds an `evolution_history: List[Dict]` field to `CommunitySummary`, updates serialization, and populates diffs in `rebuild_communities()` by Jaccard-matching old vs new communities before overwriting `self.communities`.

**Tech Stack:** `leidenalg`, `igraph` (new deps); existing `networkx`, `numpy`, `dataclasses`.

---

## Task 1: Replace Louvain with Leiden

**Files:**
- Modify: `memory_layer.py:1978-2018` (`_run_community_detection`)

### Step 1: Install Leiden dependencies

- [ ] Run:
```bash
pip install leidenalg igraph
```
Expected output: both packages install without error. Verify:
```bash
python -c "import leidenalg, igraph; print('OK')"
```
Expected: `OK`

---

### Step 2: Replace `_run_community_detection` body

- [ ] In `memory_layer.py`, replace lines 1978–2018 with:

```python
def _run_community_detection(self, G) -> Dict[str, int]:
    """
    在图 G 上运行 Leiden 社区检测（Louvain 的后继算法）。
    Leiden 保证社区内部严格连通，且支持 seed 参数确保结果可复现。

    参数:
        G: networkx.Graph 对象

    返回:
        {note_id: community_int} 映射字典
    """
    try:
        import leidenalg
        import igraph as ig
    except ImportError:
        raise ImportError(
            "leidenalg / igraph 未安装，请运行: pip install leidenalg igraph"
        )

    if len(G.nodes) == 0:
        return {}

    # 1. 运行时密度监测 (Weighted Average Degree)
    total_weight = G.size(weight='weight')
    num_nodes = len(G.nodes)
    avg_degree = (2.0 * total_weight) / num_nodes if num_nodes > 0 else 0

    # 2. 动态 resolution 计算（与原 Louvain 版本相同公式，保持连续性）
    import math
    d_base = 3.0
    alpha = 0.5
    if avg_degree > 0:
        resolution = 1.0 + alpha * math.log(1 + max(0, avg_degree - d_base) / d_base)
    else:
        resolution = 1.0
    resolution = max(0.5, min(2.5, resolution))

    # 3. networkx → igraph 转换
    # node_list 保证 index → node_id 的映射顺序
    node_list = list(G.nodes())
    ig_graph = ig.Graph.from_networkx(G)
    weights = ig_graph.es['weight'] if ig_graph.ecount() > 0 else None

    # 4. Leiden 聚类（seed=42 保证可复现）
    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=weights,
        seed=42,
        resolution_parameter=resolution,
    )

    # 5. 映射回 {node_id_str: community_int}
    return {node_list[i]: comm for i, comm in enumerate(partition.membership)}
```

---

### Step 3: Smoke-test community detection in isolation

- [ ] Run a quick sanity check (no full pipeline needed):
```bash
cd /home/isrc/myx/Mem-R-benchmark
python - <<'EOF'
import networkx as nx
import leidenalg, igraph as ig

G = nx.Graph()
for i in range(10):
    G.add_node(str(i))
for a, b, w in [("0","1",0.9),("1","2",0.8),("0","2",0.85),
                ("5","6",0.9),("6","7",0.8),("5","7",0.85),
                ("2","5",0.3)]:
    G.add_edge(a, b, weight=w)

node_list = list(G.nodes())
ig_g = ig.Graph.from_networkx(G)
weights = ig_g.es['weight'] if ig_g.ecount() > 0 else None
part = leidenalg.find_partition(
    ig_g, leidenalg.RBConfigurationVertexPartition,
    weights=weights, seed=42, resolution_parameter=1.0)
result = {node_list[i]: c for i, c in enumerate(part.membership)}
print("Partition:", result)
# 0,1,2 should be one community; 5,6,7 another
assert result["0"] == result["1"] == result["2"], "nodes 0-2 should be same community"
assert result["5"] == result["6"] == result["7"], "nodes 5-7 should be same community"
assert result["0"] != result["5"], "two groups should be different communities"
# Run twice to verify determinism
part2 = leidenalg.find_partition(
    ig_g, leidenalg.RBConfigurationVertexPartition,
    weights=weights, seed=42, resolution_parameter=1.0)
result2 = {node_list[i]: c for i, c in enumerate(part2.membership)}
assert result == result2, "Leiden must be deterministic with same seed"
print("All assertions passed. Leiden is deterministic.")
EOF
```
Expected: `All assertions passed. Leiden is deterministic.`

---

### Step 4: Delete cached graph state to force fresh rebuild

- [ ] Run:
```bash
rm -rf /home/isrc/myx/Mem-R-benchmark/cached_memories_advanced_vllm_qwen3_5-flash/graph_state_sample_0/
```
Old communities were built with Louvain. Next benchmark run will rebuild with Leiden.

---

### Step 5: Commit Task 1

- [ ] Run:
```bash
cd /home/isrc/myx/Mem-R-benchmark
git add memory_layer.py
git commit -m "feat(graphrag): replace Louvain with Leiden for deterministic community detection

- Leiden (leidenalg) guarantees internally-connected communities
- Fixed seed=42 makes results reproducible across runs
- Same dynamic resolution formula preserved for continuity
- Drops python-louvain dependency"
```

---

## Task 2: Add `evolution_history` to CommunitySummary

**Files:**
- Modify: `memory_layer.py:615-654` (`CommunitySummary` dataclass + `to_dict` + `from_dict`)
- Modify: `memory_layer.py:2126-2205` (`rebuild_communities`)

### Step 1: Add `evolution_history` field and update serialization

- [ ] In `memory_layer.py`, replace the `CommunitySummary` dataclass (lines 615–654) with:

```python
@dataclass
class CommunitySummary:
    """
    GraphRAG 社区摘要单元

    代表一组语义相关的 MemoryNote 聚类，由 LLM 生成高层摘要，
    供 Global Search 使用。

    evolution_history 记录每次 rebuild_communities() 时该社区的成员变更，
    可用于展示"话题随时间的演化轨迹"。
    每条记录格式：
        {
            "timestamp": "YYYYMMDDHHMM",   # 本次重建时间
            "total_notes": int,             # 重建时全局 note 总数
            "added_note_ids": List[str],    # 相比上次新加入的 note id
            "removed_note_ids": List[str],  # 相比上次离开的 note id
            "prev_title": str               # 上次重建时的社区标题（空串表示首次）
        }
    """
    community_id: str
    level: int                              # 0=细粒度层级
    member_note_ids: List[str]             # 隶属 note 的 id 列表
    title: str = ""                         # LLM 生成的主题标题（≤20字）
    summary: str = ""                       # LLM 生成的综合摘要（≤200字）
    keywords: List[str] = field(default_factory=list)
    embedding: Optional[Any] = None         # summary 的向量（np.ndarray）
    updated_at: str = ""
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """序列化（不含 embedding，embedding 单独存）"""
        return {
            "community_id": self.community_id,
            "level": self.level,
            "member_note_ids": self.member_note_ids,
            "title": self.title,
            "summary": self.summary,
            "keywords": self.keywords,
            "updated_at": self.updated_at,
            "evolution_history": self.evolution_history,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CommunitySummary":
        return cls(
            community_id=d["community_id"],
            level=d.get("level", 0),
            member_note_ids=d.get("member_note_ids", []),
            title=d.get("title", ""),
            summary=d.get("summary", ""),
            keywords=d.get("keywords", []),
            updated_at=d.get("updated_at", ""),
            evolution_history=d.get("evolution_history", []),
        )
```

---

### Step 2: Add `_match_old_communities` helper method

- [ ] Insert this new method **just before** `rebuild_communities` (i.e., after `_build_networkx_graph` and `_run_community_detection`), around line 2126:

```python
def _match_old_communities(
    self,
    old_communities: Dict[str, "CommunitySummary"],
    new_member_groups: Dict[str, List[str]],
) -> Dict[str, str]:
    """
    用 Jaccard 相似度将新社区映射到最相近的旧社区。

    参数:
        old_communities: 重建前的 self.communities 快照
        new_member_groups: {new_community_id -> member_note_ids}

    返回:
        {new_community_id -> old_community_id}（无匹配则不含该键）
    """
    matches: Dict[str, str] = {}
    for new_cid, new_members in new_member_groups.items():
        new_set = set(new_members)
        if not new_set:
            continue
        best_old_cid, best_score = None, 0.0
        for old_cid, old_cs in old_communities.items():
            old_set = set(old_cs.member_note_ids)
            if not old_set:
                continue
            intersection = len(new_set & old_set)
            union = len(new_set | old_set)
            jaccard = intersection / union if union > 0 else 0.0
            if jaccard > best_score:
                best_score = jaccard
                best_old_cid = old_cid
        # 只有 Jaccard ≥ 0.3 才认为是同一社区的演化，否则视为新社区
        if best_old_cid is not None and best_score >= 0.3:
            matches[new_cid] = best_old_cid
    return matches
```

---

### Step 3: Update `rebuild_communities` to record evolution diffs

- [ ] Replace `rebuild_communities` (lines 2126–2205) with:

```python
def rebuild_communities(self) -> None:
    """
    GraphRAG 社区重建：
      1. 构建 networkx 图
      2. 运行 Leiden 社区检测
      3. 为每个社区生成 LLM 摘要
      4. 更新 self.communities 和 community_embeddings
      5. 记录社区演化历史（evolution_history）

    会自动更新每条 note 的 community_id 字段。
    """
    if len(self.memories) < 2:
        return

    if not self._graph_is_dirty and len(self.communities) > 0:
        print(f"⏩ GraphRAG: 图结构未变脏，跳过社区重建")
        return

    print(f"🔄 GraphRAG: 重建社区（共 {len(self.memories)} 条记忆）...")

    # 快照旧社区，用于演化 diff
    old_communities: Dict[str, CommunitySummary] = dict(self.communities)

    # Step 1: 建图
    try:
        G = self._build_networkx_graph()
    except ImportError as e:
        print(f"⚠ {e}，跳过社区重建")
        return

    # Step 2: 社区检测
    try:
        partition = self._run_community_detection(G)   # {note_id: int}
    except ImportError as e:
        print(f"⚠ {e}，跳过社区重建")
        return

    if not partition:
        return

    # Step 3: 按社区 id 分组（排除 entity 虚拟节点）
    community_groups: Dict[int, List[str]] = {}
    for node_id, comm_int in partition.items():
        if not node_id.startswith("ent:"):
            community_groups.setdefault(comm_int, []).append(node_id)

    # 构建 new_cid -> member_ids 映射（用于匹配和 diff 计算）
    new_member_groups: Dict[str, List[str]] = {
        f"comm_{comm_int:04d}": member_ids
        for comm_int, member_ids in community_groups.items()
    }

    # Step 4: 将新社区匹配到旧社区（用于演化追踪）
    cid_to_old = self._match_old_communities(old_communities, new_member_groups)

    rebuild_ts = datetime.now().strftime("%Y%m%d%H%M")
    total_notes = len(self.memories)

    # Step 5: 为每个社区生成摘要并记录演化历史
    new_communities: Dict[str, CommunitySummary] = {}
    all_embeddings = []
    cid_order = []

    for comm_int, member_ids in community_groups.items():
        community_id = f"comm_{comm_int:04d}"

        # 更新 note.community_id
        for mid in member_ids:
            if mid in self.memories:
                self.memories[mid].community_id = community_id

        # 生成摘要
        cs = self._generate_community_summary(community_id, member_ids)

        # 计算演化 diff
        old_cid = cid_to_old.get(community_id)
        if old_cid and old_cid in old_communities:
            old_cs = old_communities[old_cid]
            old_set = set(old_cs.member_note_ids)
            new_set = set(member_ids)
            evolution_entry = {
                "timestamp": rebuild_ts,
                "total_notes": total_notes,
                "added_note_ids": sorted(new_set - old_set),
                "removed_note_ids": sorted(old_set - new_set),
                "prev_title": old_cs.title,
            }
            # 继承旧社区的演化历史，追加本次记录
            cs.evolution_history = old_cs.evolution_history + [evolution_entry]
        else:
            # 新出现的社区，记录首次创建
            cs.evolution_history = [{
                "timestamp": rebuild_ts,
                "total_notes": total_notes,
                "added_note_ids": sorted(member_ids),
                "removed_note_ids": [],
                "prev_title": "",
            }]

        new_communities[community_id] = cs

        if cs.embedding is not None:
            all_embeddings.append(cs.embedding)
            cid_order.append(community_id)

    # Step 6: 更新类属性
    self.communities = new_communities
    self.community_ids_list = cid_order
    if all_embeddings:
        self.community_embeddings = np.stack(all_embeddings, axis=0)
    else:
        self.community_embeddings = None

    # 重建完成后，重置脏标记
    self._graph_is_dirty = False

    print(f"✅ GraphRAG: 社区重建完成，共 {len(new_communities)} 个社区：")
    for cid, cs in new_communities.items():
        n_added = len(cs.evolution_history[-1]["added_note_ids"]) if cs.evolution_history else 0
        n_removed = len(cs.evolution_history[-1]["removed_note_ids"]) if cs.evolution_history else 0
        print(f"   [{cid}] {cs.title} ({len(cs.member_note_ids)} 条记忆, +{n_added}/-{n_removed})")
```

---

### Step 4: Smoke-test evolution_history serialization

- [ ] Run:
```bash
cd /home/isrc/myx/Mem-R-benchmark
python - <<'EOF'
from memory_layer import CommunitySummary

# Test round-trip serialization with evolution_history
cs = CommunitySummary(
    community_id="comm_0000",
    level=0,
    member_note_ids=["a", "b", "c"],
    title="Test Community",
    summary="A test.",
    keywords=["test"],
    updated_at="202604100900",
    evolution_history=[{
        "timestamp": "202604100900",
        "total_notes": 50,
        "added_note_ids": ["a", "b", "c"],
        "removed_note_ids": [],
        "prev_title": "",
    }]
)
d = cs.to_dict()
assert "evolution_history" in d, "evolution_history must be in to_dict()"
assert len(d["evolution_history"]) == 1

cs2 = CommunitySummary.from_dict(d)
assert cs2.evolution_history == cs.evolution_history, "round-trip must preserve evolution_history"
assert cs2.title == "Test Community"

# Test backward-compat: old dict without evolution_history
old_dict = {
    "community_id": "comm_0001",
    "level": 0,
    "member_note_ids": ["x"],
    "title": "Old",
    "summary": "Old community.",
    "keywords": [],
    "updated_at": "202601010000",
}
cs3 = CommunitySummary.from_dict(old_dict)
assert cs3.evolution_history == [], "missing key must default to empty list"

print("All serialization assertions passed.")
EOF
```
Expected: `All serialization assertions passed.`

---

### Step 5: Smoke-test `_match_old_communities`

- [ ] Run:
```bash
cd /home/isrc/myx/Mem-R-benchmark
python - <<'EOF'
from memory_layer import AgenticMemorySystem, CommunitySummary

# Build a minimal AgenticMemorySystem (no LLM needed for this test)
ms = AgenticMemorySystem.__new__(AgenticMemorySystem)
ms.communities = {}
ms.community_embeddings = None
ms.community_ids_list = []

old_communities = {
    "comm_0000": CommunitySummary(
        community_id="comm_0000", level=0,
        member_note_ids=["a","b","c","d"], title="Old A"),
    "comm_0001": CommunitySummary(
        community_id="comm_0001", level=0,
        member_note_ids=["x","y","z"], title="Old B"),
}
new_member_groups = {
    "comm_0000": ["a","b","c","d","e"],   # grew from old comm_0000
    "comm_0001": ["x","y","z","w"],       # grew from old comm_0001
    "comm_0002": ["p","q"],               # brand new
}

matches = ms._match_old_communities(old_communities, new_member_groups)
assert matches.get("comm_0000") == "comm_0000", f"expected comm_0000 match, got {matches}"
assert matches.get("comm_0001") == "comm_0001", f"expected comm_0001 match, got {matches}"
assert "comm_0002" not in matches, "brand new community must not match"
print("_match_old_communities assertions passed.")
EOF
```
Expected: `_match_old_communities assertions passed.`

---

### Step 6: Commit Task 2

- [ ] Run:
```bash
cd /home/isrc/myx/Mem-R-benchmark
git add memory_layer.py
git commit -m "feat(graphrag): add evolution_history to CommunitySummary

- CommunitySummary gains evolution_history: List[Dict] field
- Each rebuild records added/removed note IDs and prev_title
- _match_old_communities() uses Jaccard ≥ 0.3 to align old/new communities
- rebuild_communities() snapshots old state before overwriting
- Serialization is backward-compatible (missing key defaults to [])"
```

---

## Self-Review Checklist

- [x] Leiden swap: `_run_community_detection` fully replaced, same resolution formula, igraph conversion with deterministic seed
- [x] `evolution_history` field added to dataclass, `to_dict`, `from_dict`  
- [x] `_match_old_communities` helper defined before it is called in `rebuild_communities`
- [x] `rebuild_communities` snapshots `old_communities` before overwriting `self.communities`
- [x] Backward-compatible: `from_dict` uses `.get("evolution_history", [])` so old cache files load cleanly
- [x] No placeholders — every step has runnable code and expected output
- [x] Type consistency: `Dict[str, Any]` used in `evolution_history` entries, matches `List[Dict[str, Any]]` field type
