# Program-Guided Retrieval 12 Tasks

## 目标

在当前项目上增量实现一个更像论文主贡献的版本：

- 从“题型路由 + 参数切换”
- 升级为“证据程序预测 + 程序执行检索 + 证据验证 + 最小证据集生成”

当前核心入口文件：

- [runsingle.py](E:/Benchmark/Mem-R-benchmark/runsingle.py)
- [memory_layer.py](E:/Benchmark/Mem-R-benchmark/memory_layer.py)

当前关键入口位置：

- 路由入口：[runsingle.py](E:/Benchmark/Mem-R-benchmark/runsingle.py#L195)
- 检索入口：[runsingle.py](E:/Benchmark/Mem-R-benchmark/runsingle.py#L516)
- 回答入口：[runsingle.py](E:/Benchmark/Mem-R-benchmark/runsingle.py#L1028)
- RRF 检索核心：[memory_layer.py](E:/Benchmark/Mem-R-benchmark/memory_layer.py#L2832)
- 图边构建：[memory_layer.py](E:/Benchmark/Mem-R-benchmark/memory_layer.py#L1842)

## Task 1. 定义 Evidence Program 数据结构

目标：

- 把现在散落的路由参数，收敛成稳定的数据结构

建议新增文件：

- [evidence_program.py](E:/Benchmark/Mem-R-benchmark/evidence_program.py)

建议新增内容：

- `ProgramType`
- `AnswerStyle`
- `EvidenceProgram`
- `ExecutionTrace`

最低字段建议：

- `program_type`
- `answer_style`
- `max_notes`
- `max_hops`
- `channel_weights`
- `enable_ppr`
- `include_community_context`
- `need_verifier`
- `allow_abstain`

验收标准：

- 代码里不再用匿名 dict 传完整检索策略
- 可以 `print(program)` 看懂一个问题会走什么程序

## Task 2. 新增程序预测器

目标：

- 用当前已有的 heuristic + LLM route 预测 `EvidenceProgram`

主要修改文件：

- [runsingle.py](E:/Benchmark/Mem-R-benchmark/runsingle.py)

建议新增函数：

- `predict_evidence_program(self, question: str) -> EvidenceProgram`

建议复用：

- `_infer_task_attributes()`
- `_llm_route_question()`

建议输出的 program 类型先只做 5 个：

- `fact`
- `temporal`
- `multi_hop`
- `profile`
- `verify_unsupported`

验收标准：

- 给任意问题都能稳定返回一个 `EvidenceProgram`
- 不需要依赖 benchmark `category`

## Task 3. 将 `_classify_question_mode()` 降级为兼容层

目标：

- 不直接删除旧逻辑，避免一次性改太多
- 让旧逻辑暂时作为 `program -> runtime config` 的兼容层

主要修改文件：

- [runsingle.py](E:/Benchmark/Mem-R-benchmark/runsingle.py#L195)

建议做法：

- 保留 `_classify_question_mode()`
- 让它内部改为读取 `EvidenceProgram`
- 最终返回旧格式 dict，给现有调用点过渡使用

验收标准：

- 老的 `retrieve_memory(...)` 调用不崩
- 新旧接口能共存一段时间

## Task 4. 在 memory 层新增程序执行器

目标：

- 真正把“程序”映射成检索执行逻辑

主要修改文件：

- [memory_layer.py](E:/Benchmark/Mem-R-benchmark/memory_layer.py)

建议新增函数：

- `execute_evidence_program(self, program: EvidenceProgram, query: str, k: int = 10) -> Dict[str, Any]`

第一版只做程序到现有 RRF 的映射：

- `fact` -> 低噪声 RRF
- `temporal` -> 时间友好 RRF
- `multi_hop` -> RRF seed + graph expand
- `profile` -> community / ppr 更强
- `verify_unsupported` -> strict dense + bm25 only

验收标准：

- 不同 program 能真正走不同检索路径
- 输出仍兼容当前 `_format_retrieval_context()`

## Task 5. 新增最小证据压缩器

目标：

- 不再把一大堆 note 原样塞给 LLM
- 尽量只保留“最小但完整”的证据集

主要修改文件：

- [memory_layer.py](E:/Benchmark/Mem-R-benchmark/memory_layer.py)

建议新增函数：

- `compress_to_minimal_evidence(self, query: str, notes: List[MemoryNote], program: EvidenceProgram) -> List[MemoryNote]`

默认上限建议：

- `fact`: 3
- `temporal`: 3
- `profile`: 4
- `multi_hop`: 6
- `verify_unsupported`: 3

第一版规则：

- 去重
- 优先保留 query 实体直命中的 note
- 多跳题优先保留链路完整的 note 组合
- 删除重复支持同一事实的冗余 note

验收标准：

- 最终证据数量明显下降
- `raw_context` 更短、更聚焦

## Task 6. 新增多跳证据扩展器

目标：

- 让 `multi_hop` 程序不只是“调高 ppr 权重”
- 而是真正做图上的局部扩展

主要修改文件：

- [memory_layer.py](E:/Benchmark/Mem-R-benchmark/memory_layer.py#L1842)

建议新增函数：

- `expand_evidence_graph(self, seed_notes: List[MemoryNote], max_hops: int = 2, edge_types: Optional[List[str]] = None) -> List[MemoryNote]`

优先考虑的边类型：

- `semantic`
- `entity_shared`
- `temporal_entity_evolution`
- `causes`
- `results_in`
- `lexical_shared`

验收标准：

- 多跳题能从 seed 扩到更完整的因果/实体链
- 扩展结果不会失控到几十条 note

## Task 7. 新增证据验证器

目标：

- 把现在的“阈值拒答”升级成“证据支持判断”

建议新增文件：

- [evidence_verifier.py](E:/Benchmark/Mem-R-benchmark/evidence_verifier.py)

建议新增函数：

- `verify_evidence_support(question: str, evidence_notes: List[Any], program: EvidenceProgram) -> Dict[str, Any]`

返回建议字段：

- `label`: `supported / unsupported / insufficient`
- `reason`
- `supporting_spans`
- `confidence`

第一版实现建议：

- 规则过滤 + 小 LLM verifier

重点先覆盖：

- `verify_unsupported`
- `multi_hop`

验收标准：

- `category 5` 不再只靠 `rrf_confidence_threshold`
- 可以解释“为什么拒答 / 为什么可答”

## Task 8. 改造 `retrieve_memory()` 接口

目标：

- 支持 program 驱动的检索
- 保留旧接口兼容实验

主要修改文件：

- [runsingle.py](E:/Benchmark/Mem-R-benchmark/runsingle.py#L516)

建议新增参数：

- `program: Optional[EvidenceProgram] = None`

建议逻辑：

- 有 `program` -> 走 `execute_evidence_program()`
- 没 `program` -> 走现有旧路径

验收标准：

- 原有实验脚本还能跑
- 新方法可以并行对比旧方法

## Task 9. 改造 `answer_question()` 主链路

目标：

- 让主流程变成：
  `program prediction -> retrieval execution -> evidence verification -> answer generation`

主要修改文件：

- [runsingle.py](E:/Benchmark/Mem-R-benchmark/runsingle.py#L1028)

建议流程：

1. `program = predict_evidence_program(question)`
2. `retrieval_result = retrieve_memory(question, program=program)`
3. `evidence = compress_to_minimal_evidence(...)`
4. `verification = verify_evidence_support(...)`
5. unsupported -> abstain
6. supported -> generate answer

验收标准：

- 新主链能独立工作
- 可以把 program / verifier 信息写入日志

## Task 10. 把 prompt 选择改成 answer style 驱动

目标：

- 不再按 benchmark category 选 prompt
- 改成按程序里的 `answer_style`

主要修改文件：

- [runsingle.py](E:/Benchmark/Mem-R-benchmark/runsingle.py#L1074)

建议 style 先做 4 个：

- `extractive_short`
- `temporal_short`
- `reasoning_label`
- `abstain_or_span`

验收标准：

- prompt 选择不依赖 `category`
- `category-blind` 路径成立

## Task 11. 把后处理改成 answer style 驱动

目标：

- 后处理不再吃 benchmark category

主要修改文件：

- [runsingle.py](E:/Benchmark/Mem-R-benchmark/runsingle.py#L880)

建议改法：

- `postprocess_answer(question, answer_text, raw_context="", answer_style="extractive_short")`

建议规则：

- `temporal_short` -> 时间规范化
- `reasoning_label` -> 去掉多余解释
- `extractive_short` -> 尽量压短 span
- `abstain_or_span` -> 只保留拒答或最短支持 span

验收标准：

- category-blind 下仍能保留当前的 EM/F1 修正收益

## Task 12. 增加实验与日志支持

目标：

- 让新方法能清楚做 ablation 和 error analysis

主要修改文件：

- [runsingle.py](E:/Benchmark/Mem-R-benchmark/runsingle.py#L1523)

建议结果里新增字段：

- `program_type`
- `answer_style`
- `need_verifier`
- `verification_label`
- `compressed_evidence_count`
- `execution_trace`

最少实验组建议：

- `Static RRF`
- `Task-aware Routing`
- `Program Retrieval`
- `Program Retrieval + Verifier`

验收标准：

- 能看出新方法具体赢在哪类题
- 能支持论文里的 ablation 表和案例分析

## 推荐执行顺序

建议按这个顺序落地：

1. Task 1
2. Task 2
3. Task 4
4. Task 5
5. Task 8
6. Task 9
7. Task 10
8. Task 11
9. Task 7
10. Task 12
11. Task 6
12. Task 3

原因：

- 先做能跑通的主链
- 再做 verifier 和图扩展
- 最后再清理兼容层

## MVP 验收标准

如果你只做第一阶段，至少应达到：

- 能为每个问题输出 `program_type`
- 能按 program 执行不同检索路径
- 最终只向 LLM 提供压缩后的最小证据集
- 可对 `unsupported` 问题执行显式验证
- 结果文件中能看到 program 和 verifier 信息
- 能与当前 RRF 路径做对照实验

## 论文友好版主张

完成上述 12 个任务后，论文里的主张可以写成：

- 提出一个面向长对话记忆问答的 program-guided evidence retrieval 框架
- 提出一个最小完整证据集压缩策略，缓解长上下文噪声与幻觉
- 提出一个证据验证式回答机制，提升多跳题与对抗题的稳定性
