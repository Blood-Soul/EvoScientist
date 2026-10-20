# Evo 经验系统更新说明（2026-10-20）

本次更新包括可选的任务导向抽取、经验关系侧文件、结构化字段候选生成，以及图关系在检索与策略生成中的应用。原始经验记录继续使用现有文件存储；关系通过 JSON 侧文件补充，无需图数据库。

## 1. 可选任务导向抽取

新增 `TaskContext`，以任务、评价目标、数据特点、瓶颈、参考点和抽取侧重点描述下游需求。上下文作为独立的 `[task_context]` JSON 块传入 L1、直接 L2 和基于 L1 的 L2 归纳流程。

提示词明确要求：任务信息只调整筛选与表达重点，经验仍须来自论文证据；区分可迁移机制和原论文的适用边界，不得把任务需求写成论文已经验证的结论。`transferable_core` 仍描述一般机制。

不传上下文时不追加任务块，human message 与原有无任务路径一致。公共抽取提示词增加了可选上下文说明，因此这里的“一致”不表示历史 system prompt 没有变化。做公平对比时，两组应使用相同版本的提示词、相同论文全文和模型配置，仅改变任务上下文。

相关入口：

- `EvoScientist/memory/experiences/task_context.py`：数据结构、校验和渲染。
- `prompt/task_context.schema.json`：JSON schema；`prompt/task_context.examples/solar-weekly-mae.json`：实例。
- `EvoScientist/memory/experiences/extraction.py`、论文经验 worker 和主动抽取工具：可选参数传递。
- `scripts/extract_batch.py`：批量抽取入口。

任务描述示例：

```json
{
  "task": "根据分子图预测实值分子性质",
  "objective": "MAE",
  "direction": "minimize",
  "data_characteristics": ["输入包含原子与化学键信息"],
  "bottlenecks": ["需要区分局部结构与全图信息的作用"],
  "extraction_focus": ["保留可实现的机制、实验依据与适用条件"]
}
```

批量使用示例（会调用已配置的模型；论文须先转换为 Markdown）：

```bash
python scripts/extract_batch.py --bank ./bank-task --project-out P-demo --corpus ./papers --limit 5 --task-context ./task.json
python scripts/extract_batch.py --bank ./bank-neutral --project-out P-demo --corpus ./papers --limit 5
```

另外提供 `prepare_task_extraction_ab.py` 固化两臂输入、`replay_task_extraction_comparison.py` 校验外部提供的抽取响应。离线回放不等于重新调用模型生成，输出保留来源与输入哈希。

## 2. 从结构化字段生成候选关系

候选阶段同时覆盖 L1、L2；使用每条经验的有限提名，避免将所有两两组合提交给模型。词面候选保留为默认策略；通过 `--candidate-strategy field` 启用字段策略。

字段策略建立六类视图：

| 视图 | 主要字段 | 用途 |
| --- | --- | --- |
| 问题 | `task`、`trigger_context` | 发现共同问题和触发条件 |
| 方法 | `action`、`transferable_core` | 发现相近操作与机制 |
| 迁移 | `transferable_core`，缺失时回退到 `action` | 提名可能跨领域的联系 |
| 正向条件 | `applicable_when` | 检查适用条件 |
| 反向条件 | `not_applicable_when` | 发现边界与潜在冲突 |
| 结果 | `effect`、`practice_trace.feedback` | 联系结果、失败与后续处理 |

候选路线包括 `shared_problem`、`shared_method`、`boundary`、`prerequisite`、`transfer` 和 `lexical_fallback`。规则标签用于组织问题、方法和决策角色，字段引用保留具体来源。多路排名用 RRF 融合，再统一执行每条经验的提名额度和无向经验对去重。

默认每条经验最多提名 3 个对象，不强行补足低质量候选。不设置额外总量上限时，候选对数不超过 `min(N × 3, N × (N−1) / 2)`；这不是每个节点最终度数的上限，因为其他节点也可能提名它。候选生成仍可能需要库内相似度计算，这个界主要约束模型判断次数。

跨领域候选要求不同论文、不同且非空的领域标签，并要求多个结构标签轴相交，其中包括方法轴。目的在于事后发现不同领域经验的共享机制；它只表示值得进一步判断，不证明迁移成立。现有规则词表仍有领域偏向，大库性能和跨领域收益需要单独验证。

## 3. 关系类型、证据及文件存储

两阶段流程：

```text
读取原始经验 → 字段视图/词面候选 → 融合与有限提名
                                     ↓
                        模型判断 / 外部判断响应校验
                                     ↓
                      关系、证据、条件、版本写入 JSON
```

| 关系 | 含义与方向 |
| --- | --- |
| `derived_from` | L2 指向源 L1；由已有来源字段生成，不经模型推断 |
| `supports` | 源经验为目标经验提供支持 |
| `qualifies` | 源经验限定目标结论的适用边界 |
| `contradicts` | 两条结论存在冲突；对称 |
| `alternative_to` | 可以作为替代方案；对称 |
| `complements` | 在条件满足时可能互补；对称 |
| `overlaps_with` | 内容重叠；对称 |
| `specializes` | 源经验是目标经验的更具体版本 |
| `depends_on` | 使用源经验需要目标经验所述前提 |
| `fixed_by` | 失败经验指向修复经验；限定同篇论文 |

模型可以输出无关系或不确定，不要求每个候选都建边。`prompt/experience_linker.md` 给出类型定义、方向、判断规则和输出契约。校验包括端点、类型、双端字段原文引用、对称边规范化及重复关系处理。引用能验证经验记录中的文字存在，不能替代原论文核验。

默认目录如下，其中 `bank` 是传入的 memory directory，原始论文记录保留在既有项目目录中：

```text
bank/
  experiences/projects/P-demo/
    ...既有论文经验文件...
    experience_graph.json
    experience_graph.routing-index.json  # 字段候选策略生成
```

`experience_graph.json` 包含 `records`（节点快照）、`candidate_state`、`candidate_evidence`、`candidate_stats`、`edges`、`decisions`、`history`、提示词哈希、分类器标识和项目标识。字段索引保存字段视图、规则标签及其来源。

下面是帮助理解的关系示意，ID、文字和哈希均为示例，实际输出使用真实记录：

```json
{
  "from": "E-example-a",
  "to": "E-example-b",
  "relation": "qualifies",
  "claim": "历史过短时，应限制对季节模式拟合收益的预期",
  "reason": "源经验给出了目标方法的历史长度边界",
  "conditions": ["历史不足以覆盖完整季节周期"],
  "limitations": ["尚未验证其他领域是否成立"],
  "unresolved": [],
  "evidence": [
    {"id": "E-example-a", "field": "not_applicable_when", "quote": "历史不足以覆盖完整季节周期"},
    {"id": "E-example-b", "field": "action", "quote": "拟合季节模式"}
  ],
  "status": "unreviewed",
  "record_versions": {"E-example-a": "<hash>", "E-example-b": "<hash>"},
  "provenance": {
    "method": "classifier_inference",
    "prompt_hash": "<hash>",
    "classifier_id": "<classifier>",
    "source_check": "experience_record_only"
  }
}
```

有效判断只在经验内容、提示词和分类器匹配时复用；历史判断保留供追溯。图读取检查项目与经验版本，内容变化后旧图不能直接作为当前关系使用。跨项目重绑定提供严格内容匹配接口，不能仅替换 ID。

构图示例：

```bash
# 只生成候选与已有的 L2→L1 来源边，不调用模型判断
python scripts/build_experience_graph.py --bank ./bank-neutral --project P-demo --candidate-strategy field

# 对候选调用已配置模型，验证响应后落盘
python scripts/build_experience_graph.py --bank ./bank-neutral --project P-demo --candidate-strategy field --classify

# 搜索经验并查看一跳关系，不调用模型
python scripts/build_experience_graph.py --bank ./bank-neutral --project P-demo --query "short history"
```

`--neighbors` 调整单节点提名额度；`--max-pairs 0` 表示不另设全局对数上限。`--responses` 可校验外部提供的判断，不能与 `--classify` 同时使用。`scripts/evaluate_field_routing.py` 用于离线比较候选策略及路线消融，不调用模型。

## 4. 在检索和经验应用中的使用

搜索工具增加 `include_related`、`related_types` 和 `related_limit`；默认不扩展。指定后返回已有命中附近的一跳关系，仍遵守项目、领域、学科及经验级别过滤。

策略流程通过环境变量 `EVOSCIENTIST_EXPERIENCE_GRAPH_ENABLED=true` 启用，默认关闭。开启后的流程为：

1. 先按原逻辑召回经验；默认 8 个候选中保留前 4 个词面命中。
2. 使用已存图关系补充最多 4 个邻居，不足时从原召回列表补齐；边界、冲突等关系优先。
3. 候选继续进入现有任务重排，默认最多选 4 条；图邻居不会直接绕过重排进入执行。
4. 仅向策略生成器附加两端均被选中的关系，最多 8 条、总 JSON 不超过 10,000 字符，保留整条证据而非截断 JSON。
5. 策略缓存纳入图指纹、选中记录内容和当前状态；记录 `graph_retrieve`、`graph_selected` 等事件，便于确认邻居是否召回、选中和进入策略生成。

图关系在提示词中作为待核验的数据提供，保留适用条件、限制与未解决事项。缺图、过期图或无法读取时回退原检索路径。

## 5. 验证范围与后续限制

回归检查覆盖任务上下文传递与默认输入兼容、字段候选和额度、引用校验、关系方向、增量复用、图新鲜度、过滤隔离、策略缓存以及关闭开关时的原流程。A/B 工具测试另行检查同文同提示词、任务块差异和回放来源标注。测试均使用本地数据和模拟模型响应，不产生线上模型费用。

提交前验证：相关代码 Ruff 检查通过；回归与 A/B 工具测试合计 238 项通过、1 项失败。失败项为 `tests/test_paper_experience_memory.py::test_a_query_lost_to_the_tokenizer_is_reported_not_scored_silently`：它仍期待中文查询仅保留 `idea`，但当前分词器已保留中文，因而不报告退化。直接加载更新前 HEAD 的 retrieval 模块也返回空退化列表，确认该断言与基线现有行为不一致；本次未修改这项既有测试或分词器。

复现命令（在项目 Python 环境中执行，无模型调用）：

```bash
python -m pytest tests/test_experience_candidates.py tests/test_experience_field_routing.py tests/test_experience_graph.py tests/test_experience_graph_policy.py tests/test_task_context.py tests/test_task_extraction_ab.py tests/test_experience_policy.py tests/test_experience_coach.py tests/test_paper_experience_memory.py tests/test_backfill_experience_fields.py tests/test_policy_ab_harness.py -q
```

此更新证明流程可执行，不宣称已获得稳定的 benchmark 增益。后续需要多次配对实验、实际关系质量评审，以及从召回、重排到代码落地的利用率分析；跨领域候选也需要独立验证。
