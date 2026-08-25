# `experience` 分支相对 `main` 的改动说明

> 对比基准：`git diff main...experience`（当前 `HEAD`：`25ee4f5`）  
> 结果：33 个文件变更，新增约 10,931 行、删除 5 行。  
> 本文基于 Git 中实际纳入该分支、且不在 `main` 的改动整理；其中的验证报告属于交付/验证材料，不等同于运行时代码。

## 一句话结论

这个分支增加了一条“论文经验”的完整闭环：

```text
论文全文 / 研究主题
        │
        ▼
paper-experience：抽取 L1 实践经验 + L2 归纳经验
        │                         │
        │（会话缓存）             ├── 可选：promote_to_memory
        │                         ▼
        └──────────────────► EvoMemory observation 库
                                      │
                         experience-answer 检索、精读、带证据链回答
                                      │
                    solution-ab：同一问题下“经验上下文 vs 论文原文”对比
```

因此，你的印象基本正确：核心是新增了几个 skill，用于从论文中抽取可复用经验，并可选择沉淀到 EvoMemory 的长期 observation 库；不过分支也同时补齐了抽取引擎、离线导入、图谱连边、测试、提示词和展示/验证材料。

## 1. 新增和更新的 skill

### 1.1 新增 `paper-experience`：论文 → L1/L2 经验

文件：`skills/paper-experience/SKILL.md`

这是经验生产端，支持两种入口：

| 入口 | 用户给定内容 | 处理方式 |
|---|---|---|
| A：指定论文 | arXiv URL/ID、DOI 或工作区 Markdown | 确保获得全文 Markdown，然后调用抽取工具 |
| B：指定主题 | 研究问题或关键词 | 轻量检索 top-N（默认 3）论文，抓取全文后批量抽取 |

抽取结果分两层：

- **L1 Practical Experiences（实践经验）**：研究者做了什么、在什么条件下做、行动—反馈链（`practice_trace`）是什么。
- **L2 Inductive Experiences（归纳经验）**：论文作者明确给出的可迁移规律/断言，包含适用边界、置信度、因果解释和证据引用。

该 skill 还规定输出必须保留完整 narrative 和原文证据，不能把抽取结果压缩成无来源的泛泛总结；中文场景保留英文证据原文。

配套文件：

| 文件 | 作用 |
|---|---|
| `skills/paper-experience/scripts/fetch_fulltext.py` | 面向 arXiv 的全文抓取与 Markdown 转换 |
| `skills/paper-experience/scripts/promote_to_memory.py` | 将当前会话的已抽取经验写入 EvoMemory |
| `skills/paper-experience/references/experience-schema.md` | L1/L2 字段说明 |

### 1.2 新增 `experience-answer`：从经验库回答问题

文件：`skills/experience-answer/SKILL.md`

这是经验消费端。它不再抽取新论文，而是：

1. 将问题拆成 2–3 组关键词查询；
2. 用 `search_observations` 只看摘要做粗筛；
3. 对真正相关的 3–5 条调用 `read_memory` 精读；
4. 输出回答、所用经验、检索/筛选/综合过程及未覆盖缺口。

该 skill 特别约束：当前检索为 TF-IDF 关键词匹配，不能只相信分数。高分但主题跑偏的记录必须排除，库中没有的内容必须明确说“未覆盖”，不能用模型常识补写。

### 1.3 新增 `solution-ab`：验证经验是否足以支持方案设计

文件：`skills/solution-ab/SKILL.md`

此 skill 并非写入 memory，而是对经验抽取效果做 A/B 对比：在**同一问题、同一方案骨架、同一论文集合**下，分别使用：

- A 组：抽取出的 L1/L2 经验；
- B 组：论文全文（去掉参考文献与附录）。

它要求分别生成两个方案，并并列展示上下文大小、来源引用、差异和各自未覆盖项；规则明确禁止评判哪一组“更好”。配套的 `skills/solution-ab/scripts/build_contexts.py` 负责构建两组上下文并报告长度差异。

### 1.4 更新 `paper-navigator`：检索论文后可自动抽取经验

文件：`skills/paper-navigator/SKILL.md`

已有的论文检索 skill 被扩展为：在已抓到论文全文后，调用 `extract_paper_experiences` 或批量版本。这样原来的“找论文/读论文”流程可直接产出 L1/L2 经验；而 `paper-experience` 则把“只抽经验”的需求从复杂的检索评分流程中单独拆出。

## 2. 运行时代码：抽取、缓存与工具注册

### 2.1 新增抽取工具

文件：`EvoScientist/tools/paper_experience.py`（新增，866 行）

新增两个 LangChain runtime tool：

| 工具 | 用途 |
|---|---|
| `extract_paper_experiences` | 对一篇工作区内的全文 Markdown 抽取 L1 + L2 |
| `extract_paper_experiences_batch` | 对多篇论文并发抽取，默认并发数为 4，可由 `PAPER_NAV_EXPERIENCE_CONCURRENCY` 调整 |

实现要点：

- L1 与 L2 使用不同提示词并发调用配置的 auxiliary model；没有 auxiliary model 时回退主模型。
- 严格解析模型返回的 JSON；支持带代码围栏或前置文字的结果，并校验顶层结构和 `paper_id`。
- 只允许读取工作区内的 Markdown，避免工具任意读取工作区外文件。
- 统一标准化 arXiv URL/ID、DOI 等论文标识。
- 产物会格式化为可直接放入 agent 上下文的 Markdown，而不是把 JSON 原样暴露给用户。

### 2.2 会话级缓存

抽取结果写在：

```text
<MEMORIES_DIR>/paper_experiences/sessions/<session>/<paper>/{l1,l2}.json
```

缓存按**会话 + 论文**隔离，并记录论文正文 SHA-256 与提示词 SHA-256：正文或提示词变更时会自动重新抽取。L1/L2 分开持久化，因此其中一层失败时，已成功的一层可保留供重试使用。并发重复请求同一会话/论文时，有进程内锁避免重复 LLM 调用。

注意：这一步本身是会话缓存，尚不是跨会话长期记忆；长期沉淀需要下一节的 promote/import 流程。

### 2.3 注册到主 Agent

改动文件：

- `EvoScientist/EvoScientist.py`
- `EvoScientist/tools/__init__.py`

两个抽取工具被加入基础工具集，无论是否加载 MCP 都可使用。新增的 model getter 会在显式 `create_cli_agent(config=..., chat_model=...)` 场景遵从当前配置，避免模型切换后意外沿用旧的全局 auxiliary model。

此外，`pyproject.toml` 将两个抽取提示词作为包数据安装到 `share/evoscientist/prompt`，使安装后的运行环境也能找到提示词文件。

## 3. 从经验抽取到长期 Memory 的数据层

### 3.1 在线沉淀：`promote_to_memory.py`

文件：`skills/paper-experience/scripts/promote_to_memory.py`

当用户明确要求“沉淀到记忆”时，该脚本从当前会话的抽取缓存读取 JSON，写入原生 EvoMemory observation 存储：

| 抽取层级 | EvoMemory 类型 |
|---|---|
| L1 实践经验 | `procedural` |
| L2 归纳经验 | `semantic` |

支持全局或项目级 scope、指定 session 和 `--dry-run`；通过 observation 的内容 hash 去重，因此重复执行不会持续制造重复记录。

### 3.2 离线经验库导入

文件：`scripts/import_experience_bank.py`

该脚本将已有的 L1/L2 JSONL 经验库批量导入同一 EvoMemory observation 层。默认导入 L2 与 L1 的 coarse/medium 粒度，细粒度 L1 需显式 `--include-l1-fine`。同样支持 `--dry-run`、`--scope` 和幂等去重。

### 3.3 统一字段映射

文件：`scripts/experience_mapping.py`

在线 promote 与离线 import 共用同一份 L1/L2 → observation 映射，避免两条入库路径字段漂移。映射会保留：

- 摘要（用于索引）和完整 Narrative；
- L1 的 `practice_trace`、任务/适用条件；
- L2 的声明、置信度、因果解释与机制深度；
- 分类字段、关键词、来源章节和原文引文。

同时兼容旧 schema 与本分支的 v3 flat schema（如 `statement`、`confidence`、`rationale`、`evidence[]`）。

### 3.4 为批量导入记录补图谱关系

文件：`scripts/link_observations_by_rule.py`

批量导入的 observation 不经过对话期的自动 linker，因此脚本按确定性规则补充 `complements` 边：

1. 同一论文内 L1 与 L2 的互补关系优先；
2. 同一论文内同层级的不同经验；
3. 不同论文但同领域的代表节点链式连接，避免生成稠密“毛线团”图。

可先用 `--dry-run` 查看计划边，且每篇论文有默认 8 条边的上限。

## 4. 抽取提示词与 schema 升级

新增：

- `prompt/l1_extract.md`
- `prompt/l2_inductive.md`
- `prompt/README.md`

提示词把论文正文转成严格 JSON，并强调可复用性、作者证据、适用边界、置信度校准与原文引文。运行时代码和字段映射同时保留对旧字段的兼容，因此已有经验数据仍可显示和入库。

## 5. 测试与验证材料

### 自动化测试

新增测试文件：

- `tests/test_paper_experience_search.py`：解析、缓存、并发、会话隔离、工具注册等；
- `tests/test_experience_mapping.py`：L1/L2 映射与字段保留；
- `tests/test_link_observations.py`：图谱规则连边；
- `tests/test_build_contexts.py`：A/B 上下文构建。

### 交付/验证文档

新增 `docs/report/` 下的报告，包括：

- `milestone-paper-experience.md`：端到端抽取、导入与召回的里程碑记录；
- `experience-answer-verification.md`：经验库问答的命中与跑题噪音验证；
- `solution-ab-verification.md`：经验上下文与原文上下文的 A/B 验证；
- `10_query_experience_showcase.md`、`demo-briefing.md`、`demo-prompts.md`：演示查询与展示材料；
- `paper-experience-report.md`：早期/整体功能说明。

这些文档记录了当时的样例数据、配置和实测结果；它们能说明设计意图和验收情况，但不应替代对当前环境重新运行测试。

## 6. 其他改动

| 文件 | 改动 |
|---|---|
| `.gitignore` | 忽略运行产物（`runs/`、`experience/`、`artifacts/`、`sessions.db` 等），但仍保留 `skills/paper-navigator/SKILL.md` 可跟踪 |
| `EvoScientist/llm/registry.py` | 为 `custom-openai` 增加 `gpt-5.6-terra` 与 `gpt-5.6-luna` 模型条目 |

## 7. 最终按职责划分

| 目标 | 对应组件 |
|---|---|
| 找论文、评分、排序 | `paper-navigator` |
| 从指定论文/主题抽取新经验 | `paper-experience` + `extract_paper_experiences[_batch]` |
| 将本次抽取的经验长期保存 | `promote_to_memory.py` |
| 将已有 JSONL 经验库批量导入 | `import_experience_bank.py` |
| 用库中已有经验回答问题 | `experience-answer` |
| 检验“经验摘要”与“论文原文”对方案设计的差异 | `solution-ab` |
| 让批量入库经验在图中形成可导航关联 | `link_observations_by_rule.py` |

## 8. 使用时需要知道的边界

- `paper-experience` 的“沉淀到长期记忆”是可选动作，只有用户明确要求时才应执行；普通抽取只写会话缓存。
- `experience-answer` 当前依赖 TF-IDF 关键词检索，不具备语义检索兜底；查询词和人工相关性筛选很关键。
- `solution-ab` 是控制变量对比工具，输出差异而不下结论。
- 主题入口的轻量检索不是系统综述或严格选文；需要完整评分和排序时应使用 `paper-navigator`。
