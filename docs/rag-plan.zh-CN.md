# 论文原文 RAG 链路方案（已实现）

> 本文档记录**已落地**的实现，不是待办计划。若需要在此基础上继续开发，先读第 2、3、9 节：
> 第 2 节给出全貌，第 3 节是逐模块的代码地图，第 9 节列出设计细化、已知取舍与扩展点。

## 1. 背景与目标

项目已具备"抽取经验并利用"的能力：`paper-experience` skill 做前台主动抽取，
`paper-navigator` 定稿论文集后被动入队、后台 worker 抽取 L1/L2 经验，经验以
`E-*` ID 进入 EvoMemory 统一检索。

实验发现的问题：**只给 agent 看经验，效果不如直接给它看论文原文**。经验是压缩后的
判断与结论，原文是可核验的证据与细节。原链路把原文丢掉了 —— worker 在
`download_paper_text()` 里下载完整全文，抽完 L1/L2 就丢弃，只在 `metadata.json` 里
留了一个 `paper_sha256`。

本方案把已经下载到手的全文持久化、切分、建立可检索索引，并以独立工具暴露给 agent，
让 agent 在"经验给判断"之外还能"回原文取证"。

### 落地前的现状缺口

| 项 | 落地前 | 现在 |
|---|---|---|
| 全文持久化 | worker 路径完全不落盘 | `papers/projects/<project_id>/<paper_key>/` 项目隔离落盘 |
| 文本切分 | 全仓库无任何 chunk/splitter 逻辑 | `memory/papers/chunking.py`，section-aware |
| embedding | `llm/` 层只有 chat completions，无向量库依赖 | 仍然没有，且**刻意不引入**（见 1.1 决策 1） |
| 检索 | `memory/search.py` 纯 Python TF-IDF 排序器 | 原样复用，未改动其排序逻辑 |
| 可观测性 | 无 | `/debug/papers` 网页（见第 7 节） |

### 1.1 五条既定设计决策

这五条是实现的约束边界，代码里的多处注释都指回它们：

1. **检索方式：纯词法。** 复用 `memory/search.py` 的排序器。零新依赖、零 API 成本、
   与经验检索同一套排序逻辑。代价见第 9 节。
2. **工具形态：新增独立工具** `search_paper_text` + `read_paper`，**不混进**
   `search_observations`。避免上千个 chunk 在同一个排序里淹没高信号的 `E-*` 记录。
3. **入库范围：复用现有经验队列**，在抽取前顺手落盘，**零新增网络调用**。
4. **返回粒度：两级。** 先返回 chunk 定位与片段，再按需 `read_paper` 展开到整节或整篇。
5. **工具授权范围：只有** `research-agent` 与 `planner-agent` 拿到这两个工具。

---

## 2. 全貌：一篇论文的生命周期

```text
paper-navigator 定稿 / paper-experience 前台抽取
        │
        │  download_paper_text()   ← 唯一一次网络调用，原本就存在
        ▼
   全文 Markdown（已去 references）
        │
        ├──────────────► persist_paper_fulltext()   【新增，抽取之前】
        │                   ├── paper.md
        │                   ├── chunks.jsonl   C-* chunk 索引
        │                   └── metadata.json
        │
        └──────────────► L1/L2 经验抽取（原有链路，未改动）
                            └── E-* 进 EvoMemory 统一检索

agent 运行时：
   <paper_fulltext_memory> 索引块（论文级，一行一篇）注入 system prompt
        │
        ├── search_paper_text(query)  → 若干 C-* 定位 + 片段
        └── read_paper(chunk_id, expand=section|chunk|full) → 整节 / 整篇
```

**关键设计：存储对称性。** `papers/` 与 `experiences/` 用同一个 `paper_storage_key()`
命名，所以同一篇论文的经验目录与原文目录**同名**。一条 `E-*` 记录只需一次 `stat`
就能推出自己那篇论文的原文路径。这把"经验与原文互补"从一句 prompt 断言变成了一条
可导航的链接。

---

## 3. 代码地图：关键模块与行数

| 文件 | 行数 | 职责 |
|------|------|------|
| `memory/papers/chunking.py` | 245 | section-aware 切分算法，唯一的新算法 |
| `memory/papers/store.py` | 520 | 原子写、目录发现、chunk 索引加载 |
| `memory/papers/persist.py` | 100 | `persist_paper_fulltext()` 入口与配置门禁 |
| `memory/papers/retrieval.py` | 45 | 包一层 TF-IDF ranker，隔离 chunk 搜索 |
| `memory/papers/__init__.py` | 63 | 暴露对外 API（见下） |
| `memory/papers/` | **973** | **以上五个文件总和** |
| `tools/paper_rag.py` | 304 | 两个 LangChain 工具 + 一段 prompt 指令 |
| `langgraph_dev/paper_inspector.py` | 278 | WebUI debug 页面（见第 7 节） |
| `tests/test_paper_rag.py` | 1026 | 45 个测试，全通过 |
| `tests/test_paper_inspector.py` | 155 | 9 个测试，全通过 |

`memory/papers/` 包暴露的公开 API（`__init__.py:__all__`）：

- **store** — `store_paper_text`, `has_paper_text`, `read_paper_chunk`, `list_papers`, 
  `list_paper_projects`, `list_paper_chunks`（后两个供 inspector 用）
- **retrieval** — `search_paper_chunks`
- **persist** — `persist_paper_fulltext`, `paper_fulltext_settings`

与 `experiences/` 包的对称性：`store_paper_experiences` ↔ `store_paper_text`，
`paper_storage_key` 两边复用。

---

## 4. 存储布局

```text
<MEMORIES_DIR>/
├── experiences/projects/<project_id>/<paper_key>/
│   ├── l1.json
│   ├── l2.json
│   └── metadata.json
└── papers/projects/<project_id>/<paper_key>/         # 新增
    ├── paper.md          # 去 references 后的完整 Markdown 全文
    ├── chunks.jsonl      # 一行一个 chunk，含 id/section/char 区间/text
    └── metadata.json     # paper_id/url/title/sha256/chunk 数/切分版本
```

`<paper_key>` 就是 `paper_storage_key(paper_id, url)` 返回的
`<slug[:48]>-<sha256(canonical)[:12]>`，与经验库同键。

chunk ID 也用稳定哈希：`C-<sha256(f"{project_id}:{paper_key}:{chunk_index}")[:16]>`，
和 `E-*` ID 构造方式同构。`chunk_index` 是从 0 起的序号，chunk 按在原文中的出现顺序编号。

`metadata.json` 里存的字段（落盘时写入的完整集合）：
- `store_version` / `chunking_version` — 格式版本，当前都是 `1`
- `project_id`, `paper_key`, `paper_id`, `canonical_paper_id`
- `title`, `url`, `domain_arxiv`（arXiv 域时自动填）
- `paper_sha256` — 原文的 SHA256，去 references 之后、切分之前计算
- `char_count`, `chunk_count`, `section_count`
- `max_chunk_chars`, `overlap_chars` — 本次切分用的参数（可能和当前配置不同）

`chunks.jsonl` 里每一行是一个 chunk 字典：`chunk_id`, `chunk_index`, `section`,
`section_path`, `char_start`, `char_end`, `text`。`char_start`/`char_end` 是相对
`paper.md` 的字符偏移，所以 `read_paper(expand="chunk")` 能直接按区间取文本而不必重新切分。

---

## 5. 切分算法（`chunking.py`）

### 5.0 输入：已去参考文献的全文

落盘和切分拿到的文本**不含参考文献**，这不需要额外处理 ——
`download_paper_text()` 的最后一步就是 `_strip_references_section()`
（`experiences/extraction.py:151`），从 References 标题处直接截断。全文持久化复用的就是
这个已处理过的返回值，所以 `paper.md` 和所有 chunk 天然就没有参考文献。

这是对的：参考文献对检索只有噪声价值，几百条人名/年份/期刊名会污染 TF-IDF 的词频与 IDF
统计，让"检索到一段真正讨论方法的文字"变难。

匹配用的正则（`extraction.py:23`）要求 `references` / `bibliography` / `works cited`
**独占一行**，允许前面有 `#` 和编号，所以正文里出现这个词不会被误截。

### 5.1 算法

**目标：** 保留论文章节结构，让检索命中时能报出"Method > 3.2 Regularization"
这样的路径，比单纯字符窗口更可定位。

**算法概览：**
1. 正则扫 Markdown 标题（`^#{1,6}\s+...`），切出 sections，记录标题层级和 path。
   Jina 把 PDF 渲染成 ATX heading（`## Method`），与 L1/L2 evidence 的 `section`
   字段格式一致。**文档 h1 标题（论文题目）会作为所有 `section_path` 的最顶层前缀。**
2. 每个 section 如果 `≤ max_chunk_chars`，独立成一个 chunk。
3. 超长 section 按段落边界二次切分，带 `overlap_chars` 重叠。切分点选在"最后一个
   不超过一半的段落边界"（`_split_span` 逻辑），避免句子被截断。
4. 太小的 section（< `MIN_CHUNK_CHARS = 200`）和下一个 section 合并，避免标题独立成 chunk。
5. 无标题退化输入（Jina 偶尔返回纯文本）走纯字符窗口切分。

参数：
- `DEFAULT_MAX_CHUNK_CHARS = 2000` — 一个 chunk 理想尺寸
- `DEFAULT_OVERLAP_CHARS = 200` — 重叠窗口，防止跨界信息丢失
- `MIN_CHUNK_CHARS = 200` — 合并阈值，避免标题"标签"

输出：`PaperChunk` frozen dataclass，含 `chunk_id, chunk_index, section, 
section_path, char_start, char_end, text`。

**示例路径：**
- 论文标题是 h1：`Contrastive Pretraining for Catalysts`
- 某 h3 section：`### Regularization`
- 生成的 `section_path`：`"Contrastive Pretraining for Catalysts > Method > Regularization"`

这样检索时 TF-IDF 会给标题路径中的词更高权重（见第 6 节字段权重）。

---

## 6. 检索与工具

### 6.0 检索隔离：结构上的保证

`memory/papers/retrieval.py` 是**唯一**的检索入口（45 行，一个函数）。chunk 只从这里
被搜索，**永远不进** `experiences/retrieval.list_memory_documents()`。理由写在模块
docstring 里：一篇论文产出几十个 chunk，一个项目几十篇论文，如果混进共享排序，
原文会淹没 `search_observations` 本该返回的 `E-*` 记录。

这条约束有专门的测试守着 —— `test_chunks_never_appear_in_observation_search`
断言：同一个 query 下，经验记录仍然可被 `search_observations` 找到，且结果里
不出现任何 `C-*` / `record_kind == "paper_chunk"`；同时该 query 通过
`search_paper_chunks` 确实能拿到原文。

chunk 被投影成 `ObservationSearchDocument` 时的字段安排（决定 TF-IDF 权重）：
- `observation_id` = `chunk_id`（ranker 里 id 字段权重 ×5）
- `summary` = `section_path` + 首句摘要（权重 ×3）—— **把章节路径放进 summary
  是刻意的**，让"Method"、"Ablation"这类结构词能被查询命中
- `body` / `text` = chunk 全文（权重 ×1）
- `record_kind = "paper_chunk"` —— 隔离的标记位

`paper_id` 过滤发生在**排序之前**（`list_paper_chunk_documents` 里就筛掉），
这样 IDF 是在真正的候选集上算的，而不是先全局排序再过滤。

### 6.1 `search_paper_text(query, mode, paper_id, limit)`

第一级：定位。返回若干 chunk 的定位信息与片段，不返回整篇。

| 参数 | 默认 | 说明 |
|---|---|---|
| `query` | 必填 | 方法/指标/数据集/论断的关键词或短语 |
| `mode` | `ranked` | `ranked` 走 TF-IDF；`regex` 当 grep 用，非法模式退化为字面匹配 |
| `paper_id` | `""` | 限定单篇。取值来自索引块或上一次命中 |
| `limit` | 8 | 1–20 |

`query` 的描述里明确写了"检索是词法的，优先用论文自己会用的术语，第一次返回太少时
换 1–3 个措辞再试" —— 这是对纯词法方案召回弱的直接补偿（见第 9 节）。

### 6.2 `read_paper(chunk_id, paper_id, expand, max_chars)`

第二级：展开。

| `expand` | 行为 |
|---|---|
| `section`（默认） | 返回命中 chunk 所在的**整节**。理解一个结果的上下文，默认就够 |
| `chunk` | 只返回命中的那一段 |
| `full` | 返回整篇，**必须**配合 `paper_id` |

`max_chars = 0` 表示按粒度用默认上限；截断会在响应里明说，不静默丢内容。

### 6.3 索引块：agent 怎么知道有哪些论文

`memory/observations/index.py` 在 system prompt 里注入一个 `<paper_fulltext_memory>`
块，**论文级、一行一篇**（绝不是一行一个 chunk），列出可检索的论文标题与 `paper_id`。
它与已有的 `<paper_experience_memory>` 块共享同一个字符预算。

配套的 `PAPER_FULLTEXT_INSTRUCTIONS` 告诉 agent 两级检索的用法，以及"换措辞重试"的策略。

**注意：** memory worker **不**注入这段指令（`memory_worker.py` 里显式传
`enable_paper_fulltext=False`）—— worker 只写 observation，它拿不到这两个工具，
描述一个它跑不了的流程只会干扰。

### 6.4 工具授权

只有 `research-agent` 与 `planner-agent` 在 YAML 的 `tools:` 里列了这两个工具。
`EvoScientist.py` 里抽了一个 `_build_paper_tools()`，供 MCP 启用/未启用两条并行的
agent 构建路径共用 —— 这两条路径**必须保持同步**，抽函数就是为了防止只改一边。

---

## 7. 可观测性：`/debug/papers` 页面

**问题：** 怎么确认"入库真的发生了"？

**约束：** WebUI 前端代码在另一个包（`@evoscientist/webui`），改它意味着把那个仓库
拉进来，成本太高。

**做法：** langgraph dev server 已经通过 `langgraph.json` 的 `http` 键挂了一个自定义
Starlette app（原本用于 `/api/models`、`/api/teams`）。在同一个 app 上加两条路由即可 ——
同一个进程、同一个 origin，浏览器本来就在跟它说话，**零前端改动、零 CORS、不多占端口**。

| 路由 | 返回 |
|---|---|
| `GET /debug/papers` | 总览页：每个项目下的论文列表、字符数、chunk 数、section 数、切分参数 |
| `GET /debug/papers?project=<id>&paper=<key>` | 单篇：每个 chunk 的 id / section_path / 字符区间 / 全文（`<details>` 折叠） |
| `GET /debug/papers.json` | 同样的数据，供脚本检查。单篇视图**不含 chunk 全文**（避免几 MB 的 dump） |

启动 WebUI 时会在启动面板里打印 `Debug:` 一行，直接给出这个 URL。

实现上的三个要点：

1. **每个 handler 都用 `asyncio.to_thread` 包住文件操作。** langgraph dev 的
   `blockbuster` 中间件会拒绝事件循环上的阻塞 syscall，直接遍历目录会变成 500。
2. **所有值都过 `html.escape`。** 页面渲染的是下载来的论文文本，属于不可信输入 ——
   有测试专门断言 `<script>` 被转义而非插入。
3. **开关关闭时页面照常显示已存内容。** `memory_paper_fulltext_enabled = false`
   只停止新入库，不删已有数据，而"我关掉之前到底存进去了什么"恰恰是这个页面要回答的
   问题。所以开关状态是**显示**出来的，不用来 gate 页面。

面向开发者 debug，样式只有等宽字体和 `pre` 换行两条，不追求美观。

---

## 8. 配置

| 配置项 | 默认 | 环境变量 |
|---|---|---|
| `memory_paper_fulltext_enabled` | `true` | `EVOSCIENTIST_MEMORY_PAPER_FULLTEXT_ENABLED` |
| `memory_paper_chunk_max_chars` | `2000` | `EVOSCIENTIST_MEMORY_PAPER_CHUNK_MAX_CHARS` |
| `memory_paper_chunk_overlap_chars` | `200` | `EVOSCIENTIST_MEMORY_PAPER_CHUNK_OVERLAP_CHARS` |

两个数值配置都做了校验兜底：非 int、bool、或小于 1（overlap 是小于 0）时记一条
warning 并回落到默认值，不抛异常。理由与 `persist.py` 一致 —— **全文是经验抽取的
补充，不是它的前提**，配置写错不该让整条队列任务失败。

`persist_paper_fulltext()` 同样"永不抛异常"：开关关闭、写盘失败，都降级为
"这篇没有全文"，而不是让抽取失败。

---

## 9. 设计细化、修正与已知取舍

### 9.1 落盘顺序：先原文，后经验

`store_paper_text()` 里 `paper.md` 一定在 `chunks.jsonl` 之前落盘 —— chunk 记录里的
字符偏移只有在它索引的文本存在时才有意义。整条链路上，全文落盘也在经验抽取**之前**，
所以抽取失败不会连带丢掉已经下载到手的原文。

有测试守着：`test_full_text_survives_a_failed_extraction` 断言抽取返回
`{'processed': 0, 'failed': 1}`、没有经验文档产出，但**原文已存了一篇**。

所有写操作都走 `_atomic.py` 的原子写（唯一命名的临时兄弟文件 + `os.replace`），
不会出现半截文件。

### 9.2 缓存命中时补下载全文（对决策 3 的细化）

**问题：** 在全文持久化上线**之前**抽过的论文，`experiences/` 下有 `l1.json`/`l2.json`，
但 `papers/` 下没有 `paper.md`。这些论文以后每次都命中经验缓存，走不到抽取路径那次
下载，**全文缺口自己永远不会闭合**。

一开始我按决策 3 的字面意思（"零新增网络调用"）实现成只做一次 `stat`、不下载，把缺口
通过 `full_text_available: false` 报告出去，让调用方用 `refresh=true` 显式补。

**后来改成默认补下载**，理由是决策 3 真正要守的是**计费的 API 调用**（经验抽取的 LLM
token），而 `download_paper_text()` 走 Jina Reader，不计费；而且：

- 每篇**只补一次** —— 补完 `paper.md` 就在了，下次命中直接读
- 只影响**存量**数据，新论文都在抽取时落盘，不存在这个问题
- "有经验但看不到原文"对 agent 是个割裂状态，自动闭合体验更完整

`backfill_fulltext()` 的两个实现细节：
1. **共用抽取的并发信号量**（`ACTIVE_EXTRACTION_CONCURRENCY = 2`）。否则一批缓存命中的
   论文会把所有补下载同时发出去，不受限。
2. **下载失败只降级，不失败任务**。缓存的经验本身仍是有效结果，补不到全文就是
   `full_text_available: false`，任务照常算成功。有测试守着
   （`test_a_failed_backfill_still_returns_the_cached_experiences`）。

代价：这个分支在**前台工具** `extract_paper_experiences` 里，agent 同步等结果，所以老
论文第一次命中会多一次下载的等待。后台 worker 路径（`paper-navigator` 定稿后被动入队）
不受影响 —— 它本来就在抽取前落盘，不存在缺口。

### 9.3 修正：索引块截断的真实 bug

加 `<paper_fulltext_memory>` 块时发现，原有的截断逻辑会**超出字符预算**：截断提示语
（"还有 N 条未显示"）是在按预算挑完行**之后**才追加到 prefix 上的，所以一个刚好占满
预算的块，加上提示语就溢出了。这个 bug 在**已有的经验块上同样存在**。

修法是抽出 `_fit_block()` / `_fit_lines()`（`observations/index.py:83`、`:117`）：
把提示语放进 prefix **再**去挑行；先按不带提示语的 prefix 试一次，若确实发生了截断，
就带着提示语重新挑一次；若连提示语本身都放不下，则保留条目、丢掉提示语。
两个块现在都走这条统一路径。

### 9.4 已知取舍：纯词法检索的召回弱

这是决策 1 的直接代价，**没有被消除，只是被缓解**：

| 缓解手段 | 位置 |
|---|---|
| `section_path` 放进 `summary` 字段，吃 ×3 权重 | `store.py:list_paper_chunk_documents` |
| 工具描述明确要求"换 1–3 个措辞重试" | `paper_rag.py:SearchPaperTextArgs.query` |
| prompt 指令里同样写了重试策略 | `PAPER_FULLTEXT_INSTRUCTIONS` |
| 检索收在**单个函数**里，换 embedding 只改这一个文件 | `retrieval.py`（45 行） |

若日后要上向量检索：`retrieval.py` 是唯一的改动点，`llm/` 层需要新增 embeddings 接口
（目前只有 chat completions）。`chunking_version` / `store_version` 字段已经预留，
用于识别需要重建索引的旧数据。

### 9.5 安全说明

`/debug/*` 路由**没有鉴权**。这与同一个 app 上已有的 `/api/models`、`/api/teams`
一致，dev server 默认只绑 `127.0.0.1`，所以风险面没有变化。但绑到非 loopback 地址时，
`deploy/webui.py` 打印的 PUBLIC BIND 警告已经把 debug 路由一并点明了 ——
该页面会展示已存论文的原文。

---

## 10. 验证情况

| 项 | 结果 |
|---|---|
| `ruff check EvoScientist/ tests/` | 全部通过 |
| `tests/test_paper_rag.py` | 45 passed |
| `tests/test_paper_inspector.py` | 9 passed |
| 相关套件合计（含 observation / config / webui / langgraph dev） | 291 passed |

`tests/test_paper_rag.py`（1026 行）按七块组织：切分算法、存储、检索隔离、
`read_paper` 三种粒度、agent 工具、持久化接线、prompt 表面。

全量套件跑下来另有若干失败，已用 `git stash push -u` 在干净树上复现过**同样的**失败
（`test_backends.py` 里的 `python: not found` PATH 问题等），与本方案无关。

### 手工验证路径

1. 跑一次论文经验抽取（前台 `paper-experience` skill，或让 `paper-navigator` 定稿论文集
   触发后台队列）。
2. 启动 WebUI，打开启动面板里 `Debug:` 那一行给出的 `/debug/papers`。
3. 总览页应能看到项目、论文标题、字符数与 chunk 数；点进单篇能看到每个 chunk 的
   `section_path`、字符区间和全文。
4. 让 `research-agent` 或 `planner-agent` 就该论文提一个具体问题，观察它是否调用
   `search_paper_text` → `read_paper`。
