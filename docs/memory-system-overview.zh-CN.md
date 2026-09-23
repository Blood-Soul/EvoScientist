# 记忆系统改造总览：论文经验、原文 RAG 与复用策略

> 面向汇报，也面向下次接着改这块代码的自己。写的是**最终状态**——现在存什么、
> 怎么存、怎么检索、怎么被 agent 用——不是按提交顺序复述怎么改的。每节末尾指向
> 对应的详细设计文档，要看某个决策的来龙去脉去那里查。

## 0. 一句话

基线 agent 只有一种记忆：操作记忆（`O-*`，记录 agent 自己的工具和环境怎么表现），
入口是 `search_observations` / `read_memory`。本次改造新增两条能力，并各自独立
成库：

1. **论文经验**（`E-*`）：把论文读到的方法、结论、评测协议抽成结构化记录。
2. **论文原文**（`C-*`）：把下载到手的论文全文持久化、切块，供检索取证。

以及一层**复用转换**：agent 要拿经验去做决策时，不直接读原始记录（会照抄论文
自己的取值），而是先把经验改写成绑定当前任务的策略。这层转换现在由系统在**每一步
主动判断并推送**（coach），不再等 agent 自己想起来调工具。

## 1. 全貌

```text
论文 URL/ID
    │
    │ download_paper_text()  ── 唯一网络调用（Jina Reader，已去 references）
    ▼
全文 Markdown
    │
    ├──► persist_paper_fulltext()  【原文库，先落盘】
    │        └─ paper.md + chunks.jsonl + metadata.json
    │
    └──► L1/L2 并行抽取（prompt: l1_extract.md / l2_inductive.md）
             └─ 校验归一化 → 【经验库】l1.json + l2.json + metadata.json

agent 运行时，三个库各有各的检索入口：
    search_observations  ──► O-*（agent 自己的操作记忆）
    search_experience / list_experience  ──► E-*（论文经验，定位记录）
    search_paper_text / read_paper  ──► C-*（论文原文分块）
    read_memory  ──► 按 ID 统一读 O-*/E-*

复用转换有两条方向相反的入口，共用同一条流水线和同一份缓存：
    push（主 agent）：每一步 coach middleware 判断 → 推送指导（见第 6 节）
    pull（子 agent）：apply_experience 工具，被调用时才运行
```

触发经验抽取有两条路径，走的是同一套下载/抽取/存储代码：

| 路径 | 触发方式 | 特点 |
| --- | --- | --- |
| 主动（前台） | 用户明确要求提取经验，`paper-experience` skill 调 `extract_paper_experiences` | 当前请求内直接返回 |
| 被动（后台） | `paper-navigator` 定稿论文集后调 `enqueue_paper_experiences` | 入队，不阻塞检索，后台 worker 异步消费 |

详见 [论文经验抽取与利用](paper-experience-report.zh-CN.md)、
[论文原文 RAG 链路方案](rag-plan.zh-CN.md)。

## 2. 三个库分别存什么、存在哪

```text
<MEMORIES_DIR>/
├── observations/projects/<project_id>/...        O-*（agent 自己的操作记忆，原有）
├── experiences/projects/<project_id>/<paper_key>/  E-*（论文经验，新增）
│   ├── l1.json         实践经验数组
│   ├── l2.json         归纳经验数组
│   └── metadata.json   论文标题/URL/哈希/学科/支持与反驳论文列表
├── papers/projects/<project_id>/<paper_key>/       C-*（论文原文，新增）
│   ├── paper.md         去 references 后的完整 Markdown
│   ├── chunks.jsonl      一行一个 chunk：id/section/字符区间/文本
│   └── metadata.json     字符数/chunk 数/切分参数/切分版本
├── policies/projects/<project_id>/                 策略缓存（新增，见第 5 节）
│   └── policy-<hash>.json
└── profile/projects/<project_id>/
    └── PAPER_EXPERIENCES.md   由经验库派生的只读目录，供 WebUI 浏览
```

**`<paper_key>` 是两个库共用的同一个函数** `paper_storage_key(paper_id, url)`
（`slug[:48]-sha256(canonical)[:12]`），所以同一篇论文的经验目录和原文目录同名——
一次 `stat` 就能从一条 `E-*` 记录推出它那篇论文的原文在哪。

ID 构造方式：
- `E-*` = `SHA256(project:paper:level:identity)[:16]`，`identity` 由
  `statement`/`declaration`/`domain`/`task` 拼接。**内容寻址**，不含数组下标——
  重新抽取导致顺序变化不会让同一个 ID 指向另一条发现。没有可用文本时才退回下标。
- `C-*` = `SHA256(project:paper_key:chunk_index)[:16]`，构造方式与 `E-*` 同构。

所有写操作走原子写（临时文件 + `os.replace`），写一半崩溃不会留下坏文件。

## 3. 经验记录的字段（`E-*`）

模型只输出内容字段，生命周期字段（`id`/`layer`/`confidence`/`utility` 等）由
运行时注入，避免模型伪造来源或 ID。

**L1（实践经验）**：记录"在什么环境下做了什么、得到什么反馈"。

| 字段 | 说明 |
| --- | --- |
| `domain` / `task` | 研究领域 / 具体任务 |
| `statement` | 自洽陈述，≥350 词，含问题/步骤/条件/结果/边界 |
| `applicable_when` / `not_applicable_when` | 适用前提 / 排除边界 |
| `scope` | 模态、规模、模型、流水线阶段 |
| `action` / `effect` | 做法 / 观测效果（尽量带数字） |
| `practice_trace` | 3–6 个 `{action, feedback}` 对 |
| `evidence` | `{section, quote}`，quote 逐字且 ≥150 字符 |

**L2（归纳经验）**：把单篇论文的结果提升为可迁移的规律。字段基本同上，但用
`claim_type`（`property`/`relation`/`trend`/`conditional`）替代 `practice_trace`，
并加 `rationale` / `rationale_depth`（`deep`/`shallow`/null）。

**两个库共享的三个可选字段**（第一批 ~100 条记录抽取时还不存在，读取时可选）：

| 字段 | 用途 |
| --- | --- |
| `discipline` | 粗粒度学科（`cs`/`math`/`physics`/`chem`/`bio`/`med`/`materials`/`earth`/`econ`/`eng`/`other`）。模型从论文内容判断，判断不了则省略，由运行时从 `domain_arxiv` 推导，仍推不出落 `other` |
| `transferable_core` | ≤60 词，剥掉论文专有取值后剩下的因果内核。是复用策略重排阶段的描述符 |
| `bindings` | `[{name, kind}]`，论文写死的具体取值（`kind` ∈ dataset/model/scale/hyperparam/baseline/metric/toolchain/other）。`name` 必须是**具名实体**，不能是测量值（如 `91%`），否则子串匹配会在任何含这些字符的文本上误命中 |

`discipline` 是精确匹配的检索过滤字段，写入前会校验落在词表内，词表外的值让**该
条记录抽取失败**，而不是静默写入一个查不到的学科。

运行时补充字段：`id`（`E-*`）、`layer`、`paper_id`/`source_id`、`domain_arxiv`、
`confidence`（基于证据完整度估的单篇初始置信度，上限 0.85）、`utility`（预留，
当前恒为 `null`，未接入任何反馈）。

详见 [论文经验抽取与利用](paper-experience-report.zh-CN.md) 第 5 节、
[经验复用层](experience-policy.zh-CN.md) 第 2.2 节。

## 4. 论文原文怎么存、怎么切（`C-*`）

**动机**：实验发现"只给 agent 看经验，效果不如直接看论文原文"——经验是压缩后的
判断，原文是可核验的证据和细节。原链路下载完全文抽完经验就丢弃，只留一个哈希。
现在把已经下载到手的全文顺手持久化，零新增网络调用。

**切分算法**（`memory/papers/chunking.py`，section-aware）：
1. 正则扫 Markdown ATX 标题（`^#{1,6}\s+...`），按标题切出 section，记录层级
   路径（`section_path`），论文标题作为所有路径的顶层前缀。
2. `≤2000` 字符的 section 独立成一个 chunk；超长的按段落边界二次切分，带
   `200` 字符重叠；`<200` 字符的 section 与下一个合并，避免标题独立成 chunk。
3. 无标题的退化输入（Jina 偶尔返回纯文本）走纯字符窗口切分。
4. 输入已经过 `_strip_references_section()` 去掉参考文献——这是经验抽取原本就
   做的一步，全文持久化直接复用这个返回值，天然不含引用列表的噪声。

`chunk_id` 与 `char_start`/`char_end`（相对 `paper.md` 的字符偏移）一起落盘，
所以 `read_paper` 能直接按区间取文本，不必重新切分。

详见 [论文原文 RAG 链路方案](rag-plan.zh-CN.md) 第 5 节。

## 5. 检索入口：三个库、三套词汇、五个工具

**核心原则：一个入口只能有一套词汇，问题形状不同就必须拆开入口。** 三个库曾经
共用一个检索器（经验一度复用 `search_observations`），后果是：

- 经验记录硬编码存为 `SEMANTIC`/`PROJECT`，调用方按 observation 的习惯传
  `memory_type=procedural` 或 `scope=global`，会**静默清空整个经验结果集**，
  没有任何信号提示发生了这件事；
- observation 那套"该怎么做"的过程性词汇变成了经验检索的词汇，而 `E-*` 记录
  只有论文自己的**内容性**发现，过程性问法在这个库里注定没有答案。

现在每个库自己教自己的问法：

| 工具 | 检索对象 | 面向的问题 | 参数 |
| --- | --- | --- | --- |
| `search_observations` | `O-*` | 过程性："这件事该怎么做、失败了怎么办" | `memory_type`/`scope`/关键词 |
| `search_experience` | `E-*` | 内容性：关于某个**主题**，论文发现了什么 | `topic`/`method`/`task`（主题面）+ `discipline`/`domain`/`level`（精确过滤）。**不接受** `memory_type`/`scope` |
| `list_experience` | `E-*` | 不知道库里有什么、不知道怎么措辞 | `facet=discipline\|domain\|records`，逐层过滤+分页 |
| `search_paper_text` | `C-*` | 要核验原文措辞、具体数字、超出经验粒度的细节 | `query`/`mode`（ranked\|regex）/`paper_id`/`limit` |
| `read_paper` | `C-*` | 展开一个定位结果 | `chunk_id`+`expand=chunk\|section\|full` 或 `paper_id`+`expand=full` |
| `apply_experience` | `E-*` | 要**做决策**（见第 6 节）。只授予子 agent；主 agent 走 coach 推送，不持有此工具 | `task`/`state`/`max_selected`/`refresh` |
| `read_memory` | `O-*`/`E-*` | 按稳定 ID 读完整记录 | ID 的 `O-`/`E-` 前缀本身就是无歧义路由 |

`search_experience` 的三个主题面（`topic`/`method`/`task`）**分别检索再用 RRF
融合**（`RRF_K=60`），不是拼成一个长 query。拼接会让一个低信息量的 token 匹配
所有文档、"排序"退化成目录顺序，却仍然返回一整页看起来自信的结果；分别检索再
融合意味着一个文档必须在某个面上排得靠前才能出现。

`list_experience` 存在的理由是**检索是纯词法的**——TF-IDF，无 embedding。调用方
不知道库的说法就写不出能命中的 query，尤其是开放式问题往往没有唯一正确的问法。
按结构浏览（discipline → domain → records）绕开措辞问题，每层都分页并报总数，
库长到多大都能走完，不会被静默截断。

`search_paper_text` + `read_paper` 是两级检索：先返回 chunk 定位与片段（cheap），
再按需展开到整节或整篇（`expand=section` 是默认，`expand=full` 必须带
`paper_id`）。工具描述明确写了"检索是词法的，第一次返回太少就换 1-3 个措辞重试"，
这是对纯词法方案召回弱的直接补偿。

**结构性隔离**：`search_paper_text` 是 chunk 唯一的检索入口，chunk **永远不进**
经验或操作记忆的排序——否则一篇论文几十个 chunk 会淹没高信号的 `E-*`/`O-*` 记录。
有专门测试守着这条边界（`test_chunks_never_appear_in_experience_search`）。

**非拉丁语系查询的诚实边界**：ASCII query 分词是 `[a-z0-9_]+`；含 CJK 时走
**二元组**切分（主线 #485 加的），所以 `如何利用摘要完成idea的构建` 现在切出 10 个
二元组而不是只剩 `idea`。`degenerate_facets()` 按**字符**覆盖率检测残留的退化情况，
低于 `0.4` 就在响应里加 `query_warning`，写明"实际只搜了什么"，并指向
`list_experience`。它**不做翻译或查询扩展**，只报告事实。二元组是字面匹配，不是语义
匹配，中文召回的真实质量还没量过。

### system prompt 里的索引块

三个独立的字符预算块（55%/15%/30% 分配），互不挤占：

- `<observation_memory>`：逐条列出 `O-*`（agent 自己写的，量小）；
- `<paper_experience_memory>`：**只报统计**（论文数/记录数/学科分布/热门
  domain），从不逐条列——这样库从 98 条长到 10 万条，这块大小不变；
- `<paper_fulltext_memory>`：**论文级、一行一篇**（绝不是一行一个 chunk），
  列标题和 `paper_id`。

memory worker（只写 observation）不注入后两块的说明——它拿不到那两组工具，
描述一个它跑不了的流程只会干扰。

详见 [经验检索独立化改造](experience-retrieval-split.zh-CN.md) 第 2、5 节、
[论文原文 RAG 链路方案](rag-plan.zh-CN.md) 第 6 节。

## 6. 经验怎么被"用"：从照抄到重新推导

**问题**：直接把 `read_memory` 读到的 `E-*` 记录塞给 agent，agent 会照抄论文里
的具体取值。实测案例：任务是"在自有 4 万条医患对话上微调 Llama-3-8B"，检索到一
条讲 ImageNet + ResNet-50 余弦退火的经验，agent 直接在计划里写"在 ImageNet 上训
90 epoch"。根因不在检索，在**注入形态**——一条记录的 `statement` 是"某论文作者
在他们的数据集、模型上得到的数字"，可迁移的方法论和已失效的具体取值混在同一段
散文里，agent 分不清，于是全抄。这与 QCR 论文的 Full Trajectory 条件一致，其
stale-binding 错误率实测 46.9%。

**方案**：在"检索到记录"和"agent 使用记录"之间插入一层转换。

```
任务 ──► 检索(TF-IDF) ──► 重排(辅助LLM) ──► 合成(辅助LLM) ──► 策略 ──► 缓存
          8 条候选         选 3-5 条         写成结构化对象      主LLM 据此作答
```

agent 不再直接读原始记录，读的是**针对当前任务重写过的策略**；原始记录仍可通过
`read_memory` 按 ID 调阅，用于审计某条结论的证据来源。

**策略对象的九个字段**：

| 字段 | 含义 |
| --- | --- |
| `verdict` | `adopt`/`adapt`/`decline`。**`decline` 是合法且有用的答案**——"存的经验不适用" |
| `procedure` | 仍可迁移的步骤，不含源论文具体取值 |
| `rebind` | 必须重新推导的取值：`name`/`kind`/`why_bound`/`how_to_obtain`/`source_value`。**唯一故意携带源取值的字段**，`source_value` 只作出处标注，绝不作答案 |
| `preconditions` | 该策略成立的前提 |
| `declines` | 明确不迁移的部分 |
| `checks` | 收尾前要验证的项 |
| `conflicts` | 多条记录之间的分歧（不悄悄挑一条，写清分歧和判别条件） |
| `unsupported` | 记忆完全没覆盖的部分，提示转去实时检索 |
| `sources` | 每行结论可回溯到具体 `E-*` 记录 |

区别"正确复用"和"要修的 bug"：**"该调度在 ImageNet 上验证过，你的任务换成
CIFAR-10"** 是正确复用；**"在 ImageNet 上训练"** 是没重绑定，是 bug。

### 6.1 push 与 pull：谁决定"现在该复用经验"

上面那条流水线解决的是"经验以什么形态进上下文"。还剩一个问题：**谁来决定这条
流水线什么时候跑**。

`apply_experience` 是 **pull**：agent 自己要想起有这个工具、自己判断当下需不需要
经验、自己把 query 写出来。三处都会漏：

1. query 的上限就是 agent 恰好写出的那一句话，而它用的是**自己处境的词汇**，不是
   论文摘要的词汇——纯词法检索下这直接决定命中与否；
2. 这个工具不在 always-include 名单里，工具数超过阈值时 tool selector 可以把它整个
   过滤掉，复用层于是**静默停止运行**，没有任何信号；
3. 最根本的一条：agent 常常意识不到自己正在做一个决策。

**coach** 是反方向：系统在每一步替 agent 问这两个问题。

```
每个 model 调用前
    │
    ├─ 代码层短路（零模型调用）：库为空 / 上一步只是纯读工具 / facets 与上次相同
    │
    ├─ 一次 gate 调用（辅助模型）：need? + topic/method/task/state
    │     need=false ──► 什么都不注入，直接继续（这是多数情况下的正常答案）
    │
    └─ need=true ──► derive_policy(同一条流水线) ──► 渲染成指导文本
                          └─ 作为一条临时 HumanMessage 追加到 messages 末尾
                             （仅本次调用，不落 state）
```

三件事是刻意这么设计的：

- **流水线一行没改。** gate 只产出检索面和任务串，检索/重排/合成/缓存仍是
  `derive_policy`。`apply_experience` 仍注册在 tool registry 里供子 agent 调用，
  所以 push 和 pull 共享一份实现、一份缓存。
- **注入是 per-call 的，绝不落 state。** 只有返回的 `ModelResponse` 会写回 state，
  所以下一步看到的轨迹与本步**逐字节相同**，prompt cache 不失效。任何"把指导写进
  历史"或"把旧指导压成一行"的方案都会改写历史，使编辑点之后的全部 cache token 失效，
  而且是每步都失效。
- **多数步骤必须零成本。** 三个短路都在代码层，不花模型调用。单步最坏成本：短路 0、
  gate 说不要 1、重排选不出 2、完整命中且未命中策略缓存 3。

gate 产出的 `method` 面还顺手修了一个旧缺陷：RRF 融合在只有一个面时会短路，退化成
目录顺序。pull 路径只有 agent 写的那一句话（只能填 `topic`），gate 则同时给出
`topic` 和 `method`——**"改写 query"本身就是 coach 的一部分收益**。

注入的是**指导**，不是原始记录：策略本身已经是绑定目标任务的重写，把它渲染成文本
不多花任何模型调用，也没有信息损失。原始记录在这条路径上从不进入主 agent 上下文，
需要审计时用 `read_memory` 按 ID 调阅。

### 6.2 两条路径互斥，说明也必须互斥

一个 agent 只能被告知一条路径：

| agent | 持有 `apply_experience` | 被 coach 推送 | 注入的说明块 |
| --- | --- | --- | --- |
| 主 agent | 否 | 是 | `EXPERIENCE_COACH_INSTRUCTIONS`（指导会自己来、各段怎么读、它是建议而非命令） |
| `research-agent` / `planner-agent` | 是（YAML 授权） | 否 | `EXPERIENCE_POLICY_INSTRUCTIONS`（逐字段教怎么调工具） |
| memory worker | 否 | 否 | 两块都不注入 |

说明互斥不是洁癖：给被 coach 的 agent 注入工具教程，等于让它去调一个不存在的工具；
给持有工具的子 agent 注入 push 说明，等于承诺一批永远不会到达的指导。子 agent 不上
coach 是因为**每个子 agent 内部再跑一遍逐步 gate，会把辅助模型调用数乘上扇出宽度**，
而子 agent 是带着明确任务被派出去的，pull 在那里恰好够用。

详见 [经验复用层](experience-policy.zh-CN.md) 第 1、2、11 节。

## 7. 模型分工、成本与降级

**模型分工**：中间过程（coach 的 gate、重排、合成）都走辅助模型，不产出面向用户的
散文；策略是一个结构化对象，acting agent 读完它，再由**主模型**写出面向用户的回答。

**成本控制**：
- 两级筛选——重排阶段只读约 200–300 字符的紧凑描述符，完整 2500 字符
  `statement` 推迟到合成阶段才读，8 条候选能塞进一次小调用；
- 缓存，键为 `SHA256(task, sorted(选中的 E-* ID))[:16]`；任务措辞改变会产生新键
  并重新合成——因为写手是逐字读任务的；
- 按需调用——`apply_experience` 不并入 `search_experience`，前者转换记录、后者
  定位记录，工具描述要求"真正做决策时才调用"。
- coach 侧多出一次 gate 调用，用三个代码层短路和"facets 未变则复用上次指导"把它
  压回去；`need=false` 被明确写成一等答案，不是失败。

**辅助模型必须真的配一个便宜的**：`auxiliary_model` / `auxiliary_provider` 默认为
空字符串，即**回落到主模型**。没配的话 gate 和重排/合成都跑在主模型上，coach 的
成本论证不成立。

**降级行为**：复用是对实时检索的增强，不是前置条件，任何一环失败都不终止调用方
的回合。

| 情况 | 行为 |
| --- | --- |
| 检索无候选 | `status="no_candidates"`，附提示转实时检索，正常结果不是错误 |
| 重排 JSON 解析失败 | 回退到检索打分前 N 条，记录降级原因 |
| 重排选不出任何记录 | `status="no_reusable_memory"` |
| 合成输出无法解析 | 抛 `PolicyOutputError`，工具层捕获返回 `status="error"` + 提示 |
| 缓存写失败 | 记 warning，不影响本次返回 |
| gate 输出无法解析 | 判为 `need=false`（**关闭**），原因记入 trace。绝不能以"看不懂"为由开启后面两次调用 |
| gate 要检索但没给 `topic` | 降级为 `need=false`——空 topic 的词法检索返回的是目录顺序，不是召回 |
| gate 调用或 `derive_policy` 抛异常 | coach 静默跳过，注入空串，agent 回合照常进行 |
| 策略渲染为空（`no_candidates` / 无缺口的 `decline`） | 什么都不注入，比告诉 agent"记忆没意见"更省也更不误导 |

原文全文持久化同样"永不抛异常"：开关关闭、写盘失败都降级为"这篇没有全文"，不
让抽取任务失败——**全文是经验抽取的补充，不是它的前提**。

## 8. 配置

| 配置项 | 默认 | 环境变量 |
| --- | --- | --- |
| `memory_experience_policy_enabled` | `true` | `EVOSCIENTIST_MEMORY_EXPERIENCE_POLICY_ENABLED` |
| `memory_experience_policy_max_selected` | `4`（钳制 [1,6]） | `EVOSCIENTIST_MEMORY_EXPERIENCE_POLICY_MAX_SELECTED` |
| `memory_experience_coach_enabled` | `true` | `EVOSCIENTIST_MEMORY_EXPERIENCE_COACH_ENABLED` |
| `memory_experience_coach_recent_messages` | `6`（钳制 [1,30]） | `EVOSCIENTIST_MEMORY_EXPERIENCE_COACH_RECENT_MESSAGES` |
| `memory_paper_fulltext_enabled` | `true` | `EVOSCIENTIST_MEMORY_PAPER_FULLTEXT_ENABLED` |
| `memory_paper_chunk_max_chars` | `2000` | `EVOSCIENTIST_MEMORY_PAPER_CHUNK_MAX_CHARS` |
| `memory_paper_chunk_overlap_chars` | `200` | `EVOSCIENTIST_MEMORY_PAPER_CHUNK_OVERLAP_CHARS` |

关掉 `memory_experience_policy_enabled` 即回到原有行为：agent 重新直接读 `E-*`
记录；磁盘上已缓存的策略不动。开关必须同时管住**工具注册**和**说明注入**（
`middleware/memory.py`），否则关掉后 agent 会被告知去调一个不存在的工具。

关掉 `memory_experience_coach_enabled` 则回到纯 pull：coach middleware 不进栈，
`apply_experience` **重新出现在主 agent 的 base_tools 里**，说明块也换回工具教程。
tool registry 里的那份注册**始终保留**——子 agent 的 YAML `tools:` 列表是按名字对
registry 解析的。这三处（middleware、base_tools、说明块）必须同向，测试
`TestRouteExclusivity` / `TestAgentWiring` 守着这条一致性。

`memory_experience_coach_recent_messages` 是 gate 读取的轨迹尾部长度。钳制下界是
1——`0` 会让 gate 看不到这次运行做过什么；上界是 30——无上界等于每步把整条轨迹
送上辅助模型，正好抵消掉用辅助模型的理由。

工具授权：`apply_experience`/`search_experience`/`list_experience`/
`search_paper_text`/`read_paper` 只授予 `research-agent` 和 `planner-agent`
（YAML 里的 `tools:` 列表）；memory worker 恒为不启用（它只写观察）。主 agent 默认
只在 coach 关闭时才持有 `apply_experience`。

## 9. 代码地图

```text
EvoScientist/memory/
  experiences/
    extraction.py    L1/L2 下载全文 + 提示词调用 + 校验归一化
    store.py         经验落盘/内容寻址 ID/PAPER_EXPERIENCES.md 派生
    retrieval.py      search_experience_records + list + degenerate_facets
    taxonomy.py       discipline 词表与解析
    queue.py          被动入队队列
  papers/
    chunking.py        section-aware 切分算法
    store.py           原子写、目录发现、chunk 索引加载
    persist.py          persist_paper_fulltext() 入口与配置门禁
    retrieval.py         search_paper_chunks，唯一检索入口
  policy/
    schema.py           策略对象校验；VERDICTS、BINDING_KINDS
    select.py            gather_candidates（检索）+ rerank_candidates（重排）
    synthesize.py         synthesize_policy（合成）+ 容错解析
    store.py               策略缓存
    pipeline.py            derive_policy：流水线唯一对外入口
    gate.py                 push 侧的 gate：need? + 三个检索面（decide_experience_need）
    suggest.py               render_suggestion：把策略渲染成注入用的指导文本
    trace.py                  开发期调试日志
  observations/index.py      三块字符预算的 system prompt 索引拼装
  agents/paper_experience_worker.py   后台 worker（LangGraph graph）
EvoScientist/tools/
  experience_search.py    search_experience / list_experience
  experience_policy.py     apply_experience
  paper_rag.py               search_paper_text / read_paper
  paper_experience_active.py  extract_paper_experiences（前台）
  paper_experience_queue.py    enqueue_paper_experiences（后台入队）
EvoScientist/middleware/coach.py   ExperienceCoachMiddleware：逐步判断 + per-call 注入
EvoScientist/langgraph_dev/paper_inspector.py   /debug/papers WebUI 页面
prompt/
  l1_extract.md / l2_inductive.md   抽取提示词
  policy_rerank.md / policy_write.md   复用层的重排/合成提示词
  policy_gate.md                        coach 的 gate 提示词
scripts/
  backfill_experience_fields.py    存量记录补齐 transferable_core/bindings
  policy_ab.py                       复用层 A/B 验证（独立，未接入 CI）
  policy_trace_view.py                调试日志查看器（独立，临时工具）
```

## 10. 遗留（写这份文档时仍未解决）

- **A/B 数字还不能引用**：样本量不足（12 格 7–8 格死于 `APIConnectionError`），
  且"无记忆"地板线被 `AdamW`/`2e-5`/`0.1` 这类领域通用默认值污染，读不出 0。
  方向上验证过（散文条件 100% stale、策略条件 0% stale），但不是可引用的测量结果。
- `utility` 字段完全未接入：策略被采纳后的效果不会回写到经验置信度或后续排序。
- 策略缓存不按记录内容失效：同一批 `E-*` ID 更新后（如置信度聚合改变），缓存键
  不变，需要 `refresh=true` 手动重合成。
- 11 条记录仍停留在 `discipline=other`（来自 `domain_arxiv=null` 的 4 篇论文）；
  模型侧 `discipline` 字段只在未来抽取时生效，存量需重抽或专门 backfill。
- 词法检索的召回弱是刻意接受的取舍，没有被消除：若要上向量检索，
  `memory/papers/retrieval.py` 和 `memory/experiences/retrieval.py` 是各自唯一的改动点。
- 中文查询的现状需要重新量一次：主线 #485（2026-09-17）给 `_tokens()` 加了 CJK 二元组
  分词，中文 query 现在有字面二元组召回，不再只剩恰好含有的拉丁片段。因此第 5 节"只能靠
  `list_experience` 绕"这句结论已经过时，而
  `test_a_query_lost_to_the_tokenizer_is_reported_not_scored_silently` 还钉着旧行为，
  在干净树上就是红的（要换示例，不是修检测器）。
- **coach 的 gate 命中率没有实测**：现在是"每步都触发"的最朴素策略，`need=true` 的
  比例、其中真正产出非空指导的比例、以及 gate 是否有系统性偏"要"，都只有 trace 里的
  `coach_skip`/`coach_reuse`/`coach_inject` 事件可查，没有汇总统计。
- prompt cache 命中情况仍不可见：trace 没有记 `cache_read_input_tokens` /
  `cache_creation_input_tokens`，"注入不落 state 所以 cache 不失效"目前是代码层面的
  论证（只有 `ModelResponse` 写回 state），不是测出来的数字。

详见三份分文档各自的"遗留"章节：
[论文原文 RAG 方案](rag-plan.zh-CN.md) 第 9 节、
[经验复用层](experience-policy.zh-CN.md) 第 12 节、
[经验检索独立化改造](experience-retrieval-split.zh-CN.md) 第 11 节。
