# 经验利用方案（当前状态）

> 面向汇报。写的是**现在系统怎么用存下来的论文经验**——谁在什么时候决定要用、经验以
> 什么形态进入上下文、花多少钱、坏了会怎样。不按改动顺序复述过程。
>
> 涉及的实现细节与验证记录见 [经验复用层](experience-policy.zh-CN.md)、
> [经验检索独立化改造](experience-retrieval-split.zh-CN.md)、
> [记忆系统总览](memory-system-overview.zh-CN.md)。

## 1. 要解决的问题

系统已经能从论文抽出结构化经验（`E-*` 记录，L1 实践 / L2 归纳），也能检索出来。问题
出在**经验被用的那一刻**：agent 拿到记录之后会照抄论文自己的取值。

实测案例：任务是"在自有 4 万条医患对话上微调 Llama-3-8B"，检索到一条讲 ImageNet +
ResNet-50 余弦退火的经验，agent 在计划里写"在 ImageNet 上训 90 epoch，学习率 0.1 退火
到 1e-5"。抄的不只是数字，是整套与当前任务无关的绑定。

根因不在检索质量，在**注入形态**。一条 `E-*` 的 `statement` 约 2500 字符，讲的是"某篇
论文的作者在他们的数据集上、用他们的模型、得到他们的数字"——可迁移的方法论和已经失效
的具体取值混在同一段散文里。Actor 分不清二者，于是全抄。

QCR（Query-Conditioned Reuse）论文把这种注入方式称为 **Full Trajectory** 条件，实测
stale-binding 错误率 **46.9%**：近一半的复用会把源任务的取值当成答案。我们观察到的行为
与该条件一致。

还有第二个问题，它独立于形态：**没人决定什么时候该复用经验**。原方案把这件事交给 agent
自己（调不调 `apply_experience`），而 agent 常常意识不到自己正在做一个决策——它以为在
"继续写计划"，实际上正在定一个被论文测过的取值。

## 2. 方案全貌：三层

```
┌─ 定位层 ───────────────────────────────────────────────────┐
│  agent 主动检索，拿到的是候选摘要 / 论文原文               │
│    search_experience · list_experience   （E-* 结构化经验）│
│    search_paper_text · read_paper        （C-* 论文原文）  │
│    read_memory                           （按 ID 读全文）  │
└────────────────────────────────────────────────────────────┘
                          │
┌─ 转换层 ────────────────▼──────────────────────────────────┐
│  derive_policy()：唯一对外入口                             │
│  检索(RRF) → 重排(辅助LLM) → 合成(辅助LLM) → 九字段策略    │
│  把"源绑定的散文"改写成"绑定当前任务的策略"                │
└────────────────────────────────────────────────────────────┘
                          │
┌─ 触发层 ────────────────▼──────────────────────────────────┐
│  谁决定这条流水线什么时候跑（两条方向相反的入口，互斥）    │
│    push：coach middleware  每一步系统判断 → 推送指导       │
│    pull：apply_experience  子 agent 主动调用               │
└────────────────────────────────────────────────────────────┘
```

三层职责严格分开，这是整个方案的骨架：

- **定位层回答"库里有什么"**，返回候选摘要和 ID，不改写、不给建议。这一层 agent 一直
  持有，不受任何开关影响——一个能写不能读的库比没有库更糟，因为 prompt 里仍然在宣传它。
- **转换层回答"这条经验对我这个任务意味着什么"**，是 46.9% 那个问题的解药。
- **触发层回答"现在该不该复用"**，是"agent 意识不到自己在决策"那个问题的解药。

## 3. 定位层：agent 自己查什么

| 工具 | 检索对象 | 面向的问题 |
| --- | --- | --- |
| `search_experience` | `E-*` | 内容性：关于某个**主题**，论文发现了什么 |
| `list_experience` | `E-*` | 我根本不知道库里有什么、库里怎么措辞 |
| `search_paper_text` | `C-*` | 要核验原文措辞、具体数字、超出经验粒度的细节 |
| `read_paper` | `C-*` | 展开一个定位结果到整节或整篇 |
| `read_memory` | `O-*`/`E-*` | 按稳定 ID 读完整记录（前缀本身就是无歧义路由） |

**核心原则：一个入口只能有一套词汇。** 经验检索曾经复用 `search_observations`，后果是
observation 那套"这件事该怎么做"的过程性词汇变成了经验检索的词汇，而 `E-*` 只有论文自己
的内容性发现，过程性问法在这个库里注定没有答案——返回的要么是空，要么是一页看起来很自信
的噪声。拆开入口后每个库自己教自己的问法。

`search_experience` 的三个主题面（`topic`/`method`/`task`）**分别检索再用 RRF 融合**
（`RRF_K = 60`），不是拼成一个长 query。拼接会让一个低信息量的 token 匹配到所有文档、
所有文档得分相同、"排序"退化成目录顺序，却仍然返回一整页结果。另有 `discipline`/
`domain`/`level` 三个精确过滤面。**它不接受 `memory_type` 和 `scope`**——那两个描述的是
observation 的 frontmatter，对 `E-*` 传进来只可能是空操作或静默清空整个结果集。

`list_experience` 存在的理由是**检索是纯词法的**（TF-IDF，无 embedding）。调用方不知道
库里的说法就写不出能命中的 query，而开放式任务（"关于 X 有哪些发现"）根本没有唯一正确
的问法。按结构浏览（discipline → domain → records）绕开措辞问题，每层都分页并报总数，
库长到多大都能走完。

**结构性隔离**：`search_paper_text` 是 chunk 唯一的检索入口，chunk 永远不进经验或操作
记忆的排序——否则一篇论文几十个 chunk 会淹没高信号的 `E-*`/`O-*` 记录。

## 4. 转换层：`derive_policy`

`EvoScientist/memory/policy/pipeline.py`。整条流水线唯一的对外入口——push 和 pull 最终
都调这一个函数。

```
任务 ──► 检索(RRF融合) ──► 重排(辅助LLM) ──► 合成(辅助LLM) ──► 九字段策略 ──► 缓存
          8 条候选          选 ≤4 条          读完整 statement    主LLM 据此作答
```

分步说明：

1. **检索**（`gather_candidates`，走 `asyncio.to_thread`，磁盘 TF-IDF 不占事件循环）：
   `task` 和 `method` 作为两个面 RRF 融合，取 8 条候选。**空候选直接返回
   `status="no_candidates"`，一次模型调用都不花。**
2. **重排**（1 次辅助模型调用）：只读每条约 200–300 字符的紧凑描述符
   （`transferable_core`），**不读完整 `statement`**，所以 8 条候选能塞进一次小调用。
   选出最多 4 条。
3. **查缓存**：键为 `SHA256(task, sorted(选中的 E-* ID))[:16]`。命中则直接返回，省掉
   合成这次调用。任务改写措辞会产生新键并重新合成——这是正确的，因为写手是逐字读任务的。
4. **合成**（1 次辅助模型调用）：这次才读选中记录的完整 `statement`（约 2500 字符），
   写出九字段策略对象。
5. 落盘缓存（写失败只记 warning，不影响返回）。

返回一份 report：`status` / `cached` / `considered`（全部候选 ID）/ `selected`（选中 ID）
/ `selection_reason` / `policy`。

**成本**：最坏 2 次辅助模型调用；缓存命中时 1 次（重排仍要跑，因为缓存键需要
`selected_ids`）。

### 4.1 策略对象：九个字段

前六个来自 QCR，后三个是针对科研多论文复用场景的扩展。

| 字段 | 含义 |
| --- | --- |
| `verdict` | `adopt` / `adapt` / `decline`。**`decline` 是合法且有用的答案**，意思是"存的经验不适用于当前任务" |
| `procedure` | 仍然可迁移的步骤。**不得包含源论文的具体取值** |
| `rebind` | 必须重新推导的取值。每项含 `name`/`kind`/`why_bound`/`how_to_obtain`/`source_value` |
| `preconditions` | 该策略成立的前提 |
| `declines` | 明确不迁移的部分 |
| `checks` | 收尾前要验证的项 |
| `conflicts` | 多条记录之间的分歧 |
| `unsupported` | 记忆完全没覆盖的部分，提示转去实时检索 |
| `sources` | 每行结论可回溯到具体 `E-*` 记录 |

`rebind` 是整个方案的核心，也是唯一**故意携带源取值**的字段：`source_value` 只作为出处
标注和合理性锚点，**绝不作为答案**。区别是——"该调度在 ImageNet 上验证过，你的任务换成
CIFAR-10"是正确复用；"在 ImageNet 上训练"是要修的 bug。

`conflicts` 是相对 QCR 的一个必要扩展：QCR 只选一条记录，不存在分歧问题；科研记忆里论文
互相矛盾是常态，写手悄悄挑一条会掩盖最有价值的信号。`unsupported` 同理——明确说"这部分
记忆没覆盖"，比让调用方误以为已经覆盖要好。

### 4.2 抽取侧的两个配套字段

`transferable_core`（≤60 词，剥掉论文专有取值后的因果内核）给重排阶段当描述符；
`bindings`（`[{name, kind}]`，显式列出论文写死的取值）给合成阶段的 `rebind` 一个结构化
输入。两个都是**可选**字段，因为第一批约 100 条记录是在它们存在之前抽的——做成可选那批
记录不必重抽即可继续使用。存量已通过 `scripts/backfill_experience_fields.py` 从记录自己
已有的文本补齐：98/98 条 `transferable_core` 已填充，**没有任何记录仍在走截断兜底**；
363 条 `bindings`，0 条不可用。

## 5. 触发层：谁决定"现在该复用经验"

上面那条流水线解决的是"经验以什么形态进上下文"。还剩一个问题：**谁来决定它什么时候跑**。

### 5.1 pull 的三个漏点

`apply_experience` 是 **pull**：agent 自己要想起有这个工具、自己判断当下需不需要经验、
自己把 query 写出来。三处都会漏：

1. **query 的上限就是 agent 恰好写出的那一句话**，而它用的是自己处境的词汇（"搞清楚下载
   为什么卡住"），不是论文摘要的词汇。纯词法检索下这一条直接决定命中与否。
2. **这个工具不在 always-include 名单里。** 工具数超过阈值时 tool selector 可以把它整个
   过滤掉，复用层于是**静默停止运行**，没有任何信号。
3. **最根本的一条：agent 常常意识不到自己正在做一个决策。** 它以为自己在"继续写计划"，
   实际上正在定一个被论文测过的取值。

### 5.2 push：coach middleware

`EvoScientist/middleware/coach.py`。方向反过来：系统在**每个 model 调用前**替 agent 问这
两个问题。

```
每个 model 调用前（awrap_model_call）
    │
    ├─ 代码层短路（零模型调用）
    │     · messages 为空
    │     · 经验库为空 / 统计读不出来
    │     · 上一步只是纯读工具（_READ_ONLY_TOOLS）→ 还在查资料，决策还没到
    │     · 检索面与上次介入相同 → 直接复用上次那份指导
    │
    ├─ 一次 gate 调用（辅助模型，prompt/policy_gate.md）
    │     输入：本次运行的原始请求 + 轨迹尾部 6 条 + 库的统计（只有计数，不列记录）
    │     输出：reason / need / topic / method / task / state
    │     need=false ──► 什么都不注入，直接继续（多数步骤的正常答案）
    │
    └─ need=true ──► derive_policy(task, state, method)  ← 同一条流水线、同一份缓存
                        └─ render_suggestion(report) 渲染成指导文本
                             └─ 作为一条临时 HumanMessage 追加到 messages 末尾
                                （仅本次调用，不落 state）
```

**gate 把两半合成一次调用是刻意的**：`prompt/policy_gate.md` 要求 `reason` 写在 `need`
**之前**——先命名决策，再回答要不要。**说得出是什么决策，才说明真有一个决策**；命名不出
来就是 `false`。prompt 里明确写了"`false` 是多数情况下的预期答案，而且它是一等答案，不是
失败"，因为一个总说 `true` 的 gate 每步要花两次额外调用，还会把 agent 埋在它不需要的建议
里。

另外两条约束是给纯词法检索兜底的：facets 必须用**论文的词汇**写，且**必须写英文**（库是
英文的，请求可能是中文）。prompt 里也明确划了边界——工具机制、文件布局、网络失败这类
问题在 `E-*` 库里没有答案，属于另一个库，不是 gate 该管的。

### 5.3 注入的是"指导"，不是原始记录

`render_suggestion()` 把策略渲染成一个 `<experience_guidance>` 块，分段对应策略字段：

- `What transfers to this step` ← `procedure`
- `Values you must re-derive here rather than copy` ← `rebind`（`source_value` 渲染在
  推导方法**之后**，并且强制带 `provenance only` 标注）
- `This only holds if` / `What does not transfer` ← `preconditions` / `declines`
- `Papers disagree here — resolve on the condition, do not average` ← `conflicts`
- `Verify before treating this as settled` ← `checks`
- `Memory does not cover these` ← `unsupported`
- 末尾附 `E-*` ID，可用 `read_memory` 审计任意一行

**原始记录在这条路径上从不进入主 agent 的上下文。** 策略本身已经是绑定目标任务的重写，
把它渲染成文本不多花任何模型调用，也没有信息损失。渲染为空（`no_candidates` /
`no_reusable_memory` / 无缺口的 `decline`）时**什么都不注入**——这比告诉 agent"记忆没
意见"更省也更不误导。

### 5.4 三个刻意的性质

**流水线一行没改。** gate 只产出检索面和任务串，检索/重排/合成/缓存仍然是
`derive_policy`。`apply_experience` 仍注册在 tool registry 里供子 agent 调用，所以 push
和 pull 共享一份实现、一份磁盘缓存——策略缓存键只看 `(task, 选中的 ID)`，不看是谁触发的。

**注入是 per-call 的，绝不落 state。** `ModelRequest.override()` 返回新请求，而只有返回的
`ModelResponse` 会写回 graph state。所以指导对模型可见恰好一次，**下一步看到的轨迹与本步
逐字节相同，prompt cache 不失效**。任何"把指导写进历史"或"把旧指导压成一行"的方案都会
改写历史，使编辑点之后的全部 cache token 失效，而且是每步都失效。

指导追加在 `messages` 末尾而不是塞进 system message，理由同上：system prompt 是每次调用
共享的稳定前缀，把逐步变化的文本放进去等于每步重算整个前缀。

**多数步骤必须零成本。** 三个短路都在代码层，不花模型调用（库统计走 `asyncio.to_thread`，
30 秒 TTL 缓存，不在事件循环上走盘）。

| 情况 | 本步成本 |
| --- | --- |
| 被代码层短路挡掉 | **0 次**模型调用 |
| gate 说 `need=false` | 1 次 |
| 重排选不出任何记录 | 2 次 |
| 完整命中且未命中策略缓存 | 3 次 |

### 5.5 gate 顺手修掉的一个旧缺陷

RRF 融合在**只有一个检索面**时会短路，退化成目录顺序。pull 路径只有 agent 写的那一句话，
实际只能填 `topic`，正好落在这个退化分支里。gate 同时给出 `topic` 和 `method`，融合才真的
发生——**"改写 query"本身就是 coach 的一部分收益**，不只是省了 agent 一次判断。

## 6. 两条路径互斥

**一个 agent 只能被告知一条路径。**

| agent | 定位工具 | 持有 `apply_experience` | 被 coach 推送 | 注入的说明块 |
| --- | --- | --- | --- | --- |
| 主 agent | 有 | 否 | **是** | `EXPERIENCE_COACH_INSTRUCTIONS` |
| `research-agent` / `planner-agent` | 有（YAML 授权） | **是** | 否 | `EXPERIENCE_POLICY_INSTRUCTIONS` |
| memory worker | 无 | 否 | 否 | 两块都不注入 |

注意**定位工具不受这个互斥约束**：主 agent 始终持有 `search_experience` /
`list_experience` / `search_paper_text` / `read_paper`。被 coach 接管的只有**转换**这一层
（`apply_experience`），主动查库的能力一直在。

互斥不是洁癖：给被 coach 的 agent 注入工具教程，等于让它去调一个不存在的工具；给持有工具
的子 agent 注入 push 说明，等于承诺一批永远不会到达的指导。实现上
`enable_experience_coach` 与 `enable_experience_policy` 取 `and`，在
`_observation_memory_instructions()` 里二选一，三处（middleware 装配、`base_tools`、说明
块）必须同向，有测试 `TestRouteExclusivity` / `TestAgentWiring` 守着。

写测试时抓到一个真 bug：`EXPERIENCE_SEARCH_INSTRUCTIONS`（两条路径共享的检索说明）里还留
着一句"做决策时优先调 `apply_experience`"，主 agent 也会读到。已改成只陈述照抄的危害，把
机制交给后面那一块路径说明。

**子 agent 不上 coach**，因为每个子 agent 内部再跑一遍逐步 gate，会把辅助模型调用数乘上
扇出宽度；而子 agent 是带着明确任务被派出去的，pull 在那里恰好够用。

### 6.1 coach 的说明块讲了什么

`EXPERIENCE_COACH_INSTRUCTIONS` 不是工具教程，要解释的是四件事：这个 session 里**没有**
`apply_experience`，指导会自己出现；每一段怎么读，尤其 `source-side value` 是出处标注和
合理性锚点、**绝不是答案**；分歧段不许"取两个结果的平均"；它是**建议**，只由记忆库推导
而来，你在本项目里实测到的结论优先。

最后还有一句专门防一种具体的失败：**多数步骤根本不会有这个块，那表示存的经验与这一步无
关，不是库空了，也不是该去找个替代品。** 没有这句话，agent 发现"今天没收到指导"会开始
自己到处搜经验来补，反而把 coach 省下的调用又花回去。

## 7. 模型分工与降级

**模型分工**：中间过程（coach 的 gate、重排、合成）都走辅助模型，三次调用都不产出面向
用户的散文；策略是一个结构化对象，acting agent 读完它，再由**主模型**写出面向用户的回答。

**注意 `auxiliary_model` / `auxiliary_provider` 默认是空字符串**，也就是回落到主模型。
没有显式配一个便宜的辅助模型时，上面这三次调用全跑在主模型上——"用辅助模型做中间工作"的
成本论证此时不成立。这对 coach 尤其重要：gate 是每步都可能发生的。

**降级doctrine：复用是对实时检索的增强，不是前置条件。** 任何一环失败都不应该终止调用方
的回合。

| 情况 | 行为 |
| --- | --- |
| 检索无候选 | `status="no_candidates"`，附提示转实时检索。正常结果，不是错误 |
| 重排返回无法解析的 JSON | 回退到检索打分的前 N 条，`selection_reason` 记录降级原因 |
| 重排选不出任何记录 | `status="no_reusable_memory"`，提示不做经验复用直接推进 |
| 合成输出无法解析 | 抛 `PolicyOutputError`，pull 路径的工具层捕获返回 `status="error"` |
| 缓存写失败 | 记 warning，不影响本次返回 |

`rebind` / `conflicts` 里个别行格式不对时，丢掉该行而不是让整个策略失败——少一条绑定的
策略仍然有用，`verdict` 或 `procedure` 读不出来的策略则没用。

**push 侧的降级更严格一条：它必须完全隐形。** pull 路径失败时 agent 至少知道自己调了工具、
看到一个 `status`；coach 是系统替 agent 做的，agent 根本不知道它存在，所以任何失败只能
注入空串。

| 情况 | 行为 |
| --- | --- |
| gate 输出无法解析 | 判为 `need=false`（**关闭**）。绝不能因为看不懂就放行后面两次调用 |
| gate 说要但没给 `topic` | 降级为 `need=false`——空 topic 的词法检索按目录顺序返回整个库，那不是召回，是带着自信形状的噪声 |
| gate 调用抛异常 | `coach_skip`，注入空串 |
| `derive_policy` 抛异常 | `coach_skip`，注入空串 |
| 库统计读不出来 | 缓存成 `{}` 并按"库为空"短路，不重复走盘 |
| 渲染为空 | 什么都不注入，也不记入 intervention 历史 |

## 8. 配置

| 配置项 | 默认 | 环境变量 |
| --- | --- | --- |
| `memory_experience_policy_enabled` | `true` | `EVOSCIENTIST_MEMORY_EXPERIENCE_POLICY_ENABLED` |
| `memory_experience_policy_max_selected` | `4`（钳制 [1,6]） | `..._POLICY_MAX_SELECTED` |
| `memory_experience_coach_enabled` | `true` | `..._COACH_ENABLED` |
| `memory_experience_coach_recent_messages` | `6`（钳制 [1,30]） | `..._COACH_RECENT_MESSAGES` |

- 关掉 `policy_enabled`：回到原有行为，agent 重新直接读 `E-*` 记录；磁盘上已缓存的策略
  不动。coach 开关**从属于**它——policy 关掉时 coach 也不装配。
- 关掉 `coach_enabled`：回到纯 pull，coach middleware 不进栈，`apply_experience`
  **重新出现在主 agent 的 base_tools 里**，说明块也换回工具教程。tool registry 里的那份
  注册**始终保留**——子 agent 的 YAML `tools:` 列表是按名字对 registry 解析的。
- `recent_messages` 的钳制下界 1（`0` 会让 gate 看不到这次运行做过什么），上界 30（无上界
  等于每步把整条轨迹送上辅助模型，正好抵消掉用辅助模型的理由）。

## 9. 可观测性

开发期调试开关（**临时的，不进配置 schema**）：

```bash
export EVOSCIENTIST_POLICY_TRACE=1
scripts/policy_trace_view.py --memory-dir <dir> --last 5   # 只读查看器
```

一次完整的 coach 介入链路（gate → 检索 → 重排 → 合成 → 注入）共享同一个 `call_id`：

| 事件 | 含义 |
| --- | --- |
| `coach_skip` | 本步没跑 gate 或跑了但没往下走，`reason` 写明哪一种 |
| `gate` | gate 的完整决定：`need`/`reason`/`topic`/`method`/`task`/`state`，附 `raw_output` |
| `coach_reuse` | 检索面与上次相同，复用上次那份指导，零模型调用 |
| `coach_inject` | 真的注入了：检索面、`status`、是否命中缓存、选中的 `E-*` ID、指导全文 |
| `request`/`retrieve`/`rerank`/`synthesize`/`report` | 流水线内部五步（pull 路径共用） |

`need=false` 的步骤只留 `gate` 一条事件，没有 `coach_inject`——这是判断 gate 是否系统性偏
"要"的主要手段。

## 10. 验证状态

**测试**：

```bash
.venv/bin/python -m pytest tests/ -q --ignore=tests/test_backends.py
```

| 文件 | 项数 | 覆盖 |
| --- | --- | --- |
| `tests/test_experience_coach.py` | 67 | gate 输出解析、`need` 的严格读法、无 `topic` 降级、指导渲染、三个短路、per-call 注入不改动原 `messages`、coach 失败对 agent 回合隐形、两条路径互斥 |
| `tests/test_experience_policy.py` | 45 | schema 校验与容错、`transferable_core` 兜底、缓存键稳定性、`derive_policy` 全流程、调试追踪、抽取侧向后兼容 |
| `tests/test_config.py::TestExperienceCoachConfig` | 16 | 两个新开关的默认值、钳制、`MemoryControls` 透传、环境变量映射 |
| `tests/test_paper_experience_memory.py` | 36 | 内容寻址 ID、解析缓存失效、token 记忆化不改分数、discipline 词表校验 |
| `tests/test_policy_ab_harness.py` | 17 | 词边界匹配、评分器判定、桩模型端到端、actor 模型默认值从配置解析 |

**A/B 验证（转换层的效果）**：`scripts/policy_ab.py`，三个条件同一批任务、同一个 actor
模型，指标是确定性字符串匹配的 stale-binding 率（不用 LLM 裁判）。

| 条件 | n | stale% | 每份计划 stale 数 | target% |
| --- | --- | --- | --- | --- |
| A 无记忆（地板线） | 2 | 50.0% | 1.00 | 87.5% |
| B 原始 `E-*` 散文（= QCR Full Trajectory） | 2 | 100.0% | 3.00 | 75.0% |
| C 派生策略（本方案） | 1 | 0.0% | 0.00 | 100.0% |

方向上与设计意图一致：散文条件 100% stale、每份计划抄 3 个源值；派生策略 0% stale 且
target 命中 100%。**但这只是一个跑通信号，不是可引用的测量结果**，两个原因必须一起说：

1. **样本量不够**：12 格里有 7–8 格死于 `APIConnectionError`，每格只剩 n=1–2。
2. **条件 A 的绝对值不可信**："无记忆"条件也拿到 50–100% stale，而这在结构上不可能——
   看命中的是什么就清楚了：`AdamW`、`2e-5`、`0.1`，领域内任何一份像样的计划都会自己写出
   来的标准默认值，不是抄来的。所以 **B−C 的对比仍然有意义，绝对 stale% 没有意义**。

## 11. 遗留

- **A/B 数字还不能引用**（见第 10 节）：需要更稳定的网络跑满 `--repeats`，并先把
  `AdamW`/`2e-5`/`0.1` 这类领域通用值从 fixture 的 `bindings` 里剔除。
- **coach 的 gate 没有实测命中率**：现在是"每步都触发"的最朴素策略。`need=true` 的比例、
  其中真正产出非空指导的比例、gate 是否系统性偏"要"，都只能从 trace 的
  `gate`/`coach_inject` 事件人工数，没有汇总工具。A/B 脚本测的是 pull 路径的隔离流水线，
  不覆盖 gate。
- **prompt cache 没有可见性**：trace 没记 `cache_read_input_tokens` /
  `cache_creation_input_tokens`。"per-call 注入不落 state 所以 cache 不失效"目前是代码层面
  的论证（只有 `ModelResponse` 会写回 state），不是测出来的数字。
- **coach 内部不多轮**：gate 一次定稿检索面，不会看了候选之后再改写 query。保守起点，
  多轮的收益未评估。
- **同步执行路径上 coach 是直通的**（`wrap_model_call` 不做事）。gate 与流水线是端到端
  async，所以 push 侧目前只是 async 执行的能力。
- **`PolicyOutputError` 掩盖了两种不同的事**：辅助模型偶发抖动和真正的功能损坏在报告里
  无法区分，值得分开计数或加一次重试。
- **`utility` 字段仍未接入**：策略被采纳后的实际效果没有回写到置信度，"这条用过效果好"
  不会影响后续重排。
- **策略缓存不按记录内容失效**：同一批 `E-*` ID 的记录被更新时缓存键不变，需要
  `refresh=true` 手动重合成。
- **11 条记录停留在 `discipline=other`**：来自 4 篇 `domain_arxiv` 为 `null` 的论文，
  存量要靠重抽或专门 backfill。
