# 经验复用层：从"照抄论文"到"按当前任务重新推导"

本文档记录 2026-08 对经验检索与应用链路的改造，以及 2026-09 在它之上加的 push 侧
（coach）。两个用途：一是对外汇报，二是后续再动这块代码时先读它，不用重新翻源码。

第 1–10 节讲的是**转换本身**（经验以什么形态进上下文），第 11 节讲**谁决定它什么
时候跑**（push 侧的 coach）。两件事独立：coach 没有改动流水线里任何一行。

## 1. 问题

系统已经能从论文抽出经验（L1 实践 / L2 归纳），存成 `E-*` 记录，也能检索出来。
但 agent 拿到经验之后的行为是错的：**照抄论文里的具体取值**。

具体表现：规划一个"在自有 4 万条医患对话上微调 Llama-3-8B"的实验时，agent 检
索到一条讲 ImageNet + ResNet-50 余弦退火的经验，然后在计划里写"在 ImageNet 上
训 90 epoch，学习率 0.1 退火到 1e-5"。它抄的不只是数字，是整套与当前任务无关
的绑定。

根因不在检索，在**注入形态**。一条 `E-*` 记录的 `statement` 约 2500 字符，讲的
是"某篇论文的作者在他们的数据集上、用他们的模型、得到他们的数字"。这是一段
**源绑定（source-bound）**的叙述：可迁移的方法论和已经失效的具体取值混在同一段
散文里，一起进入 acting context。Actor 无法区分二者，于是全抄。

QCR（Query-Conditioned Reuse）论文把这种注入方式称为 **Full Trajectory** 条件，
并测出它的 stale-binding（陈旧绑定）错误率为 46.9%——近一半的复用会把源任务的
取值当成答案。我们的行为与该条件一致。

## 2. 方案

在"检索到记录"和"agent 使用记录"之间插入一层转换：把源绑定的散文，改写成
**目标绑定（target-bound）**的结构化策略。

```
任务 ──► 检索(TF-IDF) ──► 重排(辅助LLM) ──► 合成(辅助LLM) ──► 策略 ──► 缓存
          8 条候选         选 3-5 条          写成结构化对象      主LLM 据此作答
```

关键点：agent 不再读原始记录，读的是**针对当前任务重写过的策略**。原始记录仍可
通过 `read_memory` 按 ID 调阅，用于审计某条结论的证据来源。

这条流水线有两个方向相反的入口：`apply_experience`（pull，子 agent 主动调）和
coach middleware（push，系统每步判断）。两者共用这一份实现和这一份缓存，见第 11 节。

### 2.1 策略对象

九个字段，前六个来自 QCR，后三个是针对科研多论文复用场景的扩展：

| 字段 | 含义 |
| --- | --- |
| `verdict` | `adopt` / `adapt` / `decline`。**`decline` 是合法且有用的答案**，意思是"存的经验不适用于当前任务" |
| `procedure` | 仍然可迁移的步骤。**不得包含源论文的具体取值** |
| `rebind` | 必须重新推导的取值。每项含 `name`/`kind`/`why_bound`/`how_to_obtain`/`source_value` |
| `preconditions` | 该策略成立的前提 |
| `declines` | 明确不迁移的部分 |
| `checks` | 收尾前要验证的项 |
| `conflicts` | 多条记录之间的分歧（QCR 只选一条记录，不存在此问题；科研记忆里论文互相矛盾是常态，写手悄悄挑一条会掩盖最有价值的信号） |
| `unsupported` | 记忆完全没覆盖的部分，提示调用方转去实时检索，而不要误以为已被覆盖 |
| `sources` | 每行结论可回溯到具体 `E-*` 记录 |

`rebind` 是整个方案的核心，也是唯一**故意携带源取值**的字段：`source_value`
只作为出处标注，绝不作为答案。区别是——"该调度在 ImageNet 上验证过，你的任务
换成 CIFAR-10"是正确复用；"在 ImageNet 上训练"是要修的 bug。

### 2.2 抽取侧加两个可选字段

`prompt/l1_extract.md`、`prompt/l2_inductive.md` 各加两个**可选**字段：

- `transferable_core`：≤60 词，剥掉一切论文专有取值后剩下的因果内核。用作重排
  阶段的描述符——那里如果拿 `statement` 前 200 字符截断，信号很差。
- `bindings`：`[{name, kind}]`，显式列出论文固定的取值。`kind` 取值为
  `dataset`/`model`/`scale`/`hyperparam`/`baseline`/`metric`/`toolchain`/`other`。
  给合成阶段的 `rebind` 一个结构化输入，不必从散文里挖；也让 A/B 脚本能在**不用
  LLM 裁判**的前提下统计 stale-binding。

**为什么是可选**：已有约 100 条记录是在这两个字段存在之前抽的。做成可选，那批
记录不必重抽即可继续使用——`transferable_core` 缺失时回退到 `statement` 开头，
`bindings` 缺失时由合成模型自己从散文里识别。校验放宽到只接受这两个新键，其他
未知键仍然报错（见 `tests/test_experience_policy.py::test_genuinely_unknown_field_still_rejected`）。

## 3. 代码位置

```
EvoScientist/memory/policy/
  schema.py       策略对象的校验与归一化；VERDICTS、BINDING_KINDS 常量
  prompts.py      按与经验 prompt 相同的搜索顺序加载策略 prompt
  select.py       gather_candidates（检索）+ rerank_candidates（重排）
  synthesize.py   synthesize_policy（合成）+ parse_policy_json（容错解析）
  store.py        磁盘缓存与审计轨迹
  pipeline.py     derive_policy：串起全流程，唯一对外入口
  gate.py         decide_experience_need：push 侧的判断 + 检索面改写（第 11 节）
  suggest.py      render_suggestion：把策略渲染成注入用的指导文本（第 11 节）
  trace.py        开发期调试日志，见第 9 节
prompt/
  policy_rerank.md  重排 prompt
  policy_write.md   合成 prompt
  policy_gate.md    gate prompt
EvoScientist/middleware/coach.py          ExperienceCoachMiddleware（push 入口）
EvoScientist/tools/experience_policy.py   apply_experience 工具（pull 入口）
scripts/policy_ab.py                      A/B 验证脚本（独立）
scripts/policy_ab_fixture.json            A/B 用的任务与论文样本
scripts/policy_trace_view.py              调试日志查看器（独立，见第 9 节）
```

改动的既有文件：`config/settings.py`（四个开关）、`middleware/memory.py`（注入
使用说明，两块互斥）、`EvoScientist.py`（注册工具、装配 coach）、
`subagents/planner.yaml` 与 `research.yaml`（授予工具）、两个抽取 prompt、
`memory/experiences/extraction.py`（放宽校验）。

## 4. 模型分工

按"中间过程用辅助 LLM，面向用户的回答用主 LLM"划分：

- **gate**（push 侧）、**重排**、**合成**都是中间工作，走辅助模型
  （`_ensure_auxiliary_chat_model()`）。三次调用都不产出面向用户的散文。
- 策略是一个结构化对象，acting agent 读完它，再由**主模型**写出面向用户的回答。

**注意 `auxiliary_model` / `auxiliary_provider` 默认是空字符串**，也就是回落到主
模型。没有显式配一个便宜的辅助模型时，上面这三次调用全跑在主模型上——"用辅助模型
做中间工作"的成本论证此时不成立。这对 coach 尤其重要：gate 是每步都可能发生的。

## 5. 成本控制

单次合成读约 20K 字符、写约 2K。控制手段：

1. **两级筛选**。检索出 8 条候选后，重排阶段只读约 200-300 字符的紧凑描述符，
   完整的 2500 字符 `statement` 推迟到合成阶段才读。8 条候选因此能塞进一次小调用。
2. **缓存**。键为 `SHA256(task, sorted(选中的 E-* ID))[:16]`。任务改写措辞会产生
   新键并重新合成——这是正确的，因为写手是逐字读任务的，措辞变了策略就可能变。
3. **按需调用**。工具描述明确要求"在真正做决策时调用"，不是每次检索都调。这也是
   它没有并入 `search_experience` 的原因：那个工具负责**定位**记录，这个负责**转换**
   记录。两者共用同一套检索内核（`gather_candidates` 直接调
   `search_experience_records`），所以拆的是用途，不是实现 —— 不存在两份会各自漂移的
   排序逻辑。
4. **push 侧的三个代码层短路**。coach 每步要付一次 gate 调用，所以先用不花模型调用
   的判断挡掉：库为空、上一步只是纯读工具（`_READ_ONLY_TOOLS`）、检索面与上次相同
   （直接复用上次那份指导）。单步最坏成本：短路 0 次调用、gate 说不要 1 次、重排选
   不出 2 次、完整命中且未命中策略缓存 3 次。

## 6. 降级行为

复用是对实时检索的增强，不是前置条件。任何一环失败都不应该终止调用方的回合：

| 情况 | 行为 |
| --- | --- |
| 检索无候选 | `status="no_candidates"`，附提示转实时检索。这是正常结果，不是错误 |
| 重排返回无法解析的 JSON | 回退到检索打分的前 N 条，`selection_reason` 里记录降级原因 |
| 重排选不出任何记录 | `status="no_reusable_memory"`，提示不做经验复用直接推进 |
| 合成输出无法解析 | 抛 `PolicyOutputError`，由工具层捕获并返回 `status="error"` + 可用提示 |
| 缓存写失败 | 记 warning，不影响本次返回 |

`rebind` / `conflicts` 里个别行格式不对时，丢掉该行而不是让整个策略失败——少一条
绑定的策略仍然有用，`verdict` 或 `procedure` 读不出来的策略则没用。

push 侧的降级更严格一条：**它必须完全隐形**。pull 路径失败时 agent 至少知道自己调
了工具、看到一个 `status`；coach 是系统替 agent 做的，agent 根本不知道它存在，所以
任何失败只能注入空串。

| 情况 | 行为 |
| --- | --- |
| gate 输出无法解析 | 判为 `need=false`（**关闭**），`reason` 记下"gate output unreadable"。绝不能因为看不懂就放行后面两次调用 |
| gate 说要但没给 `topic` | 降级为 `need=false`——空 topic 的词法检索按目录顺序返回整个库，那不是召回，是带着自信形状的噪声 |
| gate 调用抛异常 | `coach_skip`，注入空串 |
| `derive_policy` 抛异常 | `coach_skip`，注入空串（pull 路径同样的异常会变成 `status="error"` 返回给调用方） |
| 库统计读不出来 | 缓存成 `{}` 并按"库为空"短路，不重复走盘 |
| 渲染为空（`no_candidates` / 无缺口的 `decline`） | 什么都不注入，也不记入 intervention 历史 |

## 7. 配置

```
EVOSCIENTIST_MEMORY_EXPERIENCE_POLICY_ENABLED       默认 true
EVOSCIENTIST_MEMORY_EXPERIENCE_POLICY_MAX_SELECTED  默认 4，钳制到 [1, 6]
EVOSCIENTIST_MEMORY_EXPERIENCE_COACH_ENABLED        默认 true
EVOSCIENTIST_MEMORY_EXPERIENCE_COACH_RECENT_MESSAGES 默认 6，钳制到 [1, 30]
```

前两个对应 `settings.py` 里的 `memory_experience_policy_enabled` 与
`memory_experience_policy_max_selected`。关掉 policy 开关即回到原有行为，agent 重新
直接读 `E-*` 记录；磁盘上已缓存的策略不动。coach 开关**从属于** policy 开关：policy
关掉时 coach 也不装配。

开关必须同时管住**工具注册**和**说明注入**，否则关掉配置后 agent 会被告知去调一个
不存在的工具。注入点在 `middleware/memory.py` 的 `_observation_memory_instructions()`，
由 `enable_experience_policy` 控制，四个构造点分别传入：

- 主 agent：`memory_controls.experience_policy_enabled`
- 子 agent：上述值 **且** 该子 agent 的 YAML 里确实授予了 `apply_experience`
- memory worker：恒为 `False`（它只写观察，复用是读侧的事）

coach 加了第二维：`enable_experience_coach` 决定注入哪一块说明。两块互斥，见第 11.5 节。

`memory_experience_coach_recent_messages` 是 gate 读取的轨迹尾部长度。钳制下界 1
（`0` 会让 gate 看不到这次运行做过什么），上界 30（无上界等于每步把整条轨迹送上辅助
模型，正好抵消掉用辅助模型的理由）。非 int、bool、越界值都会 warn 并重置为 6。

`max_selected` 的传递有个坑值得记一下：工具 args_schema 里这个字段的默认值必须是
`None`，不能写字面量 4。pydantic 会在模型省略该参数时用 schema 默认值填充，函数
签名上的默认值永远轮不到执行——写死字面量就会让配置项静默失效。现在由函数内
`max_selected or configured_max_selected` 解析，并在工具构造时钳制到 [1, 6]。

## 8. A/B 验证

`scripts/policy_ab.py`。三个条件，同一批任务、同一个 actor 模型：

| 条件 | 注入内容 |
| --- | --- |
| A `none` | 无记忆。地板线，记忆带来的任何收益都要先超过它 |
| B `raw` | 完整 `E-*` 散文。当前行为，即 QCR 的 Full Trajectory 条件 |
| C `policy` | `derive_policy()` 的输出 |

**指标：stale-binding 率**。fixture 里每条经验都声明了 `bindings`，所以判定是
确定性字符串匹配——计划里出现了目标任务从未提到的源取值，即为一次命中。不用
LLM 裁判，数字可复现且便宜。

配套两个指标防止误读：

- `rebind_rate`：计划提到源取值**并且**说明要重新推导。只有无限定的提及才计入
  stale——提 ImageNet 说明出处是对的。判定方式是在命中位置前后 160 字符窗口内
  查找重绑定标记词。
- `target_hit_rate`：计划是否提到了目标任务自己的取值。防止一个"什么都不说"的
  策略靠回避拿到零 stale。

### 设计上的两个注意点

**B 与 C 走同一套检索。** 否则 B 与 C 的差异会混进"检索质量"这个变量，而脚本要
隔离的是"散文 vs 策略"这一个对比。早期版本按 fixture 文件顺序注入前 4 条，结果
给医疗任务塞了视觉论文——那测的不是本方案。

**词边界匹配。** 朴素子串匹配会把 `support` 里的 `ppo`、`shifts` 里的 `sft` 当成
命中，虚高所有条件的 stale 数，掩盖真实差异。匹配模式按取值自身的首尾字符决定是否
加边界断言，这样 `1e-5`、`nDCG@10`、`5% warmup` 这类以标点结尾的取值仍能命中。

### 与主系统的耦合

脚本从 JSON fixture 播种一个临时记忆库、直接调 `derive_policy()`，不需要活跃项目，
graph 装配的回归也不会悄悄改变数字。代价是它测的是**隔离状态下的复用层**，不是
完整 agent 会话——`--memory-dir` 可以指向真实库来换取真实性。

### 运行

```bash
.venv/bin/python scripts/policy_ab.py --dry-run          # 只打印 prompt，不调 API
.venv/bin/python scripts/policy_ab.py --repeats 3        # 每格 3 个样本
.venv/bin/python scripts/policy_ab.py \
    --policy-model <辅助模型> --repeats 3 --json ab.json
```

`--policy-model` 用来固定"辅助模型做中间工作"这个变量，与 actor 模型独立。
默认 4 任务 × 3 条件 × 1 重复 = 12 次 actor 调用，`--dry-run` 会先报出这个数字。

**现状：已经用真实 LLM 跑通，但数字还不能引用。**
方向上与设计意图一致（B 散文 100% stale、每份计划 3 个源值；C 派生策略 0% stale、
target 命中 100%），但 12 格里有 7–8 格死于 `APIConnectionError`，每格只剩 n=1–2；
且条件 A"无记忆"也报出 50–100% stale——命中的是 `AdamW`、`2e-5`、`0.1` 这类领域通用
默认值，任何一份像样的计划都会自己写出来。所以 **B−C 的对比有意义，绝对 stale% 没有
意义**，要让地板线读作 0 需要先把这类通用值从 fixture 的 `bindings` 里剔除。
详见 [经验检索独立化改造](experience-retrieval-split.zh-CN.md) 第 9 节。

## 9. 开发期调试可见性

开发阶段光看端到端结果不够：召回了哪些经验、重排为什么选了这几条、合成模型
改写前的原始输出长啥样、最终塞进 agent 上下文的策略是什么——这些中间状态在
WebUI 上目前看不到。为此加了一个**临时的**、不进配置 schema 的调试开关，预期
这块 prompt 调稳定之后就会删掉。

### 开启方式

```bash
export EVOSCIENTIST_POLICY_TRACE=1
# 可选：换个位置，默认写到 <memory_dir>/policies/trace.jsonl
export EVOSCIENTIST_POLICY_TRACE_PATH=/tmp/policy_trace.jsonl
```

开启后，每次 `derive_policy()` 都会往这个文件追加 JSON Lines，一次调用对应 5 条
事件，用同一个 `call_id` 串起来：

`request`（任务/参数，含 `method` 面）→ `retrieve`（检索到的候选及其描述符）→
`rerank`（模型选了哪些、为什么，附模型原始输出）→ `synthesize`（**改写前**的模型
原始输出，以及解析后的策略对象）→ `report`（最终返回给 agent 的完整 JSON）。

push 路径在同一个文件里多出四种事件，共享同一个 `call_id`，所以一次 coach 介入的
完整链路（gate → 检索 → 重排 → 合成 → 注入）可以串起来看：

| 事件 | 含义 |
| --- | --- |
| `coach_skip` | 本步没跑 gate 或跑了但没往下走，`reason` 写明哪一种（库为空 / 纯读工具 / gate 调用失败 / `derive_policy` 失败） |
| `gate` | gate 的完整决定：`need`/`reason`/`topic`/`method`/`task`/`state`，附 `raw_output` |
| `coach_reuse` | 检索面与上次相同，复用上次那份指导，零模型调用 |
| `coach_inject` | 真的注入了：检索面、`status`、是否命中策略缓存、选中的 `E-*` ID，以及注入的指导全文 |

`need=false` 的步骤只留 `gate` 一条事件，没有 `coach_inject`——这是判断 gate 是否
系统性偏"要"的主要手段。

不设置该环境变量时 `emit_trace()` 直接返回，不落盘、无性能影响；写入失败也只
记 warning，绝不影响主流程（`tests/test_experience_policy.py::TestTrace`）。

### 看

原始 JSONL 不好直接讲，配了一个只读的小查看脚本：

```bash
scripts/policy_trace_view.py --memory-dir ~/.evoscientist/memories/<project>
scripts/policy_trace_view.py --memory-dir <dir> --last 5      # 最近 5 次调用
scripts/policy_trace_view.py --memory-dir <dir> --call-id <id>
scripts/policy_trace_view.py --memory-dir <dir> --follow      # 类似 tail -f
scripts/policy_trace_view.py --memory-dir <dir> --full        # 不截断长字段
```

每次调用按"检索 → 重排 → 合成 → 最终报告"四步渲染成可读文本，合成那一步会把
模型改写前的原始输出和最终策略对象并排列出，开会时可以直接照着讲"它召回了这
条经验，重排选中理由是 XX，改写前模型说了 XX，改写后变成了这份结构化策略"。

同样是临时工具：不进 CLI 主命令、不接配置，`scripts/` 下独立存在，预期随调试
需求结束一起清理。

## 10. 测试

```bash
.venv/bin/pytest tests/test_experience_policy.py tests/test_policy_ab_harness.py \
    tests/test_experience_coach.py -v
```

- `tests/test_experience_policy.py`（45 项）：schema 校验与容错、
  `transferable_core` 回退、缓存键稳定性（含 ID 顺序无关）、写手输出的格式容错、
  `derive_policy` 全流程（空候选 / 缓存命中 / `refresh` 绕过缓存 / 重排降级 /
  合成失败上抛）、调试追踪（默认关闭不落盘 / 开启后全链路共享一个 `call_id` /
  写入失败不影响主流程 / 自定义路径）、工具层（正常返回 / 失败返回提示 / 配置的
  `max_selected` 确实进了 prompt / 越界值被钳制）、说明注入的开关门控、以及抽取
  侧的**向后兼容**（老记录仍合法、新字段能落盘、未知字段仍被拒）。
- `tests/test_policy_ab_harness.py`（17 项）：词边界匹配、评分器四种判定、fixture
  自检、桩模型端到端（播种 → 检索 → 合成 → 渲染 → 评分 → 汇总）、单格失败不影响
  其他格、以及 actor 模型默认值从项目配置解析（其中 3 项在修复前的行为下会失败）。
- `tests/test_experience_coach.py`（67 项）：gate 输出解析（围栏 JSON / 尾随散文 /
  不可解析时抛错）、`need` 的严格读法（只有真 `True` 或 `"true"` 放行）、无 `topic`
  的降级、`task` 回落到 `topic`、指导渲染（空情况 / 各段落 / `source_value` 必须带
  "provenance only" 标注且排在推导方法之后 / 条数上限）、三个短路、"注入不改动原
  `messages`、不碰 `system_message`"、coach 失败时 agent 回合照常、以及两条路径的
  互斥（说明块、middleware 装配、`base_tools` 双向门控）。
- `tests/test_config.py::TestExperienceCoachConfig`（16 项）：两个新开关的默认值、
  `recent_messages` 的钳制、`MemoryControls` 透传、环境变量映射。

全量：`3926 passed, 27 skipped, 1 failed`（`--ignore=tests/test_backends.py`）。

已知无关失败两处，都在干净树上同样失败（`git stash -u` 后复跑确认过）：

- `tests/test_backends.py` 的 8 项——本机 PATH 只有 `python3`，没有 `python`，用
  `--ignore` 排除；
- `tests/test_paper_experience_memory.py::test_a_query_lost_to_the_tokenizer_is_reported_not_scored_silently`
  ——断言过时，与 push 侧无关：主线 #485 给分词器加了 CJK 二元组，那句中文 query 已经
  不再退化，`degenerate_facets()` 正确地不报警。详见
  [经验检索独立化改造](experience-retrieval-split.zh-CN.md) 第 5 节后记。

## 11. push 侧：coach

第 1–10 节讲的是"经验以什么形态进上下文"。这一节讲的是另一个问题：**谁决定这条
流水线什么时候跑**。流水线一行没改。

### 11.1 pull 的三个漏点

`apply_experience` 是 **pull**：agent 自己要想起有这个工具、自己判断当下需不需要
经验、自己把 query 写出来。三处都会漏：

1. **query 的上限就是 agent 恰好写出的那一句话**，而它用的是自己处境的词汇（"搞清楚
   下载为什么卡住"），不是论文摘要的词汇。纯词法检索下这一条直接决定命中与否。
2. **这个工具不在 always-include 名单里。** 工具数超过阈值时 tool selector 可以把它
   整个过滤掉，复用层于是静默停止运行，没有任何信号。
3. **最根本的一条：agent 常常意识不到自己正在做一个决策。** 它以为自己在"继续写
   计划"，实际上正在定一个被论文测过的取值。

### 11.2 coach 一步的流程

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
    │     输入：本次运行的原始请求 + 轨迹尾部 N 条 + 库的统计
    │     输出：reason / need / topic / method / task / state
    │     need=false ──► 什么都不注入，直接继续（多数步骤的正常答案）
    │
    └─ need=true ──► derive_policy(task, state, method)  ← 同一条流水线、同一份缓存
                        └─ render_suggestion(report) 渲染成指导文本
                             └─ 作为一条临时 HumanMessage 追加到 messages 末尾
                                （`additional_kwargs={"lc_source": "experience_coach"}`，
                                  仅本次调用，不落 state）
```

gate 的 prompt 把两半合成一次调用是刻意的：**说得出是什么决策，才说明真有一个决策**。
所以 `reason` 要求写在 `need` 之前——先命名决策，再回答要不要。命名不出来就是 `false`。

prompt 里另外两条约束是给纯词法检索兜底的：facets 必须用**论文的词汇**写，且**必须
写英文**（库是英文的，请求可能是中文）。

### 11.3 三个刻意的性质

**流水线一行没改。** gate 只产出检索面和任务串，检索/重排/合成/缓存仍然是
`derive_policy`。`apply_experience` 仍注册在 tool registry 里供子 agent 调用，所以
push 和 pull 共享一份实现、一份磁盘缓存——策略缓存键只看 `(task, 选中的 ID)`，不看
是谁触发的。

**注入是 per-call 的，绝不落 state。** `ModelRequest.override()` 返回新请求，只有
返回的 `ModelResponse` 会写回 graph state。所以指导对模型可见恰好一次，下一步看到的
轨迹与本步**逐字节相同**，prompt cache 不失效。任何"把指导写进历史"或"把旧指导压成
一行"的方案都会改写历史，使编辑点之后的全部 cache token 失效，而且是每步都失效。

指导追加在 `messages` 末尾而不是塞进 system message，理由同上：system prompt 是每次
调用共享的稳定前缀，把逐步变化的文本放进去等于每步重算整个前缀。

**多数步骤必须零成本。** 三个短路都在代码层，不花模型调用（库统计走
`asyncio.to_thread`，30 秒 TTL 缓存，不在事件循环上走盘）。单步最坏成本：短路 0 次
调用、gate 说不要 1 次、重排选不出 2 次、完整命中且未命中策略缓存 3 次。

### 11.4 gate 顺手修掉的一个旧缺陷

RRF 融合在**只有一个检索面**时会短路，退化成目录顺序（见
[经验检索独立化改造](experience-retrieval-split.zh-CN.md)）。pull 路径只有 agent
写的那一句话，实际只能填 `topic`，正好落在这个退化分支里。gate 同时给出 `topic` 和
`method`，融合才真的发生——**"改写 query"本身就是 coach 的一部分收益**，不只是省了
agent 一次判断。

`method` 面为此一路透到 `gather_candidates`（`pipeline.py` 的 `_gather`）。

### 11.5 两条路径互斥，说明也必须互斥

一个 agent 只能被告知一条路径：

| agent | 持有 `apply_experience` | 被 coach 推送 | 注入的说明块 |
| --- | --- | --- | --- |
| 主 agent | 否 | 是 | `EXPERIENCE_COACH_INSTRUCTIONS`：指导会自己来、各段怎么读、它是建议不是命令 |
| `research-agent` / `planner-agent` | 是（YAML 授权） | 否 | `EXPERIENCE_POLICY_INSTRUCTIONS`：逐字段教怎么调工具 |
| memory worker | 否 | 否 | 两块都不注入 |

互斥不是洁癖：给被 coach 的 agent 注入工具教程，等于让它去调一个不存在的工具；给持
有工具的子 agent 注入 push 说明，等于承诺一批永远不会到达的指导。

实现上 `enable_experience_coach` 与 `enable_experience_policy` 取 `and`，在
`_observation_memory_instructions()` 里二选一。写测试时抓到一个真 bug：
`EXPERIENCE_SEARCH_INSTRUCTIONS`（两条路径共享的检索说明）里还留着一句"做决策时优先
调 `apply_experience`"，主 agent 也会读到。已改成只陈述照抄的危害、把机制交给后面那
一块路径说明。

**子 agent 不上 coach**，因为每个子 agent 内部再跑一遍逐步 gate，会把辅助模型调用数
乘上扇出宽度；而子 agent 是带着明确任务被派出去的，pull 在那里恰好够用。

### 11.6 coach 的说明块讲了什么

`EXPERIENCE_COACH_INSTRUCTIONS` 不是工具教程，要解释的是四件事：

- 这个 session 里**没有** `apply_experience`，指导会自己出现在 `<experience_guidance>`
  块里，不用去找；
- 每一段怎么读，尤其 `source-side value` 是**出处标注和合理性锚点，绝不是答案**；
- 分歧段不许"取两个结果的平均"——分歧本身是信号；
- 它是**建议**：只由记忆库推导而来，你在本项目里实测到的结论优先。**多数步骤根本
  不会有这个块，那表示存的经验与这一步无关，不是库空了，也不是该去找个替代品。**

最后一句是防一种具体的失败：agent 发现"今天没收到指导"，于是开始自己到处搜经验来
补，反而把 coach 省下的调用又花回去。

## 12. 遗留

- A/B 数字已跑通但还不能引用：样本量不足，且条件 A 的地板线被领域通用默认值污染
  （见第 8 节）。`PolicyOutputError` 还会把辅助模型的偶发抖动和真正的功能损坏记成
  同一种失败格子，无法区分。
- `utility` 字段仍未接入：策略被采纳后的实际效果没有回写到记录置信度上，因此
  "这条经验用过效果好"这类信号目前不会影响后续重排。
- 缓存不按记录内容失效。同一批 `E-*` ID 的记录被更新（例如置信度聚合改变）时，
  缓存键不变，需要 `refresh=true` 手动重合成。
- fixture 只有 4 篇论文 4 个任务，样本量偏小；`scripts/policy_ab_fixture.json`
  可直接扩充，格式要求写在文件头 `_comment` 里。
- **coach 的 gate 没有实测命中率**：现在是"每步都触发"的最朴素策略。`need=true` 的
  比例、其中真正产出非空指导的比例、gate 是否系统性偏"要"，都只能从 trace 的
  `gate`/`coach_inject` 事件人工数，没有汇总工具。A/B 脚本测的是 pull 路径的隔离
  流水线，不覆盖 gate。
- trace 没有 prompt cache 的可见性：没记 `cache_read_input_tokens` /
  `cache_creation_input_tokens`。"per-call 注入不落 state 所以 cache 不失效"目前是
  代码层面的论证（只有返回的 `ModelResponse` 会写回 state），不是测出来的数字。
- coach 内部不多轮：gate 一次定稿检索面，不会看了候选之后再改写 query。保守起点，
  多轮的收益未评估。
- 同步执行路径上 coach 是直通的（`wrap_model_call` 不做事）。gate 与流水线是端到端
  async，从同步路径驱动就得在 middleware 里自己持有事件循环，所以 push 侧目前只是
  async 执行的能力。
