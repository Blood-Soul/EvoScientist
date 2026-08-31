# 经验复用层：从"照抄论文"到"按当前任务重新推导"

本文档记录 2026-08 对经验检索与应用链路的改造。两个用途：一是对外汇报，二是
后续再动这块代码时先读它，不用重新翻源码。

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
  trace.py        开发期调试日志，见第 8.5 节
prompt/
  policy_rerank.md  重排 prompt
  policy_write.md   合成 prompt
EvoScientist/tools/experience_policy.py   apply_experience 工具
scripts/policy_ab.py                      A/B 验证脚本（独立）
scripts/policy_ab_fixture.json            A/B 用的任务与论文样本
scripts/policy_trace_view.py              调试日志查看器（独立，见第 8.5 节）
```

改动的既有文件：`config/settings.py`（两个开关）、`middleware/memory.py`（注入
使用说明）、`EvoScientist.py`（注册工具）、`subagents/planner.yaml` 与
`research.yaml`（授予工具）、两个抽取 prompt、`memory/experiences/extraction.py`
（放宽校验）。

## 4. 模型分工

按"中间过程用辅助 LLM，面向用户的回答用主 LLM"划分：

- **重排**和**合成**都是中间工作，走辅助模型（`_ensure_auxiliary_chat_model()`）。
  两次调用都不产出面向用户的散文。
- 策略是一个结构化对象，acting agent 读完它，再由**主模型**写出面向用户的回答。

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

## 7. 配置

```
EVOSCIENTIST_MEMORY_EXPERIENCE_POLICY_ENABLED     默认 true
EVOSCIENTIST_MEMORY_EXPERIENCE_POLICY_MAX_SELECTED 默认 4，钳制到 [1, 6]
```

对应 `settings.py` 里的 `memory_experience_policy_enabled` 与
`memory_experience_policy_max_selected`。关掉开关即回到原有行为，agent 重新直接
读 `E-*` 记录；磁盘上已缓存的策略不动。

开关必须同时管住**工具注册**和**说明注入**，否则关掉配置后 agent 会被告知去调一个
不存在的工具。注入点在 `middleware/memory.py` 的 `_observation_memory_instructions()`，
由 `enable_experience_policy` 控制，四个构造点分别传入：

- 主 agent：`memory_controls.experience_policy_enabled`
- 子 agent：上述值 **且** 该子 agent 的 YAML 里确实授予了 `apply_experience`
- memory worker：恒为 `False`（它只写观察，复用是读侧的事）

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

**注意：本次改动只跑到了桩模型的端到端验证，真实 LLM 的 A/B 数字尚未采集。**
上述 stale-binding 率的对比表需要你用自己的额度跑一次才有数。

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

开启后，每次 `derive_policy()`（即每次 `apply_experience` 调用）都会往这个文件
追加 JSON Lines，一次调用对应 5 条事件，用同一个 `call_id` 串起来：

`request`（任务/参数）→ `retrieve`（检索到的候选及其描述符）→ `rerank`（模型
选了哪些、为什么，附模型原始输出）→ `synthesize`（**改写前**的模型原始输出，
以及解析后的策略对象）→ `report`（最终返回给 agent 的完整 JSON，即真正注入
上下文的内容）。

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
.venv/bin/pytest tests/test_experience_policy.py tests/test_policy_ab_harness.py -v
```

- `tests/test_experience_policy.py`（44 项）：schema 校验与容错、
  `transferable_core` 回退、缓存键稳定性（含 ID 顺序无关）、写手输出的格式容错、
  `derive_policy` 全流程（空候选 / 缓存命中 / `refresh` 绕过缓存 / 重排降级 /
  合成失败上抛）、调试追踪（默认关闭不落盘 / 开启后全链路共享一个 `call_id` /
  写入失败不影响主流程 / 自定义路径）、工具层（正常返回 / 失败返回提示 / 配置的
  `max_selected` 确实进了 prompt / 越界值被钳制）、说明注入的开关门控、以及抽取
  侧的**向后兼容**（老记录仍合法、新字段能落盘、未知字段仍被拒）。
- `tests/test_policy_ab_harness.py`（13 项）：词边界匹配、评分器四种判定、fixture
  自检、桩模型端到端（播种 → 检索 → 合成 → 渲染 → 评分 → 汇总）、单格失败不影响
  其他格。

已知无关失败：`tests/test_backends.py::...::test_execute_e2e_workspace_tier_skill`
在干净树上同样失败（本机 PATH 只有 `python3`，没有 `python`），与本次改动无关。

## 11. 遗留

- 真实 LLM 的 A/B 数字未采集（见第 8 节）。
- `utility` 字段仍未接入：策略被采纳后的实际效果没有回写到记录置信度上，因此
  "这条经验用过效果好"这类信号目前不会影响后续重排。
- 缓存不按记录内容失效。同一批 `E-*` ID 的记录被更新（例如置信度聚合改变）时，
  缓存键不变，需要 `refresh=true` 手动重合成。
- fixture 只有 4 篇论文 4 个任务，样本量偏小；`scripts/policy_ab_fixture.json`
  可直接扩充，格式要求写在文件头 `_comment` 里。
