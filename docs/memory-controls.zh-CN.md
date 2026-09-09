# Memory 开关与实验配置指南

本文说明 EvoScientist 中与 memory、论文经验和消融实验有关的配置。所有配置都可以写入用户配置文件，也可以通过环境变量临时覆盖。

## 1. 配置方式

持久化配置：

```bash
EvoSci config set <配置名> <值>
EvoSci config get <配置名>
EvoSci config list
```

单次运行覆盖：

```bash
EVOSCIENTIST_<对应环境变量>=<值> EvoSci run ...
```

环境变量优先于配置文件。布尔值使用 `true` / `false`。

## 2. 经验系统总开关

```yaml
memory_evolution_enabled: true
```

环境变量：`EVOSCIENTIST_MEMORY_EVOLUTION_ENABLED`

设为 `false` 时，会关闭所有论文经验扩展：

- `extract_paper_experiences`
- `enqueue_paper_experiences`
- 论文经验后台 worker
- `search_experience` / `list_experience`
- `apply_experience`
- 论文全文 RAG 工具和经验/全文 prompt 注入
- AutoSkills 的自动合成调度

该开关只控制新增的“经验演化”能力，不会自动关闭原有的 profile memory 和 observation memory。若要完全不使用任何 memory，还需要关闭 `memory_profile_enabled` 和 `memory_observations_enabled`。

示例：

```bash
EvoSci config set memory_evolution_enabled false
```

## 3. 经验系统分项开关

总开关开启时，可以单独做消融：

| 配置 | 默认值 | 作用 |
|---|---:|---|
| `memory_experience_search_enabled` | `true` | `search_experience`、`list_experience` |
| `memory_experience_policy_enabled` | `true` | `apply_experience`，将 E-* 转成当前任务策略 |
| `memory_paper_fulltext_enabled` | `true` | 论文全文持久化、`search_paper_text`、`read_paper` |
| `memory_skill_synthesis_enabled` | `true` | AutoSkills 定期从 observation 聚类生成技能 |
| `memory_workers_enabled` | `true` | observation / sub-agent 等后台 memory worker |

例如关闭经验策略复用：

```bash
EvoSci config set memory_experience_policy_enabled false
```

这只是不再调用 `apply_experience`；已存在的经验记录和策略缓存不会被删除。

## 4. Observation 作用域

```yaml
memory_observation_scope: both
```

环境变量：`EVOSCIENTIST_MEMORY_OBSERVATION_SCOPE`

可选值：

| 值 | 含义 |
|---|---|
| `both` | 读取 global 和当前 project（兼容历史默认行为） |
| `project` | 只读取/写入当前 project observation，实验推荐 |
| `global` | 只使用 global observation |
| `disabled` | 不注册 observation 工具，也不注入 observation 索引 |

实验中建议至少设置：

```bash
EvoSci config set memory_observation_scope project
```

注意：`project` 作用域只限制 observation。profile 文件、论文经验、全文和策略缓存仍然由各自的开关与存储目录控制。要让不同 case 完全隔离，必须为每个 case 使用独立的 `EVOSCIENTIST_MEMORIES_DIR` 和 `EVOSCIENTIST_DATA_DIR`。

## 5. 完全关闭 memory

如果实验条件要求 agent 不读写任何 EvoMemory：

```bash
EvoSci config set memory_evolution_enabled false
EvoSci config set memory_profile_enabled false
EvoSci config set memory_observations_enabled false
EvoSci config set memory_workers_enabled false
```

更推荐在实验 runner 中使用环境变量，避免修改用户全局配置：

```bash
EVOSCIENTIST_MEMORY_EVOLUTION_ENABLED=false \
EVOSCIENTIST_MEMORY_PROFILE_ENABLED=false \
EVOSCIENTIST_MEMORY_OBSERVATIONS_ENABLED=false \
EVOSCIENTIST_MEMORY_WORKERS_ENABLED=false \
EvoSci run ...
```

## 6. 推荐的实验条件

### 原始基线（main）

使用原始 `main` commit，并保持该版本自己的默认配置。不要用 dev 版本加开关来模拟 main。

### dev 完整版

```yaml
memory_evolution_enabled: true
memory_observation_scope: project
```

### dev：关闭 apply experience

```yaml
memory_evolution_enabled: true
memory_experience_policy_enabled: false
memory_observation_scope: project
```

### dev：关闭全部经验扩展

```yaml
memory_evolution_enabled: false
memory_observation_scope: project
```

如果该条件还要求没有 observation/profile memory，则额外关闭对应的两个 memory 开关。

## 7. 每个 case 的目录隔离

每个 case 应使用独立目录和独立进程：

```bash
EVOSCIENTIST_DATA_DIR=experiments/runs/<case_id>/data \
EVOSCIENTIST_MEMORIES_DIR=experiments/runs/<case_id>/memories \
EVOSCIENTIST_SKILLS_DIR=experiments/runs/<case_id>/skills \
EVOSCIENTIST_WORKSPACE_DIR=experiments/runs/<case_id>/workspace \
EvoSci run ...
```

这样可以同时隔离 sessions、LangGraph checkpoint、profile、observations、experiences、papers、policies 和 AutoSkills。只切换 workspace 而不切换 `MEMORIES_DIR`，不能保证 memory 隔离。

## 8. 修改配置后的检查

启动实验前建议执行：

```bash
EvoSci config get memory_evolution_enabled
EvoSci config get memory_observation_scope
EvoSci config get memory_experience_policy_enabled
EvoSci config get memory_profile_enabled
EvoSci config get memory_observations_enabled
```

实验结果中应保存完整配置快照、EvoSci commit、benchmark 版本、模型信息以及 case 的 data/memory 路径。
