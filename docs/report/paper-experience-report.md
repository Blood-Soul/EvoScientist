# Paper Experience 功能汇报

## 概述

Paper Experience 是 EvoScientist 的一项内生能力，嵌入在 **paper-navigator** 技能（`v3.3.1-experience`）中，用于从学术论文全文中**自动抽取结构化研究经验**。它不依赖人工标注，完全由 LLM 驱动，将论文正文转化为可检索、可复用的经验库。

## 核心组件

| 组件 | 路径 | 职责 |
|---|---|---|
| 抽取引擎 | `EvoScientist/tools/paper_experience.py` | LLM 调用、JSON 解析、缓存、格式化、Runtime Tool 注册 |
| L1 Prompt | `prompt/l1_extract.md` (v3) | 抽取"在某环境下为某目标做了什么动作、得到什么反馈" |
| L2 Prompt | `prompt/l2_inductive.md` (v12) | 抽取"在某种情境下可提出什么经验性断言" |
| 技能编排 | `skills/paper-navigator/SKILL.md` | 搜索→评分→全文下载→经验抽取的完整编排 |
| 测试 | `tests/test_paper_experience_search.py` | 14 个单元测试，覆盖解析/缓存/并发/工具注册/会话隔离 |

## L1 vs L2 经验层级

```
论文 Markdown
     │
     ├─→ L1 实践经验（怎么做）
     │    - granularity: coarse / medium / fine（1+1+3~6 条/篇）
     │    - narrative ≥450 词：自包含的实践叙述
     │    - practice_trace：操作-反馈链 [{action, feedback}]
     │    - t（任务上下文）、e（实践环境）
     │
     └─→ L2 归纳经验（什么规律、为什么）
          - 3-8 条/篇
          - declaration：去系统名的经验断言
          - claim_type：property / relation / trend / conditional
          - r（因果解释）、μ（置信度）、r_depth（机制深度）
```

## paper-navigator 中的完整流程

### 三种调用方式

| 你说 | 触发分支 | 做什么 |
|---|---|---|
| `read this paper https://arxiv.org/abs/1706.03762` | **POINT** | 单篇：直接下载全文 → 抽 L1+L2 经验 |
| `read this paper 1706.03762` | **POINT** | 同上，arXiv ID 也行 |
| `read this paper "Attention Is All You Need"` | **POINT** | 按标题匹配 → 下载全文 → 经验抽取 |
| `find papers about X` | **LIST** | 搜索 → Rubric 评分 → 排序 → 全文 + 批量经验 |
| `is there a paper that uses X for Y?` | **LIST** | 同上，单推荐模式 |
| `survey of X in 2024-2025` | **ITERATIVE** | 广度调研：3 轮搜索，最多 10 篇 Primary |
| `30+ papers on transformer architectures` | **ITERATIVE** | 同上 |

**三种分支核心区别**：

- **POINT** — 你已经知道要读哪篇论文，直接给定 URL / ID / 标题，一步到位下载全文 + 经验抽取。1 次调用。
- **LIST** — 你想找论文但不知道具体是哪篇，agent 帮你搜索、评分、排序，最后输出 top-K（3-5 篇）+ 每篇的 L1+L2 经验。2 轮搜索 + 1 轮补丁。
- **ITERATIVE** — 做综述/文献调研，广度优先多轮搜索，覆盖多个角度标签（method / task / dataset / evaluation），最多 10 篇 Primary + 1-2 篇 survey。

### LIST 分支 7 步详解

```
Step 1: Parse Intent     → 解析意图，提取核心概念和约束
Step 2: Author RUBRIC    → 制定 2-4 条评分标准（加权）+ 关键词标签
Step 3: Search           → Probe-then-Refine：Round 1 宽+窄探查 → Round 2 细化
Step 4: Triage           → PERFECT / GOOD / WEAK / IRREL 四档评级
Step 5: Saturation Gate  → 配额检查：≥1 PERFECT → 停止；否则继续搜索
Step 6: Rerank and Output→ weighted_total 排序，K 篇 Primary 论文
Step 7: Full-text        → fetch_paper.py 下载全文 →     ← ★ 经验入口
        Experience       → extract_paper_experiences_batch
        Enrichment         并发抽取 L1+L2，注入 agent context
```

### Step 7 数据流细节

```
Step 6 确定最终论文列表
  │
  ├─ fetch_paper.py --paper-id <ID> --papers-dir artifacts/papers
  │     逐篇下载全文 Markdown
  │
  └─ extract_paper_experiences_batch([{paper_file, paper_id}, ...])
        │
        ├─ extract_and_store_paper_experiences() × N（并发上限受 PAPER_NAV_EXPERIENCE_CONCURRENCY 控制）
        │    │
        │    ├─ session 缓存优先：同论文+同 prompt hash → hit（0 LLM 调用）
        │    ├─ cache miss → asyncio.gather(L1 LLM, L2 LLM) 并发调用 auxiliary model
        │    ├─ parse_experience_json() 校验 + 提取
        │    ├─ format_l1/l2_experiences() JSON → 可读 Markdown
        │    └─ 持久化到 ~/.evoscientist/memories/paper_experiences/sessions/<session>/<paper>/
        │
        └─ 合并所有论文的 rendered Markdown → agent context
```

## 缓存机制

- **会话级隔离**：同一 session 内重复读取同一论文 → cache hit（需 paper 内容 hash + prompt hash 均匹配）
- **跨会话不共享**：不同 session 各自独立抽取，互不影响
- **部分失败可续传**：L1 成功 L2 失败 → L1 已持久化，重试只走 L2

## 验证状态

- **单元测试**：14/14 全通过（无真实 LLM，mock 验证所有逻辑分支）
- **API 连通性**：`deepseek-v4-pro` via `custom-anthropic` → 200 OK，Anthropic 协议兼容
- **Agent 集成**：WebUI 端到端体验（待完成）

## 当前配置

- **Provider**：custom-anthropic（rightapi.ai 代理）
- **Model**：deepseek-v4-pro
- **环境**：WSL + uv + Python 3.12
- **Node**：v22.23.2（WebUI 需要 ≥20）
