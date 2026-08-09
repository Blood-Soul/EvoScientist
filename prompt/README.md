# Experience Bank v2 — Prompt 交付说明

用于从学术论文全文中抽取**分层研究经验**的 prompt 工程。

---

## 交付内容

| 文件 | 版本 | 用途 |
|---|---|---|
| `l1_extract.md` | v3 | L1 实践经验抽取 prompt |
| `l2_inductive.md` | v12 | L2 归纳经验抽取 prompt |
| 本文件 | — | 字段说明、运行方式、输入格式 |

---

## 经验层级总览

| 层 | 名称 | 回答的问题 | Prompt 版本 |
|---|---|---|---|
| **L1** | 实践经验 | 在某环境为某目标做了哪些动作、得到什么反馈 | v3 |
| **L2** | 归纳经验 | 某情境下可提出什么经验性断言（property / relation / trend / conditional） | v12 |
| **L3** | 机制经验 | 某现象由什么原因造成（why） | 待开发 |

完整定义见 `docs/经验定义.md`（仓库内）。

---

## L1 实践经验（Prompt v3）

**输入**：单篇论文全文（markdown），每篇输出 5-7 条经验（1 coarse + 1 medium + 3-6 fine）。

### L1 字段规范

| # | 字段 | 类型 | 说明 |
|---|---|---|---|
| 1 | `granularity` | string | `coarse` / `medium` / `fine` |
| 2 | `narrative` | string | **≥450 词**，自包含经验叙述（含背景、实践、条件、结果、边界） |
| 3 | `t` | object | 任务上下文，含 4 子字段 |
| | `t.summary` | string | 一句话：完成什么任务 + 达成什么效果（含定量结果） |
| | `t.modality` | string\|null | 数据模态，如 `"text"`, `"3D human skeleton trajectories"` |
| | `t.scale` | string\|null | 数据/模型规模，含量词 |
| | `t.constraint` | string\|null | 限定条件/失效边界 |
| 4 | `e` | string | 实践环境：数据集、模型、基线、指标、硬件、超参数 |
| 5 | `practice_trace` | array | 操作-反馈链 `[{action, feedback}]`，coarse 1 对，medium 2-3 对，fine 3-6 对 |
| 6 | `domain` | string | 智能体类别，13 选 1（见下） |
| 7 | `domain_arxiv` | string | arXiv CS 分类，40 选 1 |
| 8 | `domain_wikipedia` | string | Wikipedia AI 分类 |
| 9 | `keywords` | string | 论文关键词（优先原文 verbatim，否则 ~10 个） |
| 10 | `source_section` | string | 来源段落：`abstract` / `introduction` / `method` / `experiment` / `results` / `discussion` / `conclusion` |
| 11 | `source_quote` | string | **≥150 字符**原文引用 |
| 12 | `extraction_rationale` | string | 溯源说明：来源段落 + 证据类型 |
| * | `_paper_id` | string | arXiv ID（注入） |
| * | `_paper_group` | string | 所属组 t1–t10（注入） |
| * | `_paper_name` | string | 论文目录名（注入） |

---

## L2 归纳经验（Prompt v12）

**输入**：单篇论文全文（markdown），每篇输出 3-8 条经验。

### L2 字段规范

| # | 字段 | 类型 | 说明 |
|---|---|---|---|
| 1 | `narrative` | string | **≥450 词**，自包含经验叙述（背景、发现、条件、证据、因果机制） |
| 2 | `declaration` | string | **≤200 词**，经验断言（去系统名，用类别术语） |
| 3 | `claim_type` | string | `property` / `relation` / `trend` / `conditional` |
| 4 | `keywords` | string\|null | 论文关键词（原文 verbatim），无则 null |
| 5 | `keywords_summary` | string | ~10 个关键词短语（始终填充） |
| 6 | `domain` | string | 智能体类别，13 选 1 |
| 7 | `domain_arxiv` | string | arXiv CS 分类 |
| 8 | `domain_wikipedia` | string | Wikipedia AI 分类 |
| 9 | `domain_acm_ccs` | string | ACM CCS 2012 层级路径，用 ` → ` 连接 |
| 10 | `domain_clc` | string | 中图分类号（TP 类） |
| 11 | `context` | object | 适用上下文，含 4 子字段 |
| | `context.summary` | string | 一句话：完成什么任务 + 达成什么效果 |
| | `context.modality` | string\|null | 数据模态 |
| | `context.scale` | string\|null | 数据/模型规模 |
| | `context.constraint` | string\|null | 限定条件 |
| 12 | `μ` | string | 断言置信度：`high` / `medium` / `low` |
| 13 | `source_quote` | string | **≥150 字符**原文引用（finding + 因果解释两段可 `[...]` 拼接） |
| 14 | `source_section` | string | 来源段落 |
| 15 | `r` | string\|null | 因果解释（仅当论文原文明确说明 WHY 时填充，否则 null） |
| 16 | `μ_r` | string\|null | r 的置信度（仅当 r 存在时）：`high` / `medium` / `low` |
| 17 | `r_depth` | string\|null | 因果机制深度（仅当 r 存在时）：`deep` / `shallow` |
| 18 | `r_depth_rationale` | string\|null | r_depth 判定理由（仅当 r_depth 存在时） |
| 19 | `extraction_rationale` | string | 溯源说明 |
| * | `_paper_id` | string | arXiv ID（注入） |
| * | `_paper_group` | string | 所属组 t1–t10（注入） |
| * | `_paper_name` | string | 论文目录名（注入） |

### claim_type 四种类型

| 类型 | 含义 | 例子 |
|---|---|---|
| `property` | 某物是什么样的 | "Effective lifelong learning agents should propose context-appropriate tasks and self-drive exploration." |
| `relation` | A vs B 的静态比较 | "GRPO converges faster than PPO for search-augmented RL, but PPO is more stable." |
| `trend` | 随着 X 变化，Y 方向性变化 | "As the ratio of incorrect causal relations in pre-training data increases, LLM confidence in correct relations decreases." |
| `conditional` | 当 X 时，则 Y | "When labeled data <1000, pretraining + fine-tuning outperforms training from scratch." |

### domain 分类（13 类）

`agent_memory` / `agent_planning` / `agent_learning` / `agent_tool_use` / `agent_web_gui` / `agent_multi_agent` / `agent_science` / `agent_evaluation` / `agent_software_eng` / `agent_qa_knowledge` / `agent_safety` / `agent_domain_app` / `agent_general`

### 多分类体系（L2 独有）

L2 在 L1 的 `domain_arxiv` + `domain_wikipedia` 基础上额外输出：
- `domain_acm_ccs`：ACM CCS 2012 层级路径
- `domain_clc`：中图分类号 TP 类

---

## L1 vs L2 设计差异

| 维度 | L1 | L2 |
|---|---|---|
| 抽取对象 | 具体实践过程（做了什么、得到什么） | 归纳性断言（作者基于实验形成的判断） |
| 叙述焦点 | 方法步骤 + 实验条件 + 数值结果 | 发现 + 跨实验证据 + 因果机制 |
| 独有字段 | `granularity`, `t`, `e`, `practice_trace` | `declaration`, `claim_type`, `μ`, `r`, `μ_r`, `r_depth`, `r_depth_rationale` |
| 粒度 | 3 级（coarse/medium/fine） | 单一级别（3-8 条/篇） |
| 分类体系 | 3 种（自建 + arXiv + Wikipedia） | 5 种（自建 + arXiv + Wikipedia + ACM CCS + CLC） |
| 每篇数量 | 5-7 条 | 3-8 条 |

---

## 输入格式

两个 prompt 接受相同格式的输入。将论文全文 markdown 填入 user message：

```
[paper_id] {paper_id}

{full paper in markdown}
```

System prompt 为 `l1_extract.md` 或 `l2_inductive.md` 全文。

### 预处理管道

PDF → Markdown → 区段切分 → 输入 prompt：

1. `src/pdf_parse.py`：PDF 文本提取 + Unicode 清洗
2. `src/export_sections.py`：区段切分，输出 `sections.jsonl`
3. 论文按 `data/papers_markdown/t{1-10}/{paper_dir}/full.md` 组织

---

## 运行方式

```bash
# 环境（WSL + pip/conda）
pip install anthropic

# 配置 .env
cp .env.example .env
# 编辑 .env 填入 API key 和 base URL

# L1 全量抽取（16 并发，断点续传）
python tests/batch_l1.py -c 16 -m deepseek-v4-pro

# L2 全量抽取
python tests/batch_l2.py -c 16 -m deepseek-v4-pro
```

### 核心依赖

| 文件 | 作用 |
|---|---|
| `tests/batch_l1.py` | L1 批量抽取入口，按 t1–t10 分组，支持跳过已处理论文 |
| `tests/batch_l2.py` | L2 批量抽取入口，同上 |
| `src/llm_client.py` | LLM 客户端（Anthropic Messages API via proxy），含 `chat_auto` 自动伸缩 max_tokens |
| `src/runner.py` | 通用并发执行器（断点续传 + 流式进度） |
| `src/config.py` | 环境变量加载 + 模型注册表 |

### 关键参数

- `--model` / `-m`：模型名，默认 `deepseek-v4-pro`
- `--concurrency` / `-c`：并发数，默认 4
- `--max-tokens-start`：初始 max_tokens（默认 8000），截断自动翻倍至 cap（65536）
- `--group`：限跑指定组，如 `--group t1 t2 t3`
- `--dry-run`：预检不调 API
- JSONL 默认 append 模式，已处理论文自动跳过

---

## 输出格式

每次抽取输出一个 JSON，包含 `paper_id` 和 `experiences` 数组。批量脚本将所有单篇 JSON 展平为 JSONL（每行一条经验），写入 `_all_experiences.jsonl`。

### L1 输出示例（精简）

```json
{
  "paper_id": "2306.02178",
  "experiences": [
    {
      "granularity": "coarse",
      "narrative": "A self-contained 450+ word description...",
      "t": {
        "summary": "Deploys policy-level trajectory reflection for LLM agent self-improvement, achieving +4% average payoff over baselines across 900+ rounds.",
        "modality": "text-based game trajectories",
        "scale": "2 zero-sum games, 900+ rounds",
        "constraint": "requires recordable interaction trajectories; tested only on text-based imperfect-information games"
      },
      "e": "Two-player zero-sum text-based games. Agent uses GPT-4o as backbone...",
      "practice_trace": [
        {"action": "Deployed a policy-level agent that reflects on belief patterns...", "feedback": "Achieved +4% average game payoff..."}
      ],
      "domain": "agent_learning",
      "domain_arxiv": "cs.MA",
      "domain_wikipedia": "Multi-agent_systems",
      "keywords": "LLM agent, policy learning, self-evolution",
      "source_section": "abstract",
      "source_quote": "\"Across two zero-sum games...\"",
      "extraction_rationale": "Practice summary drawn from abstract. ..."
    }
  ]
}
```

### L2 输出示例（精简）

```json
{
  "paper_id": "2306.02178",
  "experiences": [
    {
      "narrative": "A self-contained 450+ word experience description...",
      "declaration": "Policy-level reflection produces more robust agent behavior than action-level self-correction, with the gap widening as horizon increases.",
      "claim_type": "trend",
      "keywords": null,
      "keywords_summary": "LLM agent, policy learning, self-evolution, reflection, game theory",
      "domain": "agent_learning",
      "domain_arxiv": "cs.MA",
      "domain_wikipedia": "Multi-agent_systems",
      "domain_acm_ccs": "Computing methodologies → Artificial intelligence → Distributed artificial intelligence → Multi-agent systems",
      "domain_clc": "TP181",
      "μ": "medium",
      "context": {
        "summary": "Enables LLM agents to learn behavioral strategies autonomously, achieving +4% average payoff...",
        "modality": "text-based game trajectories",
        "scale": "2 games, 900+ rounds, 3 base model variants",
        "constraint": "requires recordable interaction trajectories with delayed feedback..."
      },
      "source_quote": "\"Across two zero-sum games, policy-level reflection agents consistently outperformed...\"",
      "source_section": "experiment",
      "r": "Policy-level reflection examines belief patterns across full trajectories, enabling correction of systematic errors...",
      "μ_r": "medium",
      "r_depth": "deep",
      "r_depth_rationale": "Identifies a specific mechanism: full-trajectory pattern examination vs. isolated action correction...",
      "extraction_rationale": "The author explicitly interprets cross-game results as demonstrating a consistent outperformance pattern..."
    }
  ]
}
```
