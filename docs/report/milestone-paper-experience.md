# 里程碑报告：论文经验抽取端到端打通

## 一句话总结

从「主题 → 检索论文 → 抓全文 → L1/L2 经验抽取 → 中英双语呈现」的完整链路已在 WebUI 中跑通，抽取产物字段完整、可溯源；同时将 336 条离线经验库接入 EvoScientist 原生记忆层。

---

## 一、交付内容

### 1. 新增独立技能 `paper-experience`

把论文经验抽取从 `paper-navigator` 的附属步骤（Step 7）中解耦为独立技能。

```
skills/paper-experience/
├── SKILL.md                      # 双入口编排 + 呈现规范
├── scripts/
│   ├── fetch_fulltext.py         # arXiv 全文直连抓取
│   └── promote_to_memory.py      # 经验沉淀进 EvoMemory（待验证）
└── references/
    └── experience-schema.md      # L1/L2 字段速查
```

**两种入口**

| 入口 | 触发方式 | 流程 |
|---|---|---|
| A · 论文直给 | URL / arXiv ID / 本地 md | (抓全文) → 抽取 → 返回 |
| B · 主题驱动 | 研究主题 | 检索 top-N → 逐篇抓全文 → 批量抽取 → 返回 |

**与 paper-navigator 的分工**：paper-navigator 负责「找 + 评分 + 排序」（完整 rubric 多轮 triage）；paper-experience 负责「抽经验」，主题入口只做快速 top-N 取样，不做深筛。检索与抓取脚本复用 paper-navigator，不重造轮子。

### 2. 离线经验库接入原生 EvoMemory

`scripts/import_experience_bank.py` —— 将离线 JSONL 经验库写入 EvoScientist 原生 observation 记忆层。

| 项 | 内容 |
|---|---|
| 数据源 | `F:\experience-bank-v2\_out\{l1,l2}_batch\_all_experiences.jsonl` |
| 导入量 | **336 条**（L2 全量 234 + L1 粗/中粒度 102） |
| 类型映射 | L2 归纳断言 → `semantic`；L1 实践经验 → `procedural` |
| 落盘位置 | `~/.evoscientist/memories/observations/global/` |
| 侵入性 | 零 —— 纯新增脚本，未改核心代码 |
| 幂等性 | 内容 hash 去重，重跑不产生重复 |

**验证**：dry-run 解析 336 条零报错 → 小样 10 条落盘格式/UTF-8 正确 → `search_observations` TF-IDF 召回排序正常 → 全量导入 326 新建 + 10 幂等。

---

## 二、修复的问题（按发现顺序）

链路调通过程中定位并修复了 7 个问题，其中 5 个是原有缺陷。

### 1. 模型名前缀导致 503

**现象**：WebUI 发任何消息都报 `503 Pricing configuration is temporarily unavailable`，但 curl 直测端点是 200。

**根因**：抓包对比发现请求体差异 ——

```
EvoScientist 发:  "model": "openai/gpt-5.6-terra"   → 503
curl 发:          "model": "gpt-5.6-terra"          → 200
```

`gpt-5.6-terra` 在模型注册表里只登记了 `openai`（原生）和 `openrouter`（带 `openai/` 前缀）两条，**缺 `custom-openai` 条目**，查找时 fallback 用了带前缀的 model_id。代理端定价表查不到该名字即报 503。

**修复**：`EvoScientist/llm/models.py` 按 `custom-openai` 既有惯例（裸名无前缀）补注册 `gpt-5.6-terra` / `gpt-5.6-luna`。

### 2. tool_selector 过滤掉关键工具 ★

**现象**：agent 反复声称「当前会话没有 `execute` 工具」或「没有 `extract_paper_experiences` 接口」，进而中止或改去派发 subagent。

**根因**：`EvoScientist/middleware/tool_selector.py` 每轮用 LLM 挑选暴露给 agent 的工具，其「永不过滤」白名单原本只有 5 项：

```python
"think_tool", "task", "read_memory", "record_observation", "search_observations"
```

`execute` 与两个抽取工具都不在白名单内，会被选择器滤掉 —— **agent 报告「工具不存在」是事实，而非误判**。而 `task` 恰在白名单中，这解释了它为何总倾向派发 subagent：手上确实只剩该工具可用。

**修复**：将 `execute`、`extract_paper_experiences`、`extract_paper_experiences_batch` 钉入白名单。白名单与实际可用工具求交集（`candidates & available_names`），故在这些工具缺席的会话中不会引入问题。单测 27/27 通过。

### 3. 检索 93 秒无效退避导致超时

**现象**：agent 执行检索命令 180 秒超时（exit 124）。

**根因**：`RETRY_DELAYS = [3, 6, 12, 24, 48]` = 93 秒纯等待。无 `S2_API_KEY` 时 Semantic Scholar 必然限流，这 93 秒注定白费，之后才回退 arXiv。

**修复**：`scholar_search.py` 检测到无 `S2_API_KEY` 时直接走 arXiv，跳过注定 429 的请求。

| | 修复前 | 修复后 |
|---|---|---|
| 检索耗时 | 57–180+ 秒（常超时） | **8.4 秒** |

### 4. 全文抓取依赖不可达服务

**现象**：`fetch_paper.py` 卡死至 300 秒超时。

**根因**：全文抓取唯一路径是 Jina Reader（`r.jina.ai`），在当前网络下不可达（http=000）。而 `arxiv.org` 直连正常（200，0.3 秒）。

**修复**：新增 `fetch_paper_direct.py`（paper-navigator）/ `fetch_fulltext.py`（paper-experience，同一脚本），三级降级：

```
1. arXiv HTML  (arxiv.org/html → markdownify)   ← 首选，无需 PDF 库
2. arXiv PDF   (arxiv.org/pdf  → PyMuPDF)
3. Jina Reader                                   ← 仅非 arXiv 兜底
```

| | 原 fetch_paper | 新 fetcher |
|---|---|---|
| 耗时 | 51–65 秒 | **1.9 秒** |
| 内容量 | 50,039 字符（截断） | **120,019 字符** |

### 5. 抓到摘要页而非全文

**现象**：抓取「成功」但只有 2,996 字符。

**根因**：`/abs/` 是摘要页。L1 fine 粒度要求具体实验数值，摘要中不存在。

**修复**：fetcher 强制改写为 `/pdf/` 或 `/html/`。

| URL 形式 | 内容量 |
|---|---|
| `/abs/1706.03762` | 2,996 字符（仅摘要） |
| `/pdf/1706.03762` | **39,932 字符（全文）** |

### 6. 裸 python3 缺依赖导致静默降级 ★

**现象**：给 `fetch_paper.py` 打的直连补丁在手动测试中有效，但在 agent 流程中无效。

**根因**：agent 的 `execute` 运行裸 `/usr/bin/python3`，而 `pymupdf` / `markdownify` / `deepxiv_sdk` 均只安装在项目 venv 中。补丁的 `import pymupdf` 抛 `ImportError` → 被 `except` 捕获 → **静默降级回 Jina** → 再次卡死。表现为「补丁未生效」，实为悄然回退。

**修复**：`fetch_paper_direct.py` 与 `scholar_search.py` 在检测到当前解释器缺依赖时，**自动 re-exec 到项目 venv**。已验证裸 `python3` 调用可正常工作。

### 7. `/tmp` 路径在虚拟工作区失效

**现象**：`FileNotFoundError: ./tmp/exp_pool.jsonl`。

**根因**：agent 工作区是虚拟路径，前导 `/` 会被解析到工作区根，`/tmp/x` 变成 `./tmp/x`，目录不存在。

**修复**：SKILL.md 改用 `artifacts/search/` 并前置 `mkdir -p`。

### 8. WebUI 历史记录每次为空

**现象**：每次打开 WebUI 历史记录都是空的，但 memory 会累积。

**根因**：`default_mode: run` 使每次启动新建隔离工作区（`runs/<时间戳>/`）。实测 **37 个线程散落在 28 个工作区**中，WebUI 按当前工作区筛选线程，故历史始终为空。memory 存于全局目录（`~/.evoscientist/memories/`）不受工作区隔离，因此持续累积 —— 恰好解释了两者的反差。

**修复**：`default_mode` 改为 `daemon`（持久固定工作区）。已备份 `config.yaml.bak`。

---

## 三、呈现规范修正

抽取工具产出正确，但 agent 转述时会重构内容、丢失字段。

**首次问题**：agent 自创「可复用经验」「归纳含义」「实践做法」等标题替代 schema 字段，导致 `μ`、`r`、`μ_r`、`r_depth`、五套 `domain_*` 分类全部丢失，narrative 被压缩为数句。

**修正后的 SKILL.md 规范**：

- **narrative 是主体** —— 完整呈现，不得压缩为要点或替换为自撰摘要
- 每条前置身份信息：L1 为 `granularity` + `t.summary`；L2 为 `claim_type` + `declaration` + `μ`（有 `r` 则附上）
- 每条以 `source_section` + `source_quote` 收尾，保证可溯源
- 其余结构化字段按需展示，或当其含 narrative 未提及的数值时补充
- **要 N 条就给 N 条完整的，而非 N 条精简的**
- 禁止自创标题替代 schema 字段

**双语规则**：

| 字段 | 处理方式 |
|---|---|
| `declaration` / `t.summary` | 中文 + 括号附英文原文 |
| `narrative` | 完整中译；术语、数值、指标、数据集/模型名保留原文 |
| `source_quote` | **英文原文，绝不翻译**（证据），可附中文说明 |
| 字段标签 | 中文可 |

`μ`/`μ_r` 键名不稳定（Greek `μ` 或 ASCII `mu` 均出现过），已在规范中要求两种都读。

---

## 四、验证结果

### 端到端成功案例

**主题**：`LLM agent code generation debugging iterative refinement`
**命中论文**：RGD: Multi-LLM Based Agent Debugger via Refinement and Generation Guidance（arXiv:2410.01242）

| 检查项 | 结果 |
|---|---|
| 检索 | ✅ 8 秒，返回 8 篇 |
| 抓全文 | ✅ 秒级直连 |
| 抽取工具真实调用 | ✅ 非手工总结 |
| narrative 完整度 | ✅ 未压缩 |
| 中英对照 | ✅ 双语并给 |
| `source_quote` | ✅ 英文原文未翻译 |
| `μ` 置信度 | ✅ `high` |
| `r` 因果解释 | ✅ 有 |
| `claim_type` | ✅ `conditional` |
| 数值保留 | ✅ 97.6% / 83.4% / ablation 2.5 与 9.8 个百分点等 |

### 其他已抽取论文

| 论文 | L1 | L2 | 备注 |
|---|---|---|---|
| Attention Is All You Need (1706.03762) | 5 | 2 | 命令行验证，151 秒 |
| ReAct (2210.03629) | 5 | 3 | WebUI，granularity 分布标准 |
| Reflexion (2303.11366) | 5 | 3 | narrative 503–629 词，均 ≥450 |
| Privacy Risks in LLM Agent Memory (2502.13172) | 6 | 4 | claim_type 覆盖 property/relation/trend |
| RGD (2410.01242) | — | — | 端到端主题驱动成功案例 |

### 记忆召回验证

导入的 336 条经验经 `search_observations` 检索，TF-IDF 排序正常：

| 查询 | 命中 |
|---|---|
| `autonomous driving simulator sensor` | 相关经验 40.65 分居首 |
| `ROS Cyber RT integration bridge` | 对应 L2 断言 36.04 分精准命中 |
| `digital twin scenario testing` | 两条 procedural 经验居前 |

---

## 五、备份清单

所有原文件均已备份于原目录：

```
~/.evoscientist/skills/paper-navigator/scripts/fetch_paper.ORIGINAL.bak
~/.evoscientist/skills/paper-navigator/scripts/scholar_search.ORIGINAL.bak
~/.evoscientist/skills/paper-navigator/SKILL.md.ORIGINAL.bak
~/.config/evoscientist/config.yaml.bak
```

---

## 六、当前配置

| 项 | 值 |
|---|---|
| Provider | `custom-openai` |
| 主模型 | `gpt-5.6-terra` |
| Base URL | `https://www.rightapi.ai/codex/v1` |
| 辅助模型 | 未配置（回退主模型） |
| 工作区模式 | `daemon`（持久） |
| 环境 | WSL + uv + Python 3.12，Node v22 |

---

## 七、待办

1. **`promote_to_memory.py` 写入验证** —— 脚本的 import、缓存定位、字段映射、dry-run 均已通过，但尚未用真实抽取产物完成实际写入。现已有 5 篇论文的完整抽取缓存，可直接验证。
2. **网络稳定性** —— rightapi 在规则模式代理下偶发 `Connection error`；抽取需维持 1–3 分钟连接，比单次请求脆弱。可考虑将该域名设为直连。
3. **非 arXiv 论文** —— 目前仅 arXiv 走直连，DOI / 出版商 URL 仍依赖 Jina Reader。
4. **`μ` 键名不一致** —— 同一 prompt 输出时而 Greek 时而 ASCII。渲染与导入侧均已兼容，但下游若直接索引单一键名需注意。
