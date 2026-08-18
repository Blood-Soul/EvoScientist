---
name: experience-answer
description: "Answer a question using the accumulated EvoMemory experience bank (L1 practical + L2 inductive experiences distilled from papers), then show the evidence chain behind the answer. Trigger phrases: 用经验库回答, 查经验库, 基于已有经验怎么做, 经验库里有什么相关的, answer from the experience bank, what do we know about X, 根据积累的经验给个方案. Do NOT use for: extracting experiences from a new paper (paper-experience); comparing experience-grounded vs paper-grounded solutions (solution-ab); finding papers (paper-navigator)."
metadata:
  author: EvoScientist
  version: '1.0.0'
  tags: [research, experience, memory, retrieval, qa]
---

# Experience Answer

Answer from what the experience bank actually holds — and show the reader exactly
which experiences carried the answer. The point is not to produce a plausible
answer; it is to produce a **traceable** one, so the reader can tell what the bank
knows from what it doesn't.

```
    问题
     │
  拆检索词（TF-IDF 靠关键词，不靠语义）
     │
  search_observations → 摘要粗筛
     │
  read_memory → 精读 3-5 条
     │
  答案 + 依据链
```

> **Run these steps yourself.** This is a skill, not a dispatchable sub-agent.
> `search_observations` and `read_memory` are your own always-available tools —
> call them directly, never via the `task` tool.

---

## Step 1 — Turn the question into queries

Retrieval is **TF-IDF keyword matching with no semantic fallback**. A paraphrase
that shares no vocabulary with the stored text scores zero, so the query wording
does most of the work.

- Pull **3–6 domain-specific terms** out of the question. Drop generic words
  (`how`, `improve`, `better`) — they match everything and rank nothing.
- Write **2–3 queries from different angles** — e.g. one on mechanism, one on
  evaluation, one on failure mode. One query finds one slice.
- **Spell out synonyms yourself.** `failure` / `error` / `debug` are three
  different keys to the same idea here; the index will not connect them for you.

State the queries you chose before running them.

## Step 2 — Coarse pass: screen by summary

```
search_observations(query=<each query>, limit=8)
```

Dedupe across queries by `observation_id`, keeping the higher score. At this stage
read only the **`summary`** (one line), the score, and `memory_type`. Summaries
carry a `[coarse|medium|fine]` or `[property|relation|trend|conditional]` prefix,
which tells you the grain before you open anything.

### 判相关性靠读 summary，不靠分数

**An out-of-scope question does not come back empty — it comes back with
high-scoring noise.** Measured on this bank:

| Query | Top hit | Score |
|---|---|---|
| `CUDA kernel memory coalescing optimization` | a Polyak-Łojasiewicz inequality result | **28.1** |
| `sourdough bread fermentation temperature` | LLM sampling *temperature* in program synthesis | **14.6** |

Both scored respectably by matching stray words (`memory`, `optimization`,
`temperature`) while being entirely unrelated. Trusting the number alone here
produces a confident, fabricated answer.

So: **read each summary and ask whether it is actually about the question.** Drop
anything off-topic no matter how it scored. A score cliff (40 → 12) is a useful
hint about where to stop, never a substitute for reading.

The bank also holds papers well outside agent research (physics, optimization
theory), which is exactly why keyword-only screening misleads.

## Step 3 — Close pass: read the ones that matter

Call `read_memory` on the **3–5** observations that survived screening. Each
document runs ~9,600 characters, so opening eight of them buries the answer in
40k+ characters of mostly-irrelevant prose. Pick deliberately.

Read the sections you need rather than the whole document:

| Section | Use it for |
|---|---|
| `Narrative` | the substance — main source of the answer |
| `Practice trace` | L1's action→feedback chain, when the question wants concrete steps |
| `Causal explanation (r)` | L2's mechanism, when the question is "why" |
| `Applicability context` / `Task context` | boundary conditions — does this transfer to the user's setting? |
| `Source` / `Verbatim quote` | provenance for the evidence chain |
| `Classification` / `Keywords` | which field this came from |

Check the boundary sections against the user's situation. An experience validated
on 200 traces in ALFWorld may not carry to a production repository, and saying so
is part of the answer.

## Step 4 — Output: answer first, evidence chain after

```markdown
## 回答

<Answer the question directly. Synthesize across the experiences — do not list
them one by one. Lead with what to do or what holds, then the qualifications.>

## 依据链

### 用到的经验

| # | 经验 | 类型 | 来源 | 贡献了什么 |
|---|---|---|---|---|
| 1 | <summary 前 60 字> | procedural | arXiv:2607.16387 §results | 检查点锚定的具体做法 |
| 2 | … | semantic | arXiv:2509.25370 §method | 「为什么有效」的因果解释 |

### 推理过程

1. **检索**：用了哪些查询词，为什么这么拆
2. **筛选**：召回 N 条 → 保留 M 条；说明丢弃理由（跑题 / 分数断崖）
3. **综合**：这几条如何互补拼出答案；有分歧就写明分歧在哪，不要抹平
4. **边界**：这些经验的验证条件是否覆盖用户场景；不覆盖的部分明说

### 覆盖缺口

经验库没有覆盖的部分。明确列出 —— 不要用通用知识补齐。
```

---

## 硬约束

**只用经验库内容。** 这是本 skill 的全部意义 —— 用户想知道的是「这个库能答什么」。掺进模型自身知识，答案会变好看，但检验价值归零，而且读者无法分辨哪部分有据、哪部分是编的。库里没有就写「经验库未覆盖」。

**每个结论可溯源。** 答案里的关键论断都要能在依据链里指到具体经验和 arXiv id。写不出出处的论断，就是不该写的论断。

**召回失败要直说。** 如果筛完发现命中都不对题，回答「经验库中没有相关经验」，并列出实际召回到什么、为什么判为不相关。**这比编一个像样的答案有用得多** —— 它告诉用户库的边界在哪，而假答案会让人误信库里有这块积累。

**不做二次抽取。** 库里没有就是没有。需要新论文时提示用户跑 `paper-experience`，不要在这里现抽。

**分歧不要抹平。** 两条经验结论冲突时，写明双方各说什么、条件差异在哪。抹成一个折中答案会丢掉最有价值的信息。

## Notes

- 精读 3–5 条（约 3–5 万字符）加答案生成，预计 **1–3 分钟**。
- 库当前约 374 条，覆盖 51 篇论文，主要在 agent 方向（learning / software_eng /
  memory / evaluation / safety），另有少量其他领域。库外主题必然未覆盖 —— 这是
  事实，不是故障。
- Observations 也带 `related_observations` 图谱边。本流程不沿边扩展召回；需要时
  可在 Step 2 之后加一层，但先把关键词召回做准。

## Hand off to

| Goal | Skill |
|---|---|
| 抽取新论文的 L1/L2 经验 | `paper-experience` |
| 对比「经验 vs 原文」生成的方案 | `solution-ab` |
| 检索、评分、排序论文 | `paper-navigator` |
