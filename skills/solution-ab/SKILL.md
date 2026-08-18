---
name: solution-ab
description: "Generate two solutions to the same problem from two different contexts — extracted L1/L2 paper experiences vs. the raw paper full text — and present them side by side for human comparison. Trigger phrases: 对比经验和原文的方案, 用经验库生成方案并和原文对比, 经验 vs 原文 A/B, compare solution from experiences vs raw paper, does the extracted experience beat reading the paper, 分列对比两种上下文生成的方案. Do NOT use for: extracting experiences from a paper (paper-experience); finding a research direction or ranking research ideas (research-ideation); literature search (paper-navigator)."
metadata:
  author: EvoScientist
  version: '1.0.0'
  tags: [research, experience, solution, comparison, ablation]
---

# Solution A/B

Answer one question: **does a paper's distilled L1/L2 experiences support solution
design as well as the paper's full text does?**

Method: hold everything constant except the context, generate one solution per
context, and lay the two side by side. You present the difference; you do **not**
rank them.

```
        Problem + Paper
               │
      ┌────────┴────────┐
      ▼                 ▼
  A: 经验上下文      B: 原文上下文
  rendered.md      artifacts/papers/<id>.md
      │                 │
      ├── same prompt ──┤
      ├── same skeleton ┤
      ▼                 ▼
   方案 A            方案 B
      └────────┬────────┘
               ▼
      分列对比表 + 差异观察
```

> **Run these steps yourself.** This is a skill, not a dispatchable sub-agent —
> there is no `solution-ab` agent type, so never hand it to the `task` tool. Read
> this file and execute the steps directly with your own shell and file tools. If a
> command fails, read the error and fix it yourself (missing dir → `mkdir -p`);
> do not spawn a sub-agent to diagnose a shell error.

---

## Step 1 — Resolve inputs

You need three things: a **problem statement**, an **experience context**, and a
**raw-text context**.

**Problem statement.** A concrete, solution-shaped question ("如何让 coding agent
从执行失败中迭代改进"). If the user only gave a topic, restate it as a problem in
one sentence and show that restatement — the same statement feeds both groups.

**Which papers.** One to three; three is the ceiling. Cases:

| Situation | Action |
|---|---|
| Papers named and already extracted | Use them |
| Papers named but **not** extracted | Stop. Tell the user to run `paper-experience` on them first — do not extract here, and do not proceed with one side missing |
| Only a topic given | Run `paper-experience` Entry B to search + fetch + extract (its default is top-3), then come back to Step 2 |
| Arriving from `paper-experience` | The papers it just extracted are your input; collect their ids |

With multiple papers, **both groups must cover the same paper set** — otherwise the
difference reflects which papers each side saw, not the context format.

## Step 2 — Build both contexts

One script locates every piece, strips the noise, and merges each side:

```bash
python /skills/solution-ab/scripts/build_contexts.py --paper-ids <ID1>[,<ID2>,<ID3>]
```

It writes `artifacts/ab/context_A.md` (merged experiences) and
`artifacts/ab/context_B.md` (merged paper bodies), then prints a per-paper size
table, the A/B ratio, and a size-gap verdict.

**Works for 1–3 papers.** With several ids, each side is concatenated into one
context, so both groups see the same paper set and you generate one synthesized
solution per side — not one per paper. Keep it to 3; more makes the merged
contexts unwieldy.

**References and appendices are stripped from group B.** They are 15–46% of a
fetched paper and contribute nothing to solution design, so keeping them would
waste budget and make the size comparison misleading. Pass `--keep-tail` only if
you specifically need them.

**Read the size-gap line and act on it:**

- `OK size gap within 35% tolerance` → the sides are comparable; proceed.
- `NOTE size gap exceeds 35%` → still proceed, but **say so in the output**. A gap
  that large means the comparison is not size-matched, and the reader must know
  that before reading a difference as a property of the format.

If the script exits with `Cannot build contexts`, it names the missing side per
paper. Missing experiences → run `paper-experience` on that paper first. Never
substitute or reconstruct a missing side; a one-sided A/B is not a result.

The script also warns when a paper body is under 10k chars — that means an `/abs/`
page was fetched instead of the full paper, and group B would lose for reasons
unrelated to the question. Re-fetch before continuing.

Record the printed sizes; they go in the final output.

## Step 3 — Generate solution A, then solution B

Read `references/solution-template.md` first. Both solutions use that exact
6-section skeleton.

Generate **A first, in isolation**, then B. Same problem statement, same skeleton,
same instructions — **the context is the only thing that differs**:

```
[solution-template.md 的要求]
问题：<problem statement>
<context>
  A → artifacts/ab/context_A.md 的内容（合并后的经验条目）
  B → artifacts/ab/context_B.md 的内容（合并后的论文正文）
</context>
```

With several papers merged, each context is delimited by
`# === Paper <id> · … ===` headers. Synthesize **across** papers into one solution
per side — do not answer paper by paper. When papers disagree, say so and cite both.
Citations must carry the paper id so a claim stays traceable to its source:
`2303.11366 L2-002` for A, `2410.01242 §III-B` for B.

Rules that make the comparison mean something:

- **Isolation.** While writing A, act as if B does not exist, and vice versa. Never
  let one draft inform the other, and never write "相比另一方案…".
- **Context-only.** Use nothing but the given context — no outside knowledge, no
  other papers, no general best practice. Leaking your own priors is what destroys
  the comparison, because the difference stops being about the context.
- **Cite everything.** A cites experience ids (`L1-003`, `L2-002`); B cites paper
  locations (`§4.2`, `Table 3`, `abstract`).
- **Absence is a finding.** When the context cannot support a section, write
  `上下文未覆盖`. A gap in one group and not the other is exactly the signal the
  user is looking for — filling it in with invented content hides that signal.

If a generation fails (connection error, timeout), **stop and report which group
failed**. Do not present a one-sided result, and do not hand-write the missing
solution yourself.

For solution-design method (Pattern A cross-domain transfer, Pattern B problem
decomposition), `/skills/research-ideation/references/solution-design.md` is
available — read it only if you need the methodology; it is not required.

## Step 4 — Present side by side

Lead with the setup, so the reader knows the comparison was controlled:

```markdown
## 方案对比：<problem statement>

**论文**：<title>（arXiv:<id>）[· 多篇时逐个列出]
**A 组上下文**：抽取经验 — <N> 字符
**B 组上下文**：论文正文（已剔除 References/附录）— <M> 字符
**规模差异**：<gap>%（<A 或 B> 更长）· <在 35% 容差内 | 超出容差，对比非等长>
**控制变量**：同一问题、同一骨架、同一生成要求；唯一差异是上下文
```

Copy the size numbers from the script's output rather than re-counting. When the
script flagged the gap as over tolerance, keep that caveat in this block — a reader
who does not know the sides were unequal may misread a length-driven difference as
a property of the format.

Then the section-by-section table:

```markdown
| 维度 | A · 基于抽取经验 | B · 基于论文原文 |
|---|---|---|
| 方案概述 | … | … |
| 关键设计决策 | … | … |
| 实施步骤 | … | … |
| 验证方式 | … | … |
| 风险与边界 | … | … |
| 依据来源 | 2303.11366 L2-002, 2410.01242 L1-004 | 2410.01242 §III-B, Table I |
```

**When cells get long** (they will — 实施步骤 especially), switch to two parts:
a compact side-by-side summary table first (one or two lines per cell), then both
full solutions in sequence under `### 方案 A（完整）` / `### 方案 B（完整）`.
Never truncate a solution just to fit the table.

Close with a **descriptive** difference section:

```markdown
## 差异观察

- **仅 A 提到**：…
- **仅 B 提到**：…
- **同一决策上的分歧**：<决策> — A 主张…，B 主张…
- **数值具体度**：A 引用了 <n> 处具体数值，B 引用了 <m> 处
- **各自的未覆盖项**：A — …；B — …
```

### 不做裁判

Do not score, rank, weight, or declare a winner. No "A 更好", no "B 更全面", no
overall verdict, no recommendation of which context to use in future. State what
each solution contains and where they diverge; the judgement belongs to the user,
who asked for a side-by-side view precisely so they could make it themselves.

Describing a concrete, checkable fact is fine ("A 未提及 ablation 数据，B 引用了
Table 3 的 9.8 个百分点"). Turning that into a verdict is not.

---

## Notes

- Two long-context generations run in sequence, roughly **60–120 秒 each**, so
  budget **3–5 分钟** for one paper and **5–8 分钟** for three (merged contexts run
  ~140K / ~170K chars, so each generation is slower).
- The proxy occasionally drops long connections; with two long generations the
  failure odds stack. On failure, name the failing group and stop — a partial A/B
  is not a result.
- Contexts stay workspace-relative (`artifacts/…`). A leading-slash path resolves
  against the virtual workspace root, so `/tmp/x` becomes `./tmp/x` and fails.
- Only the two contexts above are in scope for now. The wider observation-memory
  experience bank is deliberately out of scope; the Step-2 layout leaves room to
  add a third group later without reworking the flow.

## Hand off to

| Goal | Skill |
|---|---|
| 抽取某篇论文的 L1/L2 经验 | `paper-experience` |
| 找研究方向 / 给 idea 排名 | `research-ideation` |
| 检索、评分、排序论文 | `paper-navigator` |
