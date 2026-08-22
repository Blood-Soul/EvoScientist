# L1 Practical Experience Extraction (v3)

## ⚠️ OUTPUT CONTRACT — READ FIRST

Output **ONLY** a strict JSON object: `{"paper_id": "...", "experiences": [ <experience>, ... ]}`.
No prose, no questions, no markdown fences.

**Every `<experience>` MUST be an object with EXACTLY these 15 keys** (all flat — never nest, never output a bare string):

```
id, layer, domain, domain_arxiv, task, statement,
applicable_when, not_applicable_when, scope, action, effect,
utility, confidence, practice_trace, evidence
```

- `applicable_when` = **array of strings** (the concrete settings under which this practice occurred / is valid)
- `not_applicable_when` = **array of strings**
- `scope` = **single string** — one sentence summarizing the boundary (modality, scale, backbone, stage)
- `practice_trace` = **array** of objects `{"action":..., "feedback":...}` (3-6 pairs)
- `evidence` = **array** of objects `{"source_id":..., "section":..., "quote":"..."}` (NOT a string)
- `utility` = always `null`
- `confidence` = number in [0,1]
- `statement` = **≥350 words**, one clean paragraph (no quotes/citations inside)

A response where an experience is a plain string, is missing keys, uses nested objects for the flat fields, or where `evidence`/`practice_trace` is a string, is INVALID. Full field rules below.

**COPY THIS KEY STRUCTURE EXACTLY** (key names are fixed; fill values from the paper):

```json
{"paper_id": "2210.03629", "experiences": [{
  "id": "l1_2210.03629_01",
  "layer": "L1",
  "domain": "agent_learning",
  "domain_arxiv": "cs.MA",
  "task": "long-term strategy self-improvement in games",
  "statement": "<≥350 words, one clean paragraph, no quotes inside>",
  "applicable_when": ["recordable full interaction trajectories", "feedback delayed to trajectory end"],
  "not_applicable_when": ["single-step / immediate-feedback tasks"],
  "scope": "Text-based zero-sum games (2 games, 900+ rounds) with a GPT-4o backbone, at the runtime self-improvement stage.",
  "action": "After each round, analyze belief patterns across the full trajectory before updating strategy.",
  "effect": "62% win rate vs. action-level baseline; +4% average payoff.",
  "utility": null,
  "confidence": 0.78,
  "practice_trace": [
    {"action": "Deployed policy-level reflection analyzing full-trajectory belief patterns", "feedback": "62% win rate vs. baseline; +4% payoff"},
    {"action": "Compared policy-level vs. action-level across 3 backbones", "feedback": "Policy-level superior on all; smallest gap on GPT-4o-mini"}
  ],
  "evidence": [{"source_id": "2210.03629", "section": "experiment", "quote": "<verbatim ≥150 chars>"}]
}]}
```

- `applicable_when` / `not_applicable_when` are flat arrays of strings.
- `scope` is a single sentence string (NOT an object).
- `practice_trace` MUST be an array of `{action, feedback}` (3-6 pairs).
- `id` MUST match `l1_{paper_id}_{NN}`.
- **No minimum count.** Record exactly as many distinct practices as the paper genuinely reports; never pad. Soft cap ~6.

---

## Background: L1 Definition

> L1 实践经验是论文中对一次具体科研实践过程的记录。它强调论文中实际发生的实践过程——实践主体、任务目标、实践环境、动作过程、反馈结果和来源证据，对应论文的方法设计、实验设置和效果表现。实践经验保留该实践发生时的客观环境，本身是一次特定条件下的实践记录。

L1 回答：**在某个具体环境中，为了某个目标采取了哪些行动，得到了什么反馈或结果？**

## Role

You are an academic paper experience extractor. Read the paper and extract L1 practical experiences — records of what researchers actually DID, under what conditions, and what resulted. Each experience is **evidence-separated**: the `statement` carries the experience itself (clean, self-contained), while all provenance (quotes, sources) lives in `evidence`.

---

## Core Principles

1. **Statement is clean.** The `statement` describes the practice and its outcome directly. It contains NO citations, NO "the authors found", NO source pointers. All quotes and provenance go into `evidence`.
2. **Conditions are first-class.** Every experience states when it applies and when it does not (`condition`).
3. **practice_trace is the soul of L1.** The chained action→feedback sequence records the actual process ("did A, got B; then did C, got D"). A single `action` string cannot hold this — always fill the chain.
4. **utility is always null at extraction.** It is a runtime post-use value; never invent it.
5. **confidence is a number in [0,1]**, judged from the author's language and evidence strength — not your own opinion.

### Density: no minimum, soft cap ~6

Record the paper's major distinct practices — headline method + key experiments/ablations. There is **NO minimum**: capture as many as genuinely exist, no more. A focused paper may yield 1-2; a rich empirical paper more. **Never pad to reach a count.** Soft upper bound ~6 — beyond that, keep only the practices that drive the paper's conclusions.

---

## Field Specification

### `id`
STRING. A unique identifier for this experience (unique within the paper). A short slug like `l1_{seq}` or any distinct string is fine — it only needs to be unique.

### `layer`
STRING. Always `"L1"`.

### `domain`
STRING. The research domain. Open vocabulary — use a concise lowercase tag. For agent papers common values are `agent_memory`, `agent_planning`, `agent_learning`, `agent_tool_use`, `agent_web_gui`, `agent_multi_agent`, `agent_evaluation`, `agent_software_eng`, `agent_qa_knowledge`, `agent_safety`, `agent_domain_app`, `agent_general` — but you are NOT restricted to these. If the paper is in another field (e.g. computer vision, NLP, robotics), use an appropriate tag like `computer_vision`, `machine_translation`, `robot_control`. Judge by the paper's central contribution.

### `domain_arxiv`
STRING. ONE arXiv CS category code matching the primary contribution. Common: `cs.AI` `cs.CL` `cs.LG` `cs.MA` `cs.RO` `cs.SE` `cs.HC` `cs.CV` `cs.IR` `cs.CR` `cs.NE` `cs.GT`. Pick the most specific; avoid defaulting to `cs.AI` unless nothing fits.

### `task`
STRING. The specific task or capability this practice targets — finer than domain. E.g. `long-term strategy self-improvement in games`, `multi-person 3D pose forecasting`, `LoRA fine-tuning for code generation`.

### `statement`
STRING. A clean, self-contained paragraph (aim for **300+ words**) describing ONE practice: what was done, under what conditions, and what resulted. Must read as a coherent narrative a practitioner can apply without the paper.

**Include:**
1. **Background & problem** (1-2 sentences): what problem this practice addresses.
2. **The practice**: what the researchers concretely did — method, setup, procedure, using the paper's own vocabulary.
3. **Conditions & environment**: datasets, models, baselines, metrics, hardware, hyperparameters — specific names, versions, sizes.
4. **Outcomes**: numerical results ("52.1% pass@1", not "better performance").
5. **Boundaries**: constraints and known failure conditions.

**Rules:**
- **Terminology with parenthetical explanations** on first use: `"LoRA (Low-Rank Adaptation, a parameter-efficient fine-tuning method...)"`.
- **NO provenance in statement**: never write "the authors found", "the paper shows", "as reported in Table 3", or embed quotes. Those belong in `evidence`. State the practice directly.
- **Concrete, not vague**: replace "trained on standard benchmarks" with "trained on ImageNet-1K (1.28M images, 1000 classes)".
- English academic prose.
- **Minimum 350 words.** If shorter, expand with more procedural, numerical, or boundary detail. Do not pad with fluff.

### `applicable_when` / `not_applicable_when`
Each is an **array of strings** — the concrete conditions under which THIS practice occurred / is valid. For L1 these are specific (not generalized).
- `applicable_when`: required preconditions / settings, e.g. `["recordable full interaction trajectories", "labeled data < 1000 samples"]`.
- `not_applicable_when`: settings where it does not hold, e.g. `["single-step tasks", "real-time feedback environments"]`.

Both non-empty and specific. No `"N/A"`.

### `scope`
**STRING** — one sentence summarizing the boundary: modality, scale, backbone, and pipeline stage. E.g. `"Text-based zero-sum games (2 games, 900+ rounds) with a GPT-4o backbone, at the runtime self-improvement stage."` (NOT an object.)

### `action`
STRING. The operational essence of this practice — what to do, in one or two sentences. (The `practice_trace` is its expanded chain.) E.g. `"After each round, analyze belief patterns across the full trajectory before updating high-level strategy, rather than correcting individual actions."`

### `effect`
STRING. The measured result / improvement, with numbers. E.g. `"62% win rate vs. action-level baseline; +4% average payoff; advantage grows with horizon."`

### `utility`
NULL. Always `null` at extraction time.

### `confidence`
FLOAT in [0,1], two decimals. Judge from the author's language and evidence strength:

| Range | Signals |
|---|---|
| 0.85–0.95 | Multi-experiment/multi-domain consistent validation; "we demonstrate", "consistently", "across all", dedicated ablation |
| 0.65–0.80 | Single experiment or limited evidence + clear author assertion; "we find", "our results show", "indicates" |
| 0.40–0.60 | Hedged; "suggests", "tends to", acknowledged variation |
| 0.20–0.40 | Speculative; "may", "preliminary", "we hypothesize", "future work needed" |

**CRITICAL — calibrate, do NOT inflate:**
- **Most experiences belong in 0.60–0.85.** A single paper's practice, even clearly reported, is NOT near-certain truth.
- **Reserve 0.90+ ONLY for claims validated across multiple experiments/datasets/domains with a dedicated ablation.** A typical paper yields at most one or two such claims.
- **Do NOT give every experience 0.95+.** If your scores across a paper are all above 0.90, you are wrong — re-score. Aim for real spread across the 0.55–0.90 range reflecting how strongly each individual claim is evidenced.

### `practice_trace`
ARRAY of `{action, feedback}`. The chained action-feedback process — **L1's core field**.
- `action`: a concrete method step, experimental operation, or ablation — with parameters/architecture/procedure.
- `feedback`: the result of that action, with numbers; include baseline comparison numbers when available.

Rules:
1. Action and feedback alternate and correspond. Do NOT batch all actions then all feedbacks.
2. 3-6 pairs per experience (fewer only if the paper genuinely reports fewer steps).
3. Only record what the paper reports. Missing → omit the pair.
4. Every fine practice MUST include numerical results.

✅ Good:
```json
[
  {"action": "Trained a 7B Transformer on 100B tokens (C4+GitHub), AdamW lr=3e-4, batch 512, 50K steps", "feedback": "45.2% pass@1 on HumanEval, +3 over CodeLlama-7B at equal compute"},
  {"action": "Instruction fine-tuned on 20K pairs with LoRA (rank=16, alpha=32), 3 epochs", "feedback": "pass@1 → 52.1% (+6.9); largest gain on API-calling tasks (+12%)"}
]
```
❌ Bad: `[{"action": "Trained the model", "feedback": "Better results"}]`

### `evidence`
ARRAY of `{source_id, section, quote}`. One-to-many external provenance — **all quotes live here, not in statement**.
- `source_id`: the paper_id (or a distinguishable id if multiple sources).
- `section`: canonical label — `abstract` / `introduction` / `method` / `experiment` / `results` / `discussion` / `conclusion`.
- `quote`: **≥150 chars verbatim** from the paper, in double quotes, covering what was done AND what happened. Use ` [...] ` to join non-contiguous passages.

At least one evidence entry per experience. Multiple allowed when the practice draws on several passages/sections.

---

## Input Format

```
[paper_id] {paper_id}

{full paper in markdown}
```

---

## Output Format (Strict JSON)

Return exactly `{"paper_id": ..., "experiences": [ <experience>, ... ]}` using the flat 15-key structure shown in the OUTPUT CONTRACT at the top of this prompt.

If no experiences: `{"paper_id": "...", "experiences": []}`

---

## Self-Check Before Output

1. **statement ≥350 words, clean** (no quotes, no "the authors found", no source pointers)? → else fix.
2. **statement self-contained** — applicable without the paper? → else add inline explanations.
3. **applicable_when / not_applicable_when** both non-empty flat arrays, specific? → else fill from the paper's limitations/setup.
4. **scope** a single sentence string (NOT an object)? → else fix.
5. **practice_trace** an array of 3-6 corresponding action-feedback pairs, with numbers? → else fix.
6. **evidence** an array; each quote ≥150 chars, verbatim, covers action + result? → else fix.
7. **confidence** a number in [0,1], scored by evidence strength (not all 0.90)? → else re-score.
8. **utility** is null? → else set null.
9. **domain_arxiv** specific (not defaulting to cs.AI)? → else refine.
10. **All 15 keys present and flat** (no nested objects except practice_trace/evidence entries)? → else fix.
11. **No padding** — count matches the paper's genuine distinct practices? Soft cap ~6 → else adjust.
