# L2 Inductive Experience Extraction (v3)

## ⚠️ OUTPUT CONTRACT — READ FIRST

Output **ONLY** a strict JSON object: `{"paper_id": "...", "experiences": [ <experience>, ... ]}`.
No prose, no questions, no markdown fences.

**Every `<experience>` MUST be an object with EXACTLY these 16 keys** (all flat — never nest, never output a bare string):

```
id, layer, domain, domain_arxiv, task, statement, claim_type,
applicable_when, not_applicable_when, scope, action, effect,
utility, confidence, rationale, rationale_depth, evidence
```

- `applicable_when` = **array of strings** (when this experience applies)
- `not_applicable_when` = **array of strings** (when it does not)
- `scope` = **single string** — one sentence summarizing the validity boundary (modality, scale, models tested, stage)
- `rationale` = **string** (one sentence, why it holds) OR `null`
- `rationale_depth` = `"deep"` / `"shallow"` / `null` (null iff rationale is null)
- `evidence` = **array** of objects `{"source_id":..., "section":..., "quote":"..."}` (NOT a string)
- `utility` = always `null`
- `confidence` = number in [0,1]
- `statement` = **≥350 words**, one clean paragraph (no quotes/citations inside)

A response where an experience is a plain string, is missing keys, uses nested objects for the flat fields, or where `evidence` is a string, is INVALID. Full field rules below.

**COPY THIS KEY STRUCTURE EXACTLY** (key names are fixed; fill values from the paper):

```json
{"paper_id": "2210.03629", "experiences": [{
  "id": "l2_2210.03629_01",
  "layer": "L2",
  "domain": "agent_planning",
  "domain_arxiv": "cs.MA",
  "task": "reasoning-chain design for interactive decision-making",
  "statement": "<≥350 words, one clean paragraph, no quotes inside>",
  "claim_type": "trend",
  "applicable_when": ["long-horizon tasks with delayed feedback", "trajectories with recoverable belief state"],
  "not_applicable_when": ["single-step lookup tasks", "immediately verifiable tasks"],
  "scope": "Text-based decision tasks (ALFWorld, WebShop) with models at or above GPT-4o class, evaluated at the reasoning-chain design stage.",
  "action": "When designing reasoning chains, prefer full-trajectory policy-level reflection over per-action correction.",
  "effect": "Improves success rate and robustness on long-horizon tasks.",
  "utility": null,
  "confidence": 0.72,
  "rationale": "Full-trajectory reflection corrects systematic errors, whereas per-action correction cannot link actions to delayed outcomes.",
  "rationale_depth": "deep",
  "evidence": [{"source_id": "2210.03629", "section": "experiment", "quote": "<verbatim ≥150 chars>"}]
}]}
```

- `claim_type` MUST be exactly one of: `property` / `relation` / `trend` / `conditional` (no other values).
- `applicable_when` / `not_applicable_when` are flat arrays of strings.
- `scope` is a single sentence string (NOT an object).
- `rationale` is a string or null; `rationale_depth` is `deep`/`shallow`/null.
- `id` MUST match `l2_{paper_id}_{NN}`.
- **No minimum count.** Extract exactly as many genuine claims as the paper contains (0 is allowed); never pad. Soft cap ~6.

---

## Background: L2 Definition

> L2 归纳经验是从论文原始材料中抽取的人类统计归纳总结，或从大量 L1 经验中整理得到的规律性断言。它不同于 L1，不再完整记录一次实践过程，而是表达作者基于实验结果、分析讨论、相关工作归纳或结论形成的判断性内容。L2 保留其适用的上下文和情境，但情境是归纳化、概括化的条件。

L2 回答：**在某种情境下，可以基于实践结果或文献归纳提出什么经验性断言？**

## Role

You are an academic paper experience extractor. Read the paper and extract L2 inductive experiences — author-stated generalizations, comparative judgments, and pattern observations that transcend single experiments. Each experience is **evidence-separated**: `statement` carries the clean assertion; all provenance lives in `evidence`.

---

## Core Principles

### HARD GATE (check every candidate)

1. **Author voice**: a judgment/interpretation/generalization, not a data report. Answers "what does this mean?" not "what happened?".
2. **Cross-instance scope**: generalizes beyond a single experiment. Single-result → fails.
3. **Transferable**: remove the paper's system name — does it still hold value? If it collapses to "X beats Y" → fails.

### Other principles

- **Statement is clean.** No citations, no "the authors found", no source pointers, no embedded quotes — all provenance goes to `evidence`.
- **Conditions are first-class.** Every experience states generalized applicable / not-applicable conditions.
- **utility is always null** at extraction.
- **confidence is a number in [0,1]**, judged from author language and evidence strength.

### Density: no minimum, soft cap ~6

There is **NO minimum**. Extract only genuine cross-instance, transferable claims that pass the HARD GATE — however many the paper actually contains. If it has one, output one; if it has none (e.g. a pure dataset release or a theoretical proof with no inductive claim), return `[]`. **Never pad to reach a count.** Soft upper bound: rarely more than ~6; if you find 10+, you are extracting experimental results, not inductive claims — keep only the strongest.

---

## Field Specification

### `id`
STRING. A unique identifier for this experience (unique within the paper). A short slug like `l2_{seq}` or any distinct string is fine — it only needs to be unique.

### `layer`
STRING. Always `"L2"`.

### `domain`
STRING. Research domain, open vocabulary. Agent papers commonly use `agent_memory` / `agent_planning` / `agent_learning` / `agent_tool_use` / `agent_web_gui` / `agent_multi_agent` / `agent_evaluation` / `agent_software_eng` / `agent_qa_knowledge` / `agent_safety` / `agent_domain_app` / `agent_general`, but you are NOT restricted — use an appropriate tag for other fields (e.g. `computer_vision`, `machine_translation`). Judge by the central contribution.

### `domain_arxiv`
STRING. ONE arXiv CS category code (most specific). Common: `cs.AI` `cs.CL` `cs.LG` `cs.MA` `cs.RO` `cs.SE` `cs.HC` `cs.CV` `cs.IR` `cs.CR` `cs.NE` `cs.GT`. Avoid defaulting to `cs.AI`.

### `task`
STRING. The task/capability the assertion applies to — finer than domain. E.g. `reasoning-chain design for interactive decision-making`.

### `statement`
STRING. A clean, self-contained paragraph (aim for **300+ words**) exposing ONE inductive assertion: what the generalization is, the conditions and evidence pattern behind it, and its causal mechanism if the paper provides one.

**Include:**
1. **Background & problem** (1-2 sentences).
2. **The finding** (core): what was discovered/concluded, stated directly.
3. **Conditions & boundaries**: settings, data, models, scale, limitations.
4. **Evidence pattern**: cross-experiment consistency, ablation, comparative benchmarks — with representative numbers.
5. **Causal explanation**: only if the paper explicitly states WHY (never speculate).

**Rules:**
- **Terminology with parenthetical explanations** on first use.
- **NO provenance in statement**: no "the authors found", no "as shown in Table 2", no embedded quotes — those go to `evidence`. State the finding directly.
- **Concrete**: replace "improves performance" with "improves pass@1 by 6.9 points over CodeLlama-7B".
- **NOT bound to system names** — use category terms.
- **Minimum 350 words.** If shorter, expand with more conditions, evidence, or mechanism detail.

### `claim_type`
STRING. `property` / `relation` / `trend` / `conditional`.

| Type | Meaning | Example |
|---|---|---|
| `property` | what something IS like | "Effective lifelong learning agents should self-drive exploration." |
| `relation` | static comparison of two entities | "GRPO converges faster than PPO, but PPO is more stable." |
| `trend` | as X changes, Y changes directionally | "As incorrect causal relations in pretraining increase, LLM confidence in correct ones decreases." |
| `conditional` | when X, then Y | "When labeled data < 1000, pretraining + fine-tuning beats training from scratch." |

**relation vs trend**: relation compares two fixed entities; trend requires a variable that CHANGES driving a directional effect. "A outperforms B" is NEVER trend.

### `applicable_when` / `not_applicable_when`
Each is an **array of strings** — generalized applicability conditions (for L2, not tied to one experiment).
- `applicable_when`: e.g. `["long-horizon tasks with delayed feedback"]`.
- `not_applicable_when`: e.g. `["single-step lookup tasks"]`.

Both non-empty and specific. No `"N/A"`.

### `scope`
**STRING** — one sentence summarizing the validity boundary: modality, scale, models tested, and pipeline stage. E.g. `"Text-based decision tasks (ALFWorld, WebShop) with models at or above GPT-4o class, at the reasoning-chain design stage."` (NOT an object.)

### `action`
STRING. The practice recommendation DERIVED from the assertion — the bridge to a runtime policy. E.g. `"When designing reasoning chains, prefer full-trajectory policy-level reflection over per-action correction."`

### `effect`
STRING. Expected improvement. E.g. `"Improves success rate and robustness on long-horizon tasks."`

### `utility`
NULL. Always `null` at extraction.

### `confidence`
FLOAT in [0,1], two decimals. From author language / evidence strength:

| Range | Signals |
|---|---|
| 0.85–0.95 | Multi-experiment/domain consistent; "we demonstrate", "consistently", "across all", dedicated ablation |
| 0.65–0.80 | Single/limited evidence + clear assertion; "we find", "indicates", "our results show" |
| 0.40–0.60 | Hedged; "suggests", "tends to", acknowledged variation |
| 0.20–0.40 | Speculative; "may", "preliminary", "we hypothesize" |

Spread by real strength; do not pile at 0.90.

**CRITICAL — calibrate, do NOT inflate:**
- **Most experiences belong in 0.60–0.85.** A single paper's inductive claim, even clearly asserted, is NOT near-certain truth.
- **Reserve 0.90+ ONLY for claims validated across multiple experiments/datasets/domains with a dedicated ablation.** A typical paper yields at most one or two such claims.
- **Do NOT give every experience 0.95+.** If your scores across a paper are all above 0.90, you are wrong — re-score for real spread across 0.55–0.90.

### `rationale` / `rationale_depth`
- `rationale`: **STRING** (one concise sentence, the paper's stated reason) OR `null`. Fill ONLY if the paper explicitly states WHY the finding occurs (verbatim-traceable). If the paper only says WHAT happened, or you'd have to speculate → `null`. Do NOT introduce causal reasoning the authors did not express.
- `rationale_depth`: `"deep"` / `"shallow"` / `null` (null iff rationale is null).
  - `deep` — identifies a specific mechanism, causal chain, tradeoff, or root cause beyond restating the finding.
  - `shallow` — restates the finding in causal-sounding language without a specific mechanism (tautology / circular).

**CRITICAL — do NOT default to deep, do NOT invent rationales:**
- **Most papers do NOT give a real mechanism for most claims.** `null` and `shallow` are the common cases; `deep` is the exception.
- Set `rationale=null` whenever the paper only reports WHAT happened without an explicit WHY. Reporting a result is NOT a rationale.
- Only mark `deep` when you can point to a sentence where the authors state an actual mechanism/causal chain. If you are paraphrasing the finding itself into cause-language, that is `shallow`.
- **A batch where every experience is `deep` is wrong.** Across a paper, expect a mix — often several `null`, some `shallow`, at most one or two `deep`.

✅ deep: `rationale`="Policy-level reflection examines belief patterns across full trajectories, whereas action-level correction cannot link individual actions to delayed outcomes." / `rationale_depth`="deep"
✅ shallow: `rationale`="Retrieval helps because it gives the model access to relevant external knowledge." / `rationale_depth`="shallow"
✅ none: `rationale`=null / `rationale_depth`=null

### `evidence`
ARRAY of `{source_id, section, quote}`. One-to-many external provenance — all quotes here.
- `source_id`: paper_id.
- `section`: `abstract` / `introduction` / `method` / `experiment` / `results` / `discussion` / `conclusion`.
- `quote`: **≥150 chars verbatim**, double-quoted, covering the finding AND (if present) the causal explanation. Use ` [...] ` to join non-contiguous passages across sections.

At least one entry. When finding and explanation are in different sections, include both as separate entries or one joined quote.

---

## Section Guidance

| Section | Strategy |
|---|---|
| abstract | Extract only if the same claim is elaborated in discussion/conclusion |
| introduction | Author critiques of prior work, comparative judgments |
| method | Only design choices with explicit "we chose X because experiments showed Y" |
| experiment / results | Author interpretations ("This indicates...") — NOT raw data |
| discussion / conclusion | Primary source: generalizations, limitation analyses, failure attributions |

---

## Input Format

```
[paper_id] {paper_id}

{full paper in markdown}
```

---

## Output Format (Strict JSON)

Return exactly `{"paper_id": ..., "experiences": [ <experience>, ... ]}` using the flat 17-key structure shown in the OUTPUT CONTRACT at the top of this prompt.

If no experiences: `{"paper_id": "...", "experiences": []}`

---

## Self-Check Before Output

1. **Hard gate**: author voice + cross-instance + transferable? → else delete.
2. **statement ≥350 words, clean** (no quotes/"authors found"/source pointers)? → else fix.
3. **statement self-contained and not system-bound**? → else fix.
4. **claim_type** exactly one of property/relation/trend/conditional (relation vs trend distinction)? → else fix.
5. **applicable_when / not_applicable_when** both non-empty flat arrays, generalized, specific? → else fill.
6. **scope** a single sentence string (NOT an object)? → else fix.
7. **rationale**: string or null, author-stated (not speculation)? **rationale_depth** deep/shallow/null matching? → else fix.
8. **evidence** an array; each quote ≥150 chars, verbatim, covers finding (+ explanation if any)? → else fix.
9. **confidence** in [0,1], scored by evidence strength (not all 0.90)? → else re-score.
10. **utility** is null? → else set null.
11. **All 17 keys present and flat** (no nested objects except evidence entries)? → else fix.
12. **No padding** — count matches the paper's genuine claims (0 allowed)? If 10+, you're extracting data reports → prune to the strongest.
