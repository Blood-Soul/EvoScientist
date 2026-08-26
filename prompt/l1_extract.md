# L1 Practical Experience Extraction (v4)

## Output contract

Return only a JSON object with this shape:

```json
{"experiences": [{
  "domain": "agent_learning",
  "task": "specific task",
  "statement": "A clean, self-contained practical experience.",
  "applicable_when": ["specific setting"],
  "not_applicable_when": ["boundary or excluded setting"],
  "scope": "One sentence describing modality, scale, models and pipeline stage.",
  "action": "What was concretely done.",
  "effect": "Measured result, including numbers where reported.",
  "practice_trace": [{"action": "step", "feedback": "result"}],
  "evidence": [{"section": "experiment", "quote": "verbatim quote"}]
}]}
```

Every experience MUST contain exactly these 10 fields:

```text
domain, task, statement, applicable_when, not_applicable_when, scope,
action, effect, practice_trace, evidence
```

Do NOT output `id`, `layer`, `paper_id`, `domain_arxiv`, `utility`,
`confidence`, `source_id`, `created_at`, or extraction metadata. The runtime
injects and maintains those fields. Do not output prose or Markdown fences.

## What to extract

L1 records one concrete research practice: what researchers did in a specific
environment, for a goal, and what feedback resulted. Extract only practices the
paper genuinely reports; do not pad the list. A focused paper may produce one
or two records, while a rich empirical paper should stay near six or fewer.

- `domain`: concise lowercase research domain.
- `task`: specific capability or task, finer than the domain.
- `statement`: one clean, self-contained paragraph of at least 350 words. It
  must include the problem, procedure, conditions, concrete environment,
  outcomes, and boundaries. Do not write citations, source pointers, or “the
  authors found”; provenance belongs in `evidence`.
- `applicable_when` and `not_applicable_when`: non-empty arrays of specific
  settings, preconditions, and limitations.
- `scope`: one sentence covering modality, scale, backbone/models, and pipeline
  stage.
- `action`: operational essence of the practice.
- `effect`: measured result or feedback, with numbers when available.
- `practice_trace`: the core action→feedback chain. Use 3–6 corresponding
  `{action, feedback}` objects when the paper reports enough steps; omit only
  steps the paper does not support. Fine-grained practices should include
  numerical feedback.
- `evidence`: one or more `{section, quote}` objects. `section` is one of
  `abstract`, `introduction`, `method`, `experiment`, `results`, `discussion`,
  or `conclusion`. `quote` must be verbatim, at least 150 characters, and cover
  both what was done and what happened. Do not invent evidence.

Use the paper's terminology, datasets, models, metrics, hyperparameters and
limitations. Keep statements clean and put all provenance only in evidence.

## Input

The user message begins with `[paper_id] <id>` followed by the full paper in
Markdown. The paper ID is context only; do not repeat it in the output.

If no genuine practical experience is supported, return:

```json
{"experiences": []}
```
