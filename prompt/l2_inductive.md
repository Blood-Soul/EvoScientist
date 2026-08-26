# L2 Inductive Experience Extraction (v4)

## Output contract

Return only a JSON object with this shape:

```json
{"experiences": [{
  "domain": "agent_planning",
  "task": "specific capability",
  "statement": "A clean, self-contained inductive experience.",
  "claim_type": "conditional",
  "applicable_when": ["generalized setting"],
  "not_applicable_when": ["boundary or excluded setting"],
  "scope": "One sentence describing the validity boundary.",
  "action": "Actionable implication.",
  "effect": "Observed or expected result.",
  "rationale": "Author-stated reason, or null.",
  "rationale_depth": "deep",
  "evidence": [{"section": "discussion", "quote": "verbatim quote"}]
}]}
```

Every experience MUST contain exactly these 12 fields:

```text
domain, task, statement, claim_type, applicable_when,
not_applicable_when, scope, action, effect, rationale,
rationale_depth, evidence
```

Do NOT output `id`, `layer`, `paper_id`, `domain_arxiv`, `utility`,
`confidence`, `source_id`, `created_at`, or extraction metadata. The runtime
injects and maintains those fields. Do not output prose or Markdown fences.

## What to extract

L2 expresses a transferable author-stated generalization, comparative judgment,
or pattern that goes beyond a single result. Apply all three gates:

1. It is an interpretation or generalization, not merely a data report.
2. It has cross-instance scope beyond one isolated experiment.
3. It remains useful after removing the paper's system name.

Extract only genuine claims; zero is valid and the soft cap is about six.

- `domain`: concise lowercase research domain.
- `task`: capability or task to which the claim applies.
- `statement`: one clean, self-contained paragraph of at least 350 words. State
  the finding directly and include conditions, boundaries, evidence pattern,
  representative numbers, and a causal mechanism only when the paper states
  one. Do not include citations or source pointers.
- `claim_type`: exactly `property`, `relation`, `trend`, or `conditional`.
- `applicable_when` and `not_applicable_when`: non-empty generalized conditions.
- `scope`: one sentence covering modality, scale, models tested, and stage.
- `action` and `effect`: actionable implication and observed consequence.
- `rationale`: one author-supported causal explanation, or `null` when absent.
- `rationale_depth`: `deep` when the mechanism is developed, `shallow` when
  only briefly stated, and `null` exactly when rationale is `null`.
- `evidence`: one or more `{section, quote}` objects. `section` is one of
  `abstract`, `introduction`, `method`, `experiment`, `results`, `discussion`,
  or `conclusion`. `quote` must be verbatim, at least 150 characters, and cover
  the claim plus its supporting finding. Do not invent evidence.

The runtime calculates initial evidence confidence and later aggregation may
raise or lower it using independent supporting and contradicting papers.

## Input

The user message begins with `[paper_id] <id>` followed by the full paper in
Markdown. The paper ID is context only; do not repeat it in the output.

If no claim passes all three gates, return:

```json
{"experiences": []}
```
