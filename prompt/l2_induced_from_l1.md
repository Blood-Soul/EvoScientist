# L2 Induction from L1 Trajectories (v1)

## Input

The user message begins with `[paper_id] <id>` followed by a JSON array of
this paper's own L1 records (the trajectory-level practices already
extracted from it):

```json
[{"task": "...", "trigger_context": "...", "statement": "...",
  "applicable_when": [...], "not_applicable_when": [...],
  "action": "...", "effect": "...",
  "practice_trace": [{"action": "...", "feedback": "..."}],
  "evidence": [{"section": "...", "quote": "..."}],
  "evidence_scope": {"datasets": [...], "models": [...],
    "has_ablation": true, "has_baseline_comparison": true,
    "verification_strength": "..."}, "...": "..."}]
```

You do not see the paper's raw text. Your only source is these L1 records.
Do not invent claims, conditions, or numbers that are not already present in
them.

## What to do

This is not a second, independent reading of the paper — it is induction
**over** the L1 records you were given. Only proceed when there is something
to induce:

- If only one L1 record is given, and its `applicable_when`/`not_applicable_when`
  already state the same boundary the record's own `statement` implies, there is
  nothing to add by restating it as L2. Return `{"experiences": []}` in that case.
- With two or more L1 records, look for: a pattern that recurs across their
  `action`/`effect`/`practice_trace` entries, a boundary condition one record
  makes explicit that qualifies another, or a common failure mode across their
  practice traces. The induced claim must say something none of the individual
  L1 records says on its own.

Apply the same three gates as direct L2 extraction:

1. It is an interpretation or generalization, not a restatement of one L1
   record's `statement`.
2. It has scope across more than one L1 record's specific setting (or, for a
   single sufficiently rich L1 record, across more than one of its
   `practice_trace` steps).
3. It remains useful after removing the paper's system name.

## Output contract

Return only a JSON object with this shape:

```json
{"experiences": [{
  "discipline": "cs",
  "domain": "agent_planning",
  "task": "specific capability",
  "trigger_context": "The open question the contributing L1 records jointly raise, in pre-solution wording.",
  "statement": "A clean, self-contained inductive experience.",
  "claim_type": "conditional",
  "applicable_when": ["generalized setting"],
  "not_applicable_when": ["boundary or excluded setting"],
  "scope": "One sentence describing the validity boundary.",
  "action": "Actionable implication.",
  "effect": "Observed or expected result.",
  "rationale": "Reason grounded in the L1 records, or null.",
  "rationale_depth": "deep",
  "evidence": [{"section": "l1_practice_trace", "quote": "verbatim text copied from one of the input L1 records"}],
  "evidence_scope": {
    "datasets": ["union of the contributing L1 records' datasets"],
    "models": ["union of the contributing L1 records' models"],
    "has_ablation": true,
    "has_baseline_comparison": true,
    "verification_strength": "multi-setting"
  },
  "source_l1_ids": ["l1_..._01", "l1_..._02"],
  "transferable_core": "The claim with every source-specific value stripped.",
  "bindings": [{"name": "GPT-4", "kind": "model"}]
}]}
```

Every experience MUST contain these 15 fields:

```text
domain, task, trigger_context, statement, claim_type, applicable_when,
not_applicable_when, scope, action, effect, rationale,
rationale_depth, evidence, evidence_scope, source_l1_ids
```

`source_l1_ids` is the one field this prompt adds beyond direct L2 extraction:
a non-empty array listing the `id` of every input L1 record that
this induced claim actually draws on. List only records that genuinely
contributed — not every record you were given.

Two more **optional** fields, same meaning as in direct L2 extraction:
`discipline` and `transferable_core`/`bindings` (see direct L2 prompt for the
exact rules; unchanged here).

Do NOT output `id`, `layer`, `paper_id`, `domain_arxiv`, `utility`,
`confidence`, `source_id`, `created_at`, or extraction metadata. The runtime
injects and maintains those fields. Do not output prose or Markdown fences.

## Field notes specific to this induction step

- `evidence`: since you have no raw paper text, each `quote` must be copied
  verbatim from an input L1 record's own `statement`, `action`, `effect`, or
  `evidence[].quote` field — not paraphrased. Use `section: "l1_practice_trace"`
  when quoting from `practice_trace`, or reuse the original L1 evidence
  `section` value when quoting from that record's own `evidence`.
- `rationale`: only state a mechanism if it is actually supported by comparing
  the L1 records (e.g., "both records report the same failure when X is
  absent, and neither reports it when X is present"). If you cannot point to
  that kind of cross-record support, set `rationale` to `null` and
  `rationale_depth` to `null`.
- `not_applicable_when`: preserve every boundary/failure condition already
  present in the contributing L1 records' own `not_applicable_when` or
  practice_trace failure steps. Losing a documented boundary during induction
  is a defect, not an acceptable simplification.

  But include only genuine *invalidating* conditions: a condition belongs here
  when the induced claim **stops holding** once it is true. A difficulty someone
  might hit while reproducing the work does not belong here. "The architecture
  has no intermediate feature maps to attach an auxiliary head to" invalidates
  the claim and belongs. "Batch size 8 at 256x256 exceeds available GPU memory"
  is a reproduction constraint, not a failure condition, and does not belong --
  the claim still holds for anyone with a larger GPU. Prefer a short list of
  genuinely invalidating conditions over an exhaustive list of obstacles; a bloated
  list produces spurious exclusions when the record is matched against a new
  task.

- `trigger_context`: one sentence, **at most 25 words**, naming the open
  question that the contributing L1 records *jointly* raise but none answers
  alone. This is the problem the induced claim resolves, phrased before its
  resolution is known. Do not restate a single L1 record's own trigger. It is a
  retrieval key, not a description: it is indexed in a length-capped high-weight
  field, so every word spent restating the setting pushes out a word that
  another record's key needs. Name the question and stop.

- `evidence_scope`: merge the contributing L1 records' own `evidence_scope`
  values -- union the `datasets` and `models` arrays, and set `has_ablation` /
  `has_baseline_comparison` to `true` when any contributing record has it.
  Recompute `verification_strength` from the merged result rather than copying
  it from one record. Induction can widen the *reported* setting count because
  it draws on several records, but it must not invent coverage: never list a
  dataset or model that does not appear in some contributing record's own
  `evidence_scope`. If the input L1 records lack `evidence_scope` (older
  records), set `datasets`/`models` to empty arrays, both booleans to `false`,
  and `verification_strength` to `single-setting`.

If no claim passes all three gates, return:

```json
{"experiences": []}
```
