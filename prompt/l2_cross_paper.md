# Cross-paper L2 induction

You receive a group of experience records (L1) from **several different papers**. They were grouped because they
address the same kind of problem or compare the same kind of design choice. Write the **findings that hold across
these papers**: claims more general than any single record, that a researcher working on a different dataset or model
can use to decide what to try, when, and what it will cost.

This is the step from specific to abstract. Your wording does not have to appear in the records. But every claim must
rest on the records, and you must say which ones.

## Input

The user message gives the group's context sentence and a JSON array of records. Each record has an `id`, its
`paper`, and its fields: `task`, `problem`, `method`, `requires`, `source_setting`, `limits_stated` (limits the
authors stated, each with a verbatim quote), `action`, `effect` (`compared`, `benefit`, `cost`) and `comparison`
(`controlled` is an ablation inside one paper; `vs_baseline` compares different systems).

## What to look for

- **A shared mechanism**: two or more papers find the same kind of change helps (or fails) for the same reason.
- **A boundary**: one paper's result appears under a condition and another paper's result disappears without it.
  Use `source_setting` and `requires` to find the condition that differs between papers.
- **A trend**: the same kind of change tried at different scales, data sizes or settings, and the direction is the finding.
- **A trade-off**: what one paper gains, another reports as a cost.
- **A conflict**: papers disagree. Say so and, if the records allow, what differs between their settings. Do not
  smooth a conflict into a vague claim.

Prefer findings that change a decision. "Data quality matters" is not a finding.

## Output

Return only a JSON object:

```json
{"experiences": [{
  "domain": "concise lowercase research domain",
  "task": "the kind of task, finer than the domain",
  "problem": "the design question this answers, at most 20 words, phrased before the answer is known",
  "method": ["canonical name of the design choice", "alias", "..."],
  "action": "the finding as an instruction: what to do, under which condition, and what it buys. At most 70 words.",
  "compared_options": [
    {"option": "a design choice or variant", "result": "what happened for it, with the record's number if it gives one", "from": ["r1"]}
  ],
  "mechanism": "why, as supported by the records jointly. Empty string if they do not support a reason.",
  "requires": [{"text": "a condition that must hold for the finding to apply", "basis": "inferred", "from": ["r2", "r5"]}],
  "limits_stated": [{"text": "where the finding stops holding, summarized from the records' own limits_stated", "from": ["r1", "r4"]}],
  "conflicts": [{"text": "what the papers disagree on and, if known, what differs between their settings", "from": ["r1", "r6"]}],
  "support": ["r1", "r2", "r5"]
}]}
```

## Rules

- **Each finding must rest on records from at least two different papers.** List them in `support`. A finding one
  record already states belongs to that record, not here.
- **Every `from` and `support` id must be one of the given ids.** Entries citing anything else are removed.
- **Numbers**: copy them from the records' `effect` or `source_setting` exactly. Do not add, average, convert or
  extrapolate numbers from different papers. Different papers use different datasets; their numbers are not comparable
  unless the records say so.
- **`limits_stated`**: only from the records' own `limits_stated`, and each entry must draw on at least two records'
  limits. Never take a limit from `effect.cost` (a price is not a limit) and never derive one from the mechanism. Empty
  array when the records do not support one.
- **`requires`**: conditions that line up with where the effect appears across the records. `basis` is `stated` only
  when a cited record states the condition in words; otherwise `inferred`.
- **`mechanism`**: only a reason the records jointly support. Do not import general knowledge about the technique.
- **`conflicts`**: required whenever the records disagree. Empty array when they do not.
- Weigh `controlled` evidence above `vs_baseline`: a cross-system comparison changes many things at once.
- At most 3 findings per group. Return `{"experiences": []}` when the records support nothing beyond what each
  already says.
- Do not mention papers, authors or "the study" in the text fields: write the finding, not the provenance.
