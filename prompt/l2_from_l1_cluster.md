# L2 induction from a cluster of L1 records

You receive a cluster of experience records (L1). Each records one practice a paper tested and what happened. They
were clustered because they address the same kind of problem or the same kind of design choice; they may come from
one paper or from several. Find **what these records have in common** and write it down as findings: claims more
general than any single record, that a researcher on a different dataset or model can use to decide what to try,
when, and at what cost.

An L1 record is one observation tied to its setting. A finding is the pattern several observations share. Your
wording does not have to appear in the records, but every claim must rest on them, and you must say which ones.

## Input

The user message is a JSON object. `records` is an array of records; each has an `id`, its `paper`, and its fields:
`task`, `problem`, `method`, `requires`, `source_setting`, `limits_stated` (limits the authors stated), `action`,
`effect` (`compared`, `benefit`, `cost`), `mechanism` (`text`, and `basis`: `stated` by the paper or `inferred` by the
extractor) and `comparison` (`controlled` is an ablation inside one paper; `vs_baseline` compares different systems).

`current_findings`, when present, holds the findings written for this cluster before some of its records arrived,
with their `support` and `from` ids mapped to the record ids given here. Then you are **updating**, not starting
over:

- Keep each current finding whose claim the records still support, word for word, and add the new records that bear
  it out to its `support`. Do not rephrase a claim the evidence has not changed.
- Revise a finding only where the new records change it: a condition to widen or narrow, an option, a stated limit,
  a reason. Keep everything else in it, including the conditions, options and limits the earlier records support.
- A new record that contradicts a finding goes into that finding's `conflicts`; the finding is not overwritten.
- Add a finding only for a pattern the current ones do not cover. Drop a finding only when the records show it is
  wrong as stated, not merely limited.

Return the full set of findings for the cluster: the kept, the revised and the new.

## What to look for

- **A shared mechanism**: several records find the same kind of change helps (or fails) for the same reason. A
  `stated` mechanism is stronger evidence than an `inferred` one.
- **A boundary**: the effect appears under one condition and disappears without it. Use `source_setting` and
  `requires` to find the condition that differs between records.
- **A trend**: the same kind of change at different scales, data sizes or settings; the direction is the finding.
- **A trade-off**: what one record gains, another reports as a cost.
- **A conflict**: records disagree. Say so and, if the records allow, what differs between their settings. Do not
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
  "conflicts": [{"text": "what the records disagree on and, if known, what differs between their settings", "from": ["r1", "r6"]}],
  "support": ["r1", "r2", "r5"]
}]}
```

## Rules

- **Each finding rests on at least three records**, listed in `support`. A claim that one or two records already
  state belongs to them, not here.
- **Every `from` and `support` id must be one of the given ids.** Entries citing anything else are removed.
- **Numbers**: copy them from the records exactly. Do not add, average, convert or extrapolate numbers from different
  records. Different experiments use different datasets; their numbers are not comparable unless the records say so.
- **`limits_stated`**: only from the records' own `limits_stated`, and each entry must draw on at least two records'
  limits. Never take a limit from `effect.cost` (a price is not a limit) and never derive one from the mechanism.
  Empty array when the records do not support one.
- **`requires`**: conditions that line up with where the effect appears across the records. `basis` is `stated` only
  when a cited record states the condition in words; otherwise `inferred`.
- **`mechanism`**: only a reason the records jointly support. Do not import general knowledge about the technique.
- **`conflicts`**: required whenever the records disagree. Empty array when they do not.
- Weigh `controlled` evidence above `vs_baseline`: a cross-system comparison changes many things at once.
- At most 3 findings. Return `{"experiences": []}` when the records support nothing beyond what each already says.
- Do not mention papers, authors or "the study" in the text fields: write the finding, not the provenance.
