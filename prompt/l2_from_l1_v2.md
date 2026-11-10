# L2 Induction from a Paper's L1 Records (v2)

## Input

The user message begins with `[paper_id] <id>` followed by a JSON array of this paper's own L1 records — the practices
it tested, each with a measured outcome:

```json
[{"id": "l1-1",
  "domain": "...", "task": "...", "problem": "...", "method": ["..."],
  "requires": [{"text": "...", "basis": "stated"}],
  "source_setting": {"environment": [{"property": "...", "value": "..."}],
                     "params": [{"name": "...", "value": "...", "basis": "...", "from": "..."}]},
  "limits_stated": [{"text": "...", "quote": "..."}],
  "action": "...",
  "effect": {"compared": "...", "benefit": "...", "cost": "..."},
  "comparison": "controlled", "locator": "Table 3"}]
```

You do not see the paper. These records are your only source. Do not add a claim, condition or number that is not
already in them.

## What to induce

This is induction **over** the records, not a second reading. With two or more records, look for:

- **a shared mechanism** — two or more records improve in the same way, or fail in the same way, and the same cause
  explains both, while no single record says so;
- **a boundary** — one record establishes the condition under which another record's effect appears or disappears
  (one works on unanswerable-heavy data, another stops working once the null share drops);
- **a trend across settings** — the same change was tried at several values or datasets, and the *direction* or the
  turning point is the finding;
- **a trade-off** — one record gains what another loses, and the pair says something neither says alone;
- **a failure pattern** — the same approach failed for the same reason in two different records.

Only write a record when the claim needs **at least two** of the input records, or when one rich record states a
condition that its own result does not explain. Restating one record — even in better words — is a defect. If there is
nothing to induce, return:

```json
{"experiences": []}
```

## Output contract

Return only a JSON object with this shape:

```json
{"experiences": [{
  "domain": "concise lowercase research domain",
  "task": "specific task, finer than the domain",
  "problem": "the design question this finding answers, at most 20 words, phrased before the answer is known",
  "method": ["canonical name of the design choice", "alias", "..."],
  "action": "the induced claim as an instruction: what to do, when, and what it buys. At most 60 words.",
  "compared_options": [
    {"option": "a design choice or variant the records compare", "result": "what happened for it", "from": "l1-2"}
  ],
  "mechanism": "why, as supported by comparing the records. Empty string if they do not support a reason.",
  "requires": [{"text": "a condition that must hold for the finding to apply", "basis": "inferred"}],
  "limits_stated": [{"text": "a condition in which the finding stops holding", "basis": "inferred"}],
  "origin": "induced",
  "source_l1_ids": ["l1-1", "l1-3"]
}]}
```

All keys must be present. `compared_options`, `requires` and `limits_stated` may be empty arrays; `mechanism` may be
`""`. `source_l1_ids` must be non-empty and must list only records that genuinely contributed — not every record you
were given. The example marks both conditions `inferred`, the usual case; `stated` is the exception, used only where a
contributing record argues the condition itself.

## Field rules specific to induction

**`source_l1_ids`** — the `id` strings exactly as they appear in the input. Never invent an id and never cite a record
whose content the claim does not use.

**`compared_options`** — every entry must come from a record; `from` names the record it came from. If the records
compare three settings of one knob, that is one finding with three options, not three findings. Numbers must be copied
from the source records' `effect`, not recomputed or combined (do not add, average or extrapolate two effect sizes).

**`mechanism`** — only a reason the records jointly support: "both records report the same failure when X is absent,
and neither reports it when X is present". Do not import the field's general knowledge about why a technique works.
If you cannot point to that cross-record support, use `""`.

**`requires`** — take it from the contributing records' own `requires` and `source_setting`: a condition that is true
in some records and false in others, and that lines up with where the effect appears. Do not invent a prerequisite.
`basis` is `stated` only when a contributing record says in words that the condition matters; a condition that merely
follows from how the technique works, or from the settings the records ran in, is `inferred`. **Most entries should be
`inferred`; when in doubt, use it.**

**`limits_stated`** — take it from the contributing records' `limits_stated` or from the cost side of their `effect`.
A limit is a condition in which the finding stops holding, not a reproduction difficulty. `basis` is `stated` only
when a contributing record reports it; otherwise, including a limit that follows from the mechanism above, `inferred`.

**`action`** — say what a reader should do differently, not what the paper did. If the induced claim only repeats the
action of a contributing record, there is nothing induced: drop it.

## Gates

Before writing a record, check all three; if any fails, drop the record.

1. It is an interpretation or generalization, not a restatement of one L1 record's `action`.
2. Its scope is wider than one contributing record's specific setting — or, when it rests on one record, wider than
   one step of that record's own comparison.
3. It remains useful after the paper's system name, dataset and model are removed.

Every number must appear in a contributing record. Use the records' terminology; do not mention the paper, its
authors or "the study".

Return only the JSON object.
