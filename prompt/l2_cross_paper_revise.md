# Cross-paper L2 revision

A cross-paper finding (L2) was induced from a group of experience records (L1) from several papers. New records from
other papers have joined the group. Decide what the new evidence does to the finding. You do **not** rewrite the
finding from scratch: you return one operation.

## Input

The user message gives the current finding (JSON, with `support` listing the record ids it rests on), the records it
already rests on, and the new records. Ids for all records are given; use them exactly.

## Operations

- `KEEP` — the new records are consistent with the finding and add nothing that changes it. The code adds them to
  `support` if you list them in `add_support`.
- `REVISE` — the new records change the finding: they widen or narrow its condition, add an option, add a stated
  limit, or give a reason. Return **only the fields that change**, each in full. Unchanged fields are kept as they are.
- `ADD_CONFLICT` — a new record disagrees with the finding. Return the conflict entry. The finding itself is not
  changed: a conflict is recorded next to it, never written over it.
- `RETIRE` — the new records show the finding is wrong as stated (not merely limited). Use rarely; say why.

## Output

Return only a JSON object:

```json
{"op": "REVISE",
 "add_support": ["r9"],
 "changes": {"action": "...", "limits_stated": [{"text": "...", "from": ["r1", "r9"]}]},
 "conflict": {"text": "...", "from": ["r3", "r9"]},
 "reason": "one sentence"}
```

`changes` only for `REVISE`; `conflict` only for `ADD_CONFLICT`. The same rules as induction apply to any field you
return: every `from` id must be a given record; numbers are copied exactly and never combined across papers;
`limits_stated` only summarizes the records' own stated limits (at least two records) and never comes from a cost; do
not mention papers or authors in text fields. A revised field may be shorter only when the new evidence narrows the
claim; do not drop conditions, options or limits that the earlier records still support.
