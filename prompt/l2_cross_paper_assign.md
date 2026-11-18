# Cross-paper assignment of new experience records

Existing groups of experience records each hold records from several papers that address the same kind of problem
or compare the same kind of design choice. New records have arrived. For each new record, decide which existing
group it belongs to, or that it belongs to none.

## Input

The user message is a JSON object:

```json
{"groups": [{"id": "g1", "context": "...", "examples": ["problem of a member", "..."]}],
 "new": [{"id": "n1", "paper": "...", "task": "...", "problem": "...", "method": ["..."], "action": "..."}]}
```

## Rule

A new record belongs to a group when a practitioner in the group's context would want to read it before deciding:
the same symptom, the same decision, or the same family of technique applied to the same kind of problem. Sharing a
word, a dataset or a model is not enough. When unsure, answer `null`: an unassigned record is kept and may form a new
group later.

## Output

Return only a JSON object mapping every new id to a group id or `null`:

```json
{"assign": {"n1": "g1", "n2": null}}
```
