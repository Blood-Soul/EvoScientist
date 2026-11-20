# Cross-paper grouping of experience records

You receive short experience records extracted from **different papers**. Each record is one practice a paper tested.
Group the records that answer **the same kind of problem** or **compare the same kind of design choice**, so that a
later step can induce a finding that holds across papers.

## Input

The user message is a JSON array of records:

```json
[{"id": "r1", "paper": "2101.00438", "task": "...", "problem": "...", "method": ["..."], "action": "..."}]
```

## What makes a group

- Records belong together when a practitioner facing one of them would want to read the others before deciding:
  the same symptom ("most windows contain no answer"), the same decision ("independent vs joint span endpoints"),
  or the same family of technique applied to the same kind of problem.
- Sharing a word, a dataset or a model is not enough. Two records about "pretraining" that solve unrelated problems
  are not a group.
- A group must contain records from **at least two different papers**.
- A record may stay ungrouped. Leaving a record out is better than forcing it into a weak group.
- A record belongs to at most one group.

## Output

Return only a JSON object:

```json
{"groups": [
  {"members": ["r1", "r4", "r7"],
   "context": "One sentence: the situation in which these records are relevant, phrased so a practitioner can tell whether they are in it."}
]}
```

`members` uses the ids exactly as given. Return `{"groups": []}` when no records belong together.
