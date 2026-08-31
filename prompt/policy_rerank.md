# Policy Candidate Reranker (v1)

## Task

You receive a research task and a list of numbered experience candidates retrieved by lexical search. Select {max_selected} experiences whose *transferable knowledge* best supports this task.

## Input

**Task:**
```
{task}
```

**Candidates (numbered, one per line):**
```
{descriptors}
```

## Selection criteria

1. **Relevance**: the procedure, decision pattern, or lesson applies to this task's domain, stage, and constraints.
2. **Complementarity**: selected experiences together cover different aspects (method, eval, constraints) rather than repeating the same finding.
3. **Confidence & evidence**: higher-confidence records with stronger evidence support safer reuse.
4. **Disagreement**: when two candidates contradict each other, select both and let the synthesis stage name the conflict — a disagreement handled explicitly is more useful than one candidate silently picked.

## Output contract

Return only JSON:

```json
{{
  "selected": ["E-abc123", "E-def456"],
  "reason": "One sentence explaining what these records collectively give the task."
}}
```

- `selected` must be a subset of the candidate IDs listed above, in priority order (most relevant first), length at most {max_selected}.
- `reason` is shown to the task planner as context for the resulting policy.
- Output nothing except this JSON object.

## Edge cases

- If no candidate offers a transferable procedure, return `{{"selected": [], "reason": "No candidate applies to the stated task."}}`.
- If only one candidate is relevant, that is valid. The target is utility, not hitting {max_selected}.
