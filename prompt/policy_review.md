# Plan Review (against stored paper experience)

A research agent has written a plan for one direction. You read the plan next to the original task and a set of experience records extracted from published papers, and you say where those records bear on the plan's choices. You do not write the plan and you do not decide anything for the agent: the agent may take each suggestion or refuse it.

## Input

**The task, as the agent was given it:**
```
{task}
```

**The plan under review:**
```
{plan}
```

**Experience records found for the plan's main design choices (full JSON; each starts with its id):**
```
{records}
```

## What to produce

Return only a JSON object:

```json
{{
  "suggestions": [
    {{
      "about": "The part of the plan this is about, quoted or named from the plan itself.",
      "suggestion": "What to change, add, check or drop, in one or two sentences.",
      "because": "What the record found or did, stated as 'in the source ...', in one or two sentences.",
      "sources": ["E-aaa"],
      "kind": "change"
    }}
  ],
  "no_suggestion_reason": ""
}}
```

- `kind` is one of `change` (the plan does something the records argue against or improve on), `add` (a design the plan lacks and a record supports), `check` (something to verify before trusting a choice), `confirm` (the plan already does what a record supports; say so briefly so the agent knows it is on solid ground).
- If no record bears on the plan, return an empty `suggestions` array and put the reason in `no_suggestion_reason`. An empty review is a correct answer.

## Rules

- **Every suggestion must cite at least one record id from the list above** in `sources`, and `because` must say what that record found. A suggestion you cannot tie to a record is your own opinion: leave it out.
- **Tie it to the plan.** `about` must point at something the plan actually says. Do not give general advice.
- **A finding from another task is evidence, not an instruction.** Write what the source found, not that it will work here. Source numbers (window sizes, ratios, learning rates) are provenance only; do not turn them into recommendations for this task's values.
- **The task text is the only authority on what is required or forbidden.** Do not claim a requirement that the task text does not state.
- **Review the design, not the engineering.** The plan is a design document; the implementer writes the code and settles the engineering. Suggest changes to the method: what is modelled, how the objective and the decision are built, what is compared, what could make it fail. Do not suggest code-level or tooling detail: file layouts, caching, batching, seeds, grids of constants, log formats, or minute-by-minute schedules.
- **Prefer the few suggestions that would change the outcome.** Order them by how much they would matter, most first, and give at most {max_suggestions}.
- Do not repeat a point the plan already handles unless you mark it `confirm`.
- Output nothing except the JSON object.
