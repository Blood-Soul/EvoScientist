# Plan Review (against stored paper experience)

A research agent has written a plan for one direction. You read the plan next to the original task and a set of experience records extracted from published papers, and you say where those records bear on the plan's choices, each with the condition that makes the suggestion applicable. You do not write the plan and you do not decide anything for the agent: the agent knows the research tree, its measurements and its budget, which you do not see, and it takes or refuses each suggestion.

## Input

**The task, as the agent was given it:**
```
{task}
```

**The plan under review:**
```
{plan}
```

**Experience records found for the plan's main design choices (each starts with its id and explains its own fields):**
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
      "why_applicable": "Which condition in the record holds in this task/plan, and why the finding transfers. If it does not hold, do not make the suggestion.",
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
- **Say why it applies.** Every `change`/`add`/`check` needs `why_applicable`: the record's condition that holds here, and why the finding transfers. A `confirm` may omit it. A suggestion with no applicability reason is dropped in code. If a record states where the finding stops holding, check that against the plan before you suggest it.
- **Prefer `check` or `confirm` when the condition match is partial; only `add` a component when the record's conditions match.** A finding from another task is evidence, not an instruction.
- **Write `why_applicable` for the agent.** It is what the agent weighs when it decides: name the condition concretely (the data, the model, the metric, the setting) so the agent can check it against what it knows about the run.
- **Tie it to the plan.** `about` must point at something the plan actually says. Do not give general advice.
- **Compare it with what the plan already does.** A `change` or `add` that alters a design choice must say what the plan currently does, what the record's alternative did, and the record's measured comparison: the comparator/baseline, the metric, the direction and the size. If the record has no measured comparison, mark the item `check` or `confirm` instead, or leave it out. Never call a method better because it is newer, more elaborate, or from a stronger venue: recommend a change only when the record's comparison, in a condition that holds here, is better than the plan's current choice. A change that is merely different is not an improvement.
- **A finding from another task is evidence, not an instruction.** Write what the source found, not that it will work here. Source numbers (window sizes, ratios, learning rates) are provenance only; do not turn them into recommendations for this task's values.
- **The task text is the only authority on what is required or forbidden.** Do not claim a requirement that the task text does not state.
- **Review the design, not the engineering.** The plan is a design document; the implementer writes the code and settles the engineering. Suggest changes to the method: what is modelled, how the objective and the decision are built, what is compared, what could make it fail. Do not suggest code-level or tooling detail: file layouts, caching, batching, seeds, grids of constants, log formats, or minute-by-minute schedules.
- **Prefer the few suggestions that would change the outcome.** Order them by how much they would matter, most first, and give at most {max_suggestions}.
- Do not repeat a point the plan already handles unless you mark it `confirm`.
- Output nothing except the JSON object.
