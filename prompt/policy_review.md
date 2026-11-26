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
- **Say why it applies.** Every `change`/`add`/`check` needs `why_applicable`: explain how the plan's current method corresponds to the record's *before* method (the practice it compared against) and what would carry over. Use your own understanding; exact field names, dataset, scale, metric or mechanism do not have to match. A `confirm` may omit it. A suggestion with no applicability reason is dropped in code.
- **Prefer `check` or `confirm` when the method match is vague; only `add` or `change` when the plan and the record are roughly the same kind of design at the same component level.** A finding from another task is evidence, not an instruction.
- **Write `why_applicable` for the agent.** It is what the agent weighs when it decides: name the part of the plan's method and the part of the record's before method that correspond. Do not require a shared metric, dataset or wording.
- **Tie it to the plan.** `about` must point at something the plan actually says. Do not give general advice.
- **Compare the methods, not the numbers.** A `change` or `add` may be suggested when the plan's current method and the record's before method are roughly the same design or component, so the record's after method is a plausible improvement. You do not need the same metric, split, scale, mechanism or measured effect, and you do not need to run the current plan as a control or an A/B test. Report a measured effect only as helpful context; never claim it will transfer. If the methods are not similar, do not propose the change. For a cross-field record, judge whether the objects and relations correspond; a plausible mapping is enough.
- **A finding from another task is evidence, not an instruction.** Write what the source found, not that it will work here. Source numbers (window sizes, ratios, learning rates) are provenance only; do not turn them into recommendations for this task's values.
- **The task text is the only authority on what is required or forbidden.** Do not claim a requirement that the task text does not state.
- **Review the design, not the engineering.** The plan is a design document; the implementer writes the code and settles the engineering. Suggest changes to the method: what is modelled, how the objective and the decision are built, what is compared, what could make it fail. Do not suggest code-level or tooling detail: file layouts, caching, batching, seeds, grids of constants, log formats, or minute-by-minute schedules.
- **Prefer the few suggestions that would change the outcome.** Order them by how much they would matter, most first, and give at most {max_suggestions}.
- Do not repeat a point the plan already handles unless you mark it `confirm`.
- Output nothing except the JSON object.
