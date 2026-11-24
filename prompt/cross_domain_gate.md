You are judging whether a practice recorded from one field can be carried over to a design point in another field.

The design point is what the requester's work currently needs. The current situation, when given, says what the work
has to work with — data, compute, metric, constraints. Each candidate is a practice some paper tested in its own
setting, given as: field/task, the problem it answers, what was done, why it works (mechanism), and the measured effect.

A candidate passes only when all three of these can be written from the supplied text:

1. **Object mappings (at least two).** Functional correspondences between objects of the design point and objects of
   the candidate: what each does in its own system, not a shared name, topic or attribute. For each, say
   `design-point object -> candidate object: why they play the same role`.
2. **A shared relation.** One predicate that holds among the mapped objects in both settings, with every argument
   drawn from your object mappings. Write it as `PREDICATE(arg, arg)` on each side, the same predicate name both
   times. A property of one object ("is sparse") is not a relation.
3. **A transfer path.** Name the candidate's contribution (method, formulation, measurement or finding), how the
   mappings carry it into the design point, what concretely would change in the requester's work, and that the
   current situation can support it. "Could be useful" or "both involve X" is not a path.

Also reject when the problem is the same: a practice that solves the same task in another dataset, population,
language or setting is the same problem, not a transfer.

Reject a candidate when any of these holds, and say which:

- `same_problem` — it does the same thing in a different setting.
- `surface` — the connection is a shared topic, noun, metaphor or broad similarity.
- `no_mapping` — fewer than two functional object correspondences can be written from the supplied text.
- `no_relation` — no shared predicate with mapped arguments holds across both settings.
- `no_path` — the mappings hold but nothing concrete would change in the requester's work, or the current situation
  cannot support it (a prerequisite the work does not have).
- `tool_only` — the candidate would only be used as an unchanged tool, dataset or package.

Use only the supplied text. Do not use outside knowledge, do not invent mappings, and do not assume the candidate is
true or applicable beyond what it states. Returning fewer passing candidates is correct; returning none is fine.

Return only one JSON object, no prose and no Markdown fences:

{{"judgements": [
  {{"i": 1, "transferable": true,
    "object_mappings": [
      {{"source": "design-point object", "target": "candidate object", "rationale": "shared role"}}
    ],
    "shared_relations": [
      {{"relation": "PREDICATE_NAME", "problem": "PREDICATE_NAME(q_object_A, q_object_B)",
        "solution": "PREDICATE_NAME(d_object_A, d_object_B)"}}
    ],
    "path": "candidate's contribution -> mapped structure -> what would change in the work",
    "drop_reason": null}},
  {{"i": 2, "transferable": false, "object_mappings": [], "shared_relations": [], "path": "",
    "drop_reason": "surface"}}
]}}

Every candidate index must appear exactly once.

Design point:
{design_point}

Restated need:
{goal}

Current situation:
{context}

Candidates:
{candidates}
