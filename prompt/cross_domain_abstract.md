You are given a description of what a piece of research currently needs — a design point: a difficulty, a constraint,
or a decision the work has to make.

Write two things about it.

1. `goal`: the need restated as a single question in the form "How to ...", stating only what must be achieved.
   - Remove the vocabulary of the requester's own field wherever its meaning survives without it.
   - Do not name a method, mechanism, model, algorithm or dataset.
   - Do not add anything the description does not imply.
   - Keep the conditions that make this need specific (what is constrained, what must hold), because they are what
     lets a practice from another field be recognised as fitting.

2. `passages`: {passages} short descriptions of what a practice FROM ANOTHER FIELD that meets this need would look
   like. Each is 2-3 sentences, written the way a paper in that field would describe its own practice.
   - Each passage must come from a different field than the one the requester works in, and the fields must differ
     from each other. Name the field at the start of the passage, in brackets, e.g. "[ecology] ...".
   - Use that field's own vocabulary for the objects and relations it works with; do not translate it back into the
     requester's terminology.
   - Describe a concrete practice, not a generic virtue. "Use a better loss" is not a practice; "measure the flow of
     individuals between patches before deciding where to sample" is.
   - These passages are hypotheses for finding candidate practices. They are not advice to the requester and nothing
     in them is asserted to be true.

Return only one JSON object, no prose and no Markdown fences:

{{"goal": "How to ...", "passages": ["[field] ...", "..."]}}

Design point:
{design_point}
