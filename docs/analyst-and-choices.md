# Analyst, open choices and the order of work in a tree run

Branch `feat/analyst-choices` (from local-dev `22550dd`), 2026-11-20. What changes and why.

## What was wrong

- **"Do this yourself or leave it to node-analyst"** (main prompt, `RESEARCH_TREE_REPORTS`). Since `61b77b3`
  (11-06) the analysis of a finished node is optional, so the main agent skips it and asks plan-writer directly.
  r38: 6 refine plans, 1 written by node-analyst.
- **node-analyst does two jobs** (analyse a node and write the next plan), so whatever it finds lives only at the
  top of one plan. Later plans cannot read it.
- **Undecided choices are lost.** A plan-writer is a fresh sub-agent every time. A choice a review raised and the
  writer did not settle (an alternative, a deferred design change) is recorded nowhere. r38 v2 n1 took an untested
  design change into the baseline itself (random gold occurrence) and every later node inherited a measured -1.2
  point loss (`/root/work/ablate`).
- **Why the analysis was made optional** (`61b77b3`): in r11 a mandatory analysis was a gate on refine nodes, and
  the tree stalled. The fix here keeps the analysis mandatory *as work* but never a gate: it runs in the
  background and nothing waits on it except the plan that builds on that node.

## The roles after the change

| | does | writes | reads |
|---|---|---|---|
| main agent | reads the record, picks the next step, decides plans, opens and closes nodes | decisions | tree, analyses, choices, plans |
| node-analyst | analyses ONE finished node: what it got right, what is wrong and why, measured on what the node saved; settles inference-only questions on the saved outputs; checks the literature for the weak points | `reports/analysis/<node>.md`, recorded with `tree_analysis`; open choices with `tree_choice` | the node's record and files |
| plan-writer | writes ONE plan: a new direction, or a step that builds on a node (then from that node's analysis and its open choices) | the plan; open choices with `tree_choice` | tree, the parent's analysis, choices |
| code-agent | builds a node | the node | the brief |

plan-writer no longer has a "refine without analysis" path: a plan that builds on a node is written after that
node's analysis exists. node-analyst writes no plan.

## Open choices

A choice the run has not settled: two or more ways to do one thing, a default (what the code does now), and why it
is open. Stored in the tree state (`choices`), shown by `tree_status`, carried into the brief of every plan and node
that starts from the node it is attached to.

- **Who records it:** node-analyst (what the node's failures suggest, what the review suggested and the evidence
  could not settle), plan-writer (a review suggestion that would change the design and cannot be checked on the
  data before training: the plan keeps the default; the other option becomes a choice).
- **Settled cheaply when it can be:** a choice that only changes inference (a threshold, a decision rule, an
  aggregation) is settled by node-analyst on the parent's saved outputs and recorded as settled with the numbers.
- **Otherwise it waits for a node:** a choice that needs training is tried as a refine node that changes only that
  one thing from the node it is attached to. Its plan cites the choice; `tree_finish` of that node settles it.
- **Default stays until a node shows otherwise.** An untested design change is not folded into a plan.
- **Order:** each choice carries the expected effect (low/medium/high) and the cost (inference-only / one short
  training / a full node). `tree_status` lists open choices by that; the main agent decides.

## Order of work (what runs in parallel, what waits)

One GPU node at a time (unchanged). Everything else may overlap, through `task_start` / `task_collect`:

| when | start in the background | must wait for |
|---|---|---|
| a node closes (`tree_finish`) | node-analyst on that node | nothing |
| while a node trains | plan-writer for a new direction; node-analyst on a node that closed earlier | nothing |
| a plan that builds on node X | plan-writer for it | **X's analysis** (the tree refuses `tree_plan` with parents=[X] until it exists) |
| the GPU is free and an accepted plan exists | open the node on it | nothing; do not wait for an analysis you do not need for that plan |

`tree_finish` says, in its reply, that the node's analysis is due and that it runs in the background. The hint
`_next_step` names analyses that are due and plans that are waiting for one. The refusal in `tree_plan` names the
missing analysis and how to get it.

## What is not decided by experiment

- Whether a mandatory analysis lowers the number of nodes in 6 hours. r38: node-analyst took 6.2 min, plan-writer
  1.5-32 min. Running in the background is meant to hide it; not measured.
- Whether plans built on an analysis are better. Design intent, not measured.
- Whether the expected-effect / cost ranking helps the order. The agent's own estimate; not validated.
