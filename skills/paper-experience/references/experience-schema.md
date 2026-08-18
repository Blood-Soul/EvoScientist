# L1 / L2 Experience Schema

Field reference for the experiences returned by `extract_paper_experiences`. Read
this when you need to interpret, summarize, or reformat the tool's output. The tool
already renders readable Markdown — you rarely need to touch raw fields.

## Two layers

| Layer | Question it answers | Count per paper |
|---|---|---|
| **L1 Practical** | In some environment, for some goal, what actions were taken and what feedback resulted? | 5–7 (1 coarse + 1 medium + 3–6 fine) |
| **L2 Inductive** | In some context, what generalizable assertion can be made? | 3–8 |

## L1 fields

| Field | Meaning |
|---|---|
| `granularity` | `coarse` (abstract-level) / `medium` (intro-level) / `fine` (specific experiment) |
| `narrative` | ≥450-word self-contained account: background, practice, conditions, outcomes, limits |
| `t.summary` | One line: what task + what result (with numbers) |
| `t.modality` / `t.scale` / `t.constraint` | Data type / scale / validity boundary |
| `e` | Concrete environment: datasets, models, baselines, metrics, hardware, hyperparameters |
| `practice_trace` | `[{action, feedback}]` — what was done → what resulted (numbers for fine) |
| `domain` | Agent category (`agent_learning`, `agent_domain_app`, ...) |
| `keywords` | Paper keywords (verbatim or generated) |
| `source_section` / `source_quote` | Provenance: section label + ≥150-char verbatim quote |

## L2 fields

| Field | Meaning |
|---|---|
| `narrative` | ≥450-word account: background, finding, conditions, evidence, causal mechanism |
| `declaration` | The inductive assertion (system-name-free, category terms) |
| `claim_type` | `property` / `relation` / `trend` / `conditional` |
| `context.summary` | One line: task + result the experience helps with |
| `context.modality/scale/constraint` | Applicability context |
| `μ` | Confidence in the declaration: `high` / `medium` / `low` |
| `r` | Causal explanation (verbatim from paper, else null) |
| `μ_r` / `r_depth` | Confidence in `r` / mechanism depth (`deep`/`shallow`) |
| `source_section` / `source_quote` | Provenance |

## claim_type quick guide

- **property** — what something IS like ("Statelessness of LLMs blocks multi-step consistency.")
- **relation** — static A vs B comparison ("GRPO converges faster than PPO, but PPO is more stable.")
- **conditional** — when X, then Y ("When labeled data <1000, pretrain+finetune beats from-scratch.")
- **trend** — as X changes, Y changes directionally ("More few-shot demos can reduce performance for context-limited agents.")
