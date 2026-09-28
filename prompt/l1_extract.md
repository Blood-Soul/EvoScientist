# L1 Practical Experience Extraction (v4)

## Output contract

Return only a JSON object with this shape:

```json
{"experiences": [{
  "discipline": "cs",
  "domain": "agent_learning",
  "task": "specific task",
  "trigger_context": "The obstacle that made this practice necessary, in the wording someone would use before knowing the fix.",
  "statement": "A clean, self-contained practical experience.",
  "applicable_when": ["specific setting"],
  "not_applicable_when": ["boundary or excluded setting"],
  "scope": "One sentence describing modality, scale, models and pipeline stage.",
  "action": "What was concretely done.",
  "effect": "Measured result, including numbers where reported.",
  "practice_trace": [{"action": "step", "feedback": "result"}],
  "evidence": [{"section": "experiment", "quote": "verbatim quote"}],
  "evidence_scope": {
    "datasets": ["ImageNet", "CIFAR-10"],
    "models": ["ResNet-50"],
    "has_ablation": true,
    "has_baseline_comparison": true,
    "verification_strength": "multi-setting-ablated"
  },
  "transferable_core": "The claim with every source-specific value stripped.",
  "bindings": [{"name": "ImageNet", "kind": "dataset"}]
}]}
```

Every experience MUST contain these 12 fields:

```text
domain, task, trigger_context, statement, applicable_when,
not_applicable_when, scope, action, effect, practice_trace, evidence,
evidence_scope
```

Three additional **optional** fields support downstream retrieval and reuse:

- `discipline`: the paper's broad field, exactly one of `cs`, `math`, `physics`,
  `chem`, `bio`, `med`, `materials`, `earth`, `econ`, `eng`, `other`. This is a
  deliberately coarse top-level facet used to keep a cross-disciplinary library
  from answering a chemistry question with computer-science records; `domain`
  remains the fine-grained subject. Judge it from the paper itself, not from
  where it was published. Omit the field if genuinely unclear rather than
  guessing -- the runtime then derives it from the arXiv category, and falls
  back to `other`, which stays browsable.

- `transferable_core`: ≤60 words, a paper-agnostic rephrasing of the claim
  suitable for semantic matching in multi-source contexts. Strip out all
  dataset names, model names, specific hyperparameters, and scale mentions, but
  keep the causal structure ("when X, doing Y yields Z"). For example, if the
  claim is "on ImageNet with ResNet-50, cosine LR schedule reduced overfitting
  by 3.2%", the core is "when training large vision models, using a cosine LR
  schedule reduces overfitting". Omit this field if the claim does not transfer
  meaningfully (e.g., a pure ablation of a single method's hyperparameters).

- `bindings`: an array of `{name, kind}` objects listing every source-fixed
  value in the claim. The `kind` must be one of: `dataset`, `model`, `scale`,
  `hyperparam`, `baseline`, `metric`, `toolchain`, `other`. For example:
  `[{"name": "ImageNet", "kind": "dataset"}, {"name": "ResNet-50", "kind":
  "model"}, {"name": "cosine schedule", "kind": "hyperparam"}]`. Omit this
  field if no source-fixed values appear (rare).

Do NOT output `id`, `layer`, `paper_id`, `domain_arxiv`, `utility`,
`confidence`, `source_id`, `created_at`, or extraction metadata. The runtime
injects and maintains those fields. Do not output prose or Markdown fences.

## What to extract

L1 records one concrete research practice: what researchers did in a specific
environment, for a goal, and what feedback resulted. Extract only practices the
paper genuinely reports; do not pad the list. A focused paper may produce one
or two records, while a rich empirical paper should stay near six or fewer.

- `domain`: concise lowercase research domain.
- `task`: specific capability or task, finer than the domain.
- `trigger_context`: one sentence, **at most 25 words**, naming the obstacle,
  difficulty, or open question that made this practice necessary. Write it the
  way someone would describe the situation *before* knowing the resolution --
  the symptom, not the remedy. Say "single-stage detectors miss small objects
  when the feature stride is large", not "adding an FPN improves small-object
  recall". Do not name the paper's own solution here. This field exists because
  a later agent searches with the problem it currently faces, not with the
  conclusion it has yet to reach; indexing only on outcome wording makes such
  records unreachable. It is a retrieval key, not a description: it is indexed
  in a length-capped high-weight field, so every word spent restating the
  setting or the motivation pushes out a word that another record's key needs.
  Name the problem and stop.
- `statement`: one clean, self-contained paragraph of at least 350 words. It
  must include the problem, procedure, conditions, concrete environment,
  outcomes, and boundaries. Do not write citations, source pointers, or “the
  authors found”; provenance belongs in `evidence`.
- `applicable_when` and `not_applicable_when`: non-empty arrays of specific
  settings, preconditions, and limitations.
- `scope`: one sentence covering modality, scale, backbone/models, and pipeline
  stage.
- `action`: operational essence of the practice.
- `effect`: measured result or feedback, with numbers when available.
- `practice_trace`: the core action→feedback chain. Use 3–6 corresponding
  `{action, feedback}` objects when the paper reports enough steps; omit only
  steps the paper does not support. Fine-grained practices should include
  numerical feedback.
- `evidence`: one or more `{section, quote}` objects. `section` is one of
  `abstract`, `introduction`, `method`, `experiment`, `results`, `discussion`,
  or `conclusion`. `quote` must be verbatim, at least 150 characters, and cover
  both what was done and what happened. Do not invent evidence.
- `evidence_scope`: how widely this specific claim was actually tested *in this
  paper*. An object with five keys:
  - `datasets`: array of every dataset this claim was evaluated on. Empty array
    if none is identifiable.
  - `models`: array of every model/backbone this claim was evaluated with.
  - `has_ablation`: `true` only when the paper isolates this claim's own
    variable (removes or varies it while holding the rest fixed). A paper with
    an ablation table that does not cover *this* claim is `false`.
  - `has_baseline_comparison`: `true` when the claim is supported against an
    external baseline, not only against the paper's own variants.
  - `verification_strength`: exactly one of `single-setting` (one dataset and
    one model, no ablation), `multi-setting` (more than one dataset or model,
    no ablation isolating this claim), `ablated` (this claim's variable is
    isolated, but within one setting), or `multi-setting-ablated` (both).

  Report only what the paper actually did for *this* claim. Do not inherit the
  paper's overall experimental scope: a broadly evaluated paper can still
  support an individual claim with a single observation. Understating here is
  far less harmful than overstating.

Use the paper's terminology, datasets, models, metrics, hyperparameters and
limitations. Keep statements clean and put all provenance only in evidence.

The two optional fields exist because a later agent must reuse these records on
a task with *different* datasets, models, and scales. Recording which values are
source-fixed (`bindings`) alongside what survives their removal
(`transferable_core`) lets that agent re-derive the values for its own setting
instead of copying yours. Populate them whenever the claim contains any
source-specific value; the record stays valid without them, but reuse degrades.

## Input

The user message begins with `[paper_id] <id>` followed by the full paper in
Markdown. The paper ID is context only; do not repeat it in the output.

If no genuine practical experience is supported, return:

```json
{"experiences": []}
```
