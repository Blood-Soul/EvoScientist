# Experience Extraction (v2: one level, comparison-centred)

You read one paper and write down the **practices it tested**, as records a later research agent can search, judge and
reuse. A record is worth writing when the paper *compares* something: it changes a design choice, a parameter, a
component or a procedure and reports what happened. Records are not summaries of the paper and not descriptions of its
datasets.

## Output contract

Return only a JSON object, no prose and no Markdown fences:

```json
{"experiences": [{
  "domain": "concise lowercase research domain",
  "task": "specific task, finer than the domain",
  "problem": "the difficulty this practice answers, in plain words, at most 20 words. The symptom, not the remedy.",
  "method": ["canonical name of the technique or design choice", "alias or other wording used in the field", "..."],
  "requires": [{"text": "a concrete prerequisite for using the practice", "basis": "stated"}],
  "source_setting": {
    "environment": [{"property": "what was true in the source experiment", "value": "its value"}],
    "params": [{"name": "a knob the authors set", "value": "the value they used",
                "basis": "swept", "from": "which values were tried, or what it was derived from"}]
  },
  "limits_stated": [{"text": "a limit or failure the authors themselves report", "quote": "verbatim sentence from the paper"}],
  "action": "what to do, as concrete steps in order, at most 60 words",
  "effect": {"compared": "what was compared with what", "benefit": "what improved, including analysis findings", "cost": "what got worse or what it costs"},
  "comparison": "controlled",
  "locator": "Table 3"
}]}
```

All of these keys must be present in every record. `requires`, `environment`, `params` and `limits_stated` may be empty
arrays; `effect.cost` may be `null`.

## How to fill each field

**`problem`** -- what goes wrong without the practice, in the words someone would use *before* knowing the fix
("most training windows have no answer, so the model learns to predict no answer"). Do not name the paper's own
solution here. It is a search key, not a description.

**`method`** -- the technique or design choice being tested, first its canonical name, then **aliases**: the other
words the field and other papers use for the same thing. One concept often has several names ("null instances",
"negative examples", "unanswerable questions", "no-answer windows"). Two to five aliases. This is what lets a keyword
search from a different paper's vocabulary find the record.

**`requires`** -- what the *technique itself* needs in order to be applied at all: a kind of label, an input form, a
component it plugs into, a compute budget. Not the experimental setup (the dataset, the benchmark, the evaluation
protocol): that belongs in `environment`. Only concrete, checkable items; leave it empty rather than writing
generalities such as "when the task is hard". `basis` is `stated` only if the paper says in words that the practice
needs it; a prerequisite that follows from how the practice works is `inferred`, which is the usual case. Never write
what the practice is *not* suitable for: papers rarely say, and a guess is worse than nothing.

**`source_setting.environment`** -- facts about the source experiment that a reader needs to judge whether their own
situation is similar: dataset and its properties (size, class balance, length, domain), model family and scale,
metric, compute, anything the paper reports that plausibly changes the size of the effect. Write `property: value`
pairs and copy numbers as the paper gives them. At most 8. Facts only; no inference.

**`source_setting.params`** -- the knobs the authors set, at most 4, the ones the result depends on most. List only
values a reader would have to decide for themselves. Leave out values forced by the data or the task (an output size
equal to the number of classes), statistical conventions (a 95% interval), and values that merely follow from another
listed knob (list the head count, not also the per-head width). For each, say how the value came about, using exactly
one `basis`:
- `swept`: they tried several values or alternatives and picked one. `from` lists what was tried.
- `derived`: the paper itself states a rule or a measurement the value was computed from. `from` says from what. Do not
  use it for a value that merely sounds reasonable.
- `prior_work`: taken from earlier work or a standard recipe. `from` names it if the paper does.
- `default`: a library or framework default.
- `unstated`: the paper does not say how it was chosen. **This is the right answer whenever the paper is silent.**
  Do not guess a reason. `from` is then an empty string.

**`limits_stated`** -- only restrictions on where the practice works, or conditions in which the authors report it failing
("does not help on ...", "hurts when ..."). The price paid in the tested comparison (more parameters, a worse score at
an extreme setting of the swept variable) is a `cost`, not a limit. Quote the sentence verbatim and exactly as the paper
writes it; do not rephrase a quote. If the paper states no such limit, return an empty array. Do not invent limits and
do not turn missing evidence into a limit.

**`action`** -- an instruction for someone who wants to do this: the operation and its order, including any
measurement that comes first ("measure the share of empty windows; then keep only a fraction of them"). Use the
paper's terms and the real values.

**`effect`** -- three parts, in this order.
- `compared`: what was compared with what (which variable, which levels, or which baselines). Only a comparison the paper
  actually runs. For `absolute_only` write what the number was measured on, never an imagined comparison.
- `benefit`: what improved. Give the metric change when the paper reports one, **and** report analysis findings the
  paper makes: which capability, question type, subset or condition got better, and any mechanism evidence. A finding
  without a number is fine. Write only what the paper reports.
- `cost`: what got worse, a side effect, a trade-off, or a price (time, memory, data). `null` if the paper reports none.

**`comparison`** -- how the effect was obtained, exactly one of:
- `controlled`: the same data, model and training, with one factor changed and the rest held fixed (an ablation, a
  removal, a parameter sweep, a sensitivity analysis).
- `vs_baseline`: the practice is compared with other methods, other datasets, or earlier results, so more than one
  thing differs.
- `absolute_only`: only an absolute result is reported.
Be strict: comparing different published systems, or the same model trained on different datasets, is `vs_baseline`,
not `controlled`.

**`locator`** -- where in the paper the result is: a table, figure or section number.

## What to write

- Prefer `controlled` comparisons: they are the evidence a reader can trust. Write one record per factor. Merge the
  several tables of one factor into one record.
- Write a `vs_baseline` or `absolute_only` record only when the paper introduces a concrete, reusable design choice and
  no controlled comparison covers it. Do not record a leaderboard.
- **Skip** descriptions of datasets, benchmarks and resources; background and related work; claims the paper does not
  test, including a theoretical guarantee that is proved but never run in an experiment. A paper with no tested practice (a theory paper, a survey, a dataset paper) returns `{"experiences": []}`.
- At most 8 records. Fewer, well-grounded records beat many thin ones.
- Every number in `effect` and `params` must be one the paper states. Do not compute new numbers, round them, or
  restate them in other units. If you are unsure of a value, leave it out.
- Use the paper's terminology. Do not mention the paper, its authors or "the study" in any field: write the practice,
  not the provenance.

## Input

The user message begins with `[paper_id] <id>` followed by the full paper text. The id is context only; do not repeat it.
