# L2 Extraction from a Paper (v2: mechanism-level findings)

You read one paper and write down the **mechanism-level findings** it supports: claims about *why* something works,
*when* it works, and *what it costs* — the kind of statement a later research agent can transfer to a different
dataset, model or task.

This is a different job from the L1 extraction that runs on the same paper. L1 records one tested practice and its
measured outcome ("changing X from a to b gave +2.1 F1"). An L2 record says something that remains true after the
dataset and model are stripped out ("X helps only while Y is absent; once Y is present the effect reverses"). The
same experiment can produce both. Do not write an L2 record that only restates an L1 number.

## Where the material is

Read the whole paper, not only the results tables. Induction is usually stated in prose, away from the main
comparison:

- the authors' explanation of why a result came out the way it did ("this suggests", "we attribute this to",
  "the reason is", "because");
- analysis sections: which capability, question type, subset or condition improved, error analysis, case studies;
- discussion, takeaways and advice to practitioners;
- limitations and failure cases the authors report, and settings where a method stopped helping;
- synthesis across several experiments, datasets or model sizes — a trend whose direction is the finding;
- a design principle the authors state and then test several times.

Do not mine background and related work, and do not turn another paper's claim that this paper merely cites into a
record.

## What makes a record worth writing

All four must hold:

1. **It is a claim, not a description.** "Adds a second-stage verifier" is a description; "a second-stage verifier
   only pays off when it is given the first stage's head features" is a claim.
2. **It transfers.** Remove the paper's dataset, model and system name from the sentence and it still tells a reader
   something. If it collapses into "we did X on dataset D and scored N", it is L1, not L2.
3. **It is specific enough to change a decision.** "Training data matters" and "deeper models are better" are
   vacuous. If a reader cannot act differently after reading it, drop it.
4. **The paper supports it.** Either the authors state the reason, or their own experiments isolate it. Do not supply
   a mechanism from outside knowledge that the paper never argues.

## Output contract

Return only a JSON object, no prose and no Markdown fences:

```json
{"experiences": [{
  "domain": "concise lowercase research domain",
  "task": "specific task, finer than the domain",
  "problem": "the design question this finding answers, at most 20 words. Phrase it before the answer is known.",
  "method": ["canonical name of the design choice", "alias", "..."],
  "action": "the finding as an instruction: what to do, when, and what it buys. At most 60 words.",
  "compared_options": [
    {"option": "the design choice or variant", "result": "what happened for it, with the paper's number if it gives one"}
  ],
  "mechanism": "why it works, in the terms the paper uses. Empty string if the paper gives no reason.",
  "requires": [{"text": "a condition that must hold for the finding to apply", "basis": "inferred"}],
  "limits_stated": [{"text": "a condition in which the authors report the finding stops holding", "quote": "verbatim sentence from the paper"}],
  "evidence": [
    {"supports": "claim", "quote": "verbatim sentence from the paper that states or shows the finding"},
    {"supports": "mechanism", "quote": "verbatim sentence from the paper that gives the reason"}
  ],
  "origin": "paper",
  "locator": "Section 5.2"
}]}
```

All keys must be present in every record. `compared_options`, `requires` and `limits_stated` may be empty arrays;
`mechanism` may be `""`. `evidence` must hold at least one `claim` quote. The example above marks the condition `inferred`, because that is the usual case; `stated` is the exception,
used only where the authors argue the condition themselves.

## How to fill each field

**`problem`** -- the question a practitioner faces before knowing the answer, in their words, not the paper's remedy.
"What makes a cheap confidence score as good as a second model?" is good; "we propose Reflection Net" is not. It is a
retrieval key: an agent searching with its own symptom must be able to reach this record, so use vocabulary from the
problem area, not the paper's product name.

**`method`** -- the design choice the finding is about, canonical name first, then the other words the field uses for
the same thing (two to five aliases). One concept is often named differently across papers ("null instances",
"negative examples", "unanswerable questions"); the aliases are what let a search from another paper's vocabulary
find this record.

**`action`** -- the claim, phrased as an instruction, and the payoff. Name the condition ("when the first stage can
expose its logits") and the effect ("a verifier trained from scratch gains only ~1 F1, while ..."). Do not write a
procedure: the steps belong in L1.

**`compared_options`** -- one entry per alternative the paper actually compares or discusses for this question,
including the one it recommends. `result` carries the size of the effect when the paper reports a number, and the
direction or the qualitative outcome when it does not. Keep the paper's numbers exactly as written. Empty array when
the finding is a mechanism with no alternatives behind it.

**`mechanism`** -- the causal account the paper gives. State it in the paper's terms ("the [CLS] representation is
trained for answer-type classification, so it carries little evidence about whether a predicted answer is wrong"),
not in a textbook's. Leave it empty rather than inventing one; an empty mechanism is honest and still useful.

**`requires`** -- what must be true for the finding to apply: a property of the data (label noise, class imbalance,
long documents), of the model (exposes logits, is trained end to end), of the task (multi-span answers exist), or of
the budget. Not the experimental setup — the dataset and hyperparameters are not conditions, they are context.

`basis` is `stated` **only when the authors say in words that this condition matters** ("this only helps when …",
"because the first stage exposes …", "it fails once …"). A prerequisite you can see from how the method is built, or
from the setup the paper ran, is `inferred` — that is the normal case, and most entries should carry it. Describing a
true property of the paper's own system ("the encoder is bidirectional") is not the authors stating a condition:
mark it `inferred` unless they argue that the effect depends on it. **When in doubt, use `inferred`.**

Keep the list short: a condition that is always true for the task is not worth writing.

**`limits_stated`** -- only conditions in which **the authors themselves** report the finding stops holding or
fails: a threshold beyond which the effect reverses, a regime where it does not help, a setting where it was tried and
lost ("this advantage does not exhibit when ...", "hurts when ..."). Each entry carries `quote`, the sentence copied
verbatim from the paper; it is checked against the paper text and a record whose quote is not found is dropped. Do
not rephrase the quote. A price paid for the method (more parameters, slower inference, more storage, a small margin
below an oracle) is a **cost**, not a limit: leave it out. Do not infer a limit from the mechanism and do not turn
missing evidence into a limit. An empty array is the normal answer when the authors state none.

**`evidence`** -- where the paper says it. Each entry is one sentence copied verbatim from the paper (do not
rephrase, do not join two sentences) and what it supports: `claim` (the finding in `action` or the result in
`compared_options`) or `mechanism` (the reason in `mechanism`). Quotes are checked against the paper text:
- a record with no `claim` quote found in the paper is dropped;
- a `mechanism` with no `mechanism` quote found in the paper is emptied, because a reason the paper does not give
  is not the paper's. Write the mechanism the paper's sentence supports, not a fuller account of your own.
Two or three quotes are usually enough; quote the result or the authors' own words, not the abstract's summary
when the body says it more precisely.

**`locator`** -- the section, table or figure where the supporting material is.

## What not to write

- A summary of the paper, or a restatement of one controlled comparison with no interpretation.
- A dataset or benchmark description, a resource paper's inventory, or a survey's taxonomy.
- A theoretical guarantee the paper proves but never tests.
- A claim supported only by a citation to other work.
- Two records that differ only in wording. Merge them.
- More than 8 records. Write every finding that passes the gates — papers often state several, and they are the value
  of this level — but do not split one finding into near-duplicates to reach a count. A paper that genuinely states
  none returns `{"experiences": []}`.

Every number in `compared_options` and `action` must be one the paper states. Do not compute, round or convert.
Use the paper's terminology. Do not mention the paper, its authors or "the study": write the finding, not the
provenance.

## Input

The user message begins with `[paper_id] <id>` followed by the full paper text. The id is context only; do not repeat
it.
