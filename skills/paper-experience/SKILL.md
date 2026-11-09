---
name: paper-experience
description: "Actively extract and save reusable experiences (practices a paper tested, with what they were compared against) from a specified academic paper or a small, deliberately selected set of papers on a topic. Use when the user explicitly asks to extract, learn, distill, or retain research experience from papers. Do not use for ordinary literature discovery, paper summaries, or related-work writing."
allowed-tools: "read_file think_tool execute extract_paper_experiences search_observations read_memory"
metadata:
  author: EvoScientist
  version: '1.0.0'
  tags: [research, experience, papers, memory]
---

# Paper Experience

Turn explicit paper-learning requests into reusable project memory and an
immediate answer. This is the foreground counterpart to paper-navigator's
automatic background accumulation; both paths use the same extraction prompt and
the same project-isolated experience store.

## Resolve the papers

- If the user supplies a paper URL, use that paper. Resolve a bare arXiv ID to
  `https://arxiv.org/abs/<id>` and a DOI to `https://doi.org/<doi>`; for other
  identifiers, use paper-navigator's point lookup to obtain a canonical URL.
- If the user supplies a topic, read `/skills/paper-navigator/SKILL.md` and use
  its search and relevance rules to select a small evidence-bearing set. Default
  to three papers and honor an explicit count up to five. In this caller mode,
  paper-navigator returns the final set without background enqueueing.
- Do not treat a topic request as permission to extract every search result.
  Selection quality matters more than volume.
- The foreground extraction tool requires a resolvable full-text URL for each
  paper. Prefer arXiv abstract/PDF URLs or publisher pages that Jina Reader can
  resolve. Never infer experience records from metadata or an abstract alone.

## Extract and persist

Call `extract_paper_experiences` once with the final paper set. It downloads the
paper bodies, extracts the experiences, saves successful results in the active
project's experience store, and returns one line per extracted record (use
`read_memory` on a record id for the full text). Existing stored
results are reused unless the user explicitly asks to refresh or re-extract.

Batch failures are per-paper: continue with successful papers and identify each
failed paper and its actual error. Do not invent a replacement extraction.

## Return the experiences

The experience, not a paper summary, is the deliverable.

- For every successful paper, present the returned experiences: for each, the task, what to do, what it was
  compared against and the benefit and cost, with whether the comparison was controlled. Read a record in full with
  `read_memory` when its detail is needed.
- Translate surrounding prose when the user writes
  in another language, without changing technical names, numbers, datasets, or
  metrics.
- State briefly that the records were saved to project memory, or that stored
  records were reused. Do not ask the user to run a separate promotion step.
- Do not paste full paper text or replace detailed experience narratives with a
  thin list of takeaways. If the user requests fewer items, return fewer complete
  entries rather than truncating every entry.

For a request that only asks whether prior experience exists, search it with
`search_observations` and read promising `E-*` records with `read_memory`; do not
re-extract a paper unnecessarily.
