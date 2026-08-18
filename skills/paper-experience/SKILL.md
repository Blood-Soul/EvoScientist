---
name: paper-experience
description: "Extract structured L1 practical + L2 inductive research experiences from academic papers and return the full experience content. Two entry modes: (A) a specific paper given as URL / arXiv ID / DOI / local Markdown file, or (B) a research topic — search for relevant papers, then extract. Trigger phrases: extract experiences from this paper, 抽取这篇论文的经验, L1 L2 experiences, 这篇论文的实践经验和归纳洞见, get research experiences on <topic>, 抽取关于 X 的论文经验, reusable research lessons from papers, 抽取经验并沉淀到记忆, extract and save paper experiences to memory. Do NOT use for: literature search / survey ranking / finding the best paper (use paper-navigator); Related Work sections (paper-writing)."
metadata:
  author: EvoScientist
  version: '1.0.0'
  tags: [research, experience, papers, extraction]
---

# Paper Experience

Extract reusable research experiences from papers and return them. You do **not**
search-and-rank papers here — that is paper-navigator's job. This skill turns a
paper (or a topic's top papers) into structured **L1 practical** + **L2 inductive**
experiences via the `extract_paper_experiences` tool, and hands the content back.

> **Run these steps yourself.** This is a skill, not a dispatchable sub-agent —
> there is no `paper-experience` agent type, so do not hand it to the `task` tool.
> Read this file, then execute the steps directly: run the fetch script with your
> shell tool and call `extract_paper_experiences` yourself. Delegating to a
> sub-agent loses the runtime session id the extraction tool needs for its cache.
>
> That applies to troubleshooting too: if a command fails, read the error and fix
> the command yourself (a missing directory → `mkdir -p`; a wrong path → correct
> it). Do not spawn a sub-agent to diagnose a shell error — that burns minutes and
> the sub-agent cannot see your workspace state anyway.

```
        User
         │
         ▼
   ┌── Router ──┐
   │            │
   ▼            ▼
 ENTRY A      ENTRY B
(paper given) (topic given)
   │            │
   │      search top-N (light, no rubric)
   │            │
   └──► ensure full text on disk ◄──┘
              │
   extract_paper_experiences[_batch]
              │
      return L1/L2 content  (+ optional promote to memory)
```

## What this skill produces

The `extract_paper_experiences` tool returns **rendered Markdown**, not raw JSON:

- **L1 Practical Experiences** — what researchers actually did: task context,
  environment, action→feedback trace, at coarse/medium/fine granularity.
- **L2 Inductive Experiences** — author-stated generalizations: declaration,
  claim type (property/relation/trend/conditional), confidence, causal reason.

Return that content to the user. Field reference: `references/experience-schema.md`.

---

## Router

Decide the entry mode from the user's input:

| Signal | Entry |
|---|---|
| A URL, `arXiv:xxxx.xxxxx`, bare arXiv id, DOI, or a local `.md` path | **Entry A** |
| A research topic / keywords ("experiences on <topic>", "关于 X 的论文经验") | **Entry B** |

When unsure, ask one clarifying question: a specific paper, or a topic?

---

## Entry A — paper given

**A1. Ensure full-text Markdown on disk.** The extraction tool only accepts a
Markdown file that already exists **inside the workspace** — it does not take raw
text or remote URLs. So:

- **Local `.md` already in the workspace** → use its path directly. Confirm it
  resolves inside the workspace (e.g. `artifacts/papers/xxx.md`).
- **arXiv URL or id** → use this skill's own fetcher (the default path):
  ```bash
  python /skills/paper-experience/scripts/fetch_fulltext.py --url <URL> --papers-dir artifacts/papers
  # or, for a bare id:
  python /skills/paper-experience/scripts/fetch_fulltext.py --paper-id 1706.03762 --papers-dir artifacts/papers
  ```
  It prints the saved Markdown path on stdout — that is your `paper_file`.

> **Use `fetch_fulltext.py`, not paper-navigator's `fetch_paper.py`.** The latter
> routes every fetch through `r.jina.ai`, which is blocked on many networks and
> hangs until the agent's 300s timeout. `fetch_fulltext.py` downloads
> `arxiv.org/pdf/<id>.pdf` directly and converts it locally with PyMuPDF, so it
> works without a tunnel and returns in seconds. It also always uses the **PDF**,
> not the `/abs/` page: the abstract page yields ~3k chars, the PDF ~40k — L1
> fine-grained extraction needs the body, so an /abs/ URL is rewritten to /pdf/.

**Non-arXiv papers (DOI, publisher URL):** `fetch_fulltext.py` covers arXiv only.
Fall back to `python /skills/paper-navigator/scripts/fetch_paper.py --paper-id <DOI>
--papers-dir artifacts/papers`, and warn the user it needs `r.jina.ai` to be
reachable. If it times out, stop and report it — never fabricate paper content or
substitute your own summary for a real extraction.

**A2. Extract.** Call the `extract_paper_experiences` tool with:
- `paper_file`: the saved workspace path from A1
- `paper_id`: the canonical id (arXiv id / DOI / CorpusId). Strip URL wrappers and
  version suffixes (`https://arxiv.org/abs/1706.03762v5` → `1706.03762`).

Go to **Return**.

---

## Entry B — topic given (light search, aligned with paper-navigator)

This is a **fast topic→experience path**, not a survey. It reuses paper-navigator's
search scripts but skips its rubric scoring and multi-round triage. If the user
needs careful selection, ranking, or a survey, route them to **paper-navigator**
instead.

**B1. Search** (reuse the script; S2 → arXiv fallback on missing key):
```bash
mkdir -p artifacts/search
python /skills/paper-navigator/scripts/scholar_search.py --query "<topic>" --limit 8 --output artifacts/search/exp_pool.jsonl --json
```
Read `artifacts/search/exp_pool.jsonl`.

> Write search output **inside the workspace** (`artifacts/search/…`), not to
> `/tmp`. A leading-slash path is resolved against the virtual workspace root, so
> `/tmp/x.jsonl` becomes `./tmp/x.jsonl` and fails with FileNotFoundError unless
> that directory exists. `mkdir -p` first, as shown.
>
> Without `S2_API_KEY` this call spends ~45s on Semantic Scholar rate-limit
> backoff before falling back to arXiv — that is expected, not a hang. Let it run.

**B2. Pick top-N.** Take the top `N` (default 3, honor an explicit user count) by
result order / citation count. **No rubric, no scoring** — that is the deliberate
difference from paper-navigator. State which papers you picked.

**B3. Fetch each paper's full text** (same direct fetcher as Entry A):
```bash
python /skills/paper-experience/scripts/fetch_fulltext.py --paper-id <ARXIV_ID> --papers-dir artifacts/papers
```
Collect `{paper_file, paper_id}` for every fetched paper. Prefer arXiv results from
B2 since this fetcher covers arXiv; skip a non-arXiv hit rather than stalling on
`fetch_paper.py`, and say which ones you skipped.

**B4. Batch extract.** Call `extract_paper_experiences_batch` once with:
```
papers = [{"paper_file": <PATH1>, "paper_id": <ID1>}, {"paper_file": <PATH2>, "paper_id": <ID2>}, ...]
```
It extracts concurrently (bounded by `PAPER_NAV_EXPERIENCE_CONCURRENCY`, default 4)
and reports `experience_extraction_elapsed_seconds`. A single-paper failure is
non-fatal — mark that paper unavailable and continue.

Go to **Return**.

---

## Return

Present the **full L1/L2 experience content** the tool returned:

- **Single paper**: the tool's rendered `L1 Practical Experiences` + `L2 Inductive
  Experiences` blocks, verbatim in structure. Do not paste the fetched full paper.
- **Multiple papers (Entry B)**: one section per paper (`## <paper_id> — <title>`),
  each with its L1/L2 blocks. Lead with a one-line note on which papers were covered.

### The narrative is the deliverable

Each entry's **`narrative`** is the experience — a self-contained account a
practitioner can apply without the paper. Relay it **in full**. Do not compress it
into a few bullet points, and do not replace it with a summary of your own.

Per entry, lead with the identity line, then the narrative:

- **L1**: `L1-001` · `granularity` · `t.summary`, then the full `narrative`.
- **L2**: `L2-001` · `claim_type` · `declaration` · confidence `μ`, then the full
  `narrative`. Include `r` (the causal explanation) when the tool returned one.

Close each entry with `source_section` + `source_quote` so the claim stays
traceable. Other structured fields (`e`, `practice_trace`, `context`, the
`domain*` classifications, `keywords_summary`, `μ_r`, `r_depth`) are available in
the tool output — surface them when the user asks, or when one carries a number
that the narrative omits. `μ`/`μ_r` may arrive as Greek (`μ`) or ASCII (`mu`) keys;
read both.

**When the user asks for N entries, show fewer entries — not a thinner version of
each.** A trimmed narrative defeats the purpose; the long form is what makes the
experience reusable.

### Bilingual output

When the user asks for Chinese, **you** are the translator — the tool only emits
English. Give both languages so nothing is lost in translation:

- `declaration` / `t.summary`: Chinese, with the English in parentheses after it.
- `narrative`: full Chinese translation. Keep technical terms and all numbers,
  metrics, dataset names, and model names verbatim in the original form.
- `source_quote`: **English verbatim, never translated** — it is evidence. Add a
  Chinese gloss below it if helpful.
- Field labels: Chinese is fine (实践经验 / 归纳经验 / 置信度 / 因果解释 …).

Translate faithfully; do not editorialize or add conclusions the paper did not
state.

Never infer experience content from an abstract when extraction failed — say it
was unavailable and keep going.

---

## Optional — chain into a solution A/B comparison

Extraction normally **ends here**. Continue only when the user explicitly asked to
compare solutions or to use the experiences to solve a problem — phrasings like
「并对比经验和原文的方案」/「用这些经验生成方案，和直接读原文对比」/「顺便做 A/B
对比」. A bare "抽取经验" request stops at the experiences.

When they did ask, hand off to the **`solution-ab`** skill: read
`/skills/solution-ab/SKILL.md` and run its steps yourself, passing along

- the **problem statement** (restate the user's topic as one solution-shaped
  question if they only gave a topic), and
- the **paper ids** you just extracted (1–3).

`solution-ab` generates one solution from the extracted experiences and one from
the paper bodies, then lays them out side by side. Do not attempt that comparison
inline here — its context-building and control-variable rules live in that skill,
and improvising them defeats the purpose.

---

## Optional — promote to long-term memory

Extraction alone leaves the experiences in a **session-scoped cache**: usable this
turn, invisible to every later session. Promoting them writes each experience into
the global EvoMemory observation store, where the per-turn memory preflight and
`search_observations` pick them up from then on — the same store the offline
experience bank lives in.

**Trigger.** Promote when the user asks for it, in either shape:

| Shape | Example |
|---|---|
| **Up front**, in the same request | 「抽取这篇论文的经验并沉淀到记忆」/「extract and save to memory」 |
| **Afterwards**, as a follow-up | 「把刚才抽的经验存进记忆」/「沉淀一下」 |

Both are equally valid. In the follow-up case the extraction cache is still on
disk, so **do not re-extract** — just promote the paper ids from that turn. Do not
promote unasked; a bare 抽取 request stops at the experiences.

**Run once per paper id:**

```bash
PYTHONUTF8=1 python /skills/paper-experience/scripts/promote_to_memory.py --paper-id <ID>
```

Add `--session <thread_id>` to pin one session, `--dry-run` to preview counts
without writing, or `--scope project --project-id <ID>` to keep the experiences out
of the global store. Default is global, so the experiences surface in any later
research task.

The script reads the extraction cache
(`<MEMORIES_DIR>/paper_experiences/sessions/<session>/<paper>/{l1,l2}.json`) and
writes one observation per experience — **L2 → `semantic`**, **L1 → `procedural`** —
with the narrative as the body and `source_quote` + paper id as evidence. Ids are
content hashes, so re-running is safe: unchanged experiences come back as
`Duplicate` rather than piling up copies.

**Report the counts it prints** (`Created` / `Duplicate`) per paper. If it exits
with `No cached experiences found`, the extraction did not land — say so instead of
promoting nothing silently.

---

## Notes

- Extraction is a **real LLM call** (L1+L2 concurrently, ≥450-word narratives) — a
  full paper takes roughly 1–3 minutes. The same paper in the same session hits the
  cache and returns instantly on re-extraction.
- Everything reads/writes UTF-8; the runner sets `PYTHONUTF8=1` to avoid Windows
  cp936 mojibake.
- `fetch_fulltext.py` needs `arxiv.org` reachable. It tries arXiv HTML first
  (no PDF library required), then the PDF via PyMuPDF, and only falls back to
  `r.jina.ai` for non-arXiv sources. It re-executes itself in the project venv
  when the calling interpreter lacks `httpx`/`markdownify`, so a bare `python3`
  works too. If a fetch or an extraction fails, **stop and report the failing
  step** — do not hand-write a summary in place of tool output.

## Hand off to

| Goal | Skill |
|---|---|
| 对比「经验 vs 原文」生成的方案 | `solution-ab` |
| Find / rank / survey papers | `paper-navigator` |
| Idea generation | `research-ideation` |
| Related Work section | `paper-writing` |
