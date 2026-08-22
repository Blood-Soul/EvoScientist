# 经验库检索与问答汇报 (10 Queries)


> 本报告演示了从 374 条结构化论文经验中检索相关条目，**全量展示其原始字段结构**，并由大模型基于这些上下文提供综合回答与思维链的端到端能力。

## 目录

- [代码评测指标的选择](#代码评测指标的选择)
- [RAG 检索增强的长文本切分](#rag-检索增强的长文本切分)
- [端侧多模态模型部署瓶颈](#端侧多模态模型部署瓶颈)
- [Multi-agent 多智能体协作分工](#multi-agent-多智能体协作分工)
- [纯视觉 GUI 自动化 Agent](#纯视觉-gui-自动化-agent)
- [指令微调的数据质量与数量](#指令微调的数据质量与数量)
- [LLM 处理超长上下文的机制问题](#llm-处理超长上下文的机制问题)
- [Coding Agent 的执行失败迭代修复](#coding-agent-的执行失败迭代修复)
- [Prompt Engineering 的 Few-shot 示例](#prompt-engineering-的-few-shot-示例)
- [Transformer 位置编码的长度外推](#transformer-位置编码的长度外推)

---


---

# 代码评测指标的选择

**用户提问：** 评估代码生成模型功能正确性时，用 BLEU 分数靠谱吗？应该使用什么指标？

**检索关键词：** `code generation functional correctness test cases metric BLEU`

## 回答

BLEU **不太靠谱，不应作为代码生成功能正确性的主要指标**。它衡量的是生成代码与参考实现之间的 n-gram 词面重叠，而不是程序行为是否正确。

经验表明：

- 在 HumanEval 中，正确代码和错误代码的 BLEU 分数分布有明显重叠，BLEU 无法可靠区分二者（O-a24a6a6cd69669a2）。
- 在 MBPP 中，BLEU 与测试用例通过率几乎没有相关性；变量重命名、不同控制流或不同库函数都可能让功能等价的代码具有很低的 BLEU（O-70be61083ea22f32）。
- 在 APPS 中，随着题目难度提高，BLEU 反而上升，但实际测试用例通过率下降；甚至较弱模型的 BLEU 可能高于较强模型（O-bf1c2e70a191ad1b）。

因此，评估功能正确性时应优先采用**基于执行的测试指标**：

1. **测试用例平均通过率（test-case average）**  
   对每道题统计通过的测试用例比例，再取平均，能反映部分正确程度。

2. **严格正确率（strict accuracy）**  
   统计通过全部测试用例的题目比例，适合作为“完整解决题目”的指标。

3. **多候选生成时，报告 top-k 结果**  
   APPS 经验中，使用 top-5 候选并选择通过测试的候选，可以明显提升结果。因此应明确报告单次生成和多次采样/候选生成下的测试通过表现（O-62e91f9862b0d7f1）。

4. **辅助报告语法错误率**  
   语法错误率可以帮助诊断模型失败原因，但它不能替代功能正确性指标（O-62e91f9862b0d7f1）。

实践上可以采用：

> **主指标：执行测试后的测试用例平均通过率 + 严格正确率；  
> 辅助指标：语法错误率、不同候选数量下的通过表现；  
> BLEU：仅作为开发阶段的低成本参考，不用于最终模型排名或功能正确性结论。**

需要注意，执行指标的可靠性取决于测试用例覆盖度。APPS 经验显示，只有少量测试用例的题目可能产生一定的误报；此外，现有经验主要来自 Python 的自然语言到代码任务，其他语言或更受限任务的具体效果，**经验库未覆盖**。

## 思维链 (Reasoning)

1. BLEU 只比较参考代码与生成代码的表面 n-gram 重叠，不能表示程序语义等价性；正确实现可能因变量名、控制流或库函数不同而获得低分（O-a24a6a6cd69669a2、O-70be61083ea22f32）。
2. 多项经验直接观察到 BLEU 与真实功能正确性缺乏相关性，甚至出现反相关，因此不适合作为主要指标（O-bf1c2e70a191ad1b、O-70be61083ea22f32）。
3. APPS 使用执行测试得到“测试用例平均通过率”和“严格正确率”，并将其作为主要评价指标，所以应优先采用测试驱动的功能指标（O-62e91f9862b0d7f1、O-5fc56ba6881ed0e3）。
4. 测试数量和覆盖度会影响执行评估的误报率，因此需要同时关注测试集质量及语法错误等辅助诊断信息（O-62e91f9862b0d7f1）。

## 召回的原始经验全貌 (5 条)

### 经验 ID: `O-bf1c2e70a191ad1b`

```markdown
---
id: O-bf1c2e70a191ad1b
created_at: '2026-08-16T15:52:57Z'
summary: '[relation] BLEU score is a poor substitute for test-case-based evaluation
  in code generation tasks, as it can be anti-correlated with the functional correctness
  of generated programs.'
memory_type: semantic
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l2
related_observations:
- id: O-5fc56ba6881ed0e3
  relation: complements
  reason: 'Both extracted from arXiv:2105.09938: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:02Z'
- id: O-62e91f9862b0d7f1
  relation: complements
  reason: 'Both extracted from arXiv:2105.09938: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:03Z'
---

## Observation

### Declaration
BLEU score is a poor substitute for test-case-based evaluation in code generation tasks, as it can be anti-correlated with the functional correctness of generated programs.

### Narrative
In evaluating machine-generated code, a common shortcut is to use BLEU (Bilingual Evaluation Understudy), a text-based n-gram overlap metric originally designed for machine translation. However, code generation is fundamentally different from natural language generation because the primary success criterion is functional correctness—whether the program, when executed, passes test cases—rather than surface-level lexical similarity to a reference solution. This experience exposes a critical flaw in that shortcut: on the APPS benchmark, BLEU scores are not merely noisy indicators of code quality; they can be actively misleading. Specifically, BLEU increases as problem difficulty rises, even though true model performance (measured by test case pass rate) decreases. The APPS benchmark categorizes 10,000 Python programming problems into three difficulty levels: introductory, interview, and competition. When fine-tuned models such as GPT-Neo 2.7B were evaluated, their BLEU scores rose from 27.1 on introductory problems to 29.1 on interview problems and further to 29.3 on competition problems, giving the false impression that the model was improving on harder tasks. In reality, the model's ability to pass test cases dropped sharply across these same levels. This anti-correlation stems from the nature of reference solutions: harder competition-level problems tend to have longer, more specialized canonical solutions with unique and complex lexical patterns, while introductory problems have more diverse and shorter solutions. As the model's output becomes increasingly random or generic on difficult problems, it can accidentally match more n-grams with a single detailed reference than with multiple short ones, artificially inflating BLEU. Compounding this flaw, a worse model (GPT-2 0.1B) obtained higher BLEU scores than a stronger model (GPT-Neo 2.7B) on competition problems, further demonstrating that BLEU can invert the true performance ranking. The practical consequence is stark: any research that relies on BLEU to compare code generation systems risks drawing wrong conclusions about which methods are superior. The evidence from APPS is complemented by earlier observations that CodeBLEU, a variant specifically designed for code, still exhibits similar limitations when evaluated against execution-based metrics. This experience underlines a fundamental principle: for tasks where the output has a well-defined operational semantics, evaluation must be execution-based. BLEU may remain useful during early development as a cheap surrogate, but final conclusions about model performance on code generation should always be drawn from test case pass rates or similar functional metrics. It should be noted that this finding is established specifically for Python code generation from natural language, and its generalizability to other programming languages or to more restricted settings (like pseudocode-to-code) has not been tested in the source work. Nonetheless, given that code is by nature an executable artifact, the principle that surface-form metrics can diverge from correctness is likely to hold broadly.

### Applicability context
summary: Warns practitioners that BLEU is an unreliable metric for code generation, as it can increase while actual program correctness decreases; recommends test-case-based evaluation for accurate assessment of model performance.
modality: text (Python code)
scale: APPS test set (5,000 problems), three models (GPT-2 0.1B, 1.5B, GPT-Neo 2.7B)
constraint: observation limited to Python code generation from natural language; the degree of anti-correlation may vary across programming languages and problem types

### Extraction rationale
The finding is an explicit conclusion drawn from a comparative analysis of BLEU and test case accuracy across difficulty levels (Figure 6). The authors directly state that BLEU is a poor substitute and demonstrate anti-correlation with a gold-standard metric. No causal mechanism for why BLEU behaves this way is provided in the paper, so r is null.

## Why It Matters

Claim type: relation. Confidence: high. Warns practitioners that BLEU is an unreliable metric for code generation, as it can increase while actual program correctness decreases; recommends test-case-based evaluation for accurate assessment of model performance. Applies under: observation limited to Python code generation from natural language; the degree of anti-correlation may vary across programming languages and problem types

## Evidence

### Source
arXiv:2105.09938 | 2105.09938__4550c66a | group: t5 | section: experiment

### Verbatim quote
"We find that assessing model performance with BLEU is a poor substitute for evaluating with test cases. ... Observe in Figure 6 that BLEU increases as problem sources become more difficult, even though models actually perform worse on harder problems. Moreover, worse models can have similar or higher BLEU scores."

### Classification
domain: agent_evaluation
arXiv: cs.SE
Wikipedia: Software_engineering
ACM CCS: Software and its engineering → Software creation and management → Designing software
CLC: TP311

### Keywords
BLEU, code evaluation, test cases, code generation, functional correctness, metric, anti-correlation

```

### 经验 ID: `O-a24a6a6cd69669a2`

```markdown
---
id: O-a24a6a6cd69669a2
created_at: '2026-08-16T15:52:57Z'
summary: '[property] BLEU score is not a reliable indicator of functional correctness
  for code generation, as it fails to distinguish between programs that pass unit
  tests and those that do not.'
memory_type: semantic
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l2
related_observations:
- id: O-9f873b661b8bb44d
  relation: complements
  reason: 'Both extracted from arXiv:2107.03374: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:04Z'
---

## Observation

### Declaration
BLEU score is not a reliable indicator of functional correctness for code generation, as it fails to distinguish between programs that pass unit tests and those that do not.

### Narrative
Evaluating code generation models typically relies on match-based metrics such as BLEU score, which compare generated code against a reference solution using n-gram overlap. The paper challenges the suitability of BLEU for assessing functional correctness—the ability of generated code to pass unit tests. Using the HumanEval benchmark of 164 hand-written Python programming problems, the authors generate samples from a 12B-parameter Codex model and compute BLEU scores relative to the reference solution for each sample. They then separate correct solutions (those that pass all unit tests) from incorrect ones and plot the distribution of BLEU scores for both groups. The finding is that the BLEU score distributions for correct and incorrect solutions overlap significantly (Figure 8). Since an incorrect solution is guaranteed to be functionally inequivalent to the reference, the inability of BLEU to separate correct from incorrect outputs indicates that BLEU is a poor proxy for functional correctness. The reason, as articulated by the authors, is that BLEU measures surface-form similarity rather than semantic equivalence, and the space of programs that are functionally equivalent to a reference is large and diverse in surface form. An incorrect solution can have high BLEU if it shares many n-grams with the reference, while a correct but differently-styled solution can have low BLEU. This observation has direct practical implications: optimizing generation to maximize BLEU may mislead developers into selecting code that appears similar to a reference but does not actually work, while discarding correct solutions. The evidence is drawn from a single model (Codex-12B), but the underlying principle—that n-gram overlap cannot capture program semantics—is broadly applicable to any code generation system. The context is limited to the task of synthesizing Python functions from docstrings, but similar conclusions have been drawn in the program translation literature, suggesting generalizability.

### Causal explanation (r)
BLEU score measures n-gram overlap with a reference solution but does not account for the large and complex space of programs functionally equivalent to a reference, so it penalizes correct solutions that differ in surface form and rewards incorrect solutions that share superficial similarities.

confidence (μ_r): medium
depth: deep
depth rationale: Identifies a specific metric limitation: BLEU's n-gram matching cannot discriminate functionally correct from incorrect code because many program variants implement the same function, leading to overlapping score distributions.

### Applicability context
summary: Demonstrates that BLEU score is a misleading metric for evaluating code generation models, as overlapping score distributions between correct and incorrect solutions prevent reliable quality assessment.
modality: code
scale: Codex-12B on 164 HumanEval problems, multiple samples per problem
constraint: evaluated on Python function synthesis from docstrings; the general principle that surface-form overlap ignores semantic equivalence likely extends to other languages and tasks but was not tested directly

### Extraction rationale
The declaration is based on the author's explicit interpretation of overlapping BLEU distributions between correct and incorrect code samples (declaration basis). The causal explanation (r) is drawn from the same discussion where the author notes that the space of functionally equivalent programs is large and BLEU fails to capture semantic features specific to code.

## Why It Matters

Claim type: property. Confidence: medium. Demonstrates that BLEU score is a misleading metric for evaluating code generation models, as overlapping score distributions between correct and incorrect solutions prevent reliable quality assessment. Applies under: evaluated on Python function synthesis from docstrings; the general principle that surface-form overlap ignores semantic equivalence likely extends to other languages and tasks but was not tested directly

## Evidence

### Source
arXiv:2107.03374 | 2107.03374__6ec0ed49 | group: t5 | section: results

### Verbatim quote
For each problem, when we plot the distributions of BLEU scores for correct and incorrect solutions, we notice significant overlap (Figure 8). Since an incorrect solution is guaranteed to be functionally inequivalent to the reference solution, we conclude that improvements in BLEU score may not indicate improved rates of functional correctness in practice.

### Classification
domain: agent_software_eng
arXiv: cs.SE
Wikipedia: Program_synthesis
ACM CCS: Computing methodologies → Artificial intelligence → Natural language processing → Evaluation methods
CLC: TP311

### Keywords
code evaluation, BLEU score, functional correctness, code generation, evaluation metrics, program synthesis

```

### 经验 ID: `O-70be61083ea22f32`

```markdown
---
id: O-70be61083ea22f32
created_at: '2026-08-16T15:52:57Z'
summary: '[property] BLEU score between generated code and a reference implementation
  does not correlate with functional correctness for natural-language-to-code synthesis
  with large language models.'
memory_type: semantic
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l2
related_observations:
- id: O-be74d39edaf4707b
  relation: complements
  reason: 'Both extracted from arXiv:2108.07732: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:07Z'
- id: O-5fc56ba6881ed0e3
  relation: complements
  reason: Both classified under agent_evaluation, from arXiv:2105.09938 and arXiv:2108.07732
    respectively — different papers addressing the same problem area.
  linked_at: '2026-08-16T15:59:14Z'
- id: O-438463797c961c9a
  relation: complements
  reason: Both classified under agent_evaluation, from arXiv:2108.07732 and arXiv:2306.04757
    respectively — different papers addressing the same problem area.
  linked_at: '2026-08-16T15:59:14Z'
---

## Observation

### Declaration
BLEU score between generated code and a reference implementation does not correlate with functional correctness for natural-language-to-code synthesis with large language models.

### Narrative
When evaluating code generated by large language models, practitioners often rely on surface-level similarity metrics such as BLEU score, computed against a reference implementation, as a proxy for correctness. This paper tests whether BLEU correlates with actual functional correctness—whether the generated program passes supplied test cases—on the MBPP dataset. The authors compute both BLEU scores and the fraction of samples that solve each task (as determined by execution against three test cases) for the 137-billion-parameter model under few-shot prompting, and find little correlation between the two. Semantically identical programs can have drastically different n-gram overlap with a reference because of identifier renaming, alternative orderings of operations, or the use of different library functions that achieve the same result. The paper illustrates this by plotting the fraction of correct samples per task against the average BLEU score per task, showing that tasks with high BLEU scores are no more likely to be solved than those with low BLEU scores. This observation aligns with prior work on code generation benchmarks (Hendrycks et al., 2021; Chen et al., 2021) and reinforces that BLEU is an unsuitable metric for measuring functional code synthesis performance. The key implication is that any evaluation or comparison of code-synthesis systems must rely on execution-based metrics, not on textual overlap measures. The finding holds specifically for the few-shot setting with the largest model and the MBPP benchmark, but the authors cite similar results from other independent studies, suggesting this is a general property of natural-language-to-code models rather than an idiosyncrasy of a particular architecture or dataset. The reason for the poor correlation is rooted in the nature of code: unlike natural language translation, where meaning collapses into a relatively narrow set of phrasings, a single programming task admits a vast space of syntactically distinct but semantically equivalent implementations. BLEU measures surface lexical overlap, so it cannot capture equivalence of behavior. The paper’s evidence is qualitative and quantitative: the scatter plot shows no clear trend, and the authors accompany it with examples of programs that are semantically correct but lexically distant from the reference. While the finding is perhaps unsurprising to researchers familiar with program analysis, the study provides empirical confirmation at the scale of large language models, and thus serves as a cautionary note against using BLEU as a primary evaluation metric in program synthesis research.

### Causal explanation (r)
Semantically identical programs can be expressed with very different surface token sequences due to identifier renaming, alternate control flow, or different library usage, so n-gram overlap (BLEU) misses functional equivalence.

confidence (μ_r): medium

### Applicability context
summary: For assessing the output of large language models on Python programming tasks, BLEU score is not a reliable indicator of whether the generated code passes test cases, implying that execution-based evaluation is necessary.
modality: text (Python code)
scale: 137B parameter model, 500-test subset of MBPP
constraint: observed under few-shot prompting with temperature sampling; correlation might differ if models are fine-tuned to generate code that closely mimics a specific reference style

### Extraction rationale
The author presents a direct interpretive comparison between BLEU and actual correctness across all test problems, emphasizing the lack of correlation. The causal explanation is stated explicitly in the same paragraph, attributing the mismatch to semantic equivalence with low n-gram overlap.

## Why It Matters

Claim type: property. Confidence: medium. For assessing the output of large language models on Python programming tasks, BLEU score is not a reliable indicator of whether the generated code passes test cases, implying that execution-based evaluation is necessary. Applies under: observed under few-shot prompting with temperature sampling; correlation might differ if models are fine-tuned to generate code that closely mimics a specific reference style

## Evidence

### Source
arXiv:2108.07732 | 2108.07732__2d6f3c88 | group: t5 | section: results

### Verbatim quote
"We find little correlation between the two. This can be explained by the fact that semantically identical programs can potentially have very low n-gram overlap; for example, because of identifier renaming."

### Classification
domain: agent_evaluation
arXiv: cs.SE
Wikipedia: Software_engineering
ACM CCS: Software and its engineering → Software creation and management → Software development process management
CLC: TP311

### Keywords
program synthesis, evaluation metric, BLEU, functional correctness, code generation, large language models, MBPP

```

### 经验 ID: `O-62e91f9862b0d7f1`

```markdown
---
id: O-62e91f9862b0d7f1
created_at: '2026-08-16T15:52:57Z'
summary: '[medium] Constructs a comprehensive code-generation benchmark (APPS) with
  10,000 Python problems, two formats, and 130,000+ test cases; fine‑tunes GPT‑2 and
  GPT‑Neo on the training set after GitHub pretraining, showing that GPT‑Neo 2.7B
  achieves 14.7% test‑case average on introductory problems and 5.5% strict accuracy
  with top‑5 sampling, while GPT‑3 few‑shot performs near zero, and that syntax errors
  decrease exponentially with model scale and fine‑tuning.'
memory_type: procedural
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l1
related_observations:
- id: O-64cef756e32679f2
  relation: complements
  reason: 'Both extracted from arXiv:2105.09938: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:02Z'
- id: O-7f843b06fb70fa1e
  relation: complements
  reason: 'Both extracted from arXiv:2105.09938: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:02Z'
- id: O-ac4389b5bc43a0bb
  relation: complements
  reason: 'Both extracted from arXiv:2105.09938: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:02Z'
- id: O-bf1c2e70a191ad1b
  relation: complements
  reason: 'Both extracted from arXiv:2105.09938: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:03Z'
---

## Observation

### Narrative
Background: Prior code‑generation benchmarks either lacked test cases, focused on pseudocode‑to‑code translation, or used limited docstrings as specifications. To enable rigorous evaluation of whole‑program synthesis from unrestricted natural language, the researchers built APPS, a dataset of 10,000 Python programming problems with a rich set of test cases and multiple human solutions. The overall practice is to curate this benchmark, define a protocol for training and evaluating large language models, and produce a comprehensive baseline across model sizes and architectures, thereby establishing a robust performance landscape for the task.

The practice involved scraping problems from seven open‑access coding websites, cleaning and standardizing them (e.g., converting image equations to LaTeX via Mathpix, parsing HTML for proper layout, deduplicating with TF‑IDF and cosine similarity), and categorizing them into three difficulty levels: introductory (3,639 problems), interview (5,000 problems), and competition (1,361 problems). Each problem in the test set (5,000 problems) is accompanied by multiple test cases (average 21.2) that check functional correctness, enabling fully automatic evaluation without human judgment. The dataset includes both call‑based format problems (with starter code that defines a function signature) and standard‑input format problems (where the program must read from stdin and print to stdout). In total, 232,421 human‑written solutions are provided.

For the baseline evaluation, language models were first pretrained on a large corpus of Python code. GPT‑2 models were further pretrained on a curated 30 GB corpus of GitHub repositories (with star counts ≥1, after removing repositories containing keywords associated with coding challenge sites and functions with signatures overlapping APPS starter code). GPT‑Neo 2.7B had already been pretrained on the Pile dataset, which includes GitHub. All models were then fine‑tuned on the APPS training split (5,000 problems) for 10 epochs using AdamW (batch size 256, weight decay 0.05) on 8 A100 GPUs with DeepSpeed ZeRO optimization. During fine‑tuning, the loss was computed only on the generated code (excluding starter code). For inference, beam search with beam width 5 was used, and the best of five beams (top‑5) was also evaluated.

The primary evaluation metrics were test case average (the average fraction of test cases passed per problem) and strict accuracy (percentage of problems for which all test cases pass). On introductory problems, GPT‑Neo 2.7B reached a test case average of 14.68% (top‑1) and 19.89% (top‑5), and strict accuracy of 3.90% (top‑1) and 5.50% (top‑5). Performance plummeted for interview and competition problems; e.g., GPT‑Neo achieved a top‑1 test case average of 9.85% on interview and 6.54% on competition, with strict accuracy near zero. Additionally, the prevalence of syntax errors was analyzed: GPT‑Neo produced syntax errors on only ~3% of introductory problems, while the few‑shot GPT‑3 175B had ~59%. The researchers also demonstrated that the BLEU metric (computed against human solutions) did not correlate with test‑case performance; in fact, BLEU scores increased with problem difficulty while test‑case averages decreased, and worse models could achieve similar BLEU scores to better ones.

Outcomes: The APPS benchmark reveals that current language models are starting to exhibit nonzero code‑generation ability, particularly on easy problems, but remain far from human‑expert level. The performance scales with model size and fine‑tuning, and syntax errors decline exponentially as models improve. The top‑5 sampling substantially boosts results, indicating that generating multiple candidates and selecting by test‑case score is an effective strategy. The false‑positive analysis showed that problems with very few test cases (2–3) can inflate strict accuracy by ~1%, but the noise is low overall. Importantly, the study establishes that BLEU is an unreliable metric for code generation and that functional correctness tests are essential.

Boundaries and limitations: The benchmark is limited to Python; models were not tested on other languages. The few‑shot GPT‑3 evaluation used a specific prompt design that may not be optimal. The GitHub pretraining data may still contain some coding‑challenge‑style code despite the keyword filtering. The evaluation only measures functional correctness and does not consider code efficiency, readability, or security. The test cases are those originally provided by the problem sources; some problems have small output spaces (e.g., YES/NO), potentially increasing false positives. The fine‑tuning was performed with a fixed hyperparameter set and only 10 epochs; longer training or different architectures might yield better results.

### Practice environment
Dataset: APPS (sources: Codeforces, Kattis, AtCoder, Codewars, CodeChef, HackerRank, LeetCode), 5,000 training, 5,000 test problems; 131,777 test cases; two formats: call‑based and standard input. Models: GPT‑2 0.1B, GPT‑2 1.5B (further pretrained on 30 GB filtered GitHub Python); GPT‑Neo 2.7B (pretrained on the Pile). Training: fine‑tuning on APPS training set for 10 epochs, AdamW (batch 256, weight decay 0.05), 8 A100 GPUs, DeepSpeed ZeRO. Evaluation: beam search (width 5) for fine‑tuned models; metrics: test‑case average, strict accuracy, syntax error rate, BLEU. GPT‑3 175B evaluated few‑shot with two examples per format.

### Practice trace
1. Action: Curated 10,000 Python problems from multiple coding challenge websites, cleaned and standardized them (LaTeX, HTML parsing, deduplication), and organized into three difficulty levels with extensive test cases
   Feedback: Dataset contains 5,000 training problems and 5,000 test problems with 131,777 test cases, covering introductory (3,639), interview (5,000), and competition (1,361) levels
2. Action: Fine‑tuned GPT‑2 0.1B, GPT‑2 1.5B, and GPT‑Neo 2.7B on APPS training set after GitHub pretraining, and evaluated on test set using beam search (beam width 5), reporting test‑case average and strict accuracy per difficulty tier
   Feedback: GPT‑Neo 2.7B achieved test‑case average: introductory 14.68%, interview 9.85%, competition 6.54%, and strict accuracy: introductory 3.90%, interview 0.57%, competition 0.00%; performance degraded sharply with difficulty and was far below human level
3. Action: Measured syntax error rates across models and compared BLEU scores with test‑case accuracy to assess evaluation metrics
   Feedback: GPT‑Neo had ≈3% syntax errors on introductory problems, GPT‑3 had ≈59%; BLEU was anticorrelated with test‑case accuracy (e.g., GPT‑Neo BLEU increased with difficulty while test‑case average decreased), demonstrating that test cases are essential for evaluation

### Task context
modality: natural language problem descriptions to Python code
scale: 10,000 problems (5,000 training, 5,000 test) with up to 232k solutions; three difficulty tiers; models up to 2.7B parameters fine‑tuned, plus GPT‑3 175B few‑shot; 131,777 test cases
constraint: only Python is supported; test case quality varies (some problems have only 2 test cases, giving ~1% false‑positive rate); few‑shot prompts for GPT‑3 were not optimized; the GitHub pretraining decontamination process may not be exhaustive; evaluation focuses solely on functional correctness, ignoring other code quality dimensions

### Extraction rationale
Practice description synthesized from Sections 1, 3, and 4 (dataset construction, training procedure, evaluation metrics). The action sequence follows the paper’s workflow; numerical results are drawn from Table 2 and Figure 5. The quote is from the introduction and abstract, providing a verbatim statement of the core method and findings.

## Why It Matters

Granularity: medium. Scale: 10,000 problems (5,000 training, 5,000 test) with up to 232k solutions; three difficulty tiers; models up to 2.7B parameters fine‑tuned, plus GPT‑3 175B few‑shot; 131,777 test cases Applies under: only Python is supported; test case quality varies (some problems have only 2 test cases, giving ~1% false‑positive rate); few‑shot prompts for GPT‑3 were not optimized; the GitHub pretraining decontamination process may not be exhaustive; evaluation focuses solely on functional correctness, ignoring other code quality dimensions

## Evidence

### Source
arXiv:2105.09938 | 2105.09938__4550c66a | group: t5 | section: introduction

### Verbatim quote
"We introduce APPS, a benchmark for code generation from natural language specifications. [...] Our benchmark includes 10,000 problems, which range from having simple oneline solutions to being substantial algorithmic challenges. [...] We fine-tune large language models on both GitHub and our training set, and we find that the prevalence of syntax errors is decreasing exponentially as models improve. [...] Recent models such as GPT-Neo can pass approximately 20% of the test cases of introductory problems, so we find that machine learning models are now beginning to learn how to code."

### Classification
domain: agent_evaluation
arXiv: cs.SE
Wikipedia: Software_engineering

### Keywords
code generation, programming benchmarks, language models, test case evaluation, Python, fine-tuning, syntax errors, BLEU, program synthesis, natural language programming

```

### 经验 ID: `O-5fc56ba6881ed0e3`

```markdown
---
id: O-5fc56ba6881ed0e3
created_at: '2026-08-16T15:52:57Z'
summary: '[coarse] Introduces APPS, a benchmark of 10,000 Python programming problems
  with 130,000+ test cases, and evaluates fine-tuned language models, finding that
  GPT-Neo 2.7B passes ~15% of test cases on introductory problems (top-1) and ~20%
  with top-5 sampling, while GPT-3 few-shot solves only 3 out of 5,000 problems, establishing
  baseline code-generation capabilities.'
memory_type: procedural
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l1
related_observations:
- id: O-64cef756e32679f2
  relation: complements
  reason: 'Both extracted from arXiv:2105.09938: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:01Z'
- id: O-7f843b06fb70fa1e
  relation: complements
  reason: 'Both extracted from arXiv:2105.09938: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:01Z'
- id: O-ac4389b5bc43a0bb
  relation: complements
  reason: 'Both extracted from arXiv:2105.09938: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:02Z'
- id: O-bf1c2e70a191ad1b
  relation: complements
  reason: 'Both extracted from arXiv:2105.09938: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:02Z'
- id: O-14e39ba5a7b26d48
  relation: complements
  reason: Both classified under agent_evaluation, from arXiv:2005.03778 and arXiv:2105.09938
    respectively — different papers addressing the same problem area.
  linked_at: '2026-08-16T15:59:14Z'
- id: O-70be61083ea22f32
  relation: complements
  reason: Both classified under agent_evaluation, from arXiv:2105.09938 and arXiv:2108.07732
    respectively — different papers addressing the same problem area.
  linked_at: '2026-08-16T15:59:14Z'
---

## Observation

### Narrative
Background: Automated code generation from natural language is an economically significant and challenging task, yet prior benchmarks for evaluating generative models of code were limited in scope—focusing on pseudocode-to-code, short snippets, or lacking functional correctness tests. The APPS benchmark (Automated Programming Progress Standard) was created to provide a rigorous, test-case-based evaluation of how well large language models can generate arbitrary Python code given English problem descriptions. The benchmark comprises 10,000 programming problems curated from open-access coding challenge websites (including Codeforces, Kattis, AtCoder, Codewars). Each problem includes a natural language specification (with an average length of 293.2 words), multiple ground-truth human solutions, and an extensive set of test cases (131,777 in total, average 21.2 per test problem) that check the functional correctness of generated code. The problems are divided into three difficulty tiers: introductory (simple tasks like counting vowels or computing a running sum), interview (algorithmic problems involving data structures and nontrivial algorithms), and competition (national and international programming competition level). The practice includes constructing the dataset with quality control steps such as parser development for LaTeX and HTML, deduplication via TF-IDF and cosine similarity, and manual refinement by students over six months. The benchmark supports two problem formats: call‑based (with starter code and function headers) and standard‑input format (program reads from stdin and writes to stdout). To produce baseline results, the researchers pretrained GPT‑2 models on a decontaminated 30 GB corpus of Python code from GitHub repositories (with star counts ≥1, and filtered to remove coding‑challenge‑style repositories) and then fine‑tuned them on the APPS training set (5,000 problems) for 10 epochs using AdamW with batch size 256, weight decay 0.05, on 8 A100 GPUs. The largest fine‑tuned model, GPT‑Neo 2.7B (pretrained on the Pile, which includes GitHub), was also fine‑tuned similarly. Evaluation used beam search with beam width 5 and reported two metrics: “test case average” (the mean fraction of test cases passed per problem) and “strict accuracy” (the proportion of problems for which all test cases are passed). Fine‑tuned GPT‑Neo 2.7B achieved a test case average of 14.68% on introductory problems, while strict accuracy was 3.90% (top‑1). The benchmark also evaluated GPT‑3 175B in a few‑shot setting (2 examples per format) and found it solved only 3 out of 5,000 problems. Moreover, the frequency of syntax errors was measured: GPT‑Neo had very few syntax errors (~3% on introductory problems), while GPT‑3 few‑shot had about 59%, and syntax errors decreased exponentially with model size and fine‑tuning. The practice of constructing APPS and establishing these baselines provides a precise, test‑case‑driven way to track code‑generation progress; it reveals that current models are beginning to solve simple problems but are far from human‑level, with performance decreasing sharply with problem difficulty. Boundaries: The benchmark is limited to Python, and evaluation relies on the coverage of the provided test cases; some problems have only a few test cases, which led to a false‑positive rate of ~1% for problems with ≤3 test cases. The few‑shot evaluation of GPT‑3 was with a particular prompt design that may be suboptimal. The fine‑tuned models use only APPS training data and a limited GitHub pretraining corpus. The benchmark does not assess code readability, style, or efficiency beyond functional correctness.

### Practice environment
Dataset: APPS (10k problems from Codeforces, Kattis, AtCoder, Codewars; difficulties: 3,639 introductory, 5,000 interview, 1,361 competition; two formats: call‑based and standard input). Models: GPT‑2 0.1B, GPT‑2 1.5B, GPT‑Neo 2.7B (fine‑tuned); GPT‑3 175B (few‑shot). Training: pretraining on 30 GB filtered GitHub Python code; fine‑tuning on APPS train set for 10 epochs with AdamW, batch 256, weight decay 0.05, on 8 A100 GPUs using DeepSpeed ZeRO. Evaluation: beam search (beam width 5) for fine‑tuned models; few‑shot prompt with two examples for GPT‑3. Metrics: test case average, strict accuracy. Hardware: 8 A100 GPUs for fine‑tuning; GPT‑3 inference via OpenAI API.

### Practice trace
1. Action: Constructed APPS benchmark with 10,000 Python problems, 130,000+ test cases, curated from coding challenge sites, deduplicated, and split evenly into train/test; then fine‑tuned GPT‑2 0.1B, GPT‑2 1.5B, and GPT‑Neo 2.7B on the training set after GitHub pretraining, and evaluated using beam search (beam width 5) with test case averages and strict accuracy
   Feedback: GPT‑Neo 2.7B achieved test case averages of 14.68% (introductory), 9.85% (interview), 6.54% (competition); strict accuracy 3.90%, 0.57%, 0.00% respectively; GPT‑3 175B few‑shot solved only 3/5,000 problems; syntax errors decreased exponentially with model scale and fine‑tuning

### Task context
modality: text (natural language problem descriptions) to Python code
scale: 10,000 problems (5,000 train, 5,000 test), 232,421 ground-truth solutions, 131,777 test cases, models up to 2.7B parameters fine-tuned, GPT-3 175B few-shot
constraint: limited to Python; test-case coverage varies; few-shot prompt design may affect GPT-3 performance; models were not evaluated on efficiency, readability, or security aspects; decontamination of GitHub pretraining data may not be exhaustive

### Extraction rationale
Summary drawn from the abstract. The action-feedback pair reflects the paper’s headline dataset introduction and the primary quantitative findings (GPT‑Neo performance and syntax‑error trends). Numerical details are verbatim from the abstract; additional details about dataset scale and model setup are from Sections 3 and 4.

## Why It Matters

Granularity: coarse. Scale: 10,000 problems (5,000 train, 5,000 test), 232,421 ground-truth solutions, 131,777 test cases, models up to 2.7B parameters fine-tuned, GPT-3 175B few-shot Applies under: limited to Python; test-case coverage varies; few-shot prompt design may affect GPT-3 performance; models were not evaluated on efficiency, readability, or security aspects; decontamination of GitHub pretraining data may not be exhaustive

## Evidence

### Source
arXiv:2105.09938 | 2105.09938__4550c66a | group: t5 | section: abstract

### Verbatim quote
"We introduce APPS, a benchmark for code generation. [...] Our benchmark includes 10,000 problems, which range from having simple oneline solutions to being substantial algorithmic challenges. We fine-tune large language models on both GitHub and our training set, and we find that the prevalence of syntax errors is decreasing exponentially as models improve. Recent models such as GPT-Neo can pass approximately 20% of the test cases of introductory problems, so we find that machine learning models are now beginning to learn how to code."

### Classification
domain: agent_evaluation
arXiv: cs.SE
Wikipedia: Software_engineering

### Keywords
code generation, programming benchmarks, language models, test case evaluation, Python, fine-tuning, syntax errors, BLEU, program synthesis, natural language programming

```

---

# RAG 检索增强的长文本切分

**用户提问：** 在构建 RAG 系统时，长文本应该如何进行切分（Chunking）？切片大小对生成效果有什么影响？

**检索关键词：** `retrieval augmented generation chunking context length pipeline`

## 回答
经验库未覆盖 RAG 长文本切分（Chunking）的具体方法，例如应按固定长度、句子、段落、语义边界还是层级结构切分，也未提供推荐的切片大小、重叠长度或具体参数范围。

对于“切片大小对生成效果的影响”，经验库同样没有直接证据。仅可得到一项间接结论：在已编译的 LLM 管线中，检索与推理的**整体结构**会显著影响效果；多跳检索在 HotPotQA 中优于单次检索加推理的结构。但该经验没有讨论文档切片粒度是否、以及如何影响检索质量或最终生成质量。因此，不能据此推导“大切片/小切片何者更优”的结论。

## 思维链 (Reasoning)
1. 检查经验库中与 RAG、检索相关的内容：O-d376ca4a17c00e83 提到单次检索、 多跳检索及其对 HotPotQA 效果的影响。  
2. O-d376ca4a17c00e83 的关注点是管线模块组合与检索跳数，而非文档 Chunking 策略、chunk size、overlap 或这些参数对生成效果的影响。  
3. 其余经验分别涉及运动预测的输入序列长度（O-1e08f04198162f7f）和代码生成中的操作链长度（O-ec3a45b51f6b03a7），不适用于 RAG 文本切分。  
4. 因此，对具体切分方法与切片大小影响的回答只能判定为：**经验库未覆盖**。

## 召回的原始经验全貌 (3 条)

### 经验 ID: `O-d376ca4a17c00e83`

```markdown
---
id: O-d376ca4a17c00e83
created_at: '2026-08-16T15:52:57Z'
summary: '[relation] The choice of pipeline structure (e.g., adding chain‑of‑thought,
  multi‑hop retrieval, or reflection) contributes more to the performance of a compiled
  LLM pipeline than the choice of the underlying language model, as the program’s
  computational graph determines the available reasoning paths while automatic compilation
  aligns prompts to that graph.'
memory_type: semantic
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l2
related_observations:
- id: O-3540404f80d70df6
  relation: complements
  reason: 'Both extracted from arXiv:2310.03714: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:40Z'
- id: O-ecfddf48641e1b9a
  relation: complements
  reason: 'Both extracted from arXiv:2310.03714: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:41Z'
---

## Observation

### Declaration
The choice of pipeline structure (e.g., adding chain‑of‑thought, multi‑hop retrieval, or reflection) contributes more to the performance of a compiled LLM pipeline than the choice of the underlying language model, as the program’s computational graph determines the available reasoning paths while automatic compilation aligns prompts to that graph.

### Narrative
When constructing a LM pipeline for a complex task, developers face two high‑level design choices: the architectural structure of the pipeline (which modules to include and how they interact) and the specific underlying language model. The DSPy experiments reveal that structural decisions can exert a larger influence on final accuracy than the model family, provided that the pipeline is compiled. In the GSM8K case study, switching from a single‑step Predict module (vanilla) to a ChainOfThought (CoT) module more than doubles the accuracy of GPT‑3.5 in the zero‑shot setting (24.0% → 50.0%), yet both programs use exactly the same input‑output signature. Adding a multi‑chain comparison (ThoughtReflection) further raises zero‑shot accuracy to 65.0%, again with the same LM. These gains (up to +41 points) dwarf the differences observed when the same program is run on different LMs. For instance, the CoT program in zero‑shot yields 50.0% with GPT‑3.5 versus 26.7% with Llama2‑13b‑chat — a gap of 23 points, but still smaller than the gain from program structure. When both factors are combined — a well‑chosen program shape and strong LM — compounding emerges: compiling CoT with bootstrapping on GPT‑3.5 reaches 80.3%, while compiling the vanilla module on Llama2‑13b‑chat after two rounds of bootstrapping only reaches 36.5% test accuracy. The HotPotQA study provides a complementary picture: the multi‑hop retrieval program (which generates a search query at each hop and accumulates context) achieves 48.7% dev answer EM with GPT‑3.5 after bootstrapping, while a simpler CoT‑RAG program (single‑shot retrieval followed by reasoning) reaches only 42.3% under the same compiler. Meanwhile, the same multi‑hop bootstrap compiles to 42.0% for Llama2‑13b‑chat, closing the gap with the GPT‑3.5’s CoT‑RAG result. The authors themselves interpret this pattern by stating that “it’s composing the right generic modules, rather than manipulating string prompts, that improves different LMs from 4–20% accuracy to 49–88% accuracy.” This observation underlines the value of DSPy’s modularity: because modules are parameterized and optimizable independently, practitioners can quickly experiment with different pipeline topologies—adding retrieval hops, injecting reflection steps, or replacing ChainOfThought with ProgramOfThought—and compile each variant to see which structure best captures the task’s reasoning requirements. The underlying rationale is that the program’s computational graph determines the search and reasoning paths available to the LM, while bootstrapping adapts the actual prompts (the demonstrations) to that specific graph. A more expressive graph (e.g., iterative retrieval with reasoning at each step) provides the LM with a better scaffold to decompose the problem, whereas a less expressive graph (e.g., single‑step generation) limits the LM even if its pretrained knowledge is strong. Thus, investing effort in designing the right module composition and letting the compiler fill in the prompt‑level details is a more scalable and effective strategy than hand‑crafting prompts for a fixed, sub‑optimal structure.

### Applicability context
summary: Demonstrates that changing from a simple Predict to ChainOfThought or to a reflection module in a compiled pipeline raises accuracy on GSM8K by 26‑41 points for GPT‑3.5, while switching the LM on a well‑composed program results in a 23‑point gap, and that in HotPotQA a multi‑hop program outperforms a single‑step CoT‑RAG by 6+ points after bootstrapping, showing structure is the primary lever.
modality: text
scale: 2 benchmarks, 2‑4 program architectures each, 2 LMs (GPT‑3.5 and Llama2‑13b‑chat)
constraint: the primacy of structure holds when the pipeline is compiled; an unoptimized structure may underperform regardless; the claim is observed on structured reasoning tasks and may not extend to tasks where LM‑specific knowledge dominates

### Extraction rationale
The author’s interpretive statement, combined with the cross‑architecture and cross‑LM results, supports the claim that module composition is the dominant factor. The text does not provide a dedicated causal explanation for why structure matters more than model size; hence r is left null.

## Why It Matters

Claim type: relation. Confidence: medium. Demonstrates that changing from a simple Predict to ChainOfThought or to a reflection module in a compiled pipeline raises accuracy on GSM8K by 26‑41 points for GPT‑3.5, while switching the LM on a well‑composed program results in a 23‑point gap, and that in HotPotQA a multi‑hop program outperforms a single‑step CoT‑RAG by 6+ points after bootstrapping, showing structure is the primary lever. Applies under: the primacy of structure holds when the pipeline is compiled; an unoptimized structure may underperform regardless; the claim is observed on structured reasoning tasks and may not extend to tasks where LM‑specific knowledge dominates

## Evidence

### Source
arXiv:2310.03714 | 2310.03714__357266c1 | group: t10 | section: experiment

### Verbatim quote
Overall, all programs in this table are expressed by composing two to four DSPy modules and teleprompters, and they reveal overall that—in the new paradigm prescribed by DSPy—it’s composing the right generic modules, rather than manipulating string prompts, that improves different LMs from 4–20% accuracy to 49–88% accuracy.

### Classification
domain: agent_general
arXiv: cs.CL
Wikipedia: Natural_language_processing
ACM CCS: Computing methodologies → Artificial intelligence → Natural language processing
CLC: TP18

### Keywords
pipeline architecture, modular programming, language model pipelines, teleprompter, chain‑of‑thought, multi‑hop retrieval, reflection, automatic compilation, abstraction design

```

### 经验 ID: `O-1e08f04198162f7f`

```markdown
---
id: O-1e08f04198162f7f
created_at: '2026-08-16T15:52:57Z'
summary: '[trend] Progressively increasing the input sequence length during autoregressive
  inference reduces error accumulation in long-horizon motion generation compared
  to using a fixed-length input window, because early interaction information is preserved
  throughout the prediction chain.'
memory_type: semantic
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l2
related_observations:
- id: O-525a60c4924a6938
  relation: complements
  reason: 'Both extracted from arXiv:2111.12073: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:10Z'
- id: O-993b7efceb86bc28
  relation: complements
  reason: 'Both extracted from arXiv:2111.12073: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:10Z'
---

## Observation

### Declaration
Progressively increasing the input sequence length during autoregressive inference reduces error accumulation in long-horizon motion generation compared to using a fixed-length input window, because early interaction information is preserved throughout the prediction chain.

### Narrative
When autoregressive models are used for long-horizon sequence prediction, a common failure mode is error accumulation: small inaccuracies at early predicted steps compound into large deviations at later steps because the model receives its own noisy predictions as input. In the context of multi-person 3D motion prediction, this problem is amplified because early errors in predicting one person's trajectory can cascade into incorrect interaction dynamics with other persons. The Multi-Range Transformers paper introduces and validates a simple yet effective strategy to mitigate this accumulation: during training and inference, progressively increase the length of the input sequence fed to the encoders as prediction proceeds, rather than using a fixed window that slides along the generated sequence. Specifically, the model receives 1 second of history (15 frames) and predicts the next second; then, instead of taking only the last second of prediction as new input for the next step, the model sees all previously observed and predicted motion concatenated – first 1 second, then 2 seconds, then 3 seconds, and so on. This 'progressive input' strategy contrasts with the standard fixed-length approach where the model always receives a window of the most recent 1 second of motion. An ablation on CMU-Mocap (Table 6) demonstrates the impact: at a 3-second prediction horizon, the progressive strategy yields an MPJPE of 2.18 (in 0.1 m) compared to 2.91 for the fixed-length baseline, a 25% reduction in error, while at 1 second both methods are identical (0.96). The improvement emerges exclusively at longer horizons (2 seconds: 1.57 vs 1.91), confirming that the strategy specifically addresses error accumulation. The mechanism, as explained by the authors, is that maintaining early motion as input preserves crucial interactive information – the initial spatial relationships and relative trajectories that define social grouping and intent – which would otherwise be lost when old frames are discarded by a sliding window. In the multi-range architecture, this preserved information benefits both the local-range encoder (which uses DCT on each person's full motion history to enforce smoothness) and the global-range encoder (which performs self-attention across all persons and times, so longer context strengthens interaction modeling). The transformer's permutation invariance and lack of recurrence make it naturally amenable to variable-length input, removing any architectural barrier to progressive lengthening. The strategy was tested under the following specific conditions: the model predicts 15 frames per autoregressive step from an offset Δx representation; training includes random-length sampling to prevent length bias; the decoder always uses only the most recent pose as query (so the increased input only enriches the encoder features); and the total prediction extends to 3 seconds (45 frames), with error measured in world-coordinate MPJPE. The evidence is limited to CMU-Mocap data but the principle is generalizable to any autoregressive transformer for temporal sequences where long-term dependencies are important. The strategy introduces a computational cost trade-off: encoder processing time grows with accumulated sequence length, but this is acceptable for batch inference on moderate-length sequences. The authors reported this finding with moderate confidence, as it was shown via a single controlled ablation.

### Causal explanation (r)
Maintaining early motion frames as input preserves initial interaction patterns between individuals, preventing information loss that occurs with fixed-length sliding windows and thereby reducing cascading prediction errors.

confidence (μ_r): medium
depth: deep
depth rationale: Identifies a specific causal chain: discarding early frames loses interaction information → missing context causes larger prediction errors → errors accumulate over autoregressive steps. This goes beyond merely stating that progressive input improves performance.

### Applicability context
summary: Improves long-term 3D motion prediction accuracy by 25% at 3 seconds by progressively lengthening encoder input rather than using a fixed window, effectively preserving early trajectory information that prevents cascade errors.
modality: 3D skeleton joint offsets
scale: 1-3 second prediction, 15 frames per step, single dataset ablation
constraint: assumes a transformer encoder that accepts variable-length input without recurrence; tested only on smooth CMU-Mocap motions with up to 3-second horizon; computational cost grows with sequence length, limiting applicability to very long horizons

### Extraction rationale
The declaration is supported by a controlled ablation in Table 6 showing identical 1-second error but diverging errors at longer horizons. The causal explanation (r) is directly provided in the method section as the rationale for the progressive strategy design.

## Why It Matters

Claim type: trend. Confidence: medium. Improves long-term 3D motion prediction accuracy by 25% at 3 seconds by progressively lengthening encoder input rather than using a fixed window, effectively preserving early trajectory information that prevents cascade errors. Applies under: assumes a transformer encoder that accepts variable-length input without recurrence; tested only on smooth CMU-Mocap motions with up to 3-second horizon; computational cost grows with sequence length, limiting applicability to very long horizons

## Evidence

### Source
arXiv:2111.12073 | 2111.12073__ce33ea55 | group: t1 | section: method

### Verbatim quote
The advantage of such design is that when predicting longer motions, we still maintain the early motions as inputs to the model, instead of using a fixed length to predict each of the future time steps [45,44], which may cause the loss of early interactive information. Through the experiment, we find this strategy could largely reduce the error accumulation.

### Classification
domain: agent_domain_app
arXiv: cs.CV
Wikipedia: Computer_vision
ACM CCS: Computing methodologies → Artificial intelligence → Computer vision → Activity recognition and understanding
CLC: TP391.4

### Keywords
autoregressive prediction, error accumulation, progressive input, long-term motion generation, transformer inference

```

### 经验 ID: `O-ec3a45b51f6b03a7`

```markdown
---
id: O-ec3a45b51f6b03a7
created_at: '2026-08-16T15:52:57Z'
summary: '[trend] Code generation performance degrades exponentially as the number
  of chained sequential operations in a function specification increases, unlike human
  programmers who can compose operations without length-dependent difficulty.'
memory_type: semantic
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l2
related_observations:
- id: O-9f873b661b8bb44d
  relation: complements
  reason: 'Both extracted from arXiv:2107.03374: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:05Z'
---

## Observation

### Declaration
Code generation performance degrades exponentially as the number of chained sequential operations in a function specification increases, unlike human programmers who can compose operations without length-dependent difficulty.

### Narrative
Real-world programming tasks often require composing multiple simple operations into a single function. To test how well a code generation model handles this, the paper constructs synthetic programming problems by chaining together 13 basic string manipulation building blocks (e.g., 'convert to lowercase', 'remove every third character', 'replace spaces with exclamation points'). Each problem is formed by concatenating a random sequence of these building blocks into a docstring and implementing the corresponding sequential transformations. The model's ability to synthesize the correct code is measured as a function of the number of chained components. The result is striking: Codex-12B's pass rate drops exponentially with each additional building block, roughly by a factor of 2-3 per added component. This means that a relatively simple composition that a human programmer could handle with ease becomes extremely difficult for the model when the chain length exceeds a few steps. The authors explicitly note that this behavior is uncharacteristic of a human programmer, who should be able to correctly implement a chain of arbitrary length if they can handle a chain of length two. The experiment is carried out on synthetically generated tasks, so the task distribution is controlled, isolating the effect of composition length. No causal explanation is provided for why the model degrades exponentially; the paper merely observes the phenomenon. This finding has important implications for the robustness of code generation tools: they are currently unreliable for implementing functions that require chaining many simple operations sequentially, limiting their usefulness for complex but well-defined tasks. The context is specifically Python string manipulation tasks with a clearly defined and deterministic building-block set, and the model evaluated is a 12B parameter Codex fine-tuned on GitHub Python code. Whether the exponential degradation generalizes to other types of composition (e.g., arithmetic operations, data structure transformations) or to models with different architectures remains an open question.

### Applicability context
summary: Reveals that code generation reliability drops exponentially as the number of sequential operations in a docstring grows, showing that current models cannot robustly compose simple operations—a core limitation for multi-step task synthesis.
modality: code
scale: synthetic tasks built from 13 string manipulation blocks, Codex-12B
constraint: tested only on synthetic string-manipulation chains in Python; performance on other types of sequential operations (e.g., numeric, object transformations) or with different models is not assessed

### Extraction rationale
The declaration is grounded in the author's explicit interpretation of the synthetic experiment results showing exponential pass-rate drop (declaration basis). The author does not propose a causal mechanism for this degradation, so r is set to null.

## Why It Matters

Claim type: trend. Confidence: medium. Reveals that code generation reliability drops exponentially as the number of sequential operations in a docstring grows, showing that current models cannot robustly compose simple operations—a core limitation for multi-step task synthesis. Applies under: tested only on synthetic string-manipulation chains in Python; performance on other types of sequential operations (e.g., numeric, object transformations) or with different models is not assessed

## Evidence

### Source
arXiv:2107.03374 | 2107.03374__6ec0ed49 | group: t5 | section: limitations

### Verbatim quote
We find that as the number of chained building blocks in the docstring increases, model performance decreases exponentially. This behavior is uncharacteristic of a human programmer, who should be able to correctly implement a program for a chain of arbitrary length if they can do so for a chain of length two.

### Classification
domain: agent_software_eng
arXiv: cs.SE
Wikipedia: Program_synthesis
ACM CCS: Computing methodologies → Artificial intelligence → Natural language processing → Natural language generation
CLC: TP311

### Keywords
code generation, compositionality, chained operations, exponential degradation, synthetic tasks, HumanEval

```

---

# 端侧多模态模型部署瓶颈

**用户提问：** 将多模态模型部署到端侧（如手机）时，最主要的性能瓶颈是什么？有哪些工程优化手段？

**检索关键词：** `on-device deployment memory latency bottleneck quantization mobile`

## 回答

在所给经验覆盖的手机端图文多模态模型场景中，**最主要的性能瓶颈是 LLM 的预填充（prefilling）阶段**，即对**图像视觉 token 与文本 token 的联合编码**，而不是逐 token 的自回归解码。原因是图像会产生较多视觉 token，这些 token 需经过整个 LLM 的多层计算与注意力操作；而解码可利用 KV Cache，且优化后吞吐已可超过人类阅读速度。实测中，手机端编码/首 token 前延迟仍约为 3 秒，而解码可达 8.2 token/s；因此，进一步降低预填充成本对用户感知延迟更关键。〔O-4b49bfedf1b4e35a〕

此外，端侧部署还受以下基础约束影响：

- **内存容量与分页（paging）**：手机通常只有 12–16GB RAM，同时加载 ViT 和 LLM 会造成过度分页，显著拖慢图像处理。〔O-7e27554b6e7f69b1〕
- **CPU/GPU 算力较弱、CPU 核分配未充分调优**：若运行时配置不匹配芯片，解码吞吐会很低。〔O-7e27554b6e7f69b1〕
- **端侧芯片 ISA 与通用预编译二进制不匹配**：未在目标设备原生编译时，编码和解码性能明显受限。〔O-7e27554b6e7f69b1〕
- **视觉编码未利用专用 NPU**：若 ViT 也在通用处理器执行，会占用大量时间。〔O-7e27554b6e7f69b1〕
- **热限制**也是手机端的客观约束。〔O-7e27554b6e7f69b1〕

可采用的工程优化手段如下：

1. **低比特量化，优先解决内存可部署性**  
   将 16-bit 模型量化为 4-bit，可将约 16–17GB 的模型内存降至约 5GB，使 8B 级模型能够放入手机内存。〔O-7e27554b6e7f69b1〕

2. **ViT 与 LLM 顺序加载、及时释放**  
   先加载 ViT 完成图像编码，释放 ViT 后再加载 LLM 做生成，避免两者并存导致分页。该做法将图像处理时间从 45.2s 降至 31.5s。〔O-7e27554b6e7f69b1〕

3. **在目标设备上原生编译**  
   不直接使用通用预编译包，而在手机端或针对目标设备重新编译，使代码适配设备 ISA。经验中，这使编码延迟从 50.5s 降至 17.0s，解码从 1.3 提升至 3.2 token/s。〔O-7e27554b6e7f69b1〕

4. **自动搜索运行时配置与 CPU 核分配**  
   针对具体 SoC 自动调优计算参数、CPU 核使用方式等，以提升 CPU 利用率。案例中解码吞吐由 3.2 提升至 8.2 token/s。〔O-7e27554b6e7f69b1〕

5. **将视觉编码卸载到 NPU**  
   通过 NPU 后端运行 ViT，LLM 仍可在 CPU 上运行。案例中视觉编码时间由 3.7s 降至 1.3s。〔O-7e27554b6e7f69b1〕

6. **从模型/输入侧减少视觉 token 数量**  
   由于当前关键路径是 LLM 预填充，应采用更高效的视觉编码，减少送入 LLM 的视觉 token，例如动态 token 裁剪或更低 token 数的重采样器。已有经验中的自适应视觉编码可把每个切片的 1024 token 压缩到 96 token，总视觉 token 控制在 96–960。〔O-4b49bfedf1b4e35a；O-64050f57a14a2c73〕

7. **进一步加速 LLM 编码，而非只优化解码**  
   在解码已达到约 8.2 token/s 的情况下，更有价值的方向是让 GPU/NPU 参与 LLM prefilling；已有案例仅将 NPU 用于 ViT，LLM 编码仍主要在 CPU，因而这是尚未充分利用的优化空间。〔O-4b49bfedf1b4e35a〕

上述结论有适用边界：它们来自特定的 8.5B、4-bit 量化模型和手机端运行环境；不同模型的视觉 token 数、不同芯片及多轮长对话负载可能改变瓶颈分布。尤其是当前 NPU 加速在经验中仅覆盖 ViT，未覆盖 LLM 推理。〔O-4b49bfedf1b4e35a；O-7e27554b6e7f69b1〕

## 思维链 (Reasoning)

1. 先依据 **O-4b49bfedf1b4e35a** 的直接结论：多设备实测表明，端侧 MLLM 的主要计算瓶颈是图像和文本 token 的 LLM 预填充，而非逐 token 解码；因此将“首 token 前编码延迟”作为首要优化目标。  
2. 再依据 **O-7e27554b6e7f69b1**，识别支撑该瓶颈的系统因素：内存分页、ISA/编译不匹配、CPU 配置未调优、ViT 未使用 NPU，以及移动端算力和热约束。  
3. 将 **O-7e27554b6e7f69b1** 中经实测验证的部署动作整理为量化、顺序加载、原生编译、自动配置搜索和 ViT-NPU 卸载。  
4. 根据 **O-4b49bfedf1b4e35a** 对未来方向的建议，补充“减少视觉 token”和“加速 LLM 编码”的优先级；并以 **O-64050f57a14a2c73** 中的视觉 token 压缩实例说明其工程落点。  
5. 最后保留经验中的适用范围：结果依赖具体模型、量化方式、框架和旗舰手机，且 NPU 加速尚只覆盖 ViT。〔O-4b49bfedf1b4e35a；O-7e27554b6e7f69b1〕

## 召回的原始经验全貌 (5 条)

### 经验 ID: `O-7e27554b6e7f69b1`

```markdown
---
id: O-7e27554b6e7f69b1
created_at: '2026-08-16T15:52:57Z'
summary: '[conditional] A suite of end‑side deployment optimizations—sequential memory
  loading of ViT and LLM, on‑device compilation, automatic CPU‑core configuration
  search, and NPU offloading for visual encoding—can collectively reduce MLLM encoding
  latency by over 97% and boost decoding throughput more than 5× on a flagship smartphone,
  surpassing human reading speed.'
memory_type: semantic
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l2
related_observations:
- id: O-64050f57a14a2c73
  relation: complements
  reason: 'Both extracted from arXiv:2408.01800: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:59:00Z'
---

## Observation

### Declaration
A suite of end‑side deployment optimizations—sequential memory loading of ViT and LLM, on‑device compilation, automatic CPU‑core configuration search, and NPU offloading for visual encoding—can collectively reduce MLLM encoding latency by over 97% and boost decoding throughput more than 5× on a flagship smartphone, surpassing human reading speed.

### Narrative
Deploying a multimodal large language model on a consumer smartphone is extremely challenging due to severe memory constraints (typically 12–16 GB RAM), slower CPU/GPU cores, and thermal limitations. Without specialized optimizations, even a heavily quantized MLLM may take over a minute to encode an image and generate text at only 1–2 tokens per second, ruining the user experience. The paper systematically tackles this problem by applying a series of deployment optimizations on a Xiaomi 14 Pro (Snapdragon 8 Gen 3) and demonstrates that acceptable real‑world performance can be attained.

The starting point is a 4‑bit quantized MiniCPM‑Llama3‑V 2.5 model (Q4_K_M mode in the llama.cpp framework), which reduces memory usage from 16–17 GB to around 5 GB, fitting within a phone’s RAM. However, even with quantization, the baseline deployment suffers from a text encoding latency of 64.2 seconds and a decoding throughput of only 1.3 tokens per second, which is far too slow.

The first optimization, memory usage optimization, addresses the fact that loading both the vision transformer (ViT) and the large language model (LLM) simultaneously into memory causes excessive paging. Instead, the ViT is loaded first, the image is encoded, the ViT is released, and then the LLM is loaded for token generation. This sequential loading reduces image processing time from 45.2 s to 31.5 s, a 30% improvement.

Next, compilation optimization is applied: re‑compiling the model natively on the target device rather than using pre‑built binaries brings substantial speedups because the generated code better matches the phone’s instruction set architecture (ISA). This step cuts encoding latency from 50.5 s to 17.0 s and boosts decoding throughput from 1.3 to 3.2 tokens/s.

Configuration optimization further tailors the inference engine to the specific chip by automatically searching for optimal parameters (e.g., assigning computation to different CPU cores). With tuned configuration, decoding throughput jumps from 3.2 to 8.2 tokens/s, exceeding the typical human reading speed and making conversational interaction feel responsive.

Finally, NPU acceleration offloads the ViT encoding to the phone’s dedicated neural processing unit via the QNN backend, while the LLM continues to run on CPU with llama.cpp. This reduces the visual encoding time from 3.7 s to 1.3 s, cutting it by nearly two‑thirds. The cumulative effect of all optimizations transforms the previously unusable system into one that can deliver first‑token latency of a few seconds and a sustained output rate faster than a human reads.

The authors observe that these techniques work synergistically: memory optimization prevented paging, compilation aligned with the ISA, configuration tuning maximized CPU throughput, and NPU offloading freed the CPU for LLM computation. They also note that the current bottleneck shifts to LLM prefilling (encoding image and text tokens) once decoding is accelerated, suggesting that future work should focus on reducing visual token counts or further accelerating LLM encoding via GPUs/NPUs.

It is important to recognize that the reported numbers were measured on a specific flagship phone (Snapdragon 8 Gen 3) with a particular GPU and NPU. Results on mid‑range devices or those with different chip vendors (e.g., MediaTek Dimensity 9300 also tested, shown in Figure 7) may vary. The NPU acceleration currently only applies to ViT encoding; LLM inference on NPU was not explored due to framework limitations. Moreover, the decoding throughput of 8.2 tokens/s, while impressive for a phone, is still modest compared to server‑grade hardware, so complex multi‑turn reasoning tasks may feel sluggish. Nevertheless, this experience provides a concrete recipe and set of ballpark figures: with careful quantization, sequential memory management, on‑device compilation, automatic configuration search, and selective NPU offloading, an 8B‑class MLLM can indeed run on a modern smartphone with acceptable latency and speed, opening the door to genuine on‑device multimodal assistants.

### Causal explanation (r)
Sequential loading eliminates memory paging, on‑device compilation aligns code to the phone’s ISA, configuration search maximizes CPU utilisation, and NPU offloading leverages dedicated hardware for visual encoding, each removing a different bottleneck.

confidence (μ_r): medium
depth: deep
depth rationale: Identifies the specific mechanism (paging, ISA mismatch, sub‑optimal CPU allocation, lack of NPU usage) behind each optimisation, linking the performance gains to clear engineering causes.

### Applicability context
summary: Enables a GPT‑4V‑level 8.5B MLLM to run on a Xiaomi 14 Pro with encoding latency reduced from 64.2s to ~3s and decoding throughput increased from 1.3 to 8.2 tokens/s, making mobile multimodal interaction practical.
modality: text + images (multimodal inference)
scale: Xiaomi 14 Pro (Snapdragon 8 Gen 3), vivo X00 Pro (Dimensity 9300), MacBook Pro M1; model 8.5B (Q4_K_M quantization)
constraint: results measured on two specific flagship smartphones; NPU acceleration only for ViT (not LLM); decoding throughput of 8.2 tokens/s may still be insufficient for very long conversations; requires device‑specific compilation and configuration tuning

### Extraction rationale
The author presents quantitative results for each optimisation technique and directly links the performance improvements to the described engineering changes. The causal reasoning is drawn from the deployment section (5.3).

## Why It Matters

Claim type: conditional. Confidence: high. Enables a GPT‑4V‑level 8.5B MLLM to run on a Xiaomi 14 Pro with encoding latency reduced from 64.2s to ~3s and decoding throughput increased from 1.3 to 8.2 tokens/s, making mobile multimodal interaction practical. Applies under: results measured on two specific flagship smartphones; NPU acceleration only for ViT (not LLM); decoding throughput of 8.2 tokens/s may still be insufficient for very long conversations; requires device‑specific compilation and configuration tuning

## Evidence

### Source
arXiv:2408.01800 | 2408.01800__9876f844 | group: t8 | section: end-side deployment

### Verbatim quote
"Memory usage optimization: sequential loading [...] results in a notable reduction of image processing time from 45.2s to 31.5s. Compilation optimization: encoding latency reduced from 50.5s to 17.0s, decoding throughput from 1.3 tokens/s to 3.2 tokens/s. Configuration optimization: decoding throughput surged from 3.2 tokens/s to an impressive 8.2 tokens/s, surpassing the typical human reading speed. NPU acceleration: visual encoding time decreasing from 3.7s to 1.3s."

### Classification
domain: agent_general
arXiv: cs.LG
Wikipedia: Deep_learning
ACM CCS: Computing methodologies → Artificial intelligence → Natural language processing → Natural language generation
CLC: TP18

### Keywords
end-side MLLM, efficient multimodal model, adaptive visual encoding, hallucination reduction, multilingual multimodal, mobile deployment, token compression, RLHF alignment, small language model, GPT-4V level

```

### 经验 ID: `O-4b49bfedf1b4e35a`

```markdown
---
id: O-4b49bfedf1b4e35a
created_at: '2026-08-16T15:52:57Z'
summary: '[property] On end‑side devices, the primary speed bottleneck of MLLM inference
  is LLM prefilling (encoding image and text tokens), not token‑by‑token decoding,
  making the reduction of visual token counts or the acceleration of LLM encoding
  the most impactful levers for improving user‑perceived latency.'
memory_type: semantic
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l2
related_observations:
- id: O-64050f57a14a2c73
  relation: complements
  reason: 'Both extracted from arXiv:2408.01800: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:59:00Z'
- id: O-e92ec79b387d047b
  relation: complements
  reason: 'Both extracted from arXiv:2408.01800: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:59:01Z'
- id: O-04175ca36ee69ef2
  relation: complements
  reason: Both classified under agent_general, from arXiv:2407.07726 and arXiv:2408.01800
    respectively — different papers addressing the same problem area.
  linked_at: '2026-08-16T15:59:20Z'
- id: O-18d9aa97167f94de
  relation: complements
  reason: Both classified under agent_general, from arXiv:2408.01800 and arXiv:2408.08435
    respectively — different papers addressing the same problem area.
  linked_at: '2026-08-16T15:59:20Z'
---

## Observation

### Declaration
On end‑side devices, the primary speed bottleneck of MLLM inference is LLM prefilling (encoding image and text tokens), not token‑by‑token decoding, making the reduction of visual token counts or the acceleration of LLM encoding the most impactful levers for improving user‑perceived latency.

### Narrative
When deploying multimodal large language models (MLLMs) on resource‑constrained end‑side devices such as smartphones, understanding where the computational bottlenecks reside is essential for guiding optimization efforts. The paper breaks down the inference process into two main phases: (1) LLM prefilling, which encodes the input image and text tokens to produce the initial hidden states; and (2) token‑by‑token decoding, which generates the output sequence auto‑regressively. Intuitively, one might expect decoding to dominate the user‑perceived latency because it involves many sequential steps. However, the experimental results presented in Figure 7 for multiple devices reveal a different picture.

Across all tested devices—Xiaomi 14 Pro (Snapdragon 8 Gen 3), vivo X00 Pro (Mediatek Dimensity 9300), and MacBook Pro (M1)—the encoding latency (the time from receiving the image to the start of decoding) is the primary bottleneck, not the decoding throughput. After applying the full suite of deployment optimizations, the Xiaomi 14 Pro achieves a decoding throughput of 8.2 tokens/s, which already surpasses the typical human reading speed. In contrast, the encoding latency remains at approximately 3 seconds (including model loading and visual encoding), implying that the model spends most of its time before the first token is generated. Even on the Mac M1, which has a more powerful CPU/GPU, the encoding latency is still around 3.5 seconds while decoding throughput easily exceeds 10 tokens/s (Figure 7b). This pattern shows that the gains from further speeding up decoding would be marginal from the user’s perspective, whereas reducing prefilling time would directly make the interaction feel snappier.

The paper attributes this bottleneck to the large number of visual tokens that must be processed by the LLM during prefilling. Although the adaptive visual encoding method compresses images to a modest token count (96–960 tokens), these tokens still need to be processed alongside the text prompt by the entire LLM stack, which involves many transformer layers and attention operations. Decoding, in contrast, processes only one token at a time and benefits from the KV cache. Therefore, the prefilling stage is computationally heavier and becomes the limiting factor on devices with weaker compute.

This observation leads to a clear strategic recommendation: research and engineering efforts aimed at improving end‑side MLLM responsiveness should focus on reducing the LLM prefilling cost. Two promising directions are highlighted: (1) developing more efficient visual encoding methods that generate even fewer visual tokens (e.g., dynamic token pruning or hard‑coded resamplers with lower token counts), and (2) better leveraging hardware accelerators like GPUs or NPUs for the LLM encoding step, which currently runs predominantly on the CPU in the studied setup. The authors note that while NPU acceleration was successfully applied to ViT encoding, the LLM prefilling stage was not offloaded and remains on CPU, representing an untapped opportunity.

It should be noted that this bottleneck profile is specific to the MiniCPM‑Llama3‑V 2.5 architecture and the execution environment (llama.cpp, 4‑bit quantization). Other models with even fewer visual tokens or with different attention mechanisms might shift the balance. Moreover, the measurements reflect single‑turn inference; in multi‑turn conversational settings, the combined decoding time over many turns could become more significant. Nevertheless, for the common scenario of answering a question about a photo, the experience reliably pinpoints prefilling as the critical path and offers clear guidance for the community: cutting visual token count and accelerating LLM encoding are the keys to real‑time on‑device multimodal interaction.

### Applicability context
summary: Characterizes the inference bottleneck of an 8.5B MLLM on smartphones, showing that decoding throughput (8.2 tokens/s) already exceeds human reading speed, while encoding latency (~3s) dominates, directing optimization efforts toward reducing visual tokens or speeding up LLM prefilling.
modality: text + images
scale: Xiaomi 14 Pro, vivo X00 Pro, MacBook Pro M1; model 8.5B Q4_K_M; encoding latency ~3-3.5s, decoding throughput 8.2-13.5 tokens/s
constraint: profile observed for a specific 8B MLLM with quantized llama.cpp deployment; may differ for models with drastically different visual token counts; did not measure multi‑turn overhead; LLM prefilling on NPU/GPU was not explored

### Extraction rationale
The author directly states the bottleneck observation based on latency and throughput measurements in Figure 7, and draws design recommendations; no explicit causal explanation for the prefilling dominance is given beyond the implicit computational load of processing many tokens at once.

## Why It Matters

Claim type: property. Confidence: medium. Characterizes the inference bottleneck of an 8.5B MLLM on smartphones, showing that decoding throughput (8.2 tokens/s) already exceeds human reading speed, while encoding latency (~3s) dominates, directing optimization efforts toward reducing visual tokens or speeding up LLM prefilling. Applies under: profile observed for a specific 8B MLLM with quantized llama.cpp deployment; may differ for models with drastically different visual token counts; did not measure multi‑turn overhead; LLM prefilling on NPU/GPU was not explored

## Evidence

### Source
arXiv:2408.01800 | 2408.01800__9876f844 | group: t8 | section: end-side deployment

### Verbatim quote
"Upon analyzing Fig. 7, it becomes evident that the current computation bottleneck primarily stems from LLM prefilling, which mainly involves encoding image and text tokens for LLM inference. Promising research directions involve developing more efficient visual encoding methods with fewer visual tokens, and better leveraging GPU/NPU acceleration for LLM encoding."

### Classification
domain: agent_general
arXiv: cs.LG
Wikipedia: Deep_learning
ACM CCS: Computing methodologies → Artificial intelligence → Natural language processing → Natural language generation
CLC: TP18

### Keywords
end-side MLLM, efficient multimodal model, adaptive visual encoding, hallucination reduction, multilingual multimodal, mobile deployment, token compression, RLHF alignment, small language model, GPT-4V level

```

### 经验 ID: `O-64050f57a14a2c73`

```markdown
---
id: O-64050f57a14a2c73
created_at: '2026-08-16T15:52:57Z'
summary: '[coarse] Develops an efficient MLLM series deployable on mobile phones through
  adaptive visual encoding, multi‑stage training, RLAIF‑V alignment, and end‑side
  deployment optimizations, achieving 65.1 on OpenCompass (outperforming GPT‑4V‑1106
  at 63.5) while using only 8.5B parameters.'
memory_type: procedural
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l1
related_observations:
- id: O-4b49bfedf1b4e35a
  relation: complements
  reason: 'Both extracted from arXiv:2408.01800: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:59:00Z'
- id: O-75981c82413e8234
  relation: complements
  reason: 'Both extracted from arXiv:2408.01800: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:59:00Z'
- id: O-7e27554b6e7f69b1
  relation: complements
  reason: 'Both extracted from arXiv:2408.01800: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:59:00Z'
- id: O-86be11708612b644
  relation: complements
  reason: 'Both extracted from arXiv:2408.01800: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:59:00Z'
- id: O-9498080307de9251
  relation: complements
  reason: 'Both extracted from arXiv:2408.01800: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:59:01Z'
- id: O-9c7be9446f814f9c
  relation: complements
  reason: 'Both extracted from arXiv:2408.01800: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:59:01Z'
- id: O-51ed81308e03abc2
  relation: complements
  reason: Both classified under agent_domain_app, from arXiv:2407.17490 and arXiv:2408.01800
    respectively — different papers addressing the same problem area.
  linked_at: '2026-08-16T15:59:13Z'
---

## Observation

### Narrative
This experience describes the design, training, and deployment of the MiniCPM-V series—a family of efficient multimodal large language models (MLLMs) specifically engineered to run on end-side devices such as mobile phones while achieving performance comparable to or surpassing strong commercial models like GPT-4V-1106 and Gemini Pro. The core problem is that state‑of‑the‑art MLLMs typically contain billions of parameters and demand prohibitive computational resources, limiting their deployment to cloud servers and excluding mobile, offline, and privacy‑sensitive scenarios. MiniCPM-V addresses this through a combination of architectural innovations, a three‑stage pre‑training recipe, supervised fine‑tuning (SFT), alignment with RLAIF‑V, and a suite of end‑side deployment optimizations. The latest model, MiniCPM‑Llama3‑V 2.5 (8.5B parameters), is built on Llama3‑Instruct 8B and adopts an adaptive high‑resolution visual encoding scheme: input images are partitioned into slices whose layout respects the original aspect ratio, each slice is resized to match the ViT pre‑training area, position embeddings are interpolated, and a shared compression layer (a perceiver resampler with one‑layer cross‑attention) squeezes the per‑slice visual tokens (1024) down to 96 queries. A spatial schema with special tokens (<slice> and <\slice>) and newline separators informs the LLM of the slice arrangement. Pre‑training proceeds in three stages: Stage‑1 warms up the compression layer on 200M image‑caption pairs at 224×224; Stage‑2 trains the full visual encoder at 448×448 on another 200M captions; Stage‑3 introduces the adaptive encoding strategy and adds OCR data to enhance fine‑grained perception, still keeping the LLM frozen. To improve data quality, noisy captions are rewritten into QA pairs by an auxiliary LLM fine‑tuned on GPT‑4 annotations. Training efficiency is further boosted by packing multiple samples into fixed‑length sequences, giving a 2–3× speedup. Supervised fine‑tuning uses a two‑part dataset: Part‑1 focuses on short‑response recognition tasks; Part‑2 injects long‑response, instruction‑following, and multilingual data. All parameters are unfrozen during SFT. For trustworthiness, RLAIF‑V generates 10 responses from the policy model, decomposes each into atomic claims, scores them with an open‑source MLLM (LLaVA‑NeXT‑Yi 34B), and constructs 6K preference pairs for DPO. Multilingual capability is extended to over 30 languages by pre‑training only on English/Chinese multimodal data and then using a lightweight multilingual SFT set (<0.5% of total SFT data). On the deployment side, the 16‑bit model is quantized to 4‑bit using GGML’s Q4_K_M to ~5GB, and a series of device‑targeted optimizations are applied on a Xiaomi 14 Pro: sequential loading of ViT and LLM reduces image encoding time from 45.2s to 31.5s; on‑device compilation cuts encoding latency from 50.5s to 17.0s and improves decoding throughput from 1.3 to 3.2 tokens/s; automatic configuration search pushes throughput to 8.2 tokens/s (above human reading speed); and NPU acceleration via QNN lowers ViT encoding to 1.3s, with end‑to‑end text encoding reaching 1.3 tokens/s decoding on the same device. These practices collectively yield an MLLM that, on the OpenCompass benchmark (11 tasks), scores 65.1, outperforming GPT‑4V‑1106 (63.5), Gemini Pro (62.9) and Claude 3 Opus (57.7), while achieving an OCRBench score of 725 (vs. 645 for GPT‑4V) and a response‑level hallucination rate of 10.3% on Object HalBench (vs. 13.6% for GPT‑4V). The practice is bounded by the compute of current flagship phones; the model still relies on a 8.5B‑parameter LLM, so small devices with <12GB RAM require aggressive quantization; real‑time interaction remains limited by LLM pre‑fill (image+text encoding), and the current deployment prioritises CPU/NPU, while GPU acceleration on phones is not yet competitive. The visual encoder is frozen after pre‑training, and the adaptive encoding introduces an upper bound of 1.8M pixels, which may be insufficient for extremely high‑resolution inputs (e.g., 4K diagrams).

### Practice environment
Visual encoder: SigLIP SoViT‑400m/14. LLM: Llama3‑Instruct 8B (or MiniCPM 2B in earlier versions). Pre‑training data: 410M English captions (COCO, CC3M/12M, LAION, etc.) + 110M Chinese captions + 39M OCR/knowledge (WIT, SynthText, etc.) + 11M Chinese OCR. SFT: Part‑1 5.3M samples (short VQA, captioning, OCR, grounding); Part‑2 3M (long‑response, instruction, text‑only). RLAIF‑V: 6K preference pairs from 3K images, OmniLMM/LLaVA‑NeXT‑Yi 34B as scorer. Multilingual SFT: 90K samples across 36 languages. Deployment targets: Xiaomi 14 Pro (Snapdragon 8 Gen 3, 12‑16GB RAM), vivo X00 Pro (Dimensity 9300), MacBook Pro M1. Evaluation: OpenCompass (MME, MMBench, MMMU, MathVista, LLaVA Bench, etc.), OCRBench, TextVQA, DocVQA, Object HalBench. Quantization: GGML Q4_K_M. Frameworks: llama.cpp, QNN (NPU). Hyperparameters: adaptive encoding slice limit N<10; compression layer query number 96; pre‑training batch packing; DPO pairs 6K.

### Practice trace
1. Action: Integrated adaptive visual encoding that partitions high‑resolution images into slices, resizes to match ViT pre‑training area, interpolates position embeddings, compresses each slice from 1024 to 96 tokens, and adds spatial schema tokens to indicate slice positions
   Feedback: Enabled processing of 1.8M‑pixel images at any aspect ratio with far fewer visual tokens (96–960) than competitors (e.g., 1728–2880 for LLaVA‑NeXT‑Llama3‑8B), contributing to strong OCR performance (OCRBench 725, TextVQA 76.6, DocVQA 84.8) and overall OpenCompass 65.1
2. Action: Trained with three‑stage pre‑training (compression layer warmup → 448×448 encoder training → adaptive high‑resolution with OCR data), followed by two‑part SFT (short‑response then long‑response/instruct), and applied RLAIF‑V using the policy model itself to generate 10 candidate responses, decomposed into atomic claims scored by an open‑source MLLM, and trained with DPO on 6K preference pairs
   Feedback: Achieved OpenCompass 65.1, surpassing GPT‑4V‑1106 (63.5), Gemini Pro (62.9); hallucination rate on Object HalBench dropped to 10.3% response‑level (GPT‑4V: 13.6%) without harming general capabilities; OpenCompass even improved by 0.6 points after RLAIF‑V
3. Action: Applied a suite of end‑side deployment optimizations on Snapdragon 8 Gen 3 (Xiaomi 14 Pro): 4‑bit quantization → sequential ViT/LLM loading → on‑device compilation → automatic config search → NPU acceleration for ViT
   Feedback: Reduced model memory to ~5GB; image processing time fell from 45.2s to 31.5s; encoding latency reduced from 50.5s to 17.0s; decoding throughput rose from 1.3 to 8.2 tokens/s (above human reading speed); NPU ViT encoding reached 1.3s, making the model practically usable on a mobile phone

### Task context
modality: images and text (multimodal)
scale: 3 models released (2.8B and 8.5B params); 11+ benchmarks; 1.8M pixel max input; deployed on Snapdragon 8 Gen 3 phone; pre‑training on 570M image‑text pairs; SFT on 5.3M samples
constraint: requires at least 5GB RAM after 4‑bit quantization; real‑time speed limited by LLM pre‑fill; highest performance currently on Snapdragon 8 Gen 3 with NPU; adaptive encoding caps at ~1.8M pixels; multilingual generalisation depends on base LLM’s language ability

### Extraction rationale
Main practice synthesised from the abstract, Section 3 (architecture), Section 4 (training), Section 5 (deployment) and Section 6 (experiments, Tables 4–5). The action–feedback narrative integrates the paper’s headline result: OpenCompass 65.1, superior to proprietary models, plus the deployment results in Fig. 6, which demonstrate the cumulative impact of optimization techniques. All numbers are taken directly from the paper’s tables and figures.

## Why It Matters

Granularity: coarse. Scale: 3 models released (2.8B and 8.5B params); 11+ benchmarks; 1.8M pixel max input; deployed on Snapdragon 8 Gen 3 phone; pre‑training on 570M image‑text pairs; SFT on 5.3M samples Applies under: requires at least 5GB RAM after 4‑bit quantization; real‑time speed limited by LLM pre‑fill; highest performance currently on Snapdragon 8 Gen 3 with NPU; adaptive encoding caps at ~1.8M pixels; multilingual generalisation depends on base LLM’s language ability

## Evidence

### Source
arXiv:2408.01800 | 2408.01800__9876f844 | group: t8 | section: abstract

### Verbatim quote
"MiniCPM-Llama3-V 2.5 has several notable features: (1) Strong performance, outperforming GPT-4V-1106, Gemini Pro and Claude 3 on OpenCompass, a comprehensive evaluation over 11 popular benchmarks, (2) strong OCR capability and 1.8M pixel high-resolution image perception at any aspect ratio, (3) trustworthy behavior with low hallucination rates, (4) multilingual support for 30+ languages, and (5) efficient deployment on mobile phones. [...] More importantly, MiniCPM-V can be viewed as a representative example of a promising trend."

### Classification
domain: agent_domain_app
arXiv: cs.CV
Wikipedia: Computer_vision

### Keywords
multimodal large language model, efficient MLLM, on‑device deployment, adaptive visual encoding, model quantization, vision‑language model, end‑side AI, mobile AI, OCR, hallucination reduction, multilingual MLLM

```

### 经验 ID: `O-e92ec79b387d047b`

```markdown
---
id: O-e92ec79b387d047b
created_at: '2026-08-16T15:52:57Z'
summary: '[medium] Constructs an 8.5B vision‑language model through adaptive visual
  encoding, multi‑stage pre‑training, RLAIF‑V hallucination reduction, and multilingual
  SFT, reaching 65.1 on OpenCompass and enabling mobile phone deployment.'
memory_type: procedural
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l1
related_observations:
- id: O-4b49bfedf1b4e35a
  relation: complements
  reason: 'Both extracted from arXiv:2408.01800: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:59:01Z'
- id: O-75981c82413e8234
  relation: complements
  reason: 'Both extracted from arXiv:2408.01800: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:59:02Z'
---

## Observation

### Narrative
This experience details the construction of MiniCPM‑Llama3‑V 2.5, an 8.5B‑parameter multimodal model that attains GPT‑4V‑level performance on the OpenCompass benchmark while being deployable on a mobile phone. The goal is to reconcile high multimodal capability with severe end‑side resource constraints. The practice hinges on three pillars: adaptive visual encoding, a carefully orchestrated training pipeline, and a deployment‑oriented optimization stack. For visual encoding, images are split into slices based on an ideal slice count N<10, using a score function that selects the slice partition (m, n) that best preserves the ViT’s pre‑training aspect ratio. Each slice is resized to an area equal to the ViT pre‑training resolution’s area, position embeddings are interpolated, and the original image is appended as an extra slice. A one‑layer cross‑attention compression module then reduces the per‑slice 1024 tokens to 96 visual queries (960 max for 10 slices), supplemented by a spatial schema with <slice> and <\slice> wrappers and row‑separating newline tokens. Training unfolds in three pre‑training stages: first, the compression layer alone is warmed up on 200M image‑caption pairs at 224×224; next, the whole visual encoder is trained at 448×448 on another 200M captions; finally, both the encoder and compression layer adapt to high‑resolution and OCR data (39M English + 11M Chinese OCR samples) with the adaptive encoding strategy, keeping the LLM frozen. Noisy web captions are pre‑processed by an LLM fine‑tuned on GPT‑4‑annotated seed examples, converting them into clean QA pairs. Data packing combines multiple samples into fixed‑length sequences, accelerating pre‑training by 2–3×. Supervised fine‑tuning then mixes short‑response recognition tasks (5.3M samples) with longer, instruction‑rich data (3M samples, including 90K multilingual entries in 30+ languages) and unfreezes all parameters. To combat hallucination, RLAIF‑V is employed: the policy model generates 10 responses per instruction, Llama‑3 8B breaks each into atomic claims, and LLaVA‑NeXT‑Yi 34B evaluates each claim. The resulting scores form 6K DPO preference pairs, yielding a hallucination rate of 10.3% response‑level on Object HalBench (vs. 13.6% for GPT‑4V). Multilingual ability is obtained by limiting pre‑training to English/Chinese and then performing a small multilingual SFT pass, which boosts per‑language scores by over 25 points across French, German, Japanese, Korean, etc. On the deployment side, 4‑bit quantization, sequential loading, on‑device compilation, auto‑configuration, and NPU offloading turn the model into a functioning mobile assistant. The end result is an OpenCompass score of 65.1, exceeding GPT‑4V‑1106 (63.5), despite having far fewer visual tokens and a model size small enough to run on consumer phones. The practice is constrained by the need for a modern smartphone with NPU (e.g., Snapdragon 8 Gen 3) to reach usable speeds; the 1.8M‑pixel encoding ceiling limits ultra‑high‑resolution documents, and the multilingual performance is uneven, depending on the base LLM’s language proficiency and the volume of available SFT data per language.

### Practice environment
Base LLM: Llama3‑Instruct 8B. Vision encoder: SigLIP SoViT‑400m/14. Compression layer: perceiver resampler (one‑layer cross‑attention) with 96 queries per slice. Pre‑training data: 410M English captions, 110M Chinese captions, 39M English OCR, 11M Chinese OCR. SFT: Part‑1 5.3M, Part‑2 3M. RLAIF‑V scorer: LLaVA‑NeXT‑Yi 34B. DPO on 6K pairs. Multilingual SFT: 90K samples in 30+ languages. Deployment device: Xiaomi 14 Pro (Snapdragon 8 Gen 3). Quantization: GGML Q4_K_M. Evaluation: OpenCompass (MME, MMBench, MMMU, MathVista, LLaVA Bench, RealWorldQA), OCRBench, TextVQA, DocVQA, Object HalBench. Compilation & frameworks: llama.cpp, QNN. Additional devices for testing: vivo X00 Pro, MacBook Pro M1.

### Practice trace
1. Action: Adopted adaptive visual encoding (image slicing, resolution‑adaptive resizing, position embedding interpolation, token compression to 96 per slice, spatial schema) and trained the visual modules through three pre‑training stages, then applied caption rewriting, data packing, and two‑part SFT with all parameters unlocked
   Feedback: OpenCompass score 65.1, outperforming proprietary models (GPT‑4V‑1106: 63.5, Gemini Pro: 62.9, Claude 3 Opus: 57.7) while using only 96–960 visual tokens; OCRBench 725 (higher than GPT‑4V’s 645)
2. Action: Implemented RLAIF‑V by generating 10 responses from the policy model, decomposing into atomic claims, scoring with LLaVA‑NeXT‑Yi 34B, and constructing 6K DPO pairs
   Feedback: Response‑level hallucination rate reduced to 10.3% (GPT‑4V: 13.6%) without sacrificing general performance; OpenCompass even improved by 0.6 points
3. Action: Performed multilingual SFT with only 90K samples (<0.5% of total SFT data) after English/Chinese pre‑training
   Feedback: Performance on 9 sample languages rose by >25 points each; the model demonstrated multilingual capability across 30+ languages, e.g., German 76.5, Japanese 88.0

### Task context
modality: images and text (multimodal)
scale: 8.5B parameter model; pre‑training on 570M multimodal samples; SFT on ~8.3M samples; RLAIF‑V uses 6K preference pairs; deployed on Snapdragon 8 Gen 3 phone; benchmarks: 11 tasks (OpenCompass), OCRBench, TextVQA, DocVQA
constraint: optimal deployment requires a recent flagship SoC with NPU; visual encoding bounded at 1.8M pixels; multilingual quality varies with base LLM language capability; real‑time interaction bottlenecked by LLM pre‑fill; model size (8.5B) still requires 5GB RAM post‑quant.

### Extraction rationale
The medium practice draws from the introduction and the overview of method sections (Sec. 3–5). The core technique descriptions (adaptive visual encoding, three‑stage pre‑training, RLAIF‑V, multilingual SFT) are extracted from Sections 3.2, 4.1, 4.3, and the multilingual generalization paragraph. The key quantitative results (OpenCompass 65.1, OCRBench 725, hallucination 10.3%) are taken from Tables 4, 5 and 6, and the multilingual improvement figures from Table 7.

## Why It Matters

Granularity: medium. Scale: 8.5B parameter model; pre‑training on 570M multimodal samples; SFT on ~8.3M samples; RLAIF‑V uses 6K preference pairs; deployed on Snapdragon 8 Gen 3 phone; benchmarks: 11 tasks (OpenCompass), OCRBench, TextVQA, DocVQA Applies under: optimal deployment requires a recent flagship SoC with NPU; visual encoding bounded at 1.8M pixels; multilingual quality varies with base LLM language capability; real‑time interaction bottlenecked by LLM pre‑fill; model size (8.5B) still requires 5GB RAM post‑quant.

## Evidence

### Source
arXiv:2408.01800 | 2408.01800__9876f844 | group: t8 | section: introduction

### Verbatim quote
"MiniCPM-Llama3-V 2.5 has several notable features: (1) Strong performance, outperforming GPT-4V-1106, Gemini Pro and Claude 3 on OpenCompass, a comprehensive evaluation over 11 popular benchmarks, (2) strong OCR capability and 1.8M pixel high-resolution image perception at any aspect ratio, (3) trustworthy behavior with low hallucination rates, (4) multilingual support for 30+ languages, and (5) efficient deployment on mobile phones."

### Classification
domain: agent_domain_app
arXiv: cs.CV
Wikipedia: Computer_vision

### Keywords
multimodal large language model, efficient MLLM, on‑device deployment, adaptive visual encoding, model quantization, vision‑language model, end‑side AI, mobile AI, OCR, hallucination reduction, multilingual MLLM

```

### 经验 ID: `O-a01838f335181e0f`

```markdown
---
id: O-a01838f335181e0f
created_at: '2026-08-16T15:52:57Z'
summary: '[conditional] For large-scale training datasets, Python-based tokenization
  becomes a performance bottleneck; replacing it with a highly-optimized Rust-based
  tokenizer significantly accelerates both training and deployment preprocessing.'
memory_type: semantic
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l2
related_observations:
- id: O-62ecfe7e54349a68
  relation: complements
  reason: 'Both extracted from arXiv:1910.03771: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:57:49Z'
- id: O-7522ef7e0a145e3c
  relation: complements
  reason: 'Both extracted from arXiv:1910.03771: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:57:49Z'
---

## Observation

### Declaration
For large-scale training datasets, Python-based tokenization becomes a performance bottleneck; replacing it with a highly-optimized Rust-based tokenizer significantly accelerates both training and deployment preprocessing.

### Narrative
In the machine learning lifecycle, tokenization—converting raw text into sequences of integer indices that can be processed by a neural model—is a preprocessing step that runs both during training and inference. In early versions of the Transformers library, tokenization was implemented entirely in Python. While this was sufficient for small‑scale experiments, the authors observed that when training on very large text corpora (e.g., C4, Wikipedia, or other datasets used for pretraining), the Python‑based tokenization became a severe bottleneck, prolonging training times and slowing down deployment data pipelines. This led to a key design change: the library adopted a dedicated tokenization library, `tokenizers`, written in Rust, a systems programming language that compiles to native code and offers fine‑grained control over memory and concurrency without a garbage collector. The Rust tokenizer is deeply optimized for the tokenization algorithms commonly used in Transformers—Byte‑Pair Encoding (BPE), WordPiece, SentencePiece, and Unigram—each of which involves repeated splitting and merging operations over large vocabularies. Because tokenization is an embarrassingly parallel operation (each sequence can be tokenized independently), the Rust implementation exploits multi‑threading to process batches of text concurrently, dramatically reducing wall‑clock time. The paper states explicitly that “For training on very large datasets, Python‑based tokenization is often undesirably slow” and that the switch to Rust “speeds up the tokenization procedure both during training and deployment.” Although the paper does not provide quantitative speedup figures, the decision to re‑implement the tokenization infrastructure in Rust for all model architectures underscores a concrete, practically validated improvement. The Rust library covers all tokenizer variants available in the Transformers ecosystem and exposes a Python interface via bindings, so the user experiences no API change—the same high‑level `encode()` and `decode()` calls remain, but the underlying execution is now in native code. This change effectively removed the tokenization bottleneck for large‑scale workloads, making it feasible to preprocess billions of tokens in a reasonable time. The broader implication is that in data‑intensive ML pipelines, scripting‑language components can become the limiting factor, and migrating performance‑critical preprocessing to a compiled language can yield substantial end‑to‑end throughput gains. The experience applies generally to any NLP practitioner training or fine‑tuning Transformer models on large corpora, not just those within the Hugging Face ecosystem, though the actual speedup may depend on dataset size, vocabulary size, and hardware. The library’s open‑source nature allowed the community to benefit immediately, and the Rust tokenizer became the default backend in the latest release. The authors offer no further causal analysis—the performance advantage of Rust over Python for CPU‑bound string processing is well known—but the design choice was clearly motivated by empirical observation of the slowdown, making it a practical generalization for similar engineering contexts.

### Applicability context
summary: Shows that when training or deploying with very large text corpora, using a Rust‑based tokenizer (Tokenizers library) instead of Python‑based tokenization removes a major speed bottleneck, enabling faster data preprocessing.
modality: text tokenization
scale: large-scale training datasets (e.g., Wikipedia, C4); library-wide adoption
constraint: the speedup claim is not quantified in the paper; benefits are most significant for very large datasets and may be less pronounced for small data; relies on multi‑threading potential of Rust.

### Extraction rationale
The authors explicitly state that Python tokenization is slow for large datasets and that a Rust-based library improves speed; this judgment is based on the team’s experience developing the library. No causal mechanism is elaborated; r is null.

## Why It Matters

Claim type: conditional. Confidence: medium. Shows that when training or deploying with very large text corpora, using a Rust‑based tokenizer (Tokenizers library) instead of Python‑based tokenization removes a major speed bottleneck, enabling faster data preprocessing. Applies under: the speedup claim is not quantified in the paper; benefits are most significant for very large datasets and may be less pronounced for small data; relies on multi‑threading potential of Rust.

## Evidence

### Source
arXiv:1910.03771 | 1910.03771__6b2567a6 | group: t7 | section: Library Design

### Verbatim quote
For training on very large datasets, Python-based tokenization is often undesirably slow. In the most recent release, Transformers switched its implementation to use a highly-optimized tokenization library ... written in Rust to speed up the tokenization procedure both during training and deployment.

### Classification
domain: agent_general
arXiv: cs.CL
Wikipedia: Natural_language_processing
ACM CCS: Computing methodologies → Artificial intelligence → Natural language processing
CLC: TP391.1

### Keywords
Hugging Face Transformers, tokenization, Rust tokenizer, large-scale datasets, preprocessing speed, NLP pipeline, BPE, WordPiece

```

---

# Multi-agent 多智能体协作分工

**用户提问：** 设计 Multi-agent 系统时，应该如何划分智能体角色？有哪些被验证有效的协作拓扑结构？

**检索关键词：** `multi-agent collaboration team topology role assignment`

## 回答

### 1. 智能体角色划分：按任务能力与协作功能构建候选池

经验表明，不应只按“看起来合理”的人工先验固定团队，而应先构建一个**具有差异化能力和工具的候选智能体池**，再根据实际协作贡献选择任务团队。角色可以从两类维度划分：

#### （1）按领域专业能力划分

针对不同任务配置领域角色，例如：

- **通用推理**：数学家、程序员、律师、历史学家、经济学家、心理学家、医生；
- **代码生成**：Python 助手、算法开发者、计算机科学家、程序员、软件架构师；
- **决策与搜索**：搜索优化器、预算分析师、产品探索者、指令分析师、描述阅读者、决策者、结果估计者。

这类角色的作用是提供不同的问题分析视角，避免所有智能体产生高度相似的答案。

#### （2）按协作流程功能划分

尤其在代码任务中，角色不应只有“解题者”，还应包括：

- **方案生成者**：程序员、算法开发者、软件架构师；
- **工具增强者**：Python 助手、单元测试器、语法检查器；
- **审查与反思者**：代码反思者、调试器、质量管理者；
- **选择与协调者**：排序器、决策反思者。

经验显示，代码生成任务需要“编写者 + 审查者”的平衡组合，而不是单纯增加更多生成智能体。

#### （3）角色划分应作为候选池设计，而不是最终团队配置

角色提示词并不能保证智能体实际按照预期角色工作。因此，更有效的做法是：

1. 为任务建立包含相关和互补能力的候选池；
2. 进行一轮初步协作试验；
3. 让智能体对前序输出进行评分；
4. 通过跨时间步的前向—反向消息传递计算 **Agent Importance Score**；
5. 选择贡献最高的 Top-k 智能体组成最终团队。

在经验中的典型配置是：

- 通用推理：约 7 个候选，优化后通常保留约 4 个，甚至 3 个智能体也可能足够；
- 代码生成：约 12 个候选，优化后约 8 个，并保持编写者与审查者的平衡；
- 决策任务：约 8 个候选，优化后约 4 个。

因此，角色划分的核心原则是：**能力多样性用于探索，实际贡献用于最终选拔，团队规模根据任务决定。**

需要注意的是，候选池中必须至少有一部分智能体与任务相关。如果大多数或全部候选者都不匹配任务，重要性评分也难以选出高质量成员。

---

### 2. 被经验验证有效的协作拓扑

#### （1）时间前馈网络 T-FFN：推荐的基础拓扑

DyLAN 将多智能体协作建模为 **Temporal Feed-Forward Network（T-FFN，时间前馈网络）**：

- 每一层表示一个交互时间步；
- 智能体在当前时间步生成答案或评价；
- 信息沿时间方向向后续智能体传播；
- 不要求所有智能体在每一轮都互相通信。

这种结构相比固定通信图更容易进行动态裁剪和重组，是经验中验证的核心拓扑。

#### （2）动态团队重组：每轮保留高质量智能体

在任务求解阶段，使用 LLM 排序器对当前智能体输出进行排序，并动态停用表现较差的智能体。例如实验中每次保留排名靠前的 2 个智能体，使后续 T-FFN 只沿高质量路径扩展。

这种“逐轮筛选—继续传播”的拓扑具有两个作用：

- **质量控制**：阻止幻觉、逻辑错误或低质量代码继续影响后续协作；
- **计算节省**：减少后续轮次参与的智能体和 API 调用。

消融结果显示，去除团队重组后：

- HumanEval Pass@1 从 82.9% 降至 76.2%；
- WebShop reward 从 68.3 降至 66.0；
- MATH 和 MMLU 也出现下降。

因此，对于代码生成、开放式决策等中间答案质量差异较大的任务，动态重组尤其重要。

#### （3）动态重组 + 共识早停

在动态拓扑上增加早停机制：

- 当超过三分之二的剩余智能体达成一致时终止；
- 分类任务可使用答案完全一致作为标准；
- 代码生成任务使用 BLEU 大于 0.9 等一致性标准。

经验表明，早停主要带来**计算成本下降**，而团队重组主要带来**答案质量提升**。移除早停会使 API 调用增加约 11%–66%，但准确率有时变化不大。

#### （4）两阶段拓扑：团队优化阶段 + 任务求解阶段

较完整的有效结构是：

```text
候选智能体池
      │
      ▼
初步协作试验
      │
前向传播 + 同伴评分 + 反向重要性聚合
      │
      ▼
Top-k 任务团队
      │
      ▼
动态 T-FFN 协作
      │
动态淘汰低质量智能体
      │
      ▼
达到共识后早停
```

该拓扑同时解决了：

- 初始角色选择不准确；
- 固定团队带来的噪声；
- 所有智能体持续参与导致的成本过高；
- 多轮协作无法及时结束等问题。

实验中，优化后的团队相比完整固定团队不仅准确率更高，而且 API 调用更少。例如代码生成中，团队从 12 个候选缩减为 8 个后，Pass@1 从 76.2% 提升到 82.9%，API 调用从 23.04 降到 16.85。

#### （5）排序器应优先考虑 listwise 拓扑

在每轮需要从多个输出中选择 Top-k 时，经验支持使用**一次性 listwise 排序**，而不是大量 pairwise 比较：

- listwise：一次输入多个候选，直接选择最佳输出；
- pairwise：两两比较，调用次数近似随候选数平方增长。

在 MMLU 上，listwise 达到 70.5% 准确率，与多种 pairwise 方法相近，但平均 API 调用仅 4.39 次，而 LLM-Blender pairwise 为 19.27 次。因此，在候选规模中等且输出能够放入上下文窗口时，listwise 是更高效的动态筛选方式。

---

### 3. 实际设计建议

一个基于经验的默认方案是：

1. **建立多样候选池**：覆盖领域专家、方案生成、工具执行、验证审查和协调角色；
2. **不要直接固定最终团队**：先用少量代表性样本做初步协作；
3. **按实际贡献选人**：使用 Agent Importance Score，而非仅依据角色名称；
4. **采用时间前馈式通信**：让输出按轮次向后传播；
5. **每轮动态筛选**：使用 listwise 排序器保留 Top-k，停用低质量智能体；
6. **设置共识早停**：在达到足够一致性后结束协作；
7. **按任务调整团队结构**：
   - 推理：少量互补专家；
   - 代码：生成者、测试者、调试者、审查者保持平衡；
   - 决策：搜索、约束分析、结果评估和反思角色组合；
8. **定期重新优化**：任务分布变化或候选行为变化时，重新进行初步试验。

经验库未覆盖其他常见拓扑（如星型、全连接辩论、环形、层级管理型或黑板型）在相同条件下的系统性验证结果，因此不能据此判断它们是否同样有效。已有经验主要验证的是 **动态 T-FFN、动态团队重组、两阶段团队优化，以及 listwise 动态排序**。

## 思维链 (Reasoning)

1. 角色示例与“候选池应具备多样能力”来自 O-08aa0ba9a4c84851、O-c2f45ce260d01d51；这些经验列出了推理、代码和决策任务中的具体角色，并指出代码任务需要编写者与审查者的平衡。
2. “不能仅依赖人工角色先验，应根据实际贡献选人”来自 O-38b673a9ef763cae：Agent Importance Score 通过初步协作中的同伴评分和前向—反向传播选择 Top-k。
3. T-FFN 及其时间步信息传播结构来自 O-c2f45ce260d01d51 和 O-08aa0ba9a4c84851。
4. 动态淘汰低质量智能体的有效性及其对准确率的贡献来自 O-88dab6fc1b7a972e。
5. 共识早停主要降低 API 成本、团队重组主要改善正确率的区分来自 O-88dab6fc1b7a972e。
6. listwise 排序在准确率相近时显著降低调用次数的结论来自 O-fd7e965472d3588b。
7. 对其他拓扑结构“经验库未覆盖”的判断，是因为所给经验只报告了 DyLAN 的动态 T-FFN 及相关筛选机制，没有提供其他拓扑的系统比较结果。

## 召回的原始经验全貌 (5 条)

### 经验 ID: `O-08aa0ba9a4c84851`

```markdown
---
id: O-08aa0ba9a4c84851
created_at: '2026-08-16T15:52:57Z'
summary: '[coarse] Deploys a two-stage dynamic multi-agent collaboration framework
  (DyLAN) that automatically selects a task-oriented team of LLM agents via Agent
  Importance Score and coordinates them with dynamic communication structures, achieving
  up to 25.0% accuracy improvement on MMLU subjects and outperforming strong baselines
  on code generation, decision-making, and reasoning tasks with moderate computational
  cost.'
memory_type: procedural
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l1
related_observations:
- id: O-38b673a9ef763cae
  relation: complements
  reason: 'Both extracted from arXiv:2310.02170: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:36Z'
- id: O-5f6f376490029eea
  relation: complements
  reason: 'Both extracted from arXiv:2310.02170: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:36Z'
- id: O-88dab6fc1b7a972e
  relation: complements
  reason: 'Both extracted from arXiv:2310.02170: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:36Z'
- id: O-fd7e965472d3588b
  relation: complements
  reason: 'Both extracted from arXiv:2310.02170: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:36Z'
---

## Observation

### Narrative
In complex domains ranging from code generation to decision-making, a single large language model (LLM) agent often lacks the breadth of expertise to solve all subtasks optimally. Multi-agent collaboration has emerged as a solution, but prior work uses a fixed number of agents and static communication structures, which cannot adapt to the varying requirements of different tasks. Dynamic LLM-Powered Agent Network (DyLAN) addresses this by treating multi-agent collaboration as a temporal feed-forward network (T-FFN) where each layer corresponds to a time step of communication. DyLAN operates in two stages: Team Optimization and Task Solving. First, a preliminary trial is run on a given query using a pool of candidate agents equipped with diverse role prompts (e.g., mathematician, programmer, lawyer) or tools (e.g., code interpreter). During this trial, an LLM-based ranker dynamically deactivates low-performing agents at each time step, expanding the T-FFN adaptively. After the trial, the framework computes an Agent Importance Score for each candidate via a forward-backward message passing algorithm that aggregates peer ratings across layers—similar to backpropagation in neural networks. The top-k agents with the highest scores are then selected to form a task-oriented team. In the second stage, the optimized team collaborates to solve the query, again using dynamic agent team reformation with early stopping based on answer consistency (exact match for classification, BLEU score >0.9 for code generation). The overall practice is to automatically prune and restructure a multi-agent team per task, without manual role assignment or fixed communication graphs. This approach was evaluated on four benchmark types: code generation (HumanEval, measured by Pass@1), decision making (WebShop, measured by reward and success rate), general reasoning (MMLU, 57 subjects, accuracy), and arithmetic reasoning (MATH, 5000 questions, accuracy). All experiments used GPT-3.5-turbo-0301 as the backbone, with GPT-4-0613 also tested on code generation. Baselines included single execution, LLM-Blender, LLM Debate, Reflexion, CodeT, LATS, CAMEL, AgentVerse, PHP, ReAct, and BOLAA. DyLAN consistently outperformed these baselines while requiring fewer API calls. On code generation, DyLAN achieved 82.9% Pass@1, a gain of 9.7 points over single execution, with only 16.85 API calls compared to 48.0 for LATS. On WebShop decision making, the reward was 68.3 (vs. 50.6 direct execution) and success rate 42%. On MMLU, overall accuracy reached 70.5%, a 4.1-point improvement over single execution and 1.2 points above LLM Debate, with API calls reduced from 12.00 to 4.39. On MATH, with Complex CoT prompting, accuracy was 37.6% (+3.5 over single execution). The team optimization stage itself was crucial: before optimization, a full 12-agent team on HumanEval achieved only 76.2% Pass@1 with 23.04 API calls, while the 8-agent optimized team reached 82.9% with 16.85 calls. On specific MMLU subjects like college mathematics, agent selection improved accuracy by up to 25.0%. This practice is bounded by the need for a diverse candidate pool; if most agents are irrelevant to the task, the agent importance score may not effectively differentiate contributors. The method was only validated on text-based domains with GPT-3.5 and GPT-4; its applicability to other model families or multimodal tasks is unknown. Additionally, the team optimization stage adds an initial overhead, though the selected team is reusable across many queries in the same domain, amortizing the cost.

### Practice environment
Datasets: HumanEval (164 function-level completion problems with unit tests), WebShop (50 item search instructions with product browsing), MMLU (57 subjects, 1/5 of test set, ~2,800 questions), MATH (5000 problems across 7 subareas). Baselines: Single Execution, LLM-Blender, LLM Debate, Reflexion, CodeT, LATS, CAMEL, AgentVerse, PHP, ReAct, BOLAA, ReAct-SC. Metrics: Pass@1 (code), reward and success rate (decision making), accuracy (reasoning). Backbone models: GPT-3.5-turbo-0301 (all tasks), GPT-4-0613 (CG), code-davinci-002 for CodeT. DyLAN configuration: T-FFN with up to 4-6 time steps, 4 agents after team optimization, listwise LLM ranker for team reformation, early stopping at >=2/3 consensus. Agent roles: Mathematician, Programmer, Lawyer, Historian, Economist, Psychologist, Doctor for GR; Python Assistant, Algorithm Developer, Computer Scientist, Programmer, Coding Artist, Software Architect, Unit Tester, Syntax Checker, Code Reflector, Debugger, Quality Manager, Ranker for CG; Search Optimizer, Budget Analyst, Product Explorer, Instruction Analyst, Description Reader, Decision Maker, Decision Reflector, Result Estimator for DM.

### Practice trace
1. Action: Built DyLAN with a two-stage paradigm: (1) Team Optimization stage runs a preliminary collaboration trial on the query using all candidates, applies a forward-backward message passing algorithm to compute Agent Importance Scores (peer ratings aggregated via backward propagation through the T-FFN layers), then selects the top-k agents; (2) Task Solving stage deploys the optimized team in a T-FFN with dynamic agent team reformation (LLM ranker deactivates low-performing agents each time step) and early stopping based on answer consistency.
   Feedback: DyLAN outperformed single-execution and multi-agent baselines across tasks: 82.9% Pass@1 on HumanEval (+9.7 vs single execution, vs 81.1% LATS with 48 API calls); 68.3 reward on WebShop (+17.7 vs direct execution, vs 66.0 BOLAA with 32.4 API calls); 70.5% accuracy on MMLU (+4.1 vs single execution, vs 69.3% LLM Debate with 12.0 API calls); 37.6% on MATH with Complex CoT (+3.5 vs single execution). Team optimization contributed up to 25.0% accuracy gain on specific MMLU subjects (e.g., college mathematics from 40.0 to 65.0).

### Task context
modality: text-based agent interactions across multiple task domains
scale: 4 benchmark tasks (code generation, decision making, general reasoning, arithmetic reasoning); agent pools of 7–12 candidates; datasets: HumanEval (164 problems), WebShop (50 test environments), MMLU (57 subjects, 20% of test set), MATH (5000 problems); models: GPT-3.5-turbo-0301, GPT-4-0613
constraint: requires a pool of candidate LLM agents with distinct role prompts; team optimization stage requires a preliminary trial on a sample of queries; early stopping relies on answer consistency threshold (exact match for classification, BLEU>0.9 for code); tested only with GPT-3.5 and GPT-4 backbones; performance degrades when majority of candidates are unrelated to the task domain

### Extraction rationale
Coarse experience drawn from the abstract and overall results. The practice description is a synthesis of the two-stage paradigm (Team Optimization + Task Solving) and the core innovation (Agent Importance Score, T-FFN, dynamic reformation). Numerical feedback values are from Table 2-4 and Table 5, and the 25.0% improvement is from Table 7. The single action-feedback pair encapsulates the entire framework's implementation and its headline result.

## Why It Matters

Granularity: coarse. Scale: 4 benchmark tasks (code generation, decision making, general reasoning, arithmetic reasoning); agent pools of 7–12 candidates; datasets: HumanEval (164 problems), WebShop (50 test environments), MMLU (57 subjects, 20% of test set), MATH (5000 problems); models: GPT-3.5-turbo-0301, GPT-4-0613 Applies under: requires a pool of candidate LLM agents with distinct role prompts; team optimization stage requires a preliminary trial on a sample of queries; early stopping relies on answer consistency threshold (exact match for classification, BLEU>0.9 for code); tested only with GPT-3.5 and GPT-4 backbones; performance degrades when majority of candidates are unrelated to the task domain

## Evidence

### Source
arXiv:2310.02170 | 2310.02170__bc1a5533 | group: t10 | section: abstract

### Verbatim quote
"We propose automatically selecting a team of agents from candidates to collaborate in a dynamic communication structure toward different tasks and domains. ... DyLAN outperforms strong baselines in code generation, decision-making, general reasoning, and arithmetic reasoning tasks with moderate computational cost. On specific subjects in MMLU, selecting a team of agents in the team optimization stage improves accuracy by up to 25.0% in DyLAN."

### Classification
domain: agent_multi_agent
arXiv: cs.MA
Wikipedia: Multi-agent_systems

### Keywords
multi-agent collaboration, large language models, team optimization, dynamic communication structure, temporal feed-forward networks, agent importance score, code generation, decision-making, reasoning, agent collaboration

```

### 经验 ID: `O-38b673a9ef763cae`

```markdown
---
id: O-38b673a9ef763cae
created_at: '2026-08-16T15:52:57Z'
summary: '[conditional] Selecting a team of LLM-based agents based on their actual
  contributions during a preliminary collaboration trial—quantified by an unsupervised
  Agent Importance Score that aggregates peer ratings across communication steps—improves
  task performance and reduces computational cost compared to using a static, hand-crafted
  team, with gains being largest when the initial candidate pool contains both relevant
  and irrelevant agents.'
memory_type: semantic
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l2
related_observations:
- id: O-08aa0ba9a4c84851
  relation: complements
  reason: 'Both extracted from arXiv:2310.02170: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:36Z'
- id: O-c2f45ce260d01d51
  relation: complements
  reason: 'Both extracted from arXiv:2310.02170: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:36Z'
---

## Observation

### Declaration
Selecting a team of LLM-based agents based on their actual contributions during a preliminary collaboration trial—quantified by an unsupervised Agent Importance Score that aggregates peer ratings across communication steps—improves task performance and reduces computational cost compared to using a static, hand-crafted team, with gains being largest when the initial candidate pool contains both relevant and irrelevant agents.

### Narrative
Large language model (LLM) agents often achieve better results when collaborating in teams, but commonly used approaches fix the set of agents and their communication structure before the task begins. This practice relies on human priors about which roles will be useful, without verifying whether those agents actually make positive contributions during collaboration. The Dynamic LLM-Powered Agent Network (DyLAN) framework addresses this by introducing an unsupervised metric—Agent Importance Score—that quantifies how much each agent contributes in a preliminary collaboration trial, and then selects the top contributors for an optimized team.

The core finding is that posterior, contribution-based agent selection (team optimization) consistently improves task performance while simultaneously reducing computational cost, compared to using the full, hand-crafted team. This holds across diverse task families: general reasoning (MMLU), arithmetic reasoning (MATH), code generation (HumanEval), and decision-making (WebShop). In DyLAN, the selection algorithm operates on a temporal feed-forward network (T-FFN), where nodes represent agents at different time steps and edges denote communication channels. After a single trial collaboration, each agent rates the solutions of its predecessors, and these peer ratings are then backward-propagated to compute an aggregate importance score per agent. The top-ranked agents form a smaller, task-oriented team for the subsequent task-solving stage.

The conditions under which this finding is robust are as follows. The initial candidate pool must contain some agents that are at least partially relevant to the task; if all agents are mismatched (e.g., all are doctors answering a programming question), the benefit diminishes because there are no genuinely contributory agents to discover. The Agent Importance Score is computed from ratings given by agents themselves—each agent is prompted to score its predecessors' responses on a 1–5 scale. Therefore, the reliability of the scores depends on the agents' ability to evaluate peer outputs, which in turn is tied to the base LLM's competence. The optimization procedure requires one round of forward-backward message passing (forward: generating responses and peer ratings; backward: aggregating importance). The cost of this trial is amortized when the same optimized team can be reused for multiple queries, and the paper shows that even using only 1% of the dataset for the trial can yield significant improvements (+3.7% over random selection on code generation). The number of agents retained after optimization is typically small (e.g., 4 out of 7 or 8 out of 12), and the optimal team size varies by task; for general reasoning, 3 agents were sufficient, while for code generation, a balanced mix of writers and reviewers was needed.

The evidence spans multiple experiments. On the MMLU general reasoning task, team optimization improved accuracy by up to 25.0% on certain subjects (e.g., college mathematics rose from 40.0% to 65.0% after selecting a team that included the mathematician agent). Across five representative subjects, overall accuracy climbed from 63.5% with the full team of 7 agents to 73.6% with the optimized 4-agent team. In code generation, Pass@1 increased from 76.2% to 82.9% while reducing API calls by 27%. On decision-making (WebShop), reward improved from 53.0 to 68.3 and success rate from 28% to 42%. The method also showed data efficiency: using 10% of the dataset for the trial still achieved 82.3% Pass@1 (vs. 82.9% with full data). Compared to random selection or human-prior selection (simulated by asking GPT-4 to choose agents based on role descriptions), Agent Importance Score consistently produced better downstream performance, demonstrating that actual in-collaboration behavior matters more than stated role descriptions.

A causal explanation for why contribution-based selection works is provided by the authors: agents often do not behave according to their assigned role prompts, and hand-crafted roles may not align with the task requirements as intended. The Agent Importance Score captures the real influence of an agent's responses on later decisions, incorporating both immediate quality and downstream impact. It filters out agents whose contributions are negatively rated by peers, thereby reducing noise and conflicting signals that would otherwise degrade the final consensus. The backward-propagation ensures that an agent who is consistently rated highly by multiple successors receives a high score, even if its role seems irrelevant from a human perspective. The authors also note that the scores correlate well with Shapley values (a theoretically grounded importance measure) in in-domain scenarios, serving as a cheaper unsupervised alternative.

Boundaries and limitations include: (1) If the candidate pool is extremely imbalanced (most agents are useless), the method may still select sub-optimal agents because there is no high-quality signal to amplify. (2) The rating process adds a small overhead to each trial, though it is far cheaper than running exhaustive combinatorial evaluations. (3) The transferability of an optimized team across different task distributions is assumed but not deeply tested; for highly specialized tasks, recomputation might be necessary. (4) The method relies on an LLM's ability to give useful peer ratings; if the model is poorly calibrated or biased (e.g., position bias), the importance scores may be distorted.

### Causal explanation (r)
Hand-crafted role prompts do not guarantee that agents will behave in a way that benefits the collaboration. Posterior selection based on actual peer-rated contributions captures real influence and filters out agents whose responses are negatively evaluated by peers, thereby reducing noise and interference that degrade the final answer.

confidence (μ_r): medium
depth: shallow
depth rationale: The explanation essentially restates the finding (that actual contributions matter more than prior roles) without identifying a specific mechanism beyond 'peer ratings filter noise'; it does not explain why peer ratings correlate with actual contribution or what specific factors cause some agents to be unhelpful.

### Applicability context
summary: Enables task-oriented selection of LLM agent teams based on their actual collaborative behavior rather than human heuristics, improving accuracy by up to 25.0% on specific MMLU subjects and Pass@1 on HumanEval from 76.2% to 82.9% while reducing API calls.
modality: text, code, and game states
scale: 4 tasks, 4–12 candidate agents per task, GPT-3.5-turbo backbone; up to 7 agents selected down to 3–4
constraint: requires a preliminary trial collaboration to compute importance scores; agent ratings are only as reliable as the underlying LLM; if all candidates are mismatched to the task, improvement is minimal; optimized team may need recalibration for changing task distributions

### Extraction rationale
The declaration is supported by consistent patterns across multiple tasks (GR, CG, DM) and explicit author statements about improvement using agent selection (e.g., 'improved accuracy by up to 25.0%'). The causal explanation (r) is drawn from the paper's discussion that 'prior selection may not grasp the actual behaviors of agents' and that agent importance scores reflect 'actual contributions ... based on peer ratings' rather than designed roles.

## Why It Matters

Claim type: conditional. Confidence: high. Enables task-oriented selection of LLM agent teams based on their actual collaborative behavior rather than human heuristics, improving accuracy by up to 25.0% on specific MMLU subjects and Pass@1 on HumanEval from 76.2% to 82.9% while reducing API calls. Applies under: requires a preliminary trial collaboration to compute importance scores; agent ratings are only as reliable as the underlying LLM; if all candidates are mismatched to the task, improvement is minimal; optimized team may need recalibration for changing task distributions

## Evidence

### Source
arXiv:2310.02170 | 2310.02170__bc1a5533 | group: t10 | section: experiment

### Verbatim quote
"By optimizing the team on 10% of the original dataset, DyLAN has demonstrated similar performance compared to using the whole dataset, with only 0.2 loss on GR and 0.6 loss on CG. [...] even with only 1% of the original dataset, DyLAN could obtain a significant improvement of +3.7 over random selection on CG. [...] agent selection in DyLAN has improved accuracy by up to 25.0% in certain subjects of the MMLU dataset ..."

### Classification
domain: agent_multi_agent
arXiv: cs.MA
Wikipedia: Multi-agent_systems
ACM CCS: Computing methodologies → Artificial intelligence → Distributed artificial intelligence → Multi-agent systems
CLC: TP18

### Keywords
multi-agent collaboration, agent selection, team optimization, importance scoring, temporal feed-forward networks, dynamic teams, peer rating, large language model agents

```

### 经验 ID: `O-c2f45ce260d01d51`

```markdown
---
id: O-c2f45ce260d01d51
created_at: '2026-08-16T15:52:57Z'
summary: '[medium] Applies a two-stage dynamic agent collaboration framework (DyLAN)
  that optimizes the team composition via Agent Importance Score from a preliminary
  trial and then solves tasks with dynamic agent team reformation, achieving performance
  gains of up to 25.0% on MMLU subjects and outperforming strong baselines across
  code generation, decision-making, and reasoning tasks with reduced computational
  cost.'
memory_type: procedural
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l1
related_observations:
- id: O-38b673a9ef763cae
  relation: complements
  reason: 'Both extracted from arXiv:2310.02170: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:36Z'
- id: O-5f6f376490029eea
  relation: complements
  reason: 'Both extracted from arXiv:2310.02170: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:37Z'
- id: O-88dab6fc1b7a972e
  relation: complements
  reason: 'Both extracted from arXiv:2310.02170: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:37Z'
- id: O-fd7e965472d3588b
  relation: complements
  reason: 'Both extracted from arXiv:2310.02170: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:38Z'
---

## Observation

### Narrative
Existing multi-agent LLM collaboration systems rely on manually designed agent teams and static communication structures, which limit adaptability to diverse tasks. DyLAN (Dynamic LLM-Powered Agent Network) treats an agent team as a temporal feed-forward network (T-FFN), where each layer represents a discrete time step of interaction, and edges between layers indicate information flow. The core practice consists of a two-stage workflow: Team Optimization and Task Solving. In Team Optimization, a preliminary collaboration trial is conducted on a given query using all candidate agents (e.g., 7 agents with expert roles for reasoning, 12 agents including code writers and reviewers for code generation). Within this trial, a forward message passing process first propagates responses; then, a backward message passing algorithm, inspired by neural network backpropagation, aggregates peer ratings across layers. Each node (agent at a time step) rates the responses from its predecessors, and these ratings are multiplied by the successors' contributions and propagated backwards, yielding an Agent Importance Score for each candidate. The top-k agents (usually 4) based on this unsupervised metric form the optimized team. In the Task Solving stage, the selected team communicates according to a dynamic structure: at each time step, an LLM ranker evaluates the current responses and keeps only the top-two performers, deactivating the others and thereby expanding the T-FFN adaptively. This process continues until an early-stopping condition—over two-thirds of remaining agents produce a consistent answer—or a maximum number of rounds is reached. The method was evaluated on four task families: code generation (HumanEval, 164 problems), decision making (WebShop, 50 test environments), general reasoning (MMLU, 57 categories), and arithmetic reasoning (MATH, 5000 problems). All systems used GPT-3.5-turbo as the backbone, with GPT-4 tested only on code generation. DyLAN demonstrated substantial performance gains: an overall MMLU accuracy of 70.5% (4.1 points above single execution, 1.2 points above LLM Debate) with only 4.39 API calls on average (vs. 12.0 for LLM Debate); 82.9% Pass@1 on HumanEval (9.7 points above single execution, outperforming LATS’s 81.1% with 48 API calls); 68.3 reward on WebShop (17.7 above direct execution, surpassing BOLAA’s 66.0 reward with 24.85 API calls vs. 32.4); and 37.6% accuracy on MATH with Complex CoT (3.5 points above single execution). Team optimization itself provided significant efficiency and accuracy improvements: before optimization, the full 12-agent team on HumanEval achieved only 76.2% with 23.04 calls, while the optimized 8-agent team reached 82.9% with 16.85 calls; on MMLU subjects, accuracy rose by up to 25.0% after agent selection. This practice is constrained by the need for a diverse initial agent pool; if no agents possess relevant expertise, the team optimization may select suboptimal members. The framework was validated only on English-text tasks with proprietary LLMs; its effectiveness with open-source models or multimodal tasks remains untested. Additionally, the agent importance score computation requires one complete collaboration trial per new domain, though the resulting team can be reused across many queries.

### Practice environment
Datasets: HumanEval (function-level coding with unit tests), WebShop (item search and purchase simulation), MMLU (multiple-choice questions across humanities, social science, STEM, other), MATH (free-response math problems). Baselines: Single Execution, LLM-Blender, LLM Debate, Reflexion, CodeT, LATS, CAMEL, AgentVerse, PHP, ReAct, BOLAA, ReAct-SC. Metrics: Pass@1 (CG), reward and success rate (DM), accuracy (GR, AR). Backbone models: GPT-3.5-turbo-0301, GPT-4-0613. DyLAN architecture: T-FFN with up to 4-6 layers; 4 agents post-optimization; listwise LLM ranker for team reformation; early stopping at ≥2/3 consensus; temperature varied per task (0 for single execution, 0.2-1.0 for others). Agent roles: for GR—Mathematician, Programmer, Lawyer, Historian, Economist, Psychologist, Doctor; for CG—Python Assistant, Algorithm Developer, Computer Scientist, Programmer, Coding Artist, Software Architect, Unit Tester (tool-augmented), Syntax Checker (tool), Code Reflector, Debugger, Quality Manager, Ranker; for DM—Search Optimizer, Budget Analyst, Product Explorer, Instruction Analyst, Description Reader, Decision Maker, Decision Reflector, Result Estimator.

### Practice trace
1. Action: Designed DyLAN with a temporal feed-forward network where each layer represents a time step; agents communicate by passing responses forward; an LLM ranker dynamically deactivates low-performing agents each round, expanding T-FFN layers. Team Optimization stage runs a preliminary trial on the query, then computes Agent Importance Scores through forward-backward message passing (nodes rate predecessors, scores propagate backward via weighted sum), selecting top-k agents.
   Feedback: DyLAN outperformed static architectures: on MMLU, accuracy 70.5% vs 69.3% for LLM Debate, with API calls 4.39 vs 12.00; on HumanEval, 82.9% Pass@1 vs 81.1% LATS, with 16.85 vs 48.00 calls. Agent selection provided up to 25.0% improvement on certain MMLU subjects (e.g., college mathematics from 40.0% to 65.0%).
2. Action: Applied team optimization to prune agent pools: on CG, reduced from 12 writers+reviewers to 8; on DM, from 8 to 4; on GR, from 7 to 4. Used the same optimized teams for all test queries within a domain.
   Feedback: Post-optimization, HumanEval Pass@1 rose from 76.2% to 82.9% while API calls fell from 23.04 to 16.85; WebShop reward jumped from 53.0 to 68.3; MMLU accuracy improved from 69.5% to 70.5% with calls dropping from 8.30 to 4.39, showing both performance gain and cost reduction.

### Task context
modality: text-based agent interactions across multiple task domains
scale: 4 task domains (CG, DM, GR, AR); agent pools of 7–12 candidates; datasets: HumanEval (164), WebShop (50 envs.), MMLU (57 subjects, ~2800 test Qs), MATH (5000); models: GPT-3.5-turbo (all), GPT-4 (CG)
constraint: requires a candidate pool of diverse LLM agents with distinct role prompts/tools; team optimization needs an initial trial on representative queries; early stopping relies on consistency threshold (exact match for classification, BLEU>0.9 for code); tested only with GPT-3.5 and GPT-4; performance may degrade when agents are all irrelevant to the task; team optimization overhead is amortized across many queries

### Extraction rationale
Medium experience constructed from Introduction (Sections 1 and 3) and overall results (Sections 4.1-4.2). The practice description details the two-stage paradigm, T-FFN formulation, forward-backward message passing for Agent Importance Score, and dynamic team reformation. Numerical outcomes are drawn from Tables 2-5 and the text. The two action-feedback pairs reflect (1) the core dynamic framework and agent selection, and (2) the concrete benefit of team optimization across tasks.

## Why It Matters

Granularity: medium. Scale: 4 task domains (CG, DM, GR, AR); agent pools of 7–12 candidates; datasets: HumanEval (164), WebShop (50 envs.), MMLU (57 subjects, ~2800 test Qs), MATH (5000); models: GPT-3.5-turbo (all), GPT-4 (CG) Applies under: requires a candidate pool of diverse LLM agents with distinct role prompts/tools; team optimization needs an initial trial on representative queries; early stopping relies on consistency threshold (exact match for classification, BLEU>0.9 for code); tested only with GPT-3.5 and GPT-4; performance may degrade when agents are all irrelevant to the task; team optimization overhead is amortized across many queries

## Evidence

### Source
arXiv:2310.02170 | 2310.02170__bc1a5533 | group: t10 | section: introduction

### Verbatim quote
"We introduce a novel framework named Dynamic LLM-Powered Agent Network (DyLAN) for LLM-powered agent collaboration, operating a two-stage paradigm: (1) Team Optimization and (2) Task Solving. During the first stage, we utilize an agent selection algorithm, based on an unsupervised metric called Agent Importance Score ... enabling the selection of best agents according to their contributions in a preliminary trial. ... Empirical results demonstrate the superior accuracy, efficiency, and stability of DyLAN across various tasks, underscoring the need for dynamic agent teams." "On specific subjects in MMLU, selecting a team of agents in the team optimization stage improves accuracy by up to 25.0% in DyLAN."

### Classification
domain: agent_multi_agent
arXiv: cs.MA
Wikipedia: Multi-agent_systems

### Keywords
multi-agent collaboration, large language models, team optimization, dynamic communication structure, temporal feed-forward networks, agent importance score, code generation, decision-making, reasoning, agent collaboration

```

### 经验 ID: `O-88dab6fc1b7a972e`

```markdown
---
id: O-88dab6fc1b7a972e
created_at: '2026-08-16T15:52:57Z'
summary: '[conditional] Dynamically dropping low-performing LLM agents during collaboration
  (agent team reformation) significantly improves final answer correctness, particularly
  for open-ended generation tasks where response quality varies widely, while early
  stopping based on majority consensus primarily reduces computational cost.'
memory_type: semantic
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l2
related_observations:
- id: O-08aa0ba9a4c84851
  relation: complements
  reason: 'Both extracted from arXiv:2310.02170: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:36Z'
- id: O-c2f45ce260d01d51
  relation: complements
  reason: 'Both extracted from arXiv:2310.02170: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:37Z'
---

## Observation

### Declaration
Dynamically dropping low-performing LLM agents during collaboration (agent team reformation) significantly improves final answer correctness, particularly for open-ended generation tasks where response quality varies widely, while early stopping based on majority consensus primarily reduces computational cost.

### Narrative
Multi-agent collaboration systems often involve multiple rounds of interaction where agents refine their answers based on feedback. A fixed set of agents participates in every round, regardless of whether they are still contributing usefully. DyLAN introduces a mechanism called agent team reformation, which dynamically drops low-performing agents during the collaboration process, keeping only the top-rated agents for subsequent rounds. This dynamic structure is complemented by an early-stopping rule that terminates the interaction when a sufficient majority of agents agree.

The key finding is that agent team reformation significantly improves final answer correctness, while early stopping primarily enhances efficiency. Without reformation, performance degrades measurably across tasks, especially for open-ended generation where the quality of intermediate responses varies widely. Reformation acts as a quality filter, preventing poor or erroneous outputs from persisting and influencing later decisions.

The conditions under which this effect is observed are as follows. The system uses an LLM ranker (a separate instance of GPT-3.5) that, at a designated time step, evaluates all agent responses from the previous step and selects the top k (k=2 in experiments) to continue to the next step. The remaining agents are deactivated, meaning their nodes and edges are removed from the T-FFN. This works across classification tasks (general reasoning, arithmetic reasoning) and generation tasks (code generation, decision-making). However, the relative importance of reformation vs. early stopping varies by task type: for classification problems where answers are discrete, early stopping can be triggered frequently because consensus is easier to reach, so reformation's marginal benefit on accuracy is smaller; for open-ended tasks like code generation, answers are heterogeneous, making early stopping less frequent and reformation more critical to filter out incorrect or syntactically flawed code.

The evidence is drawn from ablation studies. Table 6 shows that removing agent team reformation ('w/o atr') leads to a substantial performance drop: on code generation, Pass@1 falls from 82.9% to 76.2%; on decision-making, reward drops from 68.3 to 66.0; on arithmetic reasoning, accuracy drops from 35.7% to 33.8%; on general reasoning, the drop is smaller (70.5% to 69.9%) because the full-team early stopping often catches agreement anyway. Removing early stopping ('w/o es') primarily increases API calls (by 45% on AR, 66% on GR, 11% on CG, 54% on DM) while sometimes even slightly improving accuracy (on AR from 35.7% to 35.0%)—suggesting that early stopping is a cost-saving measure without hurting quality. The combination of both mechanisms (reformation + early stopping) yields a system that outperforms static debates (e.g., LLM Debate) with fewer API calls.

The authors provide a causal explanation: agent team reformation "is critical to enhance the correctness of the final answer" because it filters out agents whose temporary mistakes (such as hallucinations or logical errors) could mislead other agents in subsequent rounds. By retaining only the top-ranked responses, the system ensures that high-quality solutions propagate forward and contribute to the final consensus. Early stopping, based on a Byzantine consensus threshold (over 2/3 of agents agreeing), prevents unnecessary further rounds once a stable answer has emerged, thus saving computation. The two mechanisms together create a computationally efficient yet quality-robust collaboration framework.

Boundaries and limitations: The ranker's own quality is crucial; if the ranker is biased or poorly calibrated, it may wrongly eliminate good agents or keep bad ones. The experiments used a k=2, meaning only two agents survive per reformation step; a larger k might retain more signal but reduce efficiency. The approach assumes that the task can be meaningfully evaluated at intermediate steps; for long-horizon tasks where later steps depend on earlier choices, early elimination could remove strategically important but temporally suboptimal paths. The consistency check for open-ended outputs relied on BLEU score with a threshold, which is a coarse approximation and may not capture semantic correctness. Finally, the method was only tested with GPT-3.5 as the ranker; other LLMs with different ranking accuracies may alter the trade-off.

### Causal explanation (r)
Agent team reformation filters out agents that produce temporary mistakes (e.g., hallucinations), preventing those erroneous outputs from influencing later rounds and thereby preserving the quality of the emerging consensus.

confidence (μ_r): medium
depth: shallow
depth rationale: The explanation identifies a mechanism (filtering the flawed responses before they propagate) but does not explain why some agents produce temporary mistakes or why the remaining agents are more reliable; it is a straightforward restatement of the filtering effect rather than a deep causal chain.

### Applicability context
summary: Improves LLM multi-agent collaboration by dynamically filtering out low-quality agent responses mid-process, increasing Pass@1 on HumanEval from 76.2% to 82.9% and reward on WebShop from 66.0 to 68.3, while early stopping cuts API calls by 45–66% without harming accuracy.
modality: text, code, and game states
scale: 4 tasks, 4–12 agents, GPT-3.5-turbo backbone, reformation triggers at specific steps (e.g., t=4 for CG)
constraint: requires a reliable LLM ranker; consistency check for open-ended tasks uses BLEU threshold; k=2 agents retained per reformation step; may exclude strategically relevant but temporally suboptimal responses in long-horizon tasks

### Extraction rationale
The declaration is based on ablation results comparing performance with and without agent team reformation across four tasks, and explicit author interpretation that reformation is 'critical to enhance the correctness' (declaration basis). The causal explanation (r) is taken directly from the authors' conjecture about filtering temporary mistakes as stated in the same sentence.

## Why It Matters

Claim type: conditional. Confidence: medium. Improves LLM multi-agent collaboration by dynamically filtering out low-quality agent responses mid-process, increasing Pass@1 on HumanEval from 76.2% to 82.9% and reward on WebShop from 66.0 to 68.3, while early stopping cuts API calls by 45–66% without harming accuracy. Applies under: requires a reliable LLM ranker; consistency check for open-ended tasks uses BLEU threshold; k=2 agents retained per reformation step; may exclude strategically relevant but temporally suboptimal responses in long-horizon tasks

## Evidence

### Source
arXiv:2310.02170 | 2310.02170__bc1a5533 | group: t10 | section: experiment

### Verbatim quote
"Agent team reformation, however, is critical to enhance the correctness of the final answer. We conjecture it is because agents are filtered for temporary mistakes in LLMs, such as hallucinations, etc. [...] Removing agent team reformation (w/o atr) reduces Pass@1 on code generation from 82.9% to 76.2%, and reward on DM from 68.3 to 66.0."

### Classification
domain: agent_multi_agent
arXiv: cs.MA
Wikipedia: Multi-agent_systems
ACM CCS: Computing methodologies → Artificial intelligence → Distributed artificial intelligence → Multi-agent systems
CLC: TP18

### Keywords
dynamic communication structures, agent team reformation, early stopping, multi-agent collaboration, quality filtering, Byzantine consensus, code generation, reasoning

```

### 经验 ID: `O-fd7e965472d3588b`

```markdown
---
id: O-fd7e965472d3588b
created_at: '2026-08-16T15:52:57Z'
summary: '[relation] A listwise LLM ranker achieves comparable accuracy to pairwise
  ranking methods for selecting top agent responses in multi-agent collaboration,
  while using significantly fewer API calls (e.g., 4.39 vs 19.27 on MMLU).'
memory_type: semantic
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l2
related_observations:
- id: O-08aa0ba9a4c84851
  relation: complements
  reason: 'Both extracted from arXiv:2310.02170: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:36Z'
- id: O-c2f45ce260d01d51
  relation: complements
  reason: 'Both extracted from arXiv:2310.02170: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:38Z'
---

## Observation

### Declaration
A listwise LLM ranker achieves comparable accuracy to pairwise ranking methods for selecting top agent responses in multi-agent collaboration, while using significantly fewer API calls (e.g., 4.39 vs 19.27 on MMLU).

### Narrative
When building a multi-agent collaboration framework that requires real-time ranking of agent responses (for instance, to select which agents to retain or to combine outputs), the choice of ranking method has implications for both accuracy and computational cost. DyLAN uses an LLM ranker that evaluates all incoming responses at a given step and chooses the top k to proceed. The paper compares four ranking strategies: a listwise prompt that asks the LLM to select the best 2 solutions at once, and three pairwise methods—the original LLM-Blender pairwise ranker, Elo scoring implemented via pairwise comparisons, and a sliding-window pairwise approach.

The finding is that the listwise ranker achieves comparable final task accuracy to all pairwise methods while incurring a far lower number of API calls. On the MMLU general reasoning task, the listwise ranker yielded an overall accuracy of 70.5%, while LLM-Blender (pairwise) achieved 70.1%, Elo 70.3%, and sliding window 70.3%. However, the API calls were dramatically different: listwise required only 4.39 calls on average, whereas LLM-Blender used 19.27, Elo 19.55, and sliding window 11.40. This indicates that a single, well-prompted listwise ranking is sufficient to capture the relative quality of agent responses, and the quadratic complexity of pairwise comparisons is unnecessary when the underlying LLM can process multiple options simultaneously.

The conditions under which this holds: The experiments were conducted using GPT-3.5-turbo on the MMLU dataset, with 4–7 agents' responses to rank. The listwise prompt explicitly asks the LLM to choose the best 2 solutions and output their indices in a structured format. To mitigate positional bias, the order of responses presented to the ranker was shuffled. The task was a multiple-choice reasoning task where the correctness of the final answer can be directly measured; the ranking step is used to decide which answers to carry forward. The conclusion that listwise is more efficient than pairwise is specific to tasks where the number of items to rank is moderate and fits within the LLM's context window; for very large sets, listwise may exceed token limits or become less accurate.

The evidence is summarized in Table 14. The authors state: "different ranking methods have a relatively low impact on performance, probably because of strong discrimination ability of GPT-3.5, but pairwise ranking methods always consume higher computational cost." Thus, the listwise ranker is the most cost-effective. This insight is then adopted as the default ranker in DyLAN.

The causal mechanism: Pairwise ranking methods compare all pairs, leading to O(n²) comparisons; each comparison is a separate API call costing time and tokens. Listwise ranking leverages the LLM's ability to compare multiple items holistically within a single prompt, effectively performing relative evaluation in one forward pass. The authors hypothesize that GPT-3.5's strong discrimination ability makes it capable of producing reliable rankings even without exhaustive pairwise comparisons. Additionally, listwise ranking may capture contextual relationships among responses that pairwise isolated comparisons miss, though this is not elaborated.

Boundaries: The finding is limited to the specific LLM (GPT-3.5) and task (MMLU). For tasks where responses are long (e.g., full code blocks), a listwise prompt may exceed the model's context window, forcing the use of pairwise or other strategies. Also, if the LLM exhibits position bias or struggles with long lists, the accuracy of listwise ranking could degrade; the paper's shuffling may not fully eliminate such biases. The comparison assumes that the ranker's output is used as a hard selection (top-k), not for weighted aggregation; other paradigms might have different sensitivity to ranking quality. Finally, the scalability of listwise ranking to many more agents or to tasks where the ranker must consider very subtle differences remains untested.

### Causal explanation (r)
GPT-3.5's strong discrimination ability allows it to produce reliable rankings from a single listwise prompt, making the exhaustive O(n²) pairwise comparisons unnecessary.

confidence (μ_r): medium
depth: shallow
depth rationale: The explanation attributes the low performance gap to the LLM's 'strong discrimination ability' without unpacking what aspects of that ability make listwise equally effective; it is a high-level attribution rather than a specific mechanism.

### Applicability context
summary: Demonstrates that listwise ranking of LLM agent responses is as accurate but far cheaper than pairwise ranking, enabling efficient dynamic agent selection with only 4.39 API calls vs 19.27, without sacrificing final answer quality.
modality: text (multiple-choice answers)
scale: MMLU with 4–7 agent responses, GPT-3.5-turbo ranker
constraint: tested only on MMLU with GPT-3.5; listwise prompt must fit in context window; positional bias may be present despite shuffling; may not scale to very many responses or long text

### Extraction rationale
The declaration is based on the explicit comparison of ranking methods in Table 14 and the authors' conclusion that listwise is both accurate and efficient (declaration basis). The causal explanation (r) is taken from the phrase 'probably because of strong discrimination ability of GPT-3.5' in the same paragraph.

## Why It Matters

Claim type: relation. Confidence: medium. Demonstrates that listwise ranking of LLM agent responses is as accurate but far cheaper than pairwise ranking, enabling efficient dynamic agent selection with only 4.39 API calls vs 19.27, without sacrificing final answer quality. Applies under: tested only on MMLU with GPT-3.5; listwise prompt must fit in context window; positional bias may be present despite shuffling; may not scale to very many responses or long text

## Evidence

### Source
arXiv:2310.02170 | 2310.02170__bc1a5533 | group: t10 | section: experiment

### Verbatim quote
"We tested listwise ranker with our own prompts, pairwise GPT ranker from original LLM-Blender, Elo Score from TrueSkill also implemented with pairwise ranker, and pairwise ranker with Sliding Window algorithm. In Table 14, we show that different ranking methods have a relatively low impact on performance, probably because of strong discrimination ability of GPT-3.5, but pairwise ranking methods always consume higher computational cost. Thus, we chose a listwise ranker in our implementation of DyLAN."

### Classification
domain: agent_multi_agent
arXiv: cs.CL
Wikipedia: Information_retrieval
ACM CCS: Information systems → Information retrieval → Retrieval models and ranking
CLC: TP391.1

### Keywords
listwise ranking, pairwise ranking, LLM ranker, efficiency, multi-agent response selection, cost-effectiveness, GPT-3.5

```

---

# 纯视觉 GUI 自动化 Agent

**用户提问：** 开发 GUI Agent 时，纯视觉方案（仅依赖截图和坐标点击）可行吗？关键的前置条件和难点是什么？

**检索关键词：** `GUI agent vision only screenshot grounding click action`

## 回答

**可行，但不能把“仅依赖截图和坐标点击”理解为开箱即用或通用可靠。**经验表明，纯视觉 GUI Agent 已能在不使用 HTML、Accessibility Tree 或 View Hierarchy 的情况下完成点击、输入等自动化任务；它尤其适用于结构化界面信息不可得、不可访问或过于冗长的平台，如 iOS、桌面端等（O-69515b5a7432dcfa、O-879f04599d7cd990）。

但其可行性高度依赖于：**强 GUI grounding（界面元素定位）能力、面向 GUI 的高质量训练数据，以及针对目标任务/平台的适配训练。**仅拿通用视觉语言模型直接根据截图输出坐标，效果不足：基础 Qwen-VL 在 ScreenSpot 的平均点击准确率为 5.2%，经过 100 万级 GUI grounding 数据预训练后的 SeeClick 达到 53.4%；且 grounding 提升与下游 GUI 任务表现单调正相关（O-69515b5a7432dcfa）。

### 关键前置条件

1. **必须具备精确的视觉定位（GUI grounding）能力**  
   Agent 要能根据自然语言指令，在截图中找出正确的按钮、输入框、文本或图标，并输出落在目标边界框内的坐标。此能力是纯视觉方案的核心瓶颈，也是下游任务成功的主要驱动因素（O-69515b5a7432dcfa）。

2. **需要大规模、覆盖目标界面的 grounding 训练数据**  
   有效做法是让截图与“元素描述—坐标/边界框”配对，训练文本到坐标、文本到边界框等任务。SeeClick 使用网页与移动端数据构造了约 100 万样本的混合数据进行持续预训练（O-69515b5a7432dcfa）。  
   对移动端，AMEX 的经验还表明：除任务演示外，补充可交互元素边界框、屏幕描述和元素功能描述，能显著提升任务完成率（约 +10 个百分点）及点击/长按表现（超过 +14 个百分点）（O-90da288ba0fa6fb2、O-51ed81308e03abc2）。

3. **元素标注应描述“功能和上下文”，而不只是类别**  
   类别标签如 `ICON_PLUS` 不足以说明其具体含义；“点击以新增搜索机场”等上下文化功能描述，能更好地区分同一图标在不同页面的作用。该类标注帮助纯视觉 Agent 在未见屏幕上定位图标/控件；AMEX 训练的模型在 ScreenSpot 移动端 Icon/Widget 子集达 72.6%（O-7808996c1abe4479）。

4. **需要目标平台和任务的专门微调，以及足够的多步轨迹数据**  
   视觉 grounding 预训练不等于能稳定完成多步任务。SeeClick 在 MiniWob、AITW、Mind2Web 等任务上仍进行了任务特定微调（O-69515b5a7432dcfa）。AMEX 也通过平均 12.8 步的复杂指令—动作链覆盖真实任务流程（O-51ed81308e03abc2）。

5. **坐标执行链路本身必须稳定**  
   从已有经验可直接确认的是，模型输出目标为坐标、评价标准为预测点是否落入真实元素边界框（O-69515b5a7432dcfa）。至于不同分辨率、缩放、窗口变化、系统级坐标映射等工程校准机制，**经验库未覆盖**，但这类问题不应被误认为已由上述实验自动解决。

### 主要难点与边界

1. **小图标、无文本控件和密集界面是最难部分**  
   ScreenSpot 中约一半目标是图标或 widget。SeeClick 虽总体达到 53.4%，但在桌面与网页的图标/控件上仅约 30.0% 和 32.5%，说明细粒度定位、无文本语义理解及密集布局仍是主要短板（O-69515b5a7432dcfa）。

2. **截图分辨率可能限制小目标识别**  
   SeeClick 的输入分辨率为 448×448；经验明确指出较低分辨率可能遗漏小元素，并造成其在复杂、稠密网页及小图标上的弱势（O-69515b5a7432dcfa、O-879f04599d7cd990）。

3. **跨应用、跨域和版本变化会明显退化**  
   AMEX 的跨域评估显示，当训练与测试界面的视觉外观、交互风格不同，或应用版本发生变化时，性能会显著下降（O-51ed81308e03abc2）。因此，纯视觉方案更依赖训练覆盖与域内适配。

4. **动作空间受限时较可行，复杂交互尚是短板**  
   SeeClick 的主要限制是仅支持点击和输入，尚不覆盖拖拽、双击等复杂动作（O-69515b5a7432dcfa）。所以若产品任务以“找元素—点击/输入”为主，纯视觉路线更现实；若大量依赖复杂手势，则现有经验显示能力尚不充分。

5. **高质量人工验证数据的成本与重要性**  
   AMEX 中元素框经人工过滤，功能描述由模型生成后再经人工验证，准确率超过 97%；其收益是在这一高质量条件下观察到的。若采用噪声很大的自动元素描述或错位框，是否仍有同等收益，经验库未覆盖，且已有经验明确提示不能直接假定成立（O-90da288ba0fa6fb2）。

**结论：**纯视觉、截图+坐标点击的 GUI Agent 是一条已验证可行的路线，特别适合拿不到结构化 UI 信息的环境；但它的本质不是简单的“看图点一下”，而是依赖数据和训练支撑的视觉 grounding 系统。落地时应优先建设高质量的元素定位与功能语义数据、提高小目标/图标定位能力、针对目标平台做多步任务微调，并将首期动作范围控制在点击和输入等基础操作。

## 思维链 (Reasoning)

1. 从 SeeClick 的实验可知，纯截图输入的 GUI Agent 能执行点击和输入，并在移动、桌面、网页等场景评测，故判断“技术路线可行”；但其总体 grounding 准确率为 53.4%，下游真实网页任务步成功率为 20.8%，因此不能推断为通用且高可靠。引用：O-69515b5a7432dcfa、O-879f04599d7cd990。  
2. SeeClick 的检查点分析显示 grounding 准确率提升与三个下游任务表现持续同步提升，因此将 GUI grounding 列为首要前置条件。引用：O-69515b5a7432dcfa。  
3. SeeClick 使用约 100 万 GUI grounding 样本预训练；AMEX 通过元素框、功能描述、复杂动作链增强训练，因此归纳出数据规模、标注质量、功能语义和任务微调是关键条件。引用：O-69515b5a7432dcfa、O-90da288ba0fa6fb2、O-51ed81308e03abc2、O-7808996c1abe4479。  
4. 根据桌面/网页图标控件低准确率、448×448 分辨率限制、复杂动作缺失及跨域下降，归纳出小目标、密集页面、复杂交互和分布变化是主要难点。引用：O-69515b5a7432dcfa、O-879f04599d7cd990、O-51ed81308e03abc2。  
5. 对坐标缩放、窗口变化、执行校准等具体工程问题，所给经验没有直接实验或结论，因此标注为“经验库未覆盖”。

## 召回的原始经验全貌 (5 条)

### 经验 ID: `O-69515b5a7432dcfa`

```markdown
---
id: O-69515b5a7432dcfa
created_at: '2026-08-16T15:52:57Z'
summary: '[medium] Addresses GUI grounding for visual GUI agents by pre-training a
  9.6B LVLM on 1M automatically curated GUI grounding data, achieving 53.4% average
  click accuracy on the new ScreenSpot benchmark and demonstrating that improvements
  in grounding directly correlate with boosted performance on three downstream agent
  tasks (MiniWob, AITW, Mind2Web), with SeeClick surpassing vision‑only baselines
  using much less training data.'
memory_type: procedural
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l1
related_observations:
- id: O-589f92650cf81736
  relation: complements
  reason: 'Both extracted from arXiv:2401.10935: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:48Z'
- id: O-758437a7e4c0c1ff
  relation: complements
  reason: 'Both extracted from arXiv:2401.10935: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:48Z'
- id: O-840aa1fa1a632f0f
  relation: complements
  reason: 'Both extracted from arXiv:2401.10935: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:48Z'
- id: O-8c4d8766299925cb
  relation: complements
  reason: 'Both extracted from arXiv:2401.10935: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:48Z'
- id: O-c89c452fe414fd79
  relation: complements
  reason: 'Both extracted from arXiv:2401.10935: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:49Z'
- id: O-edf47b522504f1fe
  relation: complements
  reason: 'Both extracted from arXiv:2401.10935: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:49Z'
- id: O-49cf10d47ba5d65f
  relation: complements
  reason: Both classified under agent_web_gui, from arXiv:2307.10088 and arXiv:2401.10935
    respectively — different papers addressing the same problem area.
  linked_at: '2026-08-16T15:59:26Z'
- id: O-5186b05e33577926
  relation: complements
  reason: Both classified under agent_web_gui, from arXiv:2401.10935 and arXiv:2406.03679
    respectively — different papers addressing the same problem area.
  linked_at: '2026-08-16T15:59:26Z'
---

## Observation

### Narrative
Existing GUI agents mostly interact with environments through structured text like HTML or Android view hierarchies, which may be unavailable on certain platforms (e.g., iOS) or extremely verbose while sacrificing visual layout information. SeeClick tackles this by building a purely vision‑based GUI agent that relies solely on screenshots to perform actions. The core idea is to equip a large vision‑language model (LVLM) with GUI grounding: the ability to precisely locate screen elements (buttons, icons, text fields) given a natural language instruction. The authors first curate a large‑scale pre‑training dataset for GUI grounding. For web UIs, they crawl around 300k web pages from the latest Common Crawl snapshot and automatically extract two types of elements from the HTML: visible text content and elements with a “title” attribute (tooltip text). The coordinates (both center point and bounding box) of these elements are recorded, yielding tasks such as predicting the point coordinates of an element described by its text (text‑to‑point) and the reverse OCR‑like task of predicting the text given the coordinates. For mobile UIs, they reverse the Widget Captioning dataset (about 20k screenshots, 40k widgets, 100k natural‑language descriptions) so that the description becomes the instruction and the widget’s bounding box becomes the target; they also incorporate automatically collected elements from RICO and add UI summarization data to improve overall screen understanding. To preserve the LVLM’s general visual reasoning ability, they include 145k general vision‑language instruction‑following samples from LLaVA. The final 1‑million‑sample mixture is used to continually pre‑train the open‑source LVLM Qwen‑VL (9.6B parameters, input resolution 448×448). During pre‑training, the model sees a prompt like “In the UI, where should I click if I want to <instruction>?” followed by the screenshot, and is trained to generate the target coordinates in natural language, e.g., “click (0.49, 0.40)”. The loss is computed only on the coordinate part. Training is performed with LoRA (Low‑Rank Adaptation) applied to both the visual encoder and the language model, using AdamW with a learning rate of 3e‑5 and global batch size 64 for around 10k steps on 8 NVIDIA A100 GPUs. To benchmark the resulting grounding capability, the authors introduce ScreenSpot, a manually labeled dataset with over 600 screenshots and 1200+ instructions spanning mobile (iOS, Android), desktop (macOS, Windows), and web (development, shopping, forum, tools) platforms. Crucially, about half of the target elements are icons or widgets—elements without text—making the benchmark particularly challenging. On ScreenSpot, SeeClick achieves an average click accuracy of 53.4%, significantly outperforming both generalist LVLMs (Qwen‑VL 5.2%, GPT‑4V 16.2%) and the prior GUI‑specific model CogAgent (47.4% with almost twice the parameters). SeeClick is strongest on mobile text (78.0%) but struggles on desktop and web icons/widgets (30.0% and 32.5% respectively). For downstream GUI automation, SeeClick is fine‑tuned on three tasks: MiniWob (simplified web tasks, 2.8k episodes), AITW (Android‑in‑the‑Wild, an instruction‑wise split to avoid overfitting), and Mind2Web (real website navigation, screenshots extracted from HTML dumps). On MiniWob, despite using only 2.8k examples, SeeClick reaches a 67.0% success rate on vision tasks, beating Pix2Act (64.6% with 1.3M examples) and improving over the untrained Qwen‑VL baseline by 18.6 points. On AITW, it obtains 59.3% overall action‑matching score and a click accuracy of 66.4%, a 9‑point gain over Qwen‑VL. On Mind2Web, the step success rate nearly doubles from 11.5% to 20.8%, although it still lags behind HTML‑based methods that can choose from a candidate set instead of predicting coordinates. A direct correlation analysis (Figure 6) shows that as the grounding accuracy on ScreenSpot grows during pre‑training, the performance on all three downstream agent tasks improves monotonically, substantiating that GUI grounding is the key enabler. Limitations of SeeClick include its restricted action space (no complex actions like drag or double‑click) and the need for task‑specific fine‑tuning due to the limited scale of the base open‑source LVLM.

### Practice environment
Pre‑training data: 300k web pages from Common Crawl (text‑to‑point, text‑to‑bbox, point‑to‑text, bbox‑to‑text), mobile data from Widget Captioning (20k screenshots, 100k descriptions) and RICO, plus 48k UI summarization samples; general data: LLaVA 145k. Model: Qwen‑VL (9.6B parameters, 448×448 resolution) fine‑tuned with LoRA, unlocked visual encoder, AdamW lr=3e‑5, batch size 64, 10k steps on 8 NVIDIA A100 GPUs. ScreenSpot: 600+ screenshots across iOS, Android, macOS, Windows, web; 1200+ instructions; metric: click accuracy (point in bbox). Baselines: MiniGPT‑v2, Qwen‑VL, GPT‑4V, Fuyu, CogAgent. Agent tasks: MiniWob (50 tasks, 2.8K rollout episodes from Synapse, 50 random seeds per task, success rate); AITW (instruction‑wise split, each instruction one trajectory, screen‑wise action matching and click accuracy); Mind2Web (screenshots extracted, cropped to 1920×1080, metrics: element accuracy, operation F1, step success rate).

### Practice trace
1. Action: Curated a 1M‑sample GUI grounding dataset from web Common Crawl pages and mobile UI datasets (Widget Captioning, RICO) plus LLaVA general data, and continually pre‑trained Qwen‑VL on it.
   Feedback: ScreenSpot average click accuracy improved from the base Qwen‑VL’s 5.2% to 53.4% (a >10× relative gain), surpassing CogAgent (47.4%) and all other LVLMs.
2. Action: Fine‑tuned SeeClick on three downstream GUI agent tasks: MiniWob (2.8K episodes), AITW (instruction‑wise split), and Mind2Web (visual version).
   Feedback: MiniWob vision‑task score 67.0% (vs. Qwen‑VL 48.4%); AITW overall 59.3% and click accuracy 66.4% (Qwen‑VL 57.4%); Mind2Web step success rate 20.8% (Qwen‑VL 11.5%).
3. Action: Tracked multiple pre‑training checkpoints on ScreenSpot and downstream tasks to analyze correlation.
   Feedback: Monotonically increasing grounding accuracy on ScreenSpot was accompanied by consistent gains in MiniWob, AITW, and Mind2Web scores, confirming that GUI grounding is the primary driver of agent performance.

### Task context
modality: screenshots of GUIs paired with natural language instructions; output: (x,y) click coordinates or typed text.
scale: 1M pre‑training samples; ScreenSpot: 600+ screenshots, 1200+ instructions; MiniWob: 2.8K episodes; AITW: 5 subsets with instruction‑wise split; Mind2Web: >2000 tasks from 137 real websites.
constraint: limited action space (click, type); requires separate fine‑tuning per platform; still struggles with fine‑grained localization of icons/widgets and dense web interfaces; lower resolution (448×448) may miss small elements.

### Extraction rationale
Method description drawn from Section 3 (GUI grounding pre‑training, data construction) and Section 5 (results overview). The action sequence reflects the paper’s three main steps: pre‑training, fine‑tuning on agent tasks, and correlation analysis. Feedback values are extracted from Tables 1–4 and Figure 6, combining the abstract’s claims with concrete numbers.

## Why It Matters

Granularity: medium. Scale: 1M pre‑training samples; ScreenSpot: 600+ screenshots, 1200+ instructions; MiniWob: 2.8K episodes; AITW: 5 subsets with instruction‑wise split; Mind2Web: >2000 tasks from 137 real websites. Applies under: limited action space (click, type); requires separate fine‑tuning per platform; still struggles with fine‑grained localization of icons/widgets and dense web interfaces; lower resolution (448×448) may miss small elements.

## Evidence

### Source
arXiv:2401.10935 | 2401.10935__c405a1b5 | group: t9 | section: introduction

### Verbatim quote
"To tackle this challenge, SeeClick enhances LVLM with a GUI grounding pre-training strategy. We devise a method to automate the curation of web grounding data and adapt public mobile UI datasets to obtain mobile grounding data. [...] Evaluation results confirm SeeClick’s superiority over current LVLMs, validating the effectiveness of GUI grounding pre-training. [...] Moreover, experimental results on these three benchmarks consistently support our findings that improvement in GUI grounding directly correlates with enhanced agent task performance."

### Classification
domain: agent_web_gui
arXiv: cs.CV
Wikipedia: Human–computer_interaction

### Keywords
visual GUI agent, GUI grounding, large vision-language models, ScreenSpot, mobile UI, web UI, grounding pre-training, MiniWob, AITW, Mind2Web

```

### 经验 ID: `O-90da288ba0fa6fb2`

```markdown
---
id: O-90da288ba0fa6fb2
created_at: '2026-08-16T15:52:57Z'
summary: '[conditional] Augmenting task-oriented GUI demonstration datasets with environment-understanding
  data—specifically, human-verified element grounding and natural-language functionality
  descriptions—consistently improves a vision-based agent''s task completion accuracy,
  with especially large gains on fundamental click and long-press actions.'
memory_type: semantic
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l2
related_observations:
- id: O-51ed81308e03abc2
  relation: complements
  reason: 'Both extracted from arXiv:2407.17490: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:58Z'
- id: O-b2a93b744cf4622a
  relation: complements
  reason: 'Both extracted from arXiv:2407.17490: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:59Z'
---

## Observation

### Declaration
Augmenting task-oriented GUI demonstration datasets with environment-understanding data—specifically, human-verified element grounding and natural-language functionality descriptions—consistently improves a vision-based agent's task completion accuracy, with especially large gains on fundamental click and long-press actions.

### Narrative
Training mobile GUI agents typically requires large-scale datasets of human demonstrations annotated with step-by-step actions. However, such datasets often focus narrowly on task execution sequences and may lack explicit information about the environment—what each screen contains and what each interactive element does. The AMEX dataset provides environment-understanding data at two levels: bounding-box-based interactive element grounding and free-form natural-language descriptions of element functionalities, created through a pipeline of XML parsing, LLM-based generation, and rigorous human verification. The authors hypothesized that this kind of data, even when it does not include action traces, can improve a GUI agent's ability to perform tasks because a better grasp of the interface should translate into more accurate interactions. To evaluate this, they fine-tuned the SphAgent model on the ANDROIDCONTROL dataset under three conditions: (i) training only on ANDROIDCONTROL's task-action pairs; (ii) training on the same total number of data points but replacing 10% of the task data with AMEX's environment data (levels 1 and 2); and (iii) training on the full ANDROIDCONTROL dataset augmented with all of AMEX's environment data. The results show a clear trend: even the data-neutral replacement condition (ii) lifted overall accuracy by about 2.5% on average across both high-level and low-level task instructions, while the full addition of AMEX (iii) increased overall accuracy by roughly 10% and dramatically improved performance on click and long-press actions by over 14%. This pattern indicates that providing agents with explicit knowledge about what elements exist and what they do—beyond the action demonstrations themselves—yields an independent, additive benefit. The authors attribute this gain to the fact that environmental understanding helps the agent build a richer internal representation of the screen, making it less likely to misinterpret the function of unfamiliar elements during task execution. This experience is supported by the consistent improvements across multiple evaluation splits (in-domain, category unseen, app unseen, task unseen) and the particularly large jump in click-related actions, which are fundamental to GUI navigation. The claim is stated with medium confidence; the authors present the results as clear evidence of AMEX's effectiveness but do not isolate the specific mechanism with ablation of different data components. A boundary condition is that the benefit was observed when the environment data came from AMEX's carefully curated and human-verified annotations; automatically extracted or noisy element descriptions may not yield the same effect. Additionally, the action space of ANDROIDCONTROL differs from AMEX's level 3, so only levels 1 and 2 were used—limiting the analysis to screen and element descriptions without the longer, more complex action chains. This finding is particularly relevant for practitioners building datasets for GUI agents: investing in environment annotations is not merely a supplementary add-on but can directly improve downstream task performance, even at the cost of reducing the volume of pure task demonstrations.

### Causal explanation (r)
Explicit environment-understanding annotations allow the agent to construct a richer mental model of the interface, reducing misinterpretation of unfamiliar elements during task execution; this translates into more accurate primitive actions such as clicking.

confidence (μ_r): medium
depth: shallow
depth rationale: The explanation attributes the gain to a 'richer mental model' and reduced misinterpretation, which is a plausible restatement of the benefit rather than a mechanistic account of how the representation changes or why click actions are disproportionately improved.

### Applicability context
summary: Improves a mobile GUI agent's task completion reliability by incorporating high-quality screen and element descriptions into the training mixture, yielding a ~10% overall accuracy boost on ANDROIDCONTROL and a >14% improvement on click and long-press actions, with gains visible even when the total number of training examples is held constant.
modality: mobile screenshots with element bounding boxes and functionality text
scale: ANDROIDCONTROL test set (4 splits, 2 task levels); SphAgent-7B; training data size: 178K or 712K samples; 10% AMEX substitution vs. full addition
constraint: environment data is LLM-generated but human-verified (accuracy >97%); action spaces of the two datasets differ, so only AMEX levels 1 and 2 were used; gains are averaged across low-level (step-guided) and high-level (goal-only) task instructions; tested on a SPHINX architecture with pure vision input

### Extraction rationale
The author explicitly links the ANDROIDCONTROL experiments to the effectiveness of AMEX's multi-level annotations, with quantitative evidence from three training regimes and multiple splits (declaration basis). The casual attribution (r) is derived from the authors' discussion of why multi-level understanding helps, though it is stated in general terms rather than through dedicated causal analysis.

## Why It Matters

Claim type: conditional. Confidence: medium. Improves a mobile GUI agent's task completion reliability by incorporating high-quality screen and element descriptions into the training mixture, yielding a ~10% overall accuracy boost on ANDROIDCONTROL and a >14% improvement on click and long-press actions, with gains visible even when the total number of training examples is held constant. Applies under: environment data is LLM-generated but human-verified (accuracy >97%); action spaces of the two datasets differ, so only AMEX levels 1 and 2 were used; gains are averaged across low-level (step-guided) and high-level (goal-only) task instructions; tested on a SPHINX architecture with pure vision input

## Evidence

### Source
arXiv:2407.17490 | 2407.17490__75c8c90d | group: t9 | section: experiment

### Verbatim quote
Adding full data of AMEX level 1 and level 2 leads to an average 10% overall performance gain and strongly improves the 'click' and 'long press' actions by more than 14%. Also, the comparative results from experiment (i) and (ii) shows that even using the same number of data points, replacing down-stream instructional task data with environment understanding data would also lift the performance of agents on both low-level and high-level at an average of 2.5%. This performance gain indicates the effectiveness of our multi-level annotations.

### Classification
domain: agent_web_gui
arXiv: cs.HC
Wikipedia: Applications_of_artificial_intelligence
ACM CCS: Computing methodologies → Artificial intelligence → Distributed artificial intelligence → Intelligent agents
CLC: TP391.1

### Keywords
GUI agent, environment understanding, data augmentation, click action, mobile device, element functionality, training data mixture, ANDROIDCONTROL

```

### 经验 ID: `O-7808996c1abe4479`

```markdown
---
id: O-7808996c1abe4479
created_at: '2026-08-16T15:52:57Z'
summary: '[conditional] When training data for a vision-based mobile GUI agent includes
  natural-language functionality descriptions of interactive elements (e.g., ''Click
  to add another airport to the search criteria'') instead of or in addition to class-based
  labels (e.g., ICON_PLUS), the agent achieves substantially higher element grounding
  accuracy on unseen screens.'
memory_type: semantic
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l2
related_observations:
- id: O-51ed81308e03abc2
  relation: complements
  reason: 'Both extracted from arXiv:2407.17490: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:58Z'
- id: O-b2a93b744cf4622a
  relation: complements
  reason: 'Both extracted from arXiv:2407.17490: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:59Z'
---

## Observation

### Declaration
When training data for a vision-based mobile GUI agent includes natural-language functionality descriptions of interactive elements (e.g., 'Click to add another airport to the search criteria') instead of or in addition to class-based labels (e.g., ICON_PLUS), the agent achieves substantially higher element grounding accuracy on unseen screens.

### Narrative
Mobile GUI-control agents must interpret on-screen elements to perform tasks like booking a flight or searching for a product. Traditional datasets for training these agents label elements with abstract categories such as ICON_PLUS or TEXT, which fail to convey what a button actually does in its surrounding context—a '+' sign might mean 'add' in a calculator but 'create a new task' in a to-do list. The AMEX dataset introduces a different annotation approach: instead of class labels, each interactive element is given a natural-language description of its functionality in the current screen context (e.g., 'Click to add another airport to the search criteria'). The authors hypothesized that this context-aware functional annotation would provide a stronger learning signal for vision-based agents, enabling them to ground elements more accurately. To test this, they fine-tuned a SPHINX-based vision-language model—called SphAgent—on only AMEX's element-level data (level 1: interactive element bounding boxes filtered by human annotators; level 2: element functionality descriptions generated by GPT-4o and Gemini and verified by humans) and evaluated it on the ScreenSpot mobile subset, a benchmark that measures a model's ability to associate a given functionality description with the correct on-screen region. SphAgent achieved 72.6% accuracy on the Icon/Widget subset, surpassing four previously state-of-the-art GUI-trained large vision-language models (Fuyu-8B at 1.3%, CogAgent-18B at 24.0%, SeeClick-9.6B at 52.0%, Qwen2-VL-7B at 60.7%) and even a GPT-4V pipeline that uses an external screen parser (OmniParser, 57%). The authors note that SphAgent was trained without any specialized tricks, highlighting the strength of the functional annotations. This finding indicates that moving from categorical element typing to descriptive, contextual functionality is a highly effective strategy for improving a GUI agent's grounding capability. The causal mechanism, as implied by the authors, is that functionality descriptions capture the purpose of an element within the screen's layout, allowing the model to learn a richer, semantics-aware representation rather than relying solely on visual appearance or generic icon recognition. This advantage is particularly critical for 'compound' elements that combine images, text, and buttons, where a single class label cannot convey the composite intent. The evidence is drawn from a single, targeted evaluation on ScreenSpot, and the claim is stated with medium confidence—the authors explicitly attribute the gains to 'functionality understanding' but do not perform a formal ablation comparing class labels vs. functionality descriptions on the same architecture. However, the cross-model comparison with the same benchmark and the large performance margin strongly support the assertion. This experience is applicable to vision-only mobile agents trained without access to view hierarchy metadata, a constraint commonly encountered when apps do not expose accessibility trees. It may be less informative for setups that rely heavily on view hierarchy information or for web-based GUI agents where elements have richer HTML semantics.

### Causal explanation (r)
Functionality descriptions capture the contextual purpose of an element within the screen layout, providing a semantics-aware learning signal that generic class labels cannot offer; this helps the model disambiguate elements whose meaning depends on the surrounding GUI context.

confidence (μ_r): medium
depth: shallow
depth rationale: The explanation restates that functionality understanding improves grounding because it provides contextual semantics—essentially recasting the observation in causal terms without identifying a deeper mechanism such as how the representation space changes or why class labels fail.

### Applicability context
summary: Enables a vision-based GUI agent trained solely on screenshots to locate interactive elements more accurately by learning from context-aware functionality descriptions, achieving 72.6% on ScreenSpot mobile Icon/Widget subset, outperforming prior specialized models by a wide margin (e.g., +12.6% over Qwen2-VL and +20.6% over SeeClick).
modality: high-resolution mobile screenshots
scale: ScreenSpot mobile subset (Icon/Widget); SphAgent-7B fine-tuned on 104K AMEX screenshots; comparison with 4 LVLMs and 1 GPT-4V pipeline
constraint: applies to vision-only agents without view hierarchy input; tested only on mobile icon/widget grounding; functionality descriptions generated by LLMs and human-verified; model architecture is SPHINX-based with DINOv2 and ConvNext encoders

### Extraction rationale
The author explicitly interprets the ScreenSpot mobile results as evidence that 'functionality understanding' boosts GUI grounding, supported by a large margin across five comparison models (declaration basis). The causal reasoning (r) is drawn from the authors' motivational discussion in the introduction and method sections, where they argue that class-based labels cannot convey context-dependent meaning, though this is not validated with a controlled causal experiment.

## Why It Matters

Claim type: conditional. Confidence: medium. Enables a vision-based GUI agent trained solely on screenshots to locate interactive elements more accurately by learning from context-aware functionality descriptions, achieving 72.6% on ScreenSpot mobile Icon/Widget subset, outperforming prior specialized models by a wide margin (e.g., +12.6% over Qwen2-VL and +20.6% over SeeClick). Applies under: applies to vision-only agents without view hierarchy input; tested only on mobile icon/widget grounding; functionality descriptions generated by LLMs and human-verified; model architecture is SPHINX-based with DINOv2 and ConvNext encoders

## Evidence

### Source
arXiv:2407.17490 | 2407.17490__75c8c90d | group: t9 | section: experiment

### Verbatim quote
During evaluation, our SphAgent surpasses four LVLMs by a large margin on the 'Icon / Widget' subset without any training tricks, proving that the functionality understanding can largely boost the performance of GUI grounding for agents.

### Classification
domain: agent_web_gui
arXiv: cs.CV
Wikipedia: Intelligent_agents
ACM CCS: Computing methodologies → Artificial intelligence → Computer vision → Scene understanding
CLC: TP391.4

### Keywords
GUI agent, functionality description, element grounding, mobile device, multi-level annotation, vision-based agent, fine-tuning, ScreenSpot

```

### 经验 ID: `O-879f04599d7cd990`

```markdown
---
id: O-879f04599d7cd990
created_at: '2026-08-16T15:52:57Z'
summary: '[coarse] Pre-trains a 9.6B-parameter LVLM (Qwen-VL) on a 1M GUI grounding
  dataset to create SeeClick, a visual GUI agent that locates screen elements from
  screenshots, achieving 53.4% average grounding accuracy on ScreenSpot (surpassing
  CogAgent at 47.4%), and when fine-tuned on agent tasks, attains 67.0% on MiniWob
  (using only 0.3% of the training data of the vision baseline) and 59.3% overall
  score on AITW, with improvements in grounding directly correlating with downstream
  agent performance.'
memory_type: procedural
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l1
related_observations:
- id: O-589f92650cf81736
  relation: complements
  reason: 'Both extracted from arXiv:2401.10935: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:49Z'
- id: O-758437a7e4c0c1ff
  relation: complements
  reason: 'Both extracted from arXiv:2401.10935: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:50Z'
---

## Observation

### Narrative
Graphical User Interface (GUI) agents aim to automate tasks like booking appointments or filling forms on digital devices. Most existing agents rely on structured text representations—HTML on web, view hierarchy on Android—which can be inaccessible (e.g., on iOS or desktop) or excessively long while missing visual layout information. This paper develops SeeClick, a purely visual GUI agent that only uses screenshots to perform clicking and typing actions. The central challenge in building such an agent is GUI grounding: the ability to accurately locate screen elements (buttons, text fields, icons) from natural-language instructions. SeeClick addresses GUI grounding by pre-training a Large Vision-Language Model (LVLM, a neural network that jointly processes images and text) on a large, automatically curated dataset of GUI screenshots with element-location annotations. The dataset is constructed from two sources: web pages (300k from Common Crawl) where visible text and hover-text on interactive elements are paired with their coordinates, yielding tasks such as predicting the (x,y) coordinates of an element described by text; and mobile UIs, where the Widget Captioning dataset (20k screenshots, 40k widgets) and RICO are repurposed to provide language descriptions of elements and their bounding boxes, plus UI summarization data. To preserve general visual understanding, LLaVA instruction-following data on natural images is also included. The resulting 1‑million sample mixture is used to continually pre-train the open‑source LVLM Qwen‑VL (9.6B parameters, 448×448 input resolution) with LoRA (Low‑Rank Adaptation, a parameter‑efficient fine‑tuning method), unlocking the visual encoder gradients and training for 10k steps on 8 NVIDIA A100 GPUs. The pre‑trained model, SeeClick, learns to output numerical coordinates (e.g., “(0.49,0.40)”) in natural language given a screenshot and an instruction prompt. For evaluation, the paper introduces ScreenSpot, the first realistic GUI grounding benchmark spanning mobile (iOS, Android), desktop (macOS, Windows) and web platforms—over 600 screenshots and 1200+ instructions, half targeting iconic/widget elements, half targeting text. SeeClick achieves an average click accuracy of 53.4%, surpassing the previous GUI‑specific model CogAgent (47.4% with 18B parameters) and greatly exceeding generalist LVLMs like GPT‑4V (16.2%). On mobile text elements it reaches 78.0%; for icons/widgets, still challenging, it attains 52.0%. When adapted to downstream agent tasks with task‑specific fine‑tuning, SeeClick demonstrates the value of grounding: on MiniWob (a simplified web‑task benchmark) with only 2.8k demonstration episodes (0.3% of the data used by the prior vision‑only model Pix2Act), it obtains 67.0% success rate vs. Pix2Act’s 64.6%; on AITW (Android automation) with an instruction‑wise split to avoid overfitting, overall score is 59.3% and click accuracy 66.4%, 9 points higher than the base LVLM; on Mind2Web (real‑website navigation) seeclick nearly doubles the step success rate of Qwen‑VL (20.8% vs. 11.5%). A correlation study across pre‑training checkpoints confirms that gains in GUI grounding accuracy directly translate into higher agent task scores. The main limitations are that SeeClick only supports click and type actions (no drag, double‑click), requires fine‑tuning for multi‑step tasks, and suffers from lower resolution and smaller training data compared to some competitors, leading to weaker performance on dense web text or small icons.

### Practice environment
Qwen‑VL (9.6B parameters, 448×448 input) pre‑trained with LoRA on 1M mixture: 300k web pages from Common Crawl (text↔point/bbox tasks), mobile data from Widget Captioning (20k screenshots, 40k widgets, 100k descriptions reversed for grounding) and RICO, plus UI summarization; 145k LLaVA general data. Training: 10k steps, global batch size 64, learning rate 3e‑5, cosine annealing, 8 NVIDIA A100 GPUs. ScreenSpot benchmark: 600+ screenshots from iOS, Android, macOS, Windows, web (development, shopping, forum, tools), 1200+ instructions, metric: click accuracy (predicted point inside ground‑truth bounding box). Baselines: MiniGPT‑v2, Qwen‑VL, GPT‑4V, Fuyu, CogAgent. Downstream tasks: MiniWob (50 tasks, 2.8K rollouts from Synapse, success rate over 50 seeds), AITW (instruction‑wise split, 80% train/20% test, screen‑wise action matching), Mind2Web (extracted screenshots, cropped to 1920×1080, metrics: element accuracy, Operation F1, step success rate).

### Practice trace
1. Action: Continually pre-trained Qwen‑VL on a 1M GUI grounding dataset (web+ mobile+general) with LoRA for 10k steps, teaching the model to predict click coordinates given a screenshot and instruction.
   Feedback: SeeClick achieved 53.4% average click accuracy on ScreenSpot, outperforming CogAgent (47.4%) and GPT‑4V (16.2%); on MiniWob with 2.8K training episodes it scored 67.0%, surpassing the vision‑only Pix2Act (64.6% with 1.3M examples) and the LVLM baseline Qwen‑VL (48.4%); on AITW it improved click accuracy from Qwen‑VL’s 57.4% to 66.4%; on Mind2Web it nearly doubled step success rate from 11.5% to 20.8%.

### Task context
modality: screenshots (static GUI images) together with natural language instructions; output is a (x,y) coordinate or an action type (click/type/select/swipe).
scale: 1M pre-training samples; ScreenSpot: 600+ screenshots, 1200+ instructions; MiniWob: 2.8K episodes; AITW: instruction-wise split with 306–700 instructions per subset; Mind2Web: >2000 tasks from 137 real websites.
constraint: action space limited to click and type; multi-step tasks require separate fine‑tuning per platform; LoRA training data size (1M) and resolution (448×448) may be insufficient for dense, complex interfaces; struggles with fine‑grained icon/widget localization.

### Extraction rationale
Practice summary drawn from abstract. The action-feedback pair reflects the paper’s headline result; detailed setup and numerical context appear in the corresponding fine‑grained experiences. The source quote uses verbatim abstract text that describes both the methodology (pre‑training for GUI grounding, automated data curation) and the result (improvement on ScreenSpot, correlation with downstream tasks).

## Why It Matters

Granularity: coarse. Scale: 1M pre-training samples; ScreenSpot: 600+ screenshots, 1200+ instructions; MiniWob: 2.8K episodes; AITW: instruction-wise split with 306–700 instructions per subset; Mind2Web: >2000 tasks from 137 real websites. Applies under: action space limited to click and type; multi-step tasks require separate fine‑tuning per platform; LoRA training data size (1M) and resolution (448×448) may be insufficient for dense, complex interfaces; struggles with fine‑grained icon/widget localization.

## Evidence

### Source
arXiv:2401.10935 | 2401.10935__c405a1b5 | group: t9 | section: abstract

### Verbatim quote
"In this paper, we propose a novel visual GUI agent – SeeClick, which only relies on screenshots for task automation. [...] To tackle this challenge, we propose to enhance SeeClick with GUI grounding pre-training and devise a method to automate the curation of GUI grounding data. [...] After pre-training, SeeClick demonstrates significant improvement in ScreenSpot over various baselines. Moreover, comprehensive evaluations on three widely used benchmarks consistently support our finding that advancements in GUI grounding directly correlate with enhanced performance in downstream GUI agent tasks."

### Classification
domain: agent_web_gui
arXiv: cs.CV
Wikipedia: Human–computer_interaction

### Keywords
visual GUI agent, GUI grounding, large vision-language models, ScreenSpot, mobile UI, web UI, grounding pre-training, MiniWob, AITW, Mind2Web

```

### 经验 ID: `O-51ed81308e03abc2`

```markdown
---
id: O-51ed81308e03abc2
created_at: '2026-08-16T15:52:57Z'
summary: '[medium] Introduces AMEX, a multi‑level GUI dataset, and demonstrates that
  fine‑tuning a pure vision‑based SPHINX Agent on its annotations improves element
  grounding to 72.6% on ScreenSpot, boosts ANDROIDCONTROL task completion by ~10 percentage
  points, and raises AITW overall accuracy by 2.5 percentage points.'
memory_type: procedural
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l1
related_observations:
- id: O-32ae742330174a66
  relation: complements
  reason: 'Both extracted from arXiv:2407.17490: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:57Z'
- id: O-7808996c1abe4479
  relation: complements
  reason: 'Both extracted from arXiv:2407.17490: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:58Z'
- id: O-90da288ba0fa6fb2
  relation: complements
  reason: 'Both extracted from arXiv:2407.17490: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:58Z'
- id: O-d8424b203cb8e603
  relation: complements
  reason: 'Both extracted from arXiv:2407.17490: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:58Z'
- id: O-6932a04b4ecfc209
  relation: complements
  reason: Both classified under agent_domain_app, from arXiv:2311.03079 and arXiv:2407.17490
    respectively — different papers addressing the same problem area.
  linked_at: '2026-08-16T15:59:13Z'
- id: O-64050f57a14a2c73
  relation: complements
  reason: Both classified under agent_domain_app, from arXiv:2407.17490 and arXiv:2408.01800
    respectively — different papers addressing the same problem area.
  linked_at: '2026-08-16T15:59:13Z'
---

## Observation

### Narrative
Mobile GUI agents aim to autonomously complete tasks by interpreting screen content and interacting with on‑screen elements, yet they often struggle due to a lack of deep understanding of GUI layouts and the functional roles of interactive components. To address this, the AMEX work introduces a large‑scale, multi‑level dataset that moves beyond traditional icon classification and instead annotates screens with the actual functionality of elements in their visual context, as well as with complex instruction‑action sequences. The construction process draws on human cognitive patterns: first, a user perceives the overall screen layout, then identifies interactive regions and what they do, and finally breaks a goal into step‑by‑step actions. Following this model, AMEX provides three annotation levels. Level I: GUI interactive element grounding, where all elements that can be clicked or scrolled are marked with accurate bounding boxes, recognizing that many modern elements are compound (e.g., a card with an image, text, and a rating star) and cannot be neatly categorized as an icon or image. Level II: screen and element functionality descriptions, produced by GPT‑4o and Gemini 1.5 Pro after applying Set‑of‑Mark (SoM) — a visual prompting technique that draws numeric markers over screen regions — and then verified by human annotators (97%+ accuracy). For instance, a “plus” icon in a calculator is described as “perform addition,” while in a task manager it is described as “create a new task.” Level III: complex instructions with GUI-action chains, where each instruction averages 12.8 steps (far more than the typical 4–7 steps in prior datasets), covering realistic tasks like “Open Sephora, search for ‘foundation’, sort by highest rating, select the third result, and add it to cart.” The action space includes TAP, SCROLL, TYPE, three system‑level PRESS actions, and terminal flags; for query tasks a region‑of‑interest bounding box marks the answer location. The raw data was gathered from 192 apps using Android emulators and Appium, with screenshots and XML captured via both human operations and an autonomous exploration script. Human annotators then filtered out occluded or misaligned bounding boxes and validated all auto‑generated descriptions and instructions. Using this dataset, the authors fine‑tuned a SPHINX Agent — a vision‑language model built on internlm‑7b with DINOv2 and ConvNext visual encoders — operating purely on screenshots without any accessibility tree or view‑hierarchy data. The model was trained on 8× A100 GPUs for 4 epochs with a history window of 4 frames and a special <ICON> token. Experiments across three benchmarks demonstrated substantial gains over agents trained without AMEX. On ScreenSpot (mobile icon/widget subset) the agent achieved 72.6% accuracy, outperforming SeeClick (52.0%), Qwen2‑VL (60.7%), and even GPT‑4V with OmniParser (57%). When AMEX was mixed into ANDROIDCONTROL training data, average task completion improved by around 10 percentage points overall, with click and long‑press actions boosted by over 14 percentage points. On AITW, adding AMEX lifted overall success from 76.28% to 78.72%, with the largest improvements on General (+4.9 points) and Single (+5.4 points) tasks. The work establishes that a dataset explicitly designed to mimic human GUI understanding — via element grounding, functional context, and long instruction chains — can significantly strengthen vision‑only mobile agents, although cross‑domain evaluation exposed sensitivity to interaction style differences and app version changes.

### Practice environment
Raw data collected via Android emulators (AVD, Genymotion) and Appium. Bounding boxes parsed from XML, manually filtered. Element functionalities generated by GPT‑4o and Gemini 1.5 Pro with Set‑of‑Mark (SoM) overlay, cross‑checked between models, and verified by humans (>97% accuracy). Instructions seeded by humans, expanded by GPT, and manually filtered. Agent: SPHINX (internlm‑7b) with DINOv2 and ConvNext encoders, history window 4, <ICON> token; no accessibility information. Trained on 8× A100 80GB GPUs, 4 epochs. Evaluated on ScreenSpot (mobile icon/widget subset), AITW (General, Install, G‑Apps, Single, WebShopping), ANDROIDCONTROL (IDD, category‑unseen, app‑unseen, task‑unseen, high‑ and low‑level). Baselines: Fuyu-8B, CogAgent-18B, SeeClick-9.6B, Qwen2‑VL-7B, GPT‑4V+OmniParser, agents trained only on AITW or only on ANDROIDCONTROL.

### Practice trace
1. Action: Designed three annotation levels: interactive element grounding (clickable/scrollable boxes), functional descriptions (context‑dependent, generated by GPT‑4o & Gemini, human‑verified), and long‑chain instructions (avg 12.8 steps) collected from human demonstrations and augmented via GPT.
   Feedback: The resulting AMEX dataset contains 104K screenshots, 1.66M interactive elements, 296K verified element functionalities, and 3,046 unique complex instructions.
2. Action: Fine‑tuned a vision‑only SPHINX Agent (internlm‑7b, DINOv2+ConvNext) on AMEX and evaluated on three benchmarks: ScreenSpot, AITW, ANDROIDCONTROL.
   Feedback: Achieved 72.6% on ScreenSpot; improved ANDROIDCONTROL task completion by ~10pp overall; lifted AITW overall accuracy from 76.28% to 78.72%, with General +4.9 pp and Single +5.4 pp.

### Task context
modality: mobile GUI screenshots, XML layout, natural language instructions
scale: 104K screenshots, 192 apps, 1.66M interactive elements, 296K element functionalities, 3,046 instructions (avg 12.8 steps); 7B vision‑language model fine‑tuned on 8× A100 GPUs
constraint: predominantly English; cross‑domain performance drops sharply when visual appearance and interaction styles differ between training and test sets; offline evaluation only

### Extraction rationale
Practice description based on the introduction, which presents the full motivation and high‑level construction of the dataset and its use to train the SPHINX Agent. The action‑feedback pairs summarise the dataset creation and the subsequent agent evaluation across three external benchmarks, with numerical gains reported in Sections 4.2–4.4.

## Why It Matters

Granularity: medium. Scale: 104K screenshots, 192 apps, 1.66M interactive elements, 296K element functionalities, 3,046 instructions (avg 12.8 steps); 7B vision‑language model fine‑tuned on 8× A100 GPUs Applies under: predominantly English; cross‑domain performance drops sharply when visual appearance and interaction styles differ between training and test sets; offline evaluation only

## Evidence

### Source
arXiv:2407.17490 | 2407.17490__75c8c90d | group: t9 | section: introduction

### Verbatim quote
To address this need, we introduce a new dataset, the Android Multi-annotation EXpo (AMEX), specifically designed to advance the development of GUI agents by providing a multi-level understanding of mobile GUIs. AMEX includes three levels of annotations: (i) GUI interactive element grounding, (ii) GUI screen and element functionality descriptions, and (iii) instructions with GUI-action chains. The dataset comprises over 104K high-resolution screenshots, 21K screen descriptions with 300K element-wise functionalities, and approximately 3,000 unique complex instructions, with an average of 12.8 steps ... We train SPHINX Agents, which can serve as the baseline models for future researches on GUI agents and illustrates the effectiveness of AMEX.

### Classification
domain: agent_domain_app
arXiv: cs.HC
Wikipedia: Human–computer_interaction

### Keywords
mobile GUI agents, dataset, multi-annotation, screen understanding, element grounding, action prediction, instruction following, GUI automation, Android, visual grounding

```

---

# 指令微调的数据质量与数量

**用户提问：** 在进行指令微调（Instruction Tuning）时，数据质量和数据数量哪个更重要？少量高质量数据足够吗？

**检索关键词：** `instruction tuning data quality versus quantity high quality subset fine-tuning`

## 回答

没有绝对的“质量优先”或“数量优先”，要看任务目标：

- **对于对话式指令对齐、帮助性与安全性等行为塑造，数据质量通常比数量更重要。**  
  Llama 2-Chat 的经验表明，27,540 条经过精心标注的高质量 SFT 数据，效果可以超过数百万条质量不稳定的第三方数据。原因是高质量示例提供了更一致、更清晰的行为信号，能减少噪声、矛盾指令和不安全模式的干扰（O-0b94b448d0ba1641）。

- **少量高质量数据可以足够，但这是有条件的。**  
  “足够”主要指：让一个已有较强预训练基础的模型学会特定的对话风格、响应格式和基本安全行为；并不意味着少量数据能够覆盖所有知识、任务和分布。上述经验中的规模是约 **2.75 万条**，而不是任意几百条样本，且依赖高水平人工标注和明确的质量标准。

- **对于需要广泛任务覆盖、知识覆盖或跨分布泛化的场景，数量和多样性仍然重要。**  
  Flan-T5 使用约 1,500 万条指令数据，覆盖大量不同任务，从而获得较强的任务泛化和参数效率（O-f713add7794083c6）。UI 控制智能体的经验也显示，域内性能会随示例数量增加，但高层任务的域外泛化提升很慢，单纯堆数据也可能不足（O-5d6dbefd29ce6151）。

- **指令微调不能完全弥补预训练基础不足。**  
  在知识密集型问题和推理任务上，较弱的预训练基础会限制最终性能；即使使用高质量指令数据，也无法完全填补知识缺口（O-a2f5c4e11b5d6987）。

因此，实践上可以采用以下原则：

1. 先保证数据的**正确性、一致性、安全性和任务相关性**；
2. 再通过增加数据来补充**任务类型、语言、领域和边界案例的多样性**；
3. 若目标是特定对话风格或行为对齐，少量高质量数据可能已经很有效；
4. 若目标是广泛知识、复杂推理或域外泛化，则不能只依赖少量数据，还需要更丰富的数据和更强的预训练基础。

经验库未覆盖一个适用于所有指令微调任务的统一“质量—数量”定量规律，因此不能据此断言少量高质量数据在所有场景都足够。

## 思维链 (Reasoning)

- 根据 **O-0b94b448d0ba1641**：在对话式对齐场景中，27,540 条高质量数据超过数百万低质量数据，说明此类任务中数据质量和信噪比更关键。
- 根据 **O-f713add7794083c6**：大规模、任务多样的指令数据能够提升跨任务泛化，说明数据数量和覆盖面在广泛能力目标下仍然重要。
- 根据 **O-5d6dbefd29ce6151**：增加数据对域内任务有效，但对域外高层任务提升缓慢，说明“更多数据”不必然解决泛化问题。
- 根据 **O-a2f5c4e11b5d6987**：预训练基础决定知识密集型任务的能力上限，因此少量 SFT 不能替代充分预训练。
- 综合这些经验，结论是：**质量优先适用于特定行为对齐，但数量、多样性和预训练基础对广泛能力与泛化同样不可或缺。**

## 召回的原始经验全貌 (5 条)

### 经验 ID: `O-0b94b448d0ba1641`

```markdown
---
id: O-0b94b448d0ba1641
created_at: '2026-08-16T15:52:57Z'
summary: '[conditional] For aligning a pretrained LLM to dialogue-style instructions,
  using a small set of high-quality supervised fine-tuning examples (fewer than 30,000)
  can outperform training on millions of lower-quality third-party examples, because
  the clean signal from curated demonstrations reduces noise and contradictory instruction
  patterns.'
memory_type: semantic
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l2
related_observations:
- id: O-005e5a836f90679c
  relation: complements
  reason: 'Both extracted from arXiv:2307.09288: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:32Z'
- id: O-250c4e61079bd801
  relation: complements
  reason: 'Both extracted from arXiv:2307.09288: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:33Z'
---

## Observation

### Declaration
For aligning a pretrained LLM to dialogue-style instructions, using a small set of high-quality supervised fine-tuning examples (fewer than 30,000) can outperform training on millions of lower-quality third-party examples, because the clean signal from curated demonstrations reduces noise and contradictory instruction patterns.

### Narrative
When fine-tuning a large language model (LLM) for dialogue-style interactions, a common starting point is to leverage existing supervised instruction-tuning datasets collected from various public sources. These datasets often contain millions of prompt-response pairs, but their quality and relevance for chat-oriented assistant behavior can be highly variable. In the development of Llama 2-Chat, the authors initially employed third-party supervised fine-tuning (SFT) data (instruction–response examples used to directly teach the model desired outputs) but discovered that its lack of diversity and insufficient quality—particularly the absence of well-crafted, multi-turn, and helpful yet safe responses—limited alignment performance. They therefore shifted strategy: instead of training on millions of potentially noisy examples, they invested in collecting a relatively small corpus of high-quality human-written demonstrations. The key finding is that this shift from quantity to quality resulted in notable improvements in the model’s ability to follow dialogue instructions and generate helpful, safe answers. Specifically, the authors assembled only 27,540 carefully curated SFT annotations, and they observed that this limited set was sufficient to achieve a high level of alignment. They even found that model outputs sampled from this SFT-trained model were often competitive with the human-written reference responses, indicating that the model had internalized strong behavioral patterns from the clean data. This observation is consistent with the broader insight that, for alignment, the signal-to-noise ratio in the training data is more important than sheer data volume. The underlying reason appears to be that low-quality or inconsistent SFT examples introduce noise and contradictory patterns that confuse the model, whereas a smaller but coherent set of high-fidelity demonstrations provides a clearer target for the model to imitate. The experience is bounded by the context of dialogue-oriented chat alignment: the finding may not directly transfer to tasks requiring broad factual coverage where larger and more diverse data might still be necessary. Additionally, the quality threshold for SFT data is high—the annotations were produced by trained annotators following detailed guidelines. The data scale used is in the tens of thousands of examples, which is orders of magnitude smaller than typical instruction-tuning corpora. The result was demonstrated on the Llama 2 pretrained base models, and the authors validated the approach through iterative RLHF improvements that built upon this SFT foundation. Thus, when resources are limited and alignment to a specific conversational style is the goal, prioritizing annotation quality over quantity in the SFT stage is an effective strategy.

### Causal explanation (r)
Third-party SFT data lacks sufficient diversity and quality for dialogue-style instruction alignment, introducing noise and inconsistent patterns that confuse the model; a smaller set of curated high-quality data provides a cleaner signal for the model to imitate.

confidence (μ_r): medium
depth: shallow
depth rationale: The causal explanation essentially restates the finding—insufficient quality and diversity are the reason for inferior performance—without identifying a deeper mechanism beyond noise reduction.

### Applicability context
summary: When developing a dialogue-oriented LLM, this experience shows that collecting a small set of high-quality SFT examples (27k annotations) can yield alignment improvements beyond using millions of third-party data, as cleaner demonstrations provide a stronger behavioral signal.
modality: text
scale: 27,540 high-quality SFT annotations vs. millions of third-party examples; tested on Llama 2 models of 7B to 70B parameters
constraint: requires access to skilled human annotators for high-quality demonstration writing; the small-data strategy is demonstrated for dialogue alignment, not for broad knowledge-intensive tasks; SFT data quality is paramount and must adhere to safety and helpfulness guidelines

### Extraction rationale
The author explicitly interprets the choice to use fewer high-quality SFT examples as the cause of notable improvements, comparing millions of third-party examples against 27k curated ones. This is an explicit design decision backed by observed improvements (declaration basis). The causal explanation (r) is drawn from the author's statement that third-party data had insufficient diversity/quality, which is the reason for switching strategy, but it does not identify a deeper mechanism, so r is set to shallow.

## Why It Matters

Claim type: conditional. Confidence: medium. When developing a dialogue-oriented LLM, this experience shows that collecting a small set of high-quality SFT examples (27k annotations) can yield alignment improvements beyond using millions of third-party data, as cleaner demonstrations provide a stronger behavioral signal. Applies under: requires access to skilled human annotators for high-quality demonstration writing; the small-data strategy is demonstrated for dialogue alignment, not for broad knowledge-intensive tasks; SFT data quality is paramount and must adhere to safety and helpfulness guidelines

## Evidence

### Source
arXiv:2307.09288 | 2307.09288__3b444c4b | group: t7 | section: fine-tuning

### Verbatim quote
"We started the SFT stage with publicly available instruction tuning data, as utilized previously in Touvron et al. (2023). Quality Is All You Need. Third-party SFT data is available from many different sources, but we found that many of these have insufficient diversity and quality—in particular for aligning LLMs towards dialogue-style instructions. As a result, we focused first on collecting several thousand examples of high-quality SFT data, as illustrated in Table 5. By setting aside millions of examples from third-party datasets and using fewer but higher-quality examples from our own vendor-based annotation efforts, our results notably improved. [...] We found that SFT annotations in the order of tens of thousands was enough to achieve a high-quality result. We stopped annotating SFT after collecting a total of 27,540 annotations."

### Classification
domain: agent_general
arXiv: cs.CL
Wikipedia: Chatbots
ACM CCS: Computing methodologies → Artificial intelligence → Natural language processing → Natural language generation
CLC: TP181

### Keywords
LLM alignment, supervised fine-tuning, data quality, dialogue systems, RLHF, instruction tuning, chat model, data efficiency

```

### 经验 ID: `O-a2f5c4e11b5d6987`

```markdown
---
id: O-a2f5c4e11b5d6987
created_at: '2026-08-16T15:52:57Z'
summary: '[conditional] A strong pretraining foundation, particularly one involving
  more than one trillion training tokens, is a necessary condition for instruction-tuned
  large language models to achieve high performance on knowledge-intensive problem-solving
  and alignment tasks; models pretrained on fewer tokens cannot reach comparable levels
  even with high-quality instruction tuning.'
memory_type: semantic
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l2
related_observations:
- id: O-438463797c961c9a
  relation: complements
  reason: 'Both extracted from arXiv:2306.04757: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:31Z'
---

## Observation

### Declaration
A strong pretraining foundation, particularly one involving more than one trillion training tokens, is a necessary condition for instruction-tuned large language models to achieve high performance on knowledge-intensive problem-solving and alignment tasks; models pretrained on fewer tokens cannot reach comparable levels even with high-quality instruction tuning.

### Narrative
The foundation model—the pretrained language model that serves as the starting point for instruction tuning—plays a gate-keeping role for downstream problem-solving capability. While instruction tuning can unlock latent abilities, INSTRUCTEVAL demonstrates that a solid pretraining base, defined largely by the volume and diversity of pretraining tokens, is a necessary condition for achieving competitive performance on factual and reasoning tasks.

The evidence is clearest when comparing models of similar scale but different pretraining histories. Dolly V2 is based on Pythia‑12B, which was pretrained on 472 billion tokens from The Pile. OPT-IML builds on OPT‑30B, pretrained on a mere 180 billion tokens. Despite being the largest instructed model in the study (30B parameters), OPT-IML's average problem-solving score was only 38.7, below even the 3B Flan-T5 (47.4) and far behind Flan-UL2 (20B, 51.6), which benefited from a 1‑trillion‑token pretraining corpus. Dolly V2 fared worse, averaging 23.2. On the challenging MMLU benchmark, Dolly V2 scored 25.6% and OPT-IML 38.6%, whereas models with richer pretraining like LLaMA‑based Vicuna and Alpaca (1.4T tokens) scored around 49–58%. The pattern held across BBH, DROP, CRASS, and HumanEval: the less the pretraining data, the lower the ceiling after instruction tuning.

Alignment tasks (HHH) showed a similar dependency. Pythia‑12B pretrained on 472B tokens had a foundation HHH score of 45.1%, and Dolly V2 instruction tuning only raised it to 54.2%. In contrast, LLaMA‑13B's foundation score was 61.8%, and instruction-tuned versions like Vicuna reached 65.9%. The raw gap in pretrained knowledge limited how much instruction tuning could improve the model's ability to distinguish helpful, honest, and harmless responses.

The probable mechanism is straightforward: problem-solving benchmarks like MMLU and CRASS probe world knowledge across subjects (history, biology, law) and causal reasoning that are acquired primarily during large-scale pretraining on diverse web and book corpora. Models with insufficient pretraining tokens have not encountered enough factual material to form the necessary internal representations, so instruction tuning—which mainly teaches task format and response style—cannot fill the knowledge void. This aligns with the observation that pretraining on less than one trillion tokens consistently results in lower ceilings for knowledge-intensive tasks, regardless of the quality of subsequent instruction tuning.

### Applicability context
summary: Developing a capable instruction-tuned LLM requires a foundation model trained on at least 1T tokens; otherwise, even the best instruction data cannot overcome the knowledge deficit, as seen in OPT-IML and Dolly V2's consistent underperformance.
modality: text
scale: 10+ instructed models, pretraining token volumes from 180B to 1.4T, model sizes 3B-30B
constraint: the token threshold of ~1T is approximate; the relationship likely depends on data quality and diversity, not just quantity; tested only on English-centric Transformer models; may not apply to multilingual or code-focused pretraining.

### Extraction rationale
The declaration is a direct restatement of the author's categorical observation ('necessary condition') supported by consistent cross-model comparisons (OPT-IML vs. peers) across multiple benchmarks. No explicit causal mechanism is provided by the authors beyond the factual observation, so r is null.

## Why It Matters

Claim type: conditional. Confidence: high. Developing a capable instruction-tuned LLM requires a foundation model trained on at least 1T tokens; otherwise, even the best instruction data cannot overcome the knowledge deficit, as seen in OPT-IML and Dolly V2's consistent underperformance. Applies under: the token threshold of ~1T is approximate; the relationship likely depends on data quality and diversity, not just quantity; tested only on English-centric Transformer models; may not apply to multilingual or code-focused pretraining.

## Evidence

### Source
arXiv:2306.04757 | 2306.04757__806c2641 | group: t7 | section: experiment

### Verbatim quote
"We observe that a solid pretraining foundation is a necessary condition to perform well on the problem-solving tasks. Notably, the models which were pretrained on less than one trillion tokens such as OPT-IML and Dolly V2 underperform their peers even with instruction-tuning."

### Classification
domain: agent_evaluation
arXiv: cs.CL
Wikipedia: Natural_language_processing
ACM CCS: Computing methodologies → Artificial intelligence → Natural language processing
CLC: TP391.1

### Keywords
instruction tuning, large language model, pretraining, evaluation suite, problem-solving, alignment, scaling, foundation model

```

### 经验 ID: `O-f713add7794083c6`

```markdown
---
id: O-f713add7794083c6
created_at: '2026-08-16T15:52:57Z'
summary: '[trend] High-quality instruction tuning can reverse the flat scaling trend
  of foundation models, making smaller instructed models outperform larger un-instructed
  counterparts, because the instruction data imparts task knowledge and response formats
  that compensate for limited parameter count.'
memory_type: semantic
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l2
related_observations:
- id: O-438463797c961c9a
  relation: complements
  reason: 'Both extracted from arXiv:2306.04757: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:31Z'
---

## Observation

### Declaration
High-quality instruction tuning can reverse the flat scaling trend of foundation models, making smaller instructed models outperform larger un-instructed counterparts, because the instruction data imparts task knowledge and response formats that compensate for limited parameter count.

### Narrative
A commonly observed property of raw foundation models is that increasing model size yields diminishing returns—the scaling curve often flattens with larger parameters. INSTRUCTEVAL shows that effective instruction tuning can fundamentally alter this dynamic, turning a flat scaling trend into a steep, positive one and making smaller instructed models outperform much larger un-instructed counterparts.

The iconic example is the T5 family. The T5 foundation models (3B, 11B) without instruction tuning exhibit a nearly flat scaling curve on problem-solving and alignment benchmarks; the 11B variant provides only a modest advantage over the 3B version. After Flan instruction tuning—which augments T5 with the Flan Collection of 15M human-annotated instructions—the scaling behavior is transformed. Flan-T5 3B achieves an average problem-solving score of 47.4, already surpassing the performance level of the 11B T5 foundation model (which, while not directly reported in the paper, is implied to be much lower, as the paper explicitly states that 'the smallest version of the Flan-T5 model series outperforms the largest version of the T5 foundation model series'). On the alignment (HHH) benchmark, the improvement is even more striking: Flan-T5 3B reaches approximately 76% average accuracy, while the T5 11B foundation model sits around 52.5%. In other words, a 3B instructed model can be more helpful, honest, and harmless than an 11B raw model.

The underlying reason is that instruction tuning infuses the model with large amounts of structured, task-specific knowledge and format understanding that the foundation model lacked. The Flan Collection covers thousands of distinct tasks, effectively performing a form of meta-learning that teaches the model how to interpret and execute instructions. This compensates for the smaller capacity, enabling the model to use its parameters more efficiently for downstream generalization. The paper therefore concludes that for resource-constrained developers, investing in high-quality instruction data and effective tuning recipes is more impactful than simply scaling model size.

It is important to note that this reversal was observed primarily with encoder-decoder architectures (T5) and a massive instruction corpus. Whether the same holds for decoder-only models like LLaMA with different instruction sets is not directly compared, though the general trend of Flan-T5 outperforming larger foundation models is consistent across MMLU, BBH, and HHH. The effect may also interact with the pretraining token volume; however, the existence of the phenomenon demonstrates that instruction data can drastically improve parameter efficiency.

### Applicability context
summary: When building instruction-tuned LLMs, focusing on the scale and quality of instruction data can allow a small model (e.g., 3B) to outperform a 3× larger foundation model, offering a cost-effective path to strong performance without massive compute for pretraining.
modality: text
scale: T5 models at 3B and 11B, Flan Collection instruction data (15M examples)
constraint: demonstrated on encoder-decoder T5 with Flan Collection; generalization to other architectures (e.g., decoder-only) or smaller instruction sets requires verification; the trend may not hold if the base model is already heavily pretrained with instruction-like data.

### Extraction rationale
The declaration is derived from the author's explicit statement that instruction tuning alters the scaling trend and that 3B Flan-T5 beats 11B T5 (declaration basis). No explicit causal explanation beyond the effect of instruction data is given, so r is null.

## Why It Matters

Claim type: trend. Confidence: medium. When building instruction-tuned LLMs, focusing on the scale and quality of instruction data can allow a small model (e.g., 3B) to outperform a 3× larger foundation model, offering a cost-effective path to strong performance without massive compute for pretraining. Applies under: demonstrated on encoder-decoder T5 with Flan Collection; generalization to other architectures (e.g., decoder-only) or smaller instruction sets requires verification; the trend may not hold if the base model is already heavily pretrained with instruction-like data.

## Evidence

### Source
arXiv:2306.04757 | 2306.04757__806c2641 | group: t7 | section: discussion

### Verbatim quote
"Notably, we observe that the scaling trend of the T5 foundation model remains relatively flat, while highly effective instructed models like Flan-T5 demonstrate better scaling and parameter efficiency. Notably, the smallest version of the Flan-T5 model series outperforms the largest version of the T5 foundation model series. Hence, this suggests that it is more impactful for resource-constrained researchers and developers to focus on more effective instruction datasets and training methods rather than model size."

### Classification
domain: agent_evaluation
arXiv: cs.CL
Wikipedia: Natural_language_processing
ACM CCS: Computing methodologies → Artificial intelligence → Natural language processing
CLC: TP391.1

### Keywords
instruction tuning, scaling, large language model, evaluation suite, problem-solving, alignment, parameter efficiency

```

### 经验 ID: `O-75981c82413e8234`

```markdown
---
id: O-75981c82413e8234
created_at: '2026-08-16T15:52:57Z'
summary: '[conditional] A strong multilingual LLM pivot enables efficient cross‑lingual
  generalization of multimodal capabilities: only a very small fraction (e.g., ≤0.5%)
  of multilingual supervised fine‑tuning data is sufficient to achieve high‑quality
  multimodal interaction in 30+ languages, largely eliminating the need for per‑language
  multimodal pre‑training data.'
memory_type: semantic
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l2
related_observations:
- id: O-64050f57a14a2c73
  relation: complements
  reason: 'Both extracted from arXiv:2408.01800: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:59:00Z'
- id: O-e92ec79b387d047b
  relation: complements
  reason: 'Both extracted from arXiv:2408.01800: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:59:02Z'
---

## Observation

### Declaration
A strong multilingual LLM pivot enables efficient cross‑lingual generalization of multimodal capabilities: only a very small fraction (e.g., ≤0.5%) of multilingual supervised fine‑tuning data is sufficient to achieve high‑quality multimodal interaction in 30+ languages, largely eliminating the need for per‑language multimodal pre‑training data.

### Narrative
Multimodal large language models (MLLMs) typically require massive amounts of image–text data in each target language to perform well across languages. For low‑resource languages, collecting and cleaning such multimodal corpora can be prohibitively expensive, severely limiting the reach of these models. Traditional solutions involve extensive data curation and training for every language, a strategy that does not scale. A more efficient alternative explored in this work is to leverage the inherent multilingual capabilities of the underlying language model (LLM) to generalize visual–language alignment learned primarily from high‑resource languages to many other languages with minimal additional multimodal data.

The key finding is that a strong multilingual LLM pivot can dramatically reduce the need for per‑language multimodal training data. MiniCPM‑Llama3‑V 2.5 uses Llama‑3‑Instruct 8B as its base LLM, which already possesses multilingual text representations across dozens of languages. The entire multimodal pre‑training (aligning the vision encoder and compression layer with the LLM) was performed using only English and Chinese image–text pairs—no data from other languages. After pre‑training and the main supervised fine‑tuning (SFT) phase, a tiny amount of multilingual multimodal SFT data was added: just 90,000 examples covering 36 languages, constituting less than 0.5% of the total SFT data. This lightweight multilingual augmentation was applied after the model had already learned strong visual understanding from English and Chinese.

The results are striking. As shown in table 7 and Figure 8, the model without multilingual SFT (“w/o ML”) exhibited very poor performance on non‑English/Chinese languages, with LLaVA Bench scores as low as 13.7 for Korean and 14.4 for Thai. Adding the 0.5% multilingual SFT data lifted scores by more than 25 points in every tested language; for instance, Korean improved to 67.9, Thai to 61.9, and Japanese jumped from 13.8 to 88.0. This massive jump demonstrates that the multilingual reasoning capability already present in the LLM’s text weights can be effectively steered to generalize the vision–language knowledge with only a handful of multimodal exemplars per language. The resulting MiniCPM‑Llama3‑V 2.5 outperforms much larger models like Yi‑VL 34B on multilingual multimodal benchmarks, as illustrated in Figure 13, where it is tested on 15 typologically diverse languages including German, French, Japanese, Korean, and Spanish.

The underlying mechanism, as articulated by the authors, is that the strong multilingual LLM acts as a pivot: the visual modules are trained to output representations that align with the LLM’s internal semantic space, which is already language‑agnostic to a large degree. Once the alignment is established on a couple of high‑resource languages, the same visual tokens can elicit sensible responses in any language that the LLM understands, provided a small amount of targeted fine‑tuning teaches the model to produce answers in those languages. The authors explicitly credit earlier findings from VisCPM [41] for this insight, noting that “the integration of a multilingual LLM significantly alleviates the heavy reliance on multimodal training data in low‑resource languages.”

There are notable boundaries to this experience. The improvement is uneven across languages: those for which the base LLM has stronger representation (e.g., European languages) show higher final scores, while some languages still lag. The study also only tested text‑based visual question answering and captioning; whether the same generalization holds for fine‑grained tasks like OCR in non‑Latin scripts or for complex reasoning remains to be investigated. Furthermore, the success depends on the multilingual strength of the chosen LLM; a weaker multilingual backbone might not transfer vision‑language skills as effectively. Nevertheless, the finding offers a highly scalable recipe: for any new language, only a minuscule fraction of SFT data is needed to unlock multimodal capabilities, thus dramatically lowering the barrier for building globally deployable MLLMs.

### Causal explanation (r)
The multilingual LLM provides a shared cross‑lingual semantic space; once visual‑language alignment is learned on high‑resource languages, the same visual representations can generate responses in any language the LLM understands, and only a tiny amount of target‑language SFT is needed to teach the model to produce output in those languages.

confidence (μ_r): medium
depth: deep
depth rationale: Identifies cross‑lingual representation sharing as the causal mechanism enabling transfer, based on earlier work (VisCPM); the explanation goes beyond the observation by specifying the pivot role of the multilingual LLM.

### Applicability context
summary: Enables building multilingual multimodal assistants with minimal additional data: using a strong multilingual LLM, only 0.5% of SFT data in target languages boosts non‑English multimodal performance by 25+ points, making support for 30+ languages practical without expensive multilingual pre‑training.
modality: text + images
scale: 90k multilingual SFT samples (36 languages) out of total SFT, evaluation on LLaVA Bench translated into 15 languages, model size 8.5B
constraint: improvement magnitude varies per language, depending on the base LLM’s proficiency in that language; tested on open‑ended visual QA, not on OCR with non‑Latin scripts

### Extraction rationale
The author explicitly states the finding from VisCPM and validates it with a controlled ablation (90k multilingual SFT data, table 7). The causal explanation is directly taken from the text: ‘via a strong multilingual LLM pivot’ and the reliance reduction on multimodal data for low‑resource languages.

## Why It Matters

Claim type: conditional. Confidence: high. Enables building multilingual multimodal assistants with minimal additional data: using a strong multilingual LLM, only 0.5% of SFT data in target languages boosts non‑English multimodal performance by 25+ points, making support for 30+ languages practical without expensive multilingual pre‑training. Applies under: improvement magnitude varies per language, depending on the base LLM’s proficiency in that language; tested on open‑ended visual QA, not on OCR with non‑Latin scripts

## Evidence

### Source
arXiv:2408.01800 | 2408.01800__9876f844 | group: t8 | section: method

### Verbatim quote
"Recent findings from VisCPM [41] have shown that the multimodal capabilities can be efficiently generalized across languages via a strong multilingual LLM pivot. This solution largely alleviates the heavy reliance on multimodal data in low‑resource languages. [...] using less than 0.5% multilingual SFT data, we can see over 25 point improvement in all languages"

### Classification
domain: agent_general
arXiv: cs.CL
Wikipedia: Generative_AI
ACM CCS: Computing methodologies → Artificial intelligence → Natural language processing → Natural language generation
CLC: TP18

### Keywords
end-side MLLM, efficient multimodal model, adaptive visual encoding, hallucination reduction, multilingual multimodal, mobile deployment, token compression, RLHF alignment, small language model, GPT-4V level

```

### 经验 ID: `O-5d6dbefd29ce6151`

```markdown
---
id: O-5d6dbefd29ce6151
created_at: '2026-08-16T15:52:57Z'
summary: '[trend] For LLM‑based UI control agents fine‑tuned on human demonstrations,
  in‑domain step accuracy improves log‑linearly with training data, but out‑of‑domain
  performance scales much more slowly, requiring one to two orders of magnitude more
  episodes to achieve comparable accuracy; this discrepancy implies that fine‑tuning
  alone is insufficient for robust generalization on high‑level tasks.'
memory_type: semantic
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l2
related_observations:
- id: O-5186b05e33577926
  relation: complements
  reason: 'Both extracted from arXiv:2406.03679: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:52Z'
- id: O-a580fd47eed481a8
  relation: complements
  reason: 'Both extracted from arXiv:2406.03679: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:53Z'
---

## Observation

### Declaration
For LLM‑based UI control agents fine‑tuned on human demonstrations, in‑domain step accuracy improves log‑linearly with training data, but out‑of‑domain performance scales much more slowly, requiring one to two orders of magnitude more episodes to achieve comparable accuracy; this discrepancy implies that fine‑tuning alone is insufficient for robust generalization on high‑level tasks.

### Narrative
Background: Autonomous UI control agents powered by large language models (LLMs) promise to automate mobile tasks by interpreting natural language instructions and interacting with on-screen elements. However, without fine-tuning on human demonstrations, the absolute performance of such agents remains modest, with success rates often falling below 50% on real-world applications. A critical open question is how the quantity of demonstration data influences agent performance, both on tasks similar to those in the training set (in‑domain) and on previously unseen applications or task types (out‑of‑domain). The paper introduces ANDROIDCONTROL, a diverse dataset of 15,283 human‑collected task demonstrations spanning 833 Android apps, to systematically study this scaling behavior.

Finding: The key inductive finding is that for LLM‑based UI control agents fine‑tuned with parameter‑efficient methods (LoRA), step‑wise accuracy improves log‑linearly with the number of training episodes when evaluated in‑domain, while the improvement is dramatically slower for out‑of‑domain scenarios. Concretely, the paper extrapolates from empirical trends that reaching 95% step‑wise accuracy for low‑level instructions in‑domain would require approximately 1 million episodes; for high‑level instructions, where each task consists of multiple steps, achieving a 95% episode completion rate (assuming a 5‑step task) would demand roughly 2 million episodes. In contrast, the corresponding requirements for out‑of‑domain performance are estimated at 10 million episodes for low‑level instructions and 150 million episodes for high‑level multi‑step tasks, representing a one to two orders of magnitude increase in data needs. This scaling discrepancy suggests that while fine‑tuning alone can eventually produce robust in‑domain agents, the data cost to reach similar out‑of‑domain proficiency, especially for high‑level tasks that involve reasoning and decomposition, is likely infeasible, and additional approaches beyond simple data accumulation will be necessary for generalization.

Conditions: The agents process a textual representation of the Android accessibility tree (a flat list of UI elements with coordinates, text, and type information) without using screenshots. The base model is PaLM‑2S, fine‑tuned with LoRA (rank 4 for ≤100 episodes, rank 64 for ≥1k episodes) on a mixture of high‑level and low‑level instruction tasks from the ANDROIDCONTROL training set (13,604 episodes). Evaluation uses step‑wise accuracy with a relaxed metric that accepts equivalent actions (e.g., a click within the target element’s bounding box). The in‑domain test split (IDD) is drawn from the same distribution as the training data; three out‑of‑domain splits test robustness to unseen applications (64 apps), unseen task instructions (90 apps), and unseen app categories (68 apps from 4 categories). The scaling analysis varies the number of training episodes from 5 to all 13,604 and fits log‑linear trend lines (R² > 0.95) to extrapolate future performance.

Evidence: Figure 5 and Tables 4–5 provide the grounding. In‑domain, step accuracy rises from 26.9% (LT‑5) to 70.8% (LT‑all‑r64) for high‑level instructions and from 55.7% to 83.2% for low‑level instructions. The log‑linear relationship is clear: each 10× increase in training data adds roughly 12‑16 percentage points. Out‑of‑domain, the slope is much flatter; for low‑level instructions, the average OOD accuracy across the three splits goes from 56.4% (LT‑10) to 78.5% (LT‑all‑r64), while for high‑level instructions it climbs only from 30.2% to 58.5%. Importantly, the gap between IDD and OOD performance widens as more data is used—e.g., for LT‑10k‑r64, the high‑level instruction gap is 11.9 pp (69.6% IDD vs 57.7% app‑unseen). The trendlines fitted to these data predict that even with millions of episodes, OOD high‑level accuracy will lag far behind in‑domain performance, motivating the conclusion that fine‑tuning may not scale well out‑of‑domain for complex tasks.

Mechanism: The paper does not provide a deep causal explanation for why OOD scaling is slower. It attributes the better OOD transfer of low‑level instructions to the fact that low‑level actions (click, scroll, type) inherently share more visual and structural similarity across different apps and tasks, while high‑level task decomposition requires app‑specific knowledge and a reasoning process that does not transfer as easily when the app domain changes. In essence, the model learns surface‑level patterns for grounding low‑level commands, but for high‑level instructions it must learn how to reason about the unique workflows and UI layouts of each app, a capacity that remains tightly coupled to the training distribution and thus scales poorly with generic data accumulation.

### Applicability context
summary: Enables practitioners to estimate the amount of human demonstration data needed to reach a target performance for UI control agents, showing that in‑domain robust accuracy on low‑level instructions would require ≈1M episodes and on 5‑step high‑level tasks ≈2M episodes, while out‑of‑domain the requirements skyrocket to 10M and 150M respectively, guiding resource allocation and highlighting the need for additional generalization techniques.
modality: text-based UI accessibility trees
scale: 15,283 episodes over 833 Android apps; PaLM‑2S backbone with LoRA (rank 4‑64)
constraint: tested only on Android mobile apps with textual UI tree representations; extrapolation assumes log‑linear trend continues beyond observed range; OOD splits limited to 64‑68 apps per split; fine‑tuning done with LoRA, not full fine‑tuning; the LLM operates without access to screenshots

### Extraction rationale
The author explicitly interprets cross‑split scaling curves as a consistent pattern and provides extrapolations based on log‑linear trends (R²>0.95) (declaration basis). No causal explanation for the slower OOD scaling is offered beyond the observation that high‑level tasks are less transferable, so r is set to null.

## Why It Matters

Claim type: trend. Confidence: medium. Enables practitioners to estimate the amount of human demonstration data needed to reach a target performance for UI control agents, showing that in‑domain robust accuracy on low‑level instructions would require ≈1M episodes and on 5‑step high‑level tasks ≈2M episodes, while out‑of‑domain the requirements skyrocket to 10M and 150M respectively, guiding resource allocation and highlighting the need for additional generalization techniques. Applies under: tested only on Android mobile apps with textual UI tree representations; extrapolation assumes log‑linear trend continues beyond observed range; OOD splits limited to 64‑68 apps per split; fine‑tuning done with LoRA, not full fine‑tuning; the LLM operates without access to screenshots

## Evidence

### Source
arXiv:2406.03679 | 2406.03679__b9e89c94 | group: t9 | section: abstract

### Verbatim quote
"we find that when tested in domain fine‑tuned models outperform zero and few‑shot baselines and scale in such a way that robust performance might feasibly be obtained simply by collecting more data. Out of domain, performance scales significantly more slowly and suggests that in particular for high‑level tasks, fine‑tuning on more data alone may be insufficient for achieving robust out‑of‑domain performance. [...] We predict that to achieve 95% accuracy for in‑domain low‑level tasks, 1M episodes would be required, while 2M episodes would be required to obtain 95% episode completion rates for 5‑step high‑level tasks. Out of domain, 10M and 150M episodes would be required, respectively."

### Classification
domain: agent_web_gui
arXiv: cs.HC
Wikipedia: Intelligent_agents
ACM CCS: Computing methodologies → Artificial intelligence → Distributed artificial intelligence → Intelligent agents
CLC: TP181

### Keywords
UI control agents, fine‑tuning, scaling laws, dataset, Android, out‑of‑domain generalization, low‑level instructions, high‑level instructions, screen representation, language models

```

---

# LLM 处理超长上下文的机制问题

**用户提问：** 长文本语言模型（Long-context LLMs）在处理超长输入时，会遇到哪些注意力机制或截断带来的结构性问题？

**检索关键词：** `long context window truncation attention bottleneck extrapolation`

## 回答

长文本语言模型在处理超长输入时，基于经验库可归纳出以下由注意力设计或截断策略引起的结构性问题：

1. **位置表示的长度外推困难**  
   自注意力本身对输入排列不敏感，必须依赖位置编码来识别词序与距离。若位置表示缺乏对训练长度之外序列的外推能力，输入超过训练时见过的长度后，模型可能难以可靠地表达和利用远距离位置信息。经验指出，固定正弦位置编码因能以线性关系表达相对位移，具备向更长序列外推的性质；学习式位置嵌入在已测长度内效果相近，但未体现这一外推优势。（O-5cbdee9f397d0157）

2. **全局、无约束注意力的计算成本随长度二次增长**  
   若每个位置都要与所有其他位置计算注意力或对齐分数，计算量会随序列长度呈二次增长。输入越长，计算与资源开销越高，从而限制了模型能实际处理的上下文规模。该问题既出现在源—目标全局软对齐中，也出现在 Transformer 对多人、时间等维度拼接形成的长序列中。（O-1395ff7dffbe87fa、O-b3874315c5959518、O-d28588ffc80ce47d）

3. **为节省计算而采用单调或受限注意力，会损失跨位置重访与重排序能力**  
   如果注意力焦点被约束为只能单向前移，模型不能回看先前位置。在需要长距离重排序或跨段引用的文本任务中，这会阻碍模型建立正确的句法与语义关系。无约束注意力虽更昂贵，但能够在任意时刻访问或重新访问任意源位置；单调约束则会构成结构性限制。（O-1395ff7dffbe87fa）

4. **固定长度滑动窗口／截断会丢失早期上下文**  
   当超长输入被截成固定窗口，并在后续处理时不断丢弃较早内容，模型会失去初始信息及其与后续内容的关系。经验中的长程自回归预测显示：固定窗口丢弃早期交互信息，会使后续预测更容易发生误差累积；保留并逐步扩展历史输入可缓解该问题。对应到长文本场景，截断可能使模型无法再利用文首定义、前提、实体关系或早期约束。（O-1e08f04198162f7f）

5. **截断或压缩为单一固定长度表示，会形成信息瓶颈**  
   若长输入必须先被压缩成单个固定长度向量，细粒度信息难以被按需保留和检索，且这一问题会随输入变长而加剧。经验表明，相比固定向量，按生成步骤对源序列各位置进行加权检索的注意力机制能显著减轻长句性能退化；因此，将超长文本粗暴截断或固定压缩，会带来结构性的信息损失。（O-b3874315c5959518）

6. **保留全部历史虽减少信息丢失，却进一步放大计算负担**  
   扩展输入长度能保存早期信息、降低长程误差，但编码器处理成本会随累积长度增长。因此，长上下文机制存在固有张力：**截断可控成本但损失早期信息；保留全量历史改善信息可达性却增加注意力计算开销。**（O-1e08f04198162f7f、O-d28588ffc80ce47d）

上述经验主要直接覆盖位置编码、注意力对齐、长程自回归序列预测和 Transformer 的计算扩展性；关于其他 Long-context LLM 专用机制，经验库未覆盖。

## 思维链 (Reasoning)

1. 自注意力具有排列不变性，因此长文本中位置关系依赖位置编码；O-5cbdee9f397d0157 明确指出正弦编码可外推到更长序列，故推导出“位置表示可能成为超长长度泛化瓶颈”。  
2. O-1395ff7dffbe87fa、O-b3874315c5959518 与 O-d28588ffc80ce47d 均指出全局对齐／注意力的成本随长度二次增长，故归纳出全局注意力的可扩展性问题。  
3. O-1395ff7dffbe87fa 说明单调注意力不能回访早期位置、难以支持长距离重排序，故归纳出受限注意力的结构性表达限制。  
4. O-1e08f04198162f7f 说明固定窗口会丢弃早期信息并加剧长程误差累积，故将其对应为长文本截断造成的上下文丢失问题。  
5. O-b3874315c5959518 说明固定长度向量会成为长序列的信息瓶颈，而按位置注意力检索可缓解，故归纳出固定压缩／截断导致细粒度信息不可检索的问题。  
6. 最后结合 O-1e08f04198162f7f 的“扩展历史增加计算成本”与 O-d28588ffc80ce47d 的二次注意力成本，得出“完整保留上下文与计算可承受性之间的权衡”。

## 召回的原始经验全貌 (5 条)

### 经验 ID: `O-5cbdee9f397d0157`

```markdown
---
id: O-5cbdee9f397d0157
created_at: '2026-08-16T15:52:57Z'
summary: '[relation] Positional encodings are mandatory for non‑recurrent sequence
  models. Fixed sinusoidal encodings and learned positional embeddings yield equivalent
  translation quality, but sinusoidal encodings possess the additional property of
  allowing linear extrapolation to longer sequences, making them preferable when sequence
  length generalisation is desired.'
memory_type: semantic
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l2
related_observations:
- id: O-036453ccd61563a1
  relation: complements
  reason: 'Both extracted from arXiv:1706.03762: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:57:49Z'
---

## Observation

### Declaration
Positional encodings are mandatory for non‑recurrent sequence models. Fixed sinusoidal encodings and learned positional embeddings yield equivalent translation quality, but sinusoidal encodings possess the additional property of allowing linear extrapolation to longer sequences, making them preferable when sequence length generalisation is desired.

### Narrative
A core challenge in non‑recurrent sequence models is encoding positional information, because self‑attention by itself is permutation‑invariant. The Transformer solves this by adding explicit positional encodings to the input embeddings before feeding them into the encoder and decoder stacks. The paper experiments with two approaches: fixed sinusoidal encodings where each dimension follows a sine or cosine of different wavelengths, and learned positional embeddings that are trained as part of the model parameters. Both methods yield nearly identical performance on the English‑German translation task, as shown in Table 3 row (E) where the sinusoidal version achieves a development BLEU of 25.8 and the learned version gives 25.7. Despite this tie in accuracy, the authors advocate for the sinusoidal scheme because of its potential to extrapolate to longer sequences: for any fixed offset k, the representation of position pos+k can be expressed as a linear function of the representation at pos, a property that may help the model generalise to sequence lengths unseen during training. This inductive bias is grounded in the algebraic structure of sinusoids, and it becomes practically important in applications where inference‑time inputs exceed the maximum training length — for instance, in parsing or very long sentence translation. The experience teaches that position information is strictly necessary for a recurrent‑free architecture, but it does not have to be learned from data; a simple deterministic encoding with desirable extrapolation properties can perform equally well. Moreover, the fact that learned embeddings do not offer an advantage suggests that position is a relatively low‑complexity feature that does not require the model to expend capacity on learning it. However, the sinusoidal pattern is only one possible fixed encoding; the key insight is that the choice between fixed and learned can be made based on the desired ability to extrapolate, without sacrificing quality. In the context of large language models that follow, many variants have adopted learned positions, but the sinusoidal embedding paved the way and remains a competitive option, especially when computational efficiency or sequence length flexibility is paramount. The recommendation is therefore: when building a self‑attention model, include positional encodings; using fixed sinusoids is safe, effective, and inherently supports length extrapolation, whereas learned embeddings offer no measurable quality benefit in the tested regime.

### Causal explanation (r)
Sinusoidal encodings are chosen because they can be linearly combined to represent relative positions, enabling the model to generalise beyond the maximum training length.

confidence (μ_r): medium
depth: deep
depth rationale: The mechanism (linear representability of relative positions via trigonometric identities) is a concrete mathematical property that explains extrapolation ability.

### Applicability context
summary: Establishes that sinusoidal and learned positional encodings perform nearly identically (BLEU 25.8 vs 25.7), but sinusoids enable length extrapolation, guiding architectural choice for self‑attention models.
modality: text (token embeddings)
scale: WMT EN‑DE newstest2013 dev set; base Transformer with 6 layers, d_model=512
constraint: the equivalence holds for the tested sequence lengths (up to a few hundred tokens); learned embeddings may be advantageous for very different data distributions or extremely long sequences

### Extraction rationale
The claim is directly supported by an ablation experiment showing nearly identical BLEU, and the choice rationale is explicitly stated in Section 3.5.

## Why It Matters

Claim type: relation. Confidence: high. Establishes that sinusoidal and learned positional encodings perform nearly identically (BLEU 25.8 vs 25.7), but sinusoids enable length extrapolation, guiding architectural choice for self‑attention models. Applies under: the equivalence holds for the tested sequence lengths (up to a few hundred tokens); learned embeddings may be advantageous for very different data distributions or extremely long sequences

## Evidence

### Source
arXiv:1706.03762 | 1706.03762__574207a1 | group: t4 | section: method

### Verbatim quote
"We also experimented with using learned positional embeddings instead, and found that the two versions produced nearly identical results (see Table 3 row (E)). We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training."

### Classification
domain: agent_general
arXiv: cs.CL
Wikipedia: Natural_language_processing
ACM CCS: Computing methodologies → Artificial intelligence → Natural language processing → Machine translation
CLC: TP391.1

### Keywords
positional encoding, sinusoidal, learned embeddings, sequence length extrapolation, transformer

```

### 经验 ID: `O-1e08f04198162f7f`

```markdown
---
id: O-1e08f04198162f7f
created_at: '2026-08-16T15:52:57Z'
summary: '[trend] Progressively increasing the input sequence length during autoregressive
  inference reduces error accumulation in long-horizon motion generation compared
  to using a fixed-length input window, because early interaction information is preserved
  throughout the prediction chain.'
memory_type: semantic
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l2
related_observations:
- id: O-525a60c4924a6938
  relation: complements
  reason: 'Both extracted from arXiv:2111.12073: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:10Z'
- id: O-993b7efceb86bc28
  relation: complements
  reason: 'Both extracted from arXiv:2111.12073: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:10Z'
---

## Observation

### Declaration
Progressively increasing the input sequence length during autoregressive inference reduces error accumulation in long-horizon motion generation compared to using a fixed-length input window, because early interaction information is preserved throughout the prediction chain.

### Narrative
When autoregressive models are used for long-horizon sequence prediction, a common failure mode is error accumulation: small inaccuracies at early predicted steps compound into large deviations at later steps because the model receives its own noisy predictions as input. In the context of multi-person 3D motion prediction, this problem is amplified because early errors in predicting one person's trajectory can cascade into incorrect interaction dynamics with other persons. The Multi-Range Transformers paper introduces and validates a simple yet effective strategy to mitigate this accumulation: during training and inference, progressively increase the length of the input sequence fed to the encoders as prediction proceeds, rather than using a fixed window that slides along the generated sequence. Specifically, the model receives 1 second of history (15 frames) and predicts the next second; then, instead of taking only the last second of prediction as new input for the next step, the model sees all previously observed and predicted motion concatenated – first 1 second, then 2 seconds, then 3 seconds, and so on. This 'progressive input' strategy contrasts with the standard fixed-length approach where the model always receives a window of the most recent 1 second of motion. An ablation on CMU-Mocap (Table 6) demonstrates the impact: at a 3-second prediction horizon, the progressive strategy yields an MPJPE of 2.18 (in 0.1 m) compared to 2.91 for the fixed-length baseline, a 25% reduction in error, while at 1 second both methods are identical (0.96). The improvement emerges exclusively at longer horizons (2 seconds: 1.57 vs 1.91), confirming that the strategy specifically addresses error accumulation. The mechanism, as explained by the authors, is that maintaining early motion as input preserves crucial interactive information – the initial spatial relationships and relative trajectories that define social grouping and intent – which would otherwise be lost when old frames are discarded by a sliding window. In the multi-range architecture, this preserved information benefits both the local-range encoder (which uses DCT on each person's full motion history to enforce smoothness) and the global-range encoder (which performs self-attention across all persons and times, so longer context strengthens interaction modeling). The transformer's permutation invariance and lack of recurrence make it naturally amenable to variable-length input, removing any architectural barrier to progressive lengthening. The strategy was tested under the following specific conditions: the model predicts 15 frames per autoregressive step from an offset Δx representation; training includes random-length sampling to prevent length bias; the decoder always uses only the most recent pose as query (so the increased input only enriches the encoder features); and the total prediction extends to 3 seconds (45 frames), with error measured in world-coordinate MPJPE. The evidence is limited to CMU-Mocap data but the principle is generalizable to any autoregressive transformer for temporal sequences where long-term dependencies are important. The strategy introduces a computational cost trade-off: encoder processing time grows with accumulated sequence length, but this is acceptable for batch inference on moderate-length sequences. The authors reported this finding with moderate confidence, as it was shown via a single controlled ablation.

### Causal explanation (r)
Maintaining early motion frames as input preserves initial interaction patterns between individuals, preventing information loss that occurs with fixed-length sliding windows and thereby reducing cascading prediction errors.

confidence (μ_r): medium
depth: deep
depth rationale: Identifies a specific causal chain: discarding early frames loses interaction information → missing context causes larger prediction errors → errors accumulate over autoregressive steps. This goes beyond merely stating that progressive input improves performance.

### Applicability context
summary: Improves long-term 3D motion prediction accuracy by 25% at 3 seconds by progressively lengthening encoder input rather than using a fixed window, effectively preserving early trajectory information that prevents cascade errors.
modality: 3D skeleton joint offsets
scale: 1-3 second prediction, 15 frames per step, single dataset ablation
constraint: assumes a transformer encoder that accepts variable-length input without recurrence; tested only on smooth CMU-Mocap motions with up to 3-second horizon; computational cost grows with sequence length, limiting applicability to very long horizons

### Extraction rationale
The declaration is supported by a controlled ablation in Table 6 showing identical 1-second error but diverging errors at longer horizons. The causal explanation (r) is directly provided in the method section as the rationale for the progressive strategy design.

## Why It Matters

Claim type: trend. Confidence: medium. Improves long-term 3D motion prediction accuracy by 25% at 3 seconds by progressively lengthening encoder input rather than using a fixed window, effectively preserving early trajectory information that prevents cascade errors. Applies under: assumes a transformer encoder that accepts variable-length input without recurrence; tested only on smooth CMU-Mocap motions with up to 3-second horizon; computational cost grows with sequence length, limiting applicability to very long horizons

## Evidence

### Source
arXiv:2111.12073 | 2111.12073__ce33ea55 | group: t1 | section: method

### Verbatim quote
The advantage of such design is that when predicting longer motions, we still maintain the early motions as inputs to the model, instead of using a fixed length to predict each of the future time steps [45,44], which may cause the loss of early interactive information. Through the experiment, we find this strategy could largely reduce the error accumulation.

### Classification
domain: agent_domain_app
arXiv: cs.CV
Wikipedia: Computer_vision
ACM CCS: Computing methodologies → Artificial intelligence → Computer vision → Activity recognition and understanding
CLC: TP391.4

### Keywords
autoregressive prediction, error accumulation, progressive input, long-term motion generation, transformer inference

```

### 经验 ID: `O-b3874315c5959518`

```markdown
---
id: O-b3874315c5959518
created_at: '2026-08-16T15:52:57Z'
summary: '[coarse] Proposes an attention‑based encoder–decoder architecture for neural
  machine translation that jointly learns to align and translate, achieving BLEU scores
  comparable to the conventional phrase‑based Moses system on English‑to‑French (36.15
  vs. 35.63 on sentences with no unknown words) and dramatically outperforming a basic
  RNN encoder‑decoder, particularly on long sentences.'
memory_type: procedural
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l1
related_observations:
- id: O-1395ff7dffbe87fa
  relation: complements
  reason: 'Both extracted from arXiv:1409.0473: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:57:45Z'
- id: O-16626b8979dc08a1
  relation: complements
  reason: 'Both extracted from arXiv:1409.0473: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:57:45Z'
- id: O-6418e272cd7ca879
  relation: complements
  reason: 'Both extracted from arXiv:1409.0473: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:57:45Z'
---

## Observation

### Narrative
Neural machine translation (NMT) using an encoder–decoder architecture had recently shown promise, but the practice of compressing the entire source sentence into a single fixed‑length context vector created a bottleneck that hurt performance, especially on long sentences. The central practice introduced in this paper is an extension of that encoder–decoder framework that lets the model learn to align and translate jointly. Instead of a solitary fixed vector, the encoder produces a sequence of annotations—one for each source word—and the decoder, at each step of generating a target word, computes a distinct context vector as a weighted sum of those annotations. The weights (called attention weights and denoted α_ij) are produced by a small feed‑forward alignment model that scores how well the current decoder state matches each source annotation. The entire system—bidirectional encoder, attention‑driven decoder, and alignment model—is trained end‑to‑end on parallel corpora to maximise the conditional probability of the target sentence. This soft‑alignment mechanism lets the decoder “search” through the source sentence without committing to a hard segmentation, and it means the encoder no longer has to cram all information into one vector; information can be distributed across the annotation sequence and selectively retrieved. The practice was evaluated on the WMT’14 English‑to‑French translation task. The training corpus, after data selection, contained 348 million words from Europarl, news commentary, United Nations proceedings, and two crawled corpora. Both source and target vocabularies were limited to the 30 000 most frequent words, with all other words mapped to a special [UNK] token. The encoder used a bidirectional recurrent neural network (BiRNN) with gated hidden units; forward and backward RNNs each had 1000 hidden units, and their hidden states were concatenated to form the annotations. The decoder also used 1000 gated hidden units, and the alignment model was a single‑layer network with 1000 hidden units that took the previous decoder state and an annotation as input. The output probability was computed with a maxout layer (500 units). Two main variants were trained, one limited to sentences of up to 30 words (RNNsearch‑30) and another limited to 50 words (RNNsearch‑50), with a third variant (RNNsearch‑50*) trained longer until the development set log‑probability stopped improving. The baseline was a plain RNN Encoder–Decoder (RNNencdec) of the same size. Training used minibatch stochastic gradient descent (80 sentences per batch) with Adadelta (ε=10⁻⁶, ρ=0.95) and gradient norm clipping at 1. Each model required roughly five days on a GPU (TITAN BLACK or Quadro K‑6000). Translation was obtained by beam search that approximately maximises the conditional probability. The key outcome was that the attention‑based model dramatically outperformed the basic encoder–decoder. On the full test set of 3003 sentences, RNNsearch‑50 achieved a BLEU score of 26.75, versus 17.82 for RNNencdec‑50 (13.93 for RNNencdec‑30). When sentences containing unknown words were excluded, RNNsearch‑50* reached 36.15 BLEU, slightly exceeding the phrase‑based Moses system (35.63). Equally important, the BLEU of RNNencdec dropped rapidly once sentences exceeded 30 words, while the attention‑based models showed virtually no degradation up to 60+ words. Qualitative inspection of the attention weights revealed largely monotonic but also grammatically sensible non‑monotonic alignments, such as correctly handling adjective‑noun order differences and article selection without needing explicit NULL alignments. The practice is limited by the 30 000‑word vocabulary—unknown words are merely replaced by [UNK]—and by the O(Tx·Ty) cost of the alignment computation, though for typical translation sentences this is manageable. The approach was tested only on English‑to‑French; its effectiveness on other language pairs, particularly those requiring long‑distance reordering, remains an open question, although the soft‑alignment mechanism is designed to handle such cases.

### Practice environment
Training corpus: WMT’14 English–French parallel data (Europarl, news commentary, UN, crawled corpora) filtered to 348M words with data selection. Vocabulary: 30K most frequent words per language, remainder mapped to [UNK]. Validation: news‑test‑2012+2013 concatenated. Test: news‑test‑2014 (3003 sentences). Models: RNNsearch (attention) and RNNencdec (baseline), both with 1000 hidden units, 620‑dim word embeddings, 500‑unit maxout output layer. Training: minibatch size 80, Adadelta (ε=10⁻⁶, ρ=0.95), gradient norm clipped to max 1, trained ~5 days on NVIDIA TITAN BLACK or Quadro K‑6000. Decoding: beam search. Metrics: BLEU (all sentences and subset without [UNK]), BLEU by sentence length.

### Practice trace
1. Action: Designed an encoder–decoder with a bidirectional RNN encoder and a decoder that computes a separate context vector at each step as a weighted sum of source annotations; the weights are learned by a jointly trained feed‑forward alignment model
   Feedback: The attention model (RNNsearch‑50*) reached BLEU of 28.45 on all sentences and 36.15 on sentences without unknown words, compared to 33.30/35.63 for the phrase‑based Moses system, while the basic encoder–decoder (RNNencdec‑50) achieved only 17.82/26.71. The attention model showed no BLEU degradation on sentences up to 60+ words, unlike the basic model whose performance collapsed beyond 30 words.

### Task context
modality: text (parallel English‑French sentence pairs)
scale: WMT’14 English–French, 348M words training corpus, 30K vocabulary, test set of 3003 sentences
constraint: vocabulary limited to 30K words, all other words become [UNK]; alignment computation scales O(Tx·Ty) per sentence; only tested on English‑to‑French; beam search used for decoding may not find global optimum

### Extraction rationale
Practice description and headline result drawn from the abstract. The action‑feedback pair summarises the core contribution and its outcome as stated in the abstract and detailed in the experimental results (Table 1, Fig. 2).

## Why It Matters

Granularity: coarse. Scale: WMT’14 English–French, 348M words training corpus, 30K vocabulary, test set of 3003 sentences Applies under: vocabulary limited to 30K words, all other words become [UNK]; alignment computation scales O(Tx·Ty) per sentence; only tested on English‑to‑French; beam search used for decoding may not find global optimum

## Evidence

### Source
arXiv:1409.0473 | 1409.0473__0196c5f3 | group: t4 | section: abstract

### Verbatim quote
"In this paper, we conjecture that the use of a fixed-length vector is a bottleneck in improving the performance of this basic encoder–decoder architecture, and propose to extend this by allowing a model to automatically (soft-)search for parts of a source sentence that are relevant to predicting a target word, without having to form these parts as a hard segment explicitly. With this new approach, we achieve a translation performance comparable to the existing state-of-the-art phrase-based system on the task of English-to-French translation. Furthermore, qualitative analysis reveals that the (soft-)alignments found by the model agree well with our intuition."

### Classification
domain: agent_domain_app
arXiv: cs.CL
Wikipedia: Machine_translation

### Keywords
neural machine translation, attention mechanism, encoder-decoder, bidirectional RNN, soft alignment, sequence-to-sequence learning, recurrent neural network, English-to-French translation, beam search, BLEU

```

### 经验 ID: `O-1395ff7dffbe87fa`

```markdown
---
id: O-1395ff7dffbe87fa
created_at: '2026-08-16T15:52:57Z'
summary: '[conditional] For machine translation tasks that require substantial word
  reordering (e.g., English-to-German), an attention mechanism that can attend to
  any source position without a monotonic constraint is essential, because monotonic
  alignment cannot revisit earlier tokens and thus fails to capture long-distance
  syntactic divergences.'
memory_type: semantic
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l2
related_observations:
- id: O-992c41b41be2989e
  relation: complements
  reason: 'Both extracted from arXiv:1409.0473: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:57:44Z'
- id: O-b3874315c5959518
  relation: complements
  reason: 'Both extracted from arXiv:1409.0473: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:57:45Z'
- id: O-957cacd39cdca53e
  relation: complements
  reason: Both classified under agent_general, from arXiv:1409.0473 and arXiv:1704.08045
    respectively — different papers addressing the same problem area.
  linked_at: '2026-08-16T15:59:16Z'
---

## Observation

### Declaration
For machine translation tasks that require substantial word reordering (e.g., English-to-German), an attention mechanism that can attend to any source position without a monotonic constraint is essential, because monotonic alignment cannot revisit earlier tokens and thus fails to capture long-distance syntactic divergences.

### Narrative
An important design choice in attention-based models is whether the attention mechanism imposes a monotonic constraint on the movement of the focus, i.e., requiring that the attended positions progress only forward (or strictly increasing) across the output sequence. This paper contrasts its unconstrained soft-attention with the monotonic attention used in Graves (2013) for handwriting synthesis. The authors argue that for machine translation, a monotonic constraint is a severe limitation because translation often requires long-distance reordering, where a word early in the source sentence may need to be translated late in the target sentence, or vice versa. 

In the RNNsearch model, the attention weights α_ij can assign high probability to any source position j for any target position i, without any sequential restriction. During decoding, the model can freely shift its focus backward and forward across the source sentence. This flexibility is achieved by computing the compatibility score e_ij between the decoder's previous hidden state s_{i-1} and each encoder annotation h_j independently, then applying a softmax over all source positions. The result is that the attention can “jump” to any part of the source sentence, including revisiting earlier positions, which is essential when the target language word order differs substantially from the source language. 

The paper gives the specific example of English-to-German translation, where verbs often appear at the end of the clause and require reordering of the source words. With a monotonic attention, the decoder could only attend to a moving window, forcing it to commit to a forward-only alignment; this would make it extremely difficult to generate the correct German word order. The authors explicitly state that the monotonic approach is “a severe limitation, as (long-distance) reordering is often needed to generate a grammatically correct translation (for instance, English-to-German).” 

The RNNsearch model's qualitative alignments (on English-French) already demonstrate non-monotonic behavior, such as aligning “zone” with distant “Area” while later attending back to “Economic” and “European.” Although English-French reordering is mostly local, the model learns to perform these small jumps as needed. The need for non-monotonicity would be even more critical for language pairs with fundamentally different syntactic structures. 

This insight is not an experimental result but a conceptual comparison with related work, grounded in the nature of translation. The authors do not conduct an ablation study removing the non-monotonic capability because the model inherently allows any alignment. However, the reasoning is sound: any hard restriction on the order of attention would prevent the model from learning the necessary reorderings. The success of RNNsearch on long and complex sentences can be partly attributed to this freedom. 

One caveat is that unconstrained attention incurs a computational cost quadratic in sentence length, because the compatibility score must be computed for every (i,j) pair. For typical translation sentences (15–40 words), this is manageable, but it may limit scalability to very long sequences. The authors acknowledge this as a potential limitation for other tasks, but for translation, they consider it acceptable. Additionally, the statement about English-to-German is hypothetical and not empirically verified in this paper; the actual experiments are only on English-to-French. Nevertheless, the conceptual claim that non-monotonic attention is necessary for reordering-heavy language pairs is a well-founded inductive assertion.

### Causal explanation (r)
Monotonic attention forces the focus to move only forward, preventing the decoder from revisiting earlier source words that must be placed later in the target translation, thereby breaking the syntactic structure required for languages with different word orders.

confidence (μ_r): high
depth: deep
depth rationale: The explanation identifies the specific mechanism of enforced forward-only movement that blocks reordering, not merely stating that it doesn't work for reordering.

### Applicability context
summary: Argues that unconstrained (non-monotonic) attention is necessary for machine translation involving long-distance reorderings; without it, models like monotonic attention would fail on language pairs such as English-to-German.
modality: text
scale: conceptual claim; no specific scale beyond translation task assumption
constraint: the claim is based on logical reasoning and comparison with a monotonic model (Graves 2013), not empirically tested on a reordering-heavy language pair; computational cost of unconstrained attention grows quadratically with sentence length

### Extraction rationale
The declaration is an author-stated conceptual limitation drawn from the related work comparison. The causal explanation (r) is directly traceable to the same passage where the author explains why a monotonic constraint fails for reordering.

## Why It Matters

Claim type: conditional. Confidence: medium. Argues that unconstrained (non-monotonic) attention is necessary for machine translation involving long-distance reorderings; without it, models like monotonic attention would fail on language pairs such as English-to-German. Applies under: the claim is based on logical reasoning and comparison with a monotonic model (Graves 2013), not empirically tested on a reordering-heavy language pair; computational cost of unconstrained attention grows quadratically with sentence length

## Evidence

### Source
arXiv:1409.0473 | 1409.0473__0196c5f3 | group: t4 | section: related work

### Verbatim quote
The main difference from our approach is that, in (Graves, 2013), the modes of the weights of the annotations only move in one direction. In the context of machine translation, this is a severe limitation, as (long-distance) reordering is often needed to generate a grammatically correct translation (for instance, English-to-German).

### Classification
domain: agent_general
arXiv: cs.CL
Wikipedia: Machine_translation
ACM CCS: Computing methodologies → Artificial intelligence → Natural language processing → Machine translation
CLC: TP391.1

### Keywords
neural machine translation, attention mechanism, sequence-to-sequence, encoder-decoder, soft alignment, bidirectional RNN, long sentence translation, joint learning, alignment model, WMT

```

### 经验 ID: `O-d28588ffc80ce47d`

```markdown
---
id: O-d28588ffc80ce47d
created_at: '2026-08-16T15:52:57Z'
summary: '[medium] Decomposes multi‑person pose forecasting into body part interactions
  via a Transformer with body partitioning, social body part attention, and trajectory‑aware
  relative position encoding, achieving short‑term overall JPE of 107 mm on CMU‑Mocap
  (13–16% better than the leading multi‑person baseline) and long‑term JPE of 329 mm
  at 3.0 s (6.5% improvement), while generating more realistic and temporally coherent
  motions.'
memory_type: procedural
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l1
related_observations:
- id: O-1019fa8f0d11423e
  relation: complements
  reason: 'Both extracted from arXiv:2303.05095: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:25Z'
- id: O-5a4bc48490d51994
  relation: complements
  reason: 'Both extracted from arXiv:2303.05095: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:25Z'
- id: O-6a5ca31925879e68
  relation: complements
  reason: 'Both extracted from arXiv:2303.05095: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:25Z'
---

## Observation

### Narrative
Forecasting the future 3D poses of multiple interacting persons simultaneously is a challenging problem because people influence each other through fine‑grained body part interactions, not merely through whole‑body proximity. Existing multi‑person motion prediction methods, such as MRT, model social interaction in the temporal domain but ignore skeletal body part dynamics, leading to unrealistic poses and limited accuracy, especially over long horizons. The paper presents TBIFormer, a Transformer‑based framework that tackles this by explicitly modeling body part interactions across individuals. The approach first converts observed pose sequences into displacement trajectories to capture dynamics, then uses a Temporal Body Partition Module (TBPM) to split each skeleton into five semantic body parts (left/right arms, left/right legs, torso), perform 2D convolution with a kernel of size 10×1 to extract spatiotemporal features, and concatenate the per‑body‑part features across all persons into a single Multi‑Person Body‑Part (MPBP) sequence. This sequence is processed by a stack of three TBIFormer blocks. Inside each block, a Social Body Interaction Multi‑Head Self‑Attention (SBI‑MSA) module computes attention among all body parts across time and persons, effectively learning inter‑ and intra‑individual interaction dependencies. SBI‑MSA incorporates a novel Trajectory‑Aware Relative Position Encoding (TRPE) bias, which replaces conventional Euclidean‑distance‑based spatial encodings. TRPE is computed by a Shifted Local Dynamic Time Warping (SL‑DTW) algorithm that slides a window over root trajectories of different persons, measures similarity with a differentiable soft‑DTW cost, and maps the similarity via a piecewise logarithmic function into relative position indices. This provides discriminative spatial cues that capture actual interaction patterns—e.g., two persons moving in the same direction—without expensive body orientation calculations. Temporal positional encoding and a learnable person‑identity encoding are added to the MPBP sequence to preserve timing and identity information. Finally, a Transformer decoder takes a pooled representation of the last observed sub‑sequence as queries and attends to the output of the TBIFormer blocks, followed by fully connected layers and an Inverse Discrete Cosine Transform to regress future joint positions. The model is trained solely on the CMU‑Mocap dataset augmented with UMPM (3‑person scenes), with an ADAM optimizer (lr=0.0003, batch size 32, dropout 0.2) for 50 epochs on an NVIDIA RTX 3090 GPU. It is evaluated across unseen scenarios with varying crowd densities: MuPoTS‑3D (2–3 persons), Mix1 (6 persons), Mix2 (10 persons). For short‑term prediction, 50 frames (2.0 s) of history are used to forecast 25 frames (1.0 s); for long‑term, 15 frames (1.0 s) predict 45 frames (3.0 s). Performance is assessed by Joint Position Error (JPE, global pose error including trajectory), Aligned Position Error (APE, pose error after removing global translation), and Final Displacement Error (FDE, root position error at the last frame). Compared with the single‑person methods HRI and MSR and the multi‑person method MRT, TBIFormer consistently achieves state‑of‑the‑art results. On CMU‑Mocap (UMPM) short‑term overall JPE is 107 mm (MRT: 114), APE 76 mm (MRT: 101), FDE 74 mm (MRT: 91). On the challenging Mix2 (10 persons) dataset, JPE drops to 117 mm (MRT: 126). For long‑term prediction, the advantage becomes more pronounced: at 3.0 s, JPE is 329 mm vs. MRT‘s 352 mm, and APE is 152 mm vs. 218 mm, showing that body part attention with trajectory‑aware position encoding effectively prevents pose degradation. Ablation studies reveal that removing TBPM, TRPE, or SBI‑MSA each cause significant performance drops, and replacing TRPE with Euclidean‑based encoding yields sub‑optimal results. Qualitative visualizations show TBIFormer generates smoother, more lifelike motions that stay close to ground truth, whereas baselines tend to converge to static or distorted poses. A limitation is the quadratic attention cost w.r.t. the length of the MPBP sequence, which grows with the number of persons, making real‑time inference for large crowds challenging. The method also assumes a common 15‑joint skeleton and clean input data, and it has not been validated in heavily occluded or noisy real‑world scenes.

### Practice environment
Training data: CMU‑Mocap augmented with UMPM (3‑person sequences, displacement representation). Evaluation: MuPoTS‑3D (2–3 persons), Mix1 (6 persons, mixture of CMU‑Mocap, UMPM, 3DPW, MuPoTS‑3D), Mix2 (10 persons). Baselines: HRI (attention network), MSR (GCN), MRT (multi‑range Transformer). Metrics: JPE, APE, FDE (mm). Hardware: NVIDIA RTX 3090. Hyperparameters: ADAM, lr=0.0003, batch size 32, dropout 0.2, 50 epochs; TBPM conv 10×1 stride 1; TRPE α=1, β=9, γ=η=2000; 3 TBIFormer blocks, hidden dim 1024, d_z=64, 8 heads; Transformer decoder with same dims. Input horizons: 50 frames (2.0 s) short‑term, 15 frames (1.0 s) long‑term; output horizons: 25 frames (1.0 s), 45 frames (3.0 s).

### Practice trace
1. Action: Constructed TBPM to split skeletons into 5 body parts, apply 2D convolution, and concatenate all persons' parts into an MPBP sequence, then fed to SBI‑MSA with TRPE bias based on SL‑DTW trajectory similarity.
   Feedback: Overall short‑term JPE on CMU‑Mocap (UMPM) reached 107 mm, outperforming MRT (114 mm), HRI (129 mm), and MSR (143 mm). APE dropped from MRT‘s 101 mm to 76 mm, and FDE from 91 mm to 74 mm.
2. Action: Extended evaluation to unseen crowd densities: tested the same TBIFormer model on MuPoTS‑3D, Mix1, Mix2 without retraining.
   Feedback: On Mix2 (10 persons), TBIFormer achieved JPE 117 mm vs. MRT 126 mm; on MuPoTS‑3D JPE 195 mm vs. MRT 217 mm. The architecture generalized well to higher person counts.
3. Action: Performed long‑term prediction (up to 3.0 s) on CMU‑Mocap using only 1.0 s of history.
   Feedback: At 3.0 s, JPE was 329 mm (MRT: 352); APE 152 mm (MRT: 218); FDE 273 mm (MRT: 315). The benefits of body part interaction modeling grew with the forecast horizon.

### Task context
modality: 3D human skeleton displacement trajectories (joint positions in world coordinates)
scale: 2–10 persons; trained on CMU‑Mocap+UMPM (3 persons); tested on 4 datasets with up to 10 persons; up to 45‑frame (3.0 s) prediction horizon
constraint: quadratic attention cost w.r.t. (persons × body parts × time) limits scalability to very large crowds; requires a consistent 15‑joint skeleton across persons; trained on clean motion capture data and may degrade under severe occlusion or noisy joint estimates; relies on root trajectory similarity and may miss static positional cues

### Extraction rationale
Practice description drawn from the introduction and method overview (Sections 1, 3). The action sequence follows the paper’s procedural description; feedback values are from Tables 1 and 2 and the quantitative results narrative.

## Why It Matters

Granularity: medium. Scale: 2–10 persons; trained on CMU‑Mocap+UMPM (3 persons); tested on 4 datasets with up to 10 persons; up to 45‑frame (3.0 s) prediction horizon Applies under: quadratic attention cost w.r.t. (persons × body parts × time) limits scalability to very large crowds; requires a consistent 15‑joint skeleton across persons; trained on clean motion capture data and may degrade under severe occlusion or noisy joint estimates; relies on root trajectory similarity and may miss static positional cues

## Evidence

### Source
arXiv:2303.05095 | 2303.05095__4ca5d25e | group: t1 | section: introduction

### Verbatim quote
"Our method outperforms the state-of-the-art approaches for both short- and long-term predictions by a large margin, with 14.4% ∼ 16.5% accuracy improvement for the short-term (≤1.0 s) and 6.5% ∼ 18.2% accuracy improvement for the long-term (1.0 s ∼ 3.0 s)."

### Classification
domain: agent_domain_app
arXiv: cs.CV
Wikipedia: Computer_vision

### Keywords
multi-person pose forecasting, human motion prediction, Transformer, body part interaction, trajectory-aware encoding, social interaction, 3D human skeleton, attention mechanism, motion capture, relative position encoding

```

---

# Coding Agent 的执行失败迭代修复

**用户提问：** 如何让 Coding Agent 在遇到执行报错时能够有效地进行迭代修复，而不是陷入死循环？

**检索关键词：** `agent execution failure recovery root cause repair debugging loop`

## 回答

要让 Coding Agent 在执行报错后有效迭代修复、避免死循环，应将“反复尝试修补”改造成**有诊断、有状态、有预算、有验证门槛的闭环**：

1. **完整记录失败轨迹，并先定位最早的根因**  
   记录每一步的上下文、计划、工具调用、代码修改、静态分析与测试输出。不要对所有后续报错逐一打补丁，而应找出“若修正该步骤即可避免最终失败”的**最早关键错误**；后续错误往往只是该偏差的连锁结果。可按记忆、反思、规划、行动等模块，以及错误类型进行结构化标注。  

2. **只针对根因给出具体、可执行的修复反馈**  
   反馈应至少包含：关键步骤、责任模块、错误类型、证据、根因、修改建议及可能的级联影响。避免仅让 Agent “再想想”或无目标地自我反思；这类无引导修订容易在错误状态上持续迭代。  

3. **从关键步骤恢复执行，而非每次从头重跑**  
   在可回放的环境中保存状态，并从定位到的关键步骤重新执行。这样既保留正确的前序工作，也避免每次重新引入相同或新的偏差。若仍失败，则根据新的执行证据收紧和细化反馈，再进行下一次重跑。  

4. **设定严格的迭代上限与停止条件**  
   重跑次数必须有上限；经验中采用最多 **5 次**定向重跑。达到上限、关键步骤无法可靠定位、环境不可复现、或每轮失败模式没有实质变化时，应停止自动修复并输出诊断结果，而不是继续循环。  

5. **把测试/验证反馈控制为“少而关键”**  
   交互式修复时，优先反馈与当前问题直接相关、规模较小的失败测试、错误信息和栈追踪。不要把大量公共与隐藏测试的全部失败输出一次性塞入上下文：过长、未排序的反馈可能淹没真正有用的信号。隐藏或全量测试更适合作为候选修复后的最终验证。  

6. **每轮都运行确定性验证，并设置提交质量门**  
   将静态分析、lint、构建、相关测试作为循环内的确定性验证器；候选补丁至少应通过原始失败测试，再经过 CI 等更全面的验证。生产场景还可增加经过校准的质量判别和人工审查，而不应因一次测试通过就自动认定修复正确。  

7. **在进入循环前排除“不可修”或不应修的任务**  
   先检查失败是否可稳定复现，排除测试偶发性、任务已被人工修复、环境异常、缺失工具/验证器等情况。这些问题即使不断改代码也未必能解决，是造成无效循环的重要来源。  

可将整体流程固化为：

> 报错 → 复现与采集轨迹 → 根因定位 → 生成结构化定向反馈 → 从关键步骤恢复 → 静态分析/相关测试验证 →  
> 成功则进入全量验证与人工审核；失败则基于新证据细化反馈；达到次数或停止条件则终止并升级处理。

该方法的前提是：环境可回放、失败可复现，并且系统能够相对可靠地定位关键步骤与错误类别；若不具备这些条件，经验库未证明其能够稳定避免循环。

## 思维链 (Reasoning)

1. 经验指出，多步 Agent 失败时，修复最早的根因比修复全部表面错误、无引导自我修订或扩大采样更有效；并应从关键步骤重跑，而非盲目重启（O-989b7314c0bd7b8c）。  
2. 结构化诊断应包含关键步骤、模块、错误类型、证据、根因、修复指导和级联影响；失败后可细化反馈，但重跑次数限制为最多 5 次，因此可形成有界迭代而非无限循环（O-fea2f063ea43a2f1）。  
3. 交互式修复反馈应保持紧凑、相关；大规模测试失败输出会增大上下文并降低修复效果，而隐藏测试适合最终验证（O-866e1c7b6cf58af3）。  
4. 在工程修复中，静态分析、测试、CI 等确定性验证结果应反馈到 Agent 循环；原始失败测试、质量判断和 CI 可构成提交前质量门（O-22ad8bf1e650418f；O-350ca2d3d20d18e3）。  
5. 不可复现的失败、测试偶发性、已被修复的任务和缺失验证能力会使重复迭代无效，故应作为前置筛除或停止条件（O-350ca2d3d20d18e3）。

## 召回的原始经验全貌 (5 条)

### 经验 ID: `O-989b7314c0bd7b8c`

```markdown
---
id: O-989b7314c0bd7b8c
created_at: '2026-08-16T15:55:08Z'
summary: '[relation] For failed multi-step LLM-agent trajectories, identifying the
  earliest root-cause error and applying targeted feedback from that step generally
  improves error localization and task recovery more effectively than correcting all
  surface mistakes, unguided self-refinement, or broad rollout scaling under a matched
  compute budget.'
memory_type: semantic
scope: global
source:
  type: turn
  agent: paper-experience
  session_id: paper-experience-l2
related_observations:
- id: O-3b34b334b1e94c09
  relation: complements
  reason: 'Both extracted from arXiv:2509.25370: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:59:06Z'
- id: O-8fa714957ddd4e9b
  relation: complements
  reason: 'Both extracted from arXiv:2509.25370: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:59:07Z'
- id: O-b947b647abcdd477
  relation: complements
  reason: 'Both extracted from arXiv:2509.25370: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:59:08Z'
---

## Observation

### Declaration
For failed multi-step LLM-agent trajectories, identifying the earliest root-cause error and applying targeted feedback from that step generally improves error localization and task recovery more effectively than correcting all surface mistakes, unguided self-refinement, or broad rollout scaling under a matched compute budget.

### Narrative
When an LLM agent fails on a long or multi-step task, debugging should focus on isolating and correcting the minimal root-cause error rather than asking the agent to revise every mistake in the trajectory. A root-cause error is the earliest decision whose correction would prevent the final failure; later mistakes may be genuine errors, but they can be downstream symptoms created by the earlier deviation. This distinction is operationally important because a failed trajectory can contain many visible problems: an incorrect memory retrieval, a mistaken interpretation of feedback, an infeasible plan, an invalid action format, and repeated attempts to continue from the wrong state. Treating all of these as equally important produces noisy feedback and can cause the agent to modify behavior that was not responsible for the failure. A more useful debugging procedure first maps each decision step to interpretable error categories, then identifies the earliest critical step, and finally gives targeted, actionable feedback at that point. The proposed procedure analyzes four agent modules—memory, reflection, planning, and action—at every step using a modular taxonomy. It then asks a language model to identify the critical error from a global view of the trajectory, considering whether the error caused a cascade and whether correcting it would fundamentally redirect the rollout toward success. The revised agent re-executes the trajectory from the critical step rather than restarting blindly from the beginning. If the new rollout still fails, the feedback is refined and another re-rollout is attempted, with up to five attempts in the downstream experiments. This approach was evaluated on AgentErrorBench, a dataset of 200 expert-annotated failure trajectories from ALFWorld, WebShop, and GAIA. Compared with direct prompting, brute-force step substitution, and binary search, the debugging framework achieved 45.0% average step accuracy versus 28.0% for the strongest baseline and 24.3% average all-correct accuracy—requiring the exact step, module, and error type—versus 0.3%. The improvement was especially large on GAIA, where step accuracy reached 58.0% versus 30.0% and all-correct accuracy reached 38.0% versus 12.0%. The downstream effect was also observed across different agent backbones. On ALFWorld, success increased from 21 to 55 for GPT-4o-mini, from 48 to 74 for Qwen3-8B, and from 60 to 84 for Qwen3-Next-80B after targeted debugging. Across ALFWorld, GAIA, and WebShop, the method produced up to 26% relative improvement in task success while matching baseline token budgets. The result has boundaries: it depends on having a recordable trajectory, a sufficiently capable debugger, and a way to re-run the environment from the intervention point. It does not establish that every failure has a single root cause, and the authors note that the benchmark remains limited in scale and domain diversity. Nevertheless, for interdependent agent workflows, root-cause localization and targeted corrective feedback are more useful than unguided self-refinement or broad test-time sampling.

### Causal explanation (r)
Correcting a single root-cause mistake can redirect the trajectory toward success, while feedback grounded in the error taxonomy is targeted and forward-looking: it resolves the originating problem and shapes subsequent plans and actions.

confidence (μ_r): medium
depth: deep
depth rationale: The explanation specifies why root-cause intervention is effective: it redirects subsequent trajectory decisions and prevents the originating problem from generating further errors, adding a mechanism beyond the performance comparison.

### Applicability context
summary: Enables failed LLM agents to recover on embodied, web-navigation, and tool-use tasks by locating the earliest critical error and re-rolling out with targeted feedback, yielding up to 26% relative task-success improvement under matched token budgets.
modality: text-based trajectories containing memory, reflection, planning, action, and environment-feedback records
scale: 200 annotated trajectories across ALFWorld, WebShop, and GAIA; downstream tests with GPT-4o-mini, Qwen3-8B, and Qwen3-Next-80B; up to 5 re-rollouts
constraint: requires a replayable environment and reliable step/module/error localization; the debugging model used GPT-4.1 for detection, and the benchmark has limited domain diversity and no multimodal or safety-critical evaluation

### Extraction rationale
The declaration is supported by cross-environment detection results, downstream recovery experiments on ALFWorld, GAIA, and WebShop, comparisons with several refinement and scaling baselines, and an ablation explicitly favoring root-cause correction over surface-error correction. The causal explanation is directly stated in the method and results sections, where the authors describe how correcting the originating error changes later trajectory behavior.

## Why It Matters

Claim type: relation. Confidence: high. Enables failed LLM agents to recover on embodied, web-navigation, and tool-use tasks by locating the earliest critical error and re-rolling out with targeted feedback, yielding up to 26% relative task-success improvement under matched token budgets. Applies under: requires a replayable environment and reliable step/module/error localization; the debugging model used GPT-4.1 for detection, and the benchmark has limited domain diversity and no multimodal or safety-critical evaluation

## Evidence

### Source
arXiv:2509.25370 | section: results

### Verbatim quote
"The central intuition is that correcting a single root-cause mistake can often flip an otherwise failing trajectory into a successful one. [...] Ablation studies further confirm that focusing on root-cause errors, rather than attempting to fix every surface-level mistake, is key to efficient debugging and meaningful performance gains. [...] In particular, it achieves improvements of up to 26% on ALFWorld and strong average performance on GAIA and WebShop, underscoring its robustness across diverse environments and confirming that targeted error detection and correction can outperform broader but less focused strategies such as scaling rollouts or unguided self-revision."

### Classification
domain: agent_learning
arXiv: cs.AI
Wikipedia: Intelligent_agents
ACM CCS: Computing methodologies → Artificial intelligence → Distributed artificial intelligence → Intelligent agents
CLC: TP181

### Keywords
root-cause debugging, LLM agents, critical error detection, targeted feedback, iterative recovery, agent trajectories, self-refinement, test-time scaling, task success, error taxonomy

```

### 经验 ID: `O-fea2f063ea43a2f1`

```markdown
---
id: O-fea2f063ea43a2f1
created_at: '2026-08-16T15:55:08Z'
summary: '[fine] Applies root-cause feedback at the critical step of failed single-agent
  rollouts, increasing ALFWorld success from 21 to 55, 48 to 74, and 60 to 84 across
  GPT-4o-mini, Qwen3-8B, and Qwen3-Next-80B backbones.'
memory_type: procedural
scope: global
source:
  type: turn
  agent: paper-experience
  session_id: paper-experience-l1
related_observations:
- id: O-e2b8e0fc62e70bfd
  relation: complements
  reason: Both extracted from arXiv:2509.25370 at the L1 level, covering different
    facets of the same system.
  linked_at: '2026-08-17T05:08:39Z'
---

## Observation

### Narrative
This practice tests whether root-cause diagnosis can improve actual task completion after an LLM agent has already failed. The intervention does not simply ask the agent to reflect again or sample more complete trajectories from the beginning. Instead, AgentDebug identifies the earliest critical error, generates guidance specific to the responsible module and error type, and resumes execution from that point. A critical error is the earliest decision whose correction directly prevents the final failure; later mistakes are treated as consequences when they arise from the original deviation. The evaluation uses three single-agent benchmarks with complementary demands. ALFWorld is a text-based embodied environment in which an agent navigates household locations and manipulates objects. WebShop is a web-commerce environment in which the agent must search products, satisfy attributes such as color, size, fit, and price, and complete a transactional goal. GAIA is a knowledge-intensive benchmark in which the agent answers questions using web and other external tools. The agent rollout exposes intermediate memory, reflection, planning, and action modules. Memory retrieves relevant history, reflection interprets the previous outcome and progress, planning selects a feasible next strategy, and action executes an admissible environment operation. AgentDebug first annotates these modules with AgentErrorTaxonomy categories, then asks a debugging model to produce a structured report containing the critical step, module, error type, evidence, root cause, correction guidance, and cascading effects. The corrected agent reruns from the identified critical step rather than restarting from step one. Up to five rerollouts are allowed, and if a rerun fails, the feedback is updated with more specific forward-looking instructions. For fair comparison, baseline methods receive an equivalent total token budget. The comparison includes Self-Refine, which iteratively revises without explicit causal diagnosis; a Vanilla Debugger, which applies naive post-hoc corrections; Tree-of-Thought, which searches among alternative reasoning paths; and Best-of-NN, which samples multiple candidate trajectories and selects among them. AgentDebug is evaluated with GPT-4o-mini, Qwen3-8B, and Qwen3-Next-80B as the underlying agent backbones. On ALFWorld, the method increases success from 21 on the first attempt to 55 for GPT-4o-mini, from 48 to 74 for Qwen3-8B, and from 60 to 84 for Qwen3-Next-80B. The reported findings state that improvements reach up to 26% relative across ALFWorld, GAIA, and WebShop. The method also reports downstream performance in connection with localization quality, with 50.0% step accuracy and 42.5% All-Correct accuracy in the downstream setting described by the paper. Targeted correction is especially useful for smaller backbones, because they have more recoverable failures and benefit from guidance concentrated at the point where the trajectory diverged. The practice remains bounded by a maximum of five rerollouts, the cost of repeated inference, and the quality of the diagnostic model’s causal judgment. It is not a guarantee of recovery: if the environment fails, the task is impossible, the trajectory lacks sufficient evidence, or the detector selects a wrong upstream step, repeated rerollouts may preserve the same failure. Generalization beyond the three text-based benchmarks, especially to multimodal and safety-critical tasks, remains untested.

### Practice environment
Downstream evaluation uses ALFWorld, GAIA, and WebShop. AgentDebug starts each rerollout at the identified critical step and allows N=5 attempts. Backbones are GPT-4o-mini, Qwen3-8B, and Qwen3-Next-80B. Baselines include Self-Refine, Vanilla Debugger, Tree-of-Thought, and Best-of-NN, with maximum attempts matched by total token usage. Primary metric is task success.

### Practice trace
1. Action: Generated structured correction guidance identifying the critical step, responsible module, taxonomy error type, evidence, root cause, and expected cascading effects.
   Feedback: The targeted feedback was used to guide subsequent decisions rather than merely asking for unguided self-reflection or restarting the whole trajectory.
2. Action: Reran each failed agent trajectory from the identified critical step for up to five attempts, refining the feedback after unsuccessful reruns.
   Feedback: On ALFWorld, success increased from 21 to 55 for GPT-4o-mini, from 48 to 74 for Qwen3-8B, and from 60 to 84 for Qwen3-Next-80B.
3. Action: Compared targeted debugging with Self-Refine, Vanilla Debugger, Tree-of-Thought, and Best-of-NN under matched total token usage across ALFWorld, GAIA, and WebShop.
   Feedback: AgentDebug consistently delivered the largest gains and achieved up to 26% relative improvement in task success across the three benchmarks.

### Task context
modality: text-based embodied interaction, web navigation, and tool-using knowledge tasks
scale: 3 benchmarks and 3 backbone agents; up to 5 targeted rerollouts; comparisons with Self-Refine, Vanilla Debugger, Tree-of-Thought, and Best-of-NN
constraint: Requires a detectable critical step and enough environment state to resume from that point; limited to five attempts and incurs additional inference cost, with no validation in multimodal or safety-critical environments

### Extraction rationale
Practice drawn from Section 3.2’s iterative debugging procedure and Section 4.2’s downstream debugging experiments. The rerollout budget, backbone models, baseline comparisons, and ALFWorld success values are directly reported in the experimental setup and results; the cross-benchmark 26% relative improvement is taken from the reported findings.

## Why It Matters

Granularity: fine. Scale: 3 benchmarks and 3 backbone agents; up to 5 targeted rerollouts; comparisons with Self-Refine, Vanilla Debugger, Tree-of-Thought, and Best-of-NN Applies under: Requires a detectable critical step and enough environment state to resume from that point; limited to five attempts and incurs additional inference cost, with no validation in multimodal or safety-critical environments

## Evidence

### Source
arXiv:2509.25370 | section: results

### Verbatim quote
"On GPT-4o-mini, AgentDebug boosts success from 21 (first attempt) to 55; on Qwen3-8B, from 48 to 74; and on Qwen3-Next-80B, from 60 to 84. These results show that AgentDebug consistently outperforms all baselines and can effectively help LLM agents improve regardless of the backbone model, with especially large relative gains for smaller models."

### Classification
domain: agent_learning
arXiv: cs.AI
Wikipedia: Intelligent_agents

### Keywords
LLM agents, agent debugging, agent failures, error propagation, root-cause analysis, AgentErrorTaxonomy, Agent Error Benchmark, targeted feedback, iterative refinement, task success

```

### 经验 ID: `O-866e1c7b6cf58af3`

```markdown
---
id: "O-866e1c7b6cf58af3"
created_at: "2026-08-17T06:47:21Z"
summary: "[conditional] When using test-feedback-driven language-model repair, a focused public-test feedback loop can guide patch generation more effectively than supplying the substantially larger combined public-and-hidden failure output, while hidden tests remain valuable for final validation."
memory_type: semantic
scope: global
source:
  type: turn
  agent: "paper-experience"
  session_id: "paper-experience-l2"
---

## Observation

### Declaration
When using test-feedback-driven language-model repair, a focused public-test feedback loop can guide patch generation more effectively than supplying the substantially larger combined public-and-hidden failure output, while hidden tests remain valuable for final validation.

### Narrative
Test-suite-based program repair uses executable tests not only to decide whether a generated patch is acceptable, but also to guide the language model toward a better revision. In a conversational repair setting, the model first receives the issue description and the faulty function, then receives either a syntax-error message or the failures produced by the public test suite after each attempted patch. The experiment indicates that the feedback channel must be kept sufficiently focused: validating against a much larger combination of public and hidden tests during the conversation can reduce patch-generation effectiveness, even though the combined tests provide more information about final correctness. The relevant setting is a function-level repair task drawn from SWE-Bench Lite, where the system has oracle localization of the faulty function and can execute the public tests associated with the issue. The study compares validation using the public failing-test set alone with validation using the union of public and hidden tests. Public tests provide targeted failure information during the dialogue, while hidden tests are primarily intended to measure whether a plausible patch generalizes beyond the immediately visible behavior. The conversational experiments reached 62% public-test-valid patches for LLaMA 3.1 70B Instruct and 56% for GPT-4o-mini when the public test suite was used as the interactive validation set. When public and hidden tests were used simultaneously, the corresponding public-plus-hidden success rates fell to 47% and 46%. The paper states that the worse result with both suites may be explained by the private test suite usually being several times larger than the public suite, causing its failure output to substantially increase the size of the language-model request. This creates an important engineering tradeoff. More diagnostic tests can improve the breadth of validation, but returning every failure in the conversational prompt may overwhelm the model with redundant or weakly prioritized information, leaving less effective attention for the issue description, faulty function, and the most informative failures. The finding does not imply that hidden tests should be omitted from evaluation. The paper uses them to calculate the stricter SWE-Bench-style metric of successfully resolved issues after a plausible patch is found. Rather, the experience concerns their use inside the repair loop: a compact, targeted public-test feedback channel may be more effective for guiding generation, followed by hidden-test execution for final acceptance. The result is bounded by the experimental design. Only 92 problems were retained from SWE-Bench Lite after excluding multi-function-localized cases, Django cases with unsuitable test output, and environments that could not be set up. The tasks came from well-known Python projects, and the model was asked to rewrite only the provided function. The paper does not compare different methods for compressing, ranking, or summarizing test failures, so the evidence supports limiting feedback volume but does not identify an optimal prompt size. It also does not show that larger diagnostic contexts always harm repair; the stated explanation is specifically that private-suite outputs are larger and significantly increase request size. Practitioners building repair agents should therefore separate interactive guidance from final verification: expose concise, relevant failing-test feedback during patch revision, and reserve the complete public-plus-hidden test suite for evaluating the finished patch.

### Causal explanation (r)
The paper suggests that simultaneous use of public and private suites performs worse because private suites are usually several times larger than public suites, and their failure output significantly increases the language-model request size; hidden tests are instead run after a plausible patch is found for final validation.

confidence (μ_r): medium
depth: deep
depth rationale: The explanation identifies a concrete mechanism—larger private-suite diagnostics expand the model request—and distinguishes interactive guidance from post hoc hidden-test validation rather than merely restating that one configuration performs worse.

### Applicability context
summary: Guides language-model agents in repairing localized Python defects by using compact public-test failures during conversation and reserving hidden tests for final verification, avoiding the lower repair effectiveness observed when large combined failure outputs are fed back interactively.
modality: Python source code, unit-test outputs, public and hidden test suites
scale: 92 selected SWE-Bench Lite issues; two language models; public-test and public-plus-hidden-test conversational validation
constraint: private test suites were typically several times larger than public suites, increasing request length; the study did not test failure summarization or ranking and used oracle function localization rather than unrestricted issue resolution

### Extraction rationale
The declaration is based on the explicit RQ-1 comparison between public-test feedback and simultaneous public-plus-hidden validation, together with the reported 62%/56% public-test results and 47%/46% combined-suite results for the two models. The causal explanation comes directly from the results discussion, which attributes the degradation to the larger private-suite output and separately describes hidden tests as a final validation mechanism; because the wording is 'may be explained,' r receives medium confidence.

## Why It Matters

Claim type: conditional. Confidence: medium. Guides language-model agents in repairing localized Python defects by using compact public-test failures during conversation and reserving hidden tests for final verification, avoiding the lower repair effectiveness observed when large combined failure outputs are fed back interactively. Applies under: private test suites were typically several times larger than public suites, increasing request length; the study did not test failure summarization or ranking and used oracle function localization rather than unrestricted issue resolution

## Evidence

### Source
arXiv:2410.04485 | section: results

### Verbatim quote
"The experiment version where both public TT and private T∗T^{\*} suites are used simultaneously shown worse results. This result may be explained by the fact that private test suite size usually multiple times larger than public, and its failure output significantly increases the LLM request size. [...] In the SWE-Bench both TT and T∗T^{\*} test suites are used for patch validation. Therefore, as soon as the plausible patch is found, the hidden suite T∗T^{\*} is also run."

### Classification
domain: agent_software_eng
arXiv: cs.SE
Wikipedia: Software_engineering
ACM CCS: Software and its engineering → Software creation and management → Software development process management
CLC: TP311

### Keywords
test-suite-based program repair, test feedback, hidden tests, public tests, SWE-Bench Lite, language-model debugging, prompt length, patch validation, automatic issue resolution, Python software repair

```

### 经验 ID: `O-350ca2d3d20d18e3`

```markdown
---
id: "O-350ca2d3d20d18e3"
created_at: "2026-08-17T06:47:16Z"
summary: "[fine] Deploys a validated Llama-based test-failure repair agent with an LLM quality judge and human review, publishing 1,589 production diffs of which 405, or 25.5%, were landed over three months."
memory_type: procedural
scope: global
source:
  type: turn
  agent: "paper-experience"
  session_id: "paper-experience-l1"
---

## Observation

### Narrative
This experience records the controlled deployment of an agentic repair workflow in a production software organization, with explicit quality gates between automated patch generation and human code review. The task is to repair live test failures that would ordinarily be assigned to Meta engineers, while reducing the number of low-quality or misleading diffs shown to reviewers. The workflow begins when Meta’s rule-based Test Failure Management Bot (TFMB) detects a broken test, creates a task, assigns an owner, and triggers the Engineering Agent in a background job. Before the agent begins, TFMB supplies failure details such as the test definition, stack trace, error message, likely blame change, and ownership information. The Engineering Agent then executes a Llama-based ReAct loop (a sequential reasoning-and-tool-use process) in an internal development environment. It can inspect code, search repository structures, edit files, and run relevant tests. Static analysis and test execution are used as deterministic validators, and their outputs are fed back into the trajectory so the agent can repair lint, build, or test problems before final submission. The production configuration selected from offline ablations uses one ReAct run with both static-analysis and test-execution feedback. Once the agent produces a candidate patch, production validation proceeds in three stages. First, the originally breaking test must pass. Second, a separately calibrated LLM-as-a-Judge evaluates whether the patch resembles an acceptable human change. The judge is a smaller iCodeLlama model prompted with examples and instructed to classify a patch as Class 0, likely unacceptable, or Class 1, likely acceptable, while also producing a natural-language reason. Third, the patch enters continuous integration, which builds, tests, and validates relevant code. Only after all validation signals pass is the diff submitted for review. The review artifact includes the agent trajectory, passing test and CI signals, and the associated TFMB task, enabling a human engineer to inspect both the result and the evidence behind it. The system was progressively rolled out to 10%, 50%, and 100% of production test-failure tasks. During the three-month measurement period from February 1 through April 30, 2025, it published 1,589 diffs. Engineers reviewed 1,285, corresponding to an 80% review rate, and landed 405, corresponding to a 31.5% land rate among reviewed diffs and a 25.5% land rate over all published diffs. The judge benchmark contained 244 agent-generated patches manually labeled by engineers: 57 acceptable and 187 unacceptable. Calibration prioritized precision for unacceptable patches so that low-quality candidates would not waste reviewer time or reduce trust. The reported table gives Class 1 precision of 0.867 and recall of 0.385; the paper’s discussion also describes 72 true negatives and 11 false negatives and states that the production prompt yielded approximately 0.86 precision for the targeted class. Production feedback led to concrete changes rather than merely retrospective analysis. A sample of 100 engineer comments identified flaky or already-fixed tests, difficulty finding expert reviewers, missing tools and validators, partially correct solutions, source-versus-test ambiguity, and insufficient production context. The system added a pre-run test to verify that failures were reproducible, moved execution into a more consistent container, suppressed tasks with existing human diffs, added debugging validators, improved review descriptions, and introduced mechanisms to detect blame-change reverts and permit selected test-file changes. The workflow is not an autonomous merge system. Twenty percent of published diffs were not reviewed, and a landed change may still be imperfect even when it passes tests and the judge. The measured acceptance rate is specific to Meta’s codebase, engineering culture, reviewer availability, internal tools, and rolling test-failure stream. Flakiness, missing tools, already-resolved failures, partial repairs, and changes whose correctness depends on production flags or deployment context remain boundaries requiring human intervention.

### Practice environment
Meta production test-failure workflow from February 1 to April 30, 2025. TFMB creates tasks and triggers the Engineering Agent. The agent uses a Llama ReAct harness, internal source-code tools, static analysis, test execution, and continuous integration. A calibrated smaller iCodeLlama LLM-as-a-Judge classifies patches as acceptable or unacceptable before review. Judge evaluation uses 244 human-labeled patches, with 57 acceptable and 187 unacceptable. Production metrics are published volume, review rate, land rate among reviewed diffs, and total land rate. Qualitative analysis codes 100 engineer comments.

### Practice trace
1. Action: Progressively rolled the Engineering Agent from 10% to 50% and then 100% of live TFMB test-failure tasks, requiring the breaking test, LLM judge, and CI validation signals to pass before submitting a diff for review.
   Feedback: From February 1 through April 30, 2025, the system published 1,589 diffs; 1,285 were reviewed, an 80% review rate, and 405 were landed, a 31.5% land rate among reviewed diffs and 25.5% of all published diffs.
2. Action: Calibrated a many-shot iCodeLlama judge on 244 engineer-labeled patches, using Class 0 for likely unacceptable patches and Class 1 for likely acceptable patches, with quality filtering prioritized over maximum publication volume.
   Feedback: The benchmark contained 57 acceptable and 187 unacceptable patches; the reported Class 1 precision was 0.867 and recall was 0.385, while the paper reports approximately 0.86 precision for the targeted unacceptable-patch filtering objective.
3. Action: Coded 100 engineer comments and modified deployment safeguards in response to flakiness, already-fixed tasks, missing validators, reviewer difficulty, source-versus-test ambiguity, and insufficient production context.
   Feedback: The production system added a pre-run reproducibility test, a more consistent container, rules suppressing tasks with existing human diffs, additional debugging validators, clearer review descriptions, a revert validator, and a flag permitting selected test-file changes.

### Task context
modality: production source-code diffs, test failures, CI signals, agent trajectories, and engineer review comments
scale: 10% to 50% to 100% progressive rollout; 1,589 published diffs, 1,285 reviewed diffs, 405 landed diffs; 244 judge benchmark examples and 100 coded engineer comments
constraint: Requires reviewer availability, reliable reproduction, CI and validator coverage, calibrated quality classification, and production-context knowledge; 20% of published diffs received no review and passing tests do not guarantee complete correctness.

### Extraction rationale
The deployment and outcome sequence is drawn from Sections VII-C and IX-A, including progressive rollout, validation gates, review evidence, and production counts. Judge metrics come from Sections VI-B and VIII-C; operational refinements and limitations come from Section IX-B and the Threats to Validity section.

## Why It Matters

Granularity: fine. Scale: 10% to 50% to 100% progressive rollout; 1,589 published diffs, 1,285 reviewed diffs, 405 landed diffs; 244 judge benchmark examples and 100 coded engineer comments Applies under: Requires reviewer availability, reliable reproduction, CI and validator coverage, calibrated quality classification, and production-context knowledge; 20% of published diffs received no review and passing tests do not guarantee complete correctness.

## Evidence

### Source
arXiv:2507.18755 | section: results

### Verbatim quote
"Over a three-month period, the Engineering Agent published nearly 1.6k diffs, with 25.5% of these accepted and landed by human engineers who deemed them to be of sufficient quality to correctly solve the broken test." [...] "80% of the AI fixes were of sufficient interest/quality that engineers took time to review them. Of those that received a review, 31.5% were accepted."

### Classification
domain: agent_software_eng
arXiv: cs.SE
Wikipedia: Software_engineering

### Keywords
AI, Agents, Program Repair, Test Failures, Patch Generation, LLMs, Benchmarking, AI in Production, Engineer Feedback

```

### 经验 ID: `O-22ad8bf1e650418f`

```markdown
---
id: "O-22ad8bf1e650418f"
created_at: "2026-08-17T06:47:16Z"
summary: "[medium] Uses TFMB-triaged failures, a 15-tool Llama ReAct loop, search-and-replace patching, symbolic validation, and an LLM judge to repair Meta code, achieving 42.3% SR@1 offline and a 25.5% production land rate."
memory_type: procedural
scope: global
source:
  type: turn
  agent: "paper-experience"
  session_id: "paper-experience-l1"
---

## Observation

### Narrative
The practice addresses automated repair of diverse software failures in a large, continuously changing codebase, where a failing test can originate from a source defect, a dependency change, a test defect, a reverted change, or environmental flakiness. The Engineering Agent operationalizes program repair as an Oracle-Guided Patch Generation problem: it consumes a Specification containing failure information, produces a Patch, and uses an Oracle consisting primarily of the broken tests to determine whether the resulting program satisfies the observed requirement. A rule-based Test Failure Management Bot (TFMB) first creates and triages a task. It collects the test definition, error message, stack trace, and related failure signals; bisects code changes to identify the likely blame change and owner; groups failures with a common root cause; and applies historical sampling and statistical methods to identify intermittent tests. The resulting task is passed to a Llama-based ReAct harness. ReAct (reasoning and acting) alternates a model-generated thought with one executable action and an observation. The harness exposes 15 actions modeled on the tools available to internal engineers, including reading directories and files, searching for files, classes, methods, and code fragments, navigating to line ranges, retrieving task and diff details, editing a file, running selected tests, and exiting with a summary. The agent must wait for the observation after each command, and its trajectory of thoughts, actions, and tool results becomes context for later decisions. This allows the agent to reproduce the failure, localize the relevant code, make an edit, inspect static-analysis feedback, execute the oracle tests, and iterate until the failure is resolved or its execution budget is exhausted. Patch generation is delegated to a specialized sub-agent rather than exposing the patcher to the full tool set. The patcher receives contextual code, a natural-language instruction, and a summarized orchestration trajectory, then emits a search-and-replace diff. This format avoids arbitrary line numbers and presents unprefixed code in SEARCH and REPLACE blocks, which is closer to the code representation used during model training. Deterministic static analyzers, linters, and test executors act as symbolic validators inside the loop, while a final validation stage checks the original breaking test, runs continuous integration, and sends the candidate to an LLM-as-a-Judge. The judge is a calibrated iCodeLlama model that classifies a patch as acceptable or unacceptable according to human-engineer preferences, including concerns about style, legacy libraries, and unnecessary changes. On 123 real TFMB failures from two large monorepositories and 15 languages, the ReAct-only system solved 28.5% of tasks in one run. Static-analysis feedback alone increased SR@1 to 34.1%, test-execution feedback alone increased it to 43.9%, and the combined configuration achieved 42.3% with a 0.2% error rate and 11.8 median iterations. Five independent runs with test-execution feedback reached 61.0% solve rate. The separate 210-task PatchGen benchmark showed that search-and-replace substantially outperformed unified diff, reaching 53% versus 30% for Llama-405B with detailed instructions. A calibrated judge was used to suppress unacceptable fixes before review, prioritizing quality over volume. Production deployment over three months published 1,589 diffs; engineers reviewed 1,285, or 80%, and landed 405, or 25.5% of all generated diffs and 31.5% of reviewed diffs. The practice does not provide autonomous correctness guarantees: a passing oracle can miss behavioral or nonfunctional defects, and the benchmark excludes many test-only, build, configuration, and generated-file changes. The system depends on internal search and validation infrastructure, deterministic reproduction, reviewer availability, and enough context to distinguish fixing forward from reverting a blame change or changing the test itself.

### Practice environment
The environment is Meta's internal engineering ecosystem, with TFMB task databases, code-change bisection, repository search indexes, source checkouts, test-selection infrastructure, static analyzers, test runners, and continuous integration. The orchestrator is a Llama-based ReAct agent; the patching sub-agent is internally fine-tuned iCodeLlama-70B; the judge is a many-shot-calibrated smaller iCodeLlama model. The TF benchmark contains 123 failures across web, app, and systems code in 15 languages. Evaluation uses SR, PGR, IC, and ER; production uses published, reviewed, and landed diffs.

### Practice trace
1. Action: Constructed each repair task as a Specification-Patch-Oracle triplet, with TFMB supplying the failing test, stack trace, ownership, and blame information, then ran a one-command-at-a-time ReAct loop over repository inspection, editing, and test execution.
   Feedback: The resulting benchmark contained 123 validated real failures across two monorepositories and 15 languages; ReAct alone achieved 28.5% SR@1, while the production-balanced combined-feedback configuration achieved 42.3% SR@1.
2. Action: Separated orchestration from patch generation and prompted an internally fine-tuned iCodeLlama-70B patcher to emit unique SEARCH/REPLACE blocks, followed by static analysis, oracle tests, continuous integration, and LLM-based quality classification.
   Feedback: On 210 PatchGen tasks, search-and-replace reached 43%/51% solve rate for high-level/detailed instructions with iCodeLlama-70B and 42%/53% with Llama-405B; the judge was calibrated for high precision on unacceptable patches, reported as 0.867 precision for acceptable Class 1 in the benchmark.
3. Action: Rolled the validated workflow progressively to 10%, 50%, and 100% of live test-failure tasks, publishing candidates with their trajectories and validation evidence for standard human review.
   Feedback: During the three-month production period, 1,589 diffs were published, 1,285 were reviewed, and 405 were landed; review rate was 80%, land rate among reviewed diffs was 31.5%, and land rate over all published diffs was 25.5%.

### Task context
modality: software source code, unit-test executions, stack traces, static-analysis feedback, and human-reviewed patches
scale: Two large monorepositories; 123 failures in 15 languages; 15 agent actions; 210 PatchGen tuples; 244 judge-labeled patches; 1,589 production diffs
constraint: Offline tasks require a human-landed source-code fix and reliable failing/passing oracle tests; deployment remains dependent on CI, judge calibration, reviewer availability, and reliable reproduction of non-flaky failures.

### Extraction rationale
The core practice and problem framing are drawn from the Introduction and Engineering Agent Overview, including the Specification-Patch-Oracle representation and ReAct loop. Numerical feedback is synthesized from the benchmark and production results in Sections VIII and IX, while the patch-format and judge details come from Sections V and VI.

## Why It Matters

Granularity: medium. Scale: Two large monorepositories; 123 failures in 15 languages; 15 agent actions; 210 PatchGen tuples; 244 judge-labeled patches; 1,589 production diffs Applies under: Offline tasks require a human-landed source-code fix and reliable failing/passing oracle tests; deployment remains dependent on CI, judge calibration, reviewer availability, and reliable reproduction of non-flaky failures.

## Evidence

### Source
arXiv:2507.18755 | section: introduction

### Verbatim quote
"In this paper, we start from a failing test and create an agent that generates a code patch that passes the test." [...] "A ReAct harness ‘reasons’ to generate actions, such as running the test, searching for a file, and creating a patch. After a sequence of actions, a verification step is run which includes static analysis, testing, and an LLM based judge. The verification results are fed back into the agentic loop, and new actions can be taken."

### Classification
domain: agent_software_eng
arXiv: cs.SE
Wikipedia: Software_engineering

### Keywords
AI, Agents, Program Repair, Test Failures, Patch Generation, LLMs, Benchmarking, AI in Production, Engineer Feedback

```

---

# Prompt Engineering 的 Few-shot 示例

**用户提问：** 在 Prompt Engineering 中，Few-shot 示例的选择对最终效果影响大吗？有何系统性的方法？

**检索关键词：** `few-shot prompt example selection variance sensitivity`

## 回答

影响很大，尤其是在代码生成、复杂推理和多跳问答等任务中。经验显示：

- **示例身份本身会显著改变效果**：在 MBPP 编程任务上，仅改变 10 个 few-shot 示例的选择，测试成功率就从约 **43% 变化到 60%以上**，差异超过 **15 个百分点**（O-0f456153fa110e6a）。
- **示例数量有帮助，但选择质量更重要**：从零样本增加到 3 个示例，性能可由 43.2% 提升到 58.4%；但不同示例集合之间仍可能产生很大差异（O-0f456153fa110e6a）。
- **不同 Prompt 往往解决不同问题**：效果较弱的 Prompt 也可能解决最佳 Prompt 未解决的样本。因此，多组 Prompt 做集成，可将单个最佳 Prompt 的 59.6% 提升到 66.4%（O-0f456153fa110e6a）。

较系统的示例选择方法可以按以下流程进行：

1. **建立候选示例池**  
   从训练集或历史成功案例中收集候选示例，确保示例的输入、输出格式和目标任务一致。对于代码任务，还应关注示例是否简洁、是否展示了合适的库调用和实现模式，因为过长或重复的示例可能导致输出冗长、超出上下文限制（O-0f456153fa110e6a）。

2. **按任务覆盖度和代表性筛选**  
   不要只选“看起来容易”或表面相似的示例，应尽量覆盖不同任务类型、推理步骤和输入结构。同时避免所有示例都呈现同一种解法，否则模型可能错误地把示例中的特定数据结构或模式迁移到不适用的问题上（O-0f456153fa110e6a）。

3. **用验证集进行自动评估**  
   将示例选择视为一个可优化问题：生成多组候选 Prompt，在独立验证集上评估，并选择任务指标最优的集合。DSPy 的做法是使用不同随机种子生成多组候选示例，再依据验证指标进行选择（O-7911d1af45688f84；O-ecfddf48641e1b9a）。

4. **优先使用“成功轨迹”进行 Bootstrap**  
   不仅从人工标注示例中选，还可以让模型在训练样本上运行，保留最终结果通过验证指标的完整执行轨迹，将这些轨迹作为候选 demonstrations。这样筛选出的示例更贴合当前模型、任务和程序结构。GSM8K 上，Bootstrap few-shot 达到 80.3%，明显高于随机 few-shot 的 63.0%（O-ecfddf48641e1b9a）。

5. **迭代优化示例质量**  
   用已经优化过的 Prompt 或程序作为教师，再生成新一轮成功轨迹，进行第二轮 Bootstrap。经验中，`zero-shot → few-shot → bootstrap → bootstrap×2` 通常带来持续提升，但前提是验证指标可靠、教师模型能够生成有效轨迹（O-7911d1af45688f84）。

6. **进行多 Prompt 集成**  
   不要只保留一组示例。可以保留多个在验证集上表现较好的、且错误互补的 Prompt，在推理时采用投票或结果聚合，以降低单一示例集合带来的方差（O-0f456153fa110e6a；O-7911d1af45688f84）。

7. **针对具体模型和任务重新优化**  
   示例通常是模型和任务相关的，不能假设一套 demonstrations 能稳定迁移到不同模型、数据集或领域。若任务需要特定 API 词汇、工具本体或领域知识，few-shot 可能无法弥补知识缺口，领域微调可能比继续调整示例更有效（O-fc17597fe9a96925）。

因此，实用策略可以概括为：

> **候选池构建 → 多组示例采样 → 验证集评估 → 成功轨迹 Bootstrap → 迭代优化 → 多 Prompt 集成**

经验库没有覆盖更细粒度的通用排序准则，例如“语义相似度、难度、长度、示例顺序”各自应如何加权，以及这些方法在所有通用对话任务上的统一结论；这些部分应视为**经验库未覆盖**。

## 思维链 (Reasoning)

1. O-0f456153fa110e6a 直接报告：仅改变 few-shot 示例身份，MBPP 成功率可变化超过 15 个百分点，且不同 Prompt 的解题集合具有互补性，因此判断影响很大，并引出多 Prompt 集成。
2. O-0f456153fa110e6a 还说明：增加示例数量有效，但示例中的偶然模式可能误导模型，故提出覆盖度、代表性、简洁性和避免模式偏置等筛选原则。
3. O-7911d1af45688f84 与 O-ecfddf48641e1b9a 描述了系统化 Bootstrap 方法：用任务指标过滤成功轨迹，通过随机搜索和验证集选择 demonstrations，并可进一步迭代和集成。
4. O-fc17597fe9a96925 表明，在具有专门 API 词汇和领域本体的任务中，few-shot 不一定能弥补领域知识不足，因此补充了“必要时采用领域微调”的边界条件。

## 召回的原始经验全貌 (5 条)

### 经验 ID: `O-0f456153fa110e6a`

```markdown
---
id: O-0f456153fa110e6a
created_at: '2026-08-16T15:52:57Z'
summary: '[property] The identity of the few-shot prompt examples has a large influence
  on the fraction of programming tasks solved by a large language model, with performance
  varying by over 15 percentage points across different example sets, and different
  prompts tend to solve complementary subsets of problems.'
memory_type: semantic
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l2
related_observations:
- id: O-be74d39edaf4707b
  relation: complements
  reason: 'Both extracted from arXiv:2108.07732: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:06Z'
- id: O-dfb69f08a8675576
  relation: complements
  reason: 'Both extracted from arXiv:2108.07732: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:07Z'
- id: O-2fb51216bba7818d
  relation: complements
  reason: Both classified under agent_general, from arXiv:2105.10327 and arXiv:2108.07732
    respectively — different papers addressing the same problem area.
  linked_at: '2026-08-16T15:59:17Z'
- id: O-11c050308c4f4a67
  relation: complements
  reason: Both classified under agent_general, from arXiv:2108.07732 and arXiv:2111.12077
    respectively — different papers addressing the same problem area.
  linked_at: '2026-08-16T15:59:17Z'
---

## Observation

### Declaration
The identity of the few-shot prompt examples has a large influence on the fraction of programming tasks solved by a large language model, with performance varying by over 15 percentage points across different example sets, and different prompts tend to solve complementary subsets of problems.

### Narrative
In practical use of few-shot language models for program synthesis, the choice of which solved examples to include in the prompt can dramatically affect success rates. The paper conducts a controlled experiment around this sensitivity using the 137B parameter model on the MBPP dataset. The authors hold out 10 problems to serve as prompt examples, and by varying the random seed that selects which specific 10 problems become prompt examples, they create multiple distinct few-shot prompts. The number of test problems solved (i.e., fraction of 500 test tasks with at least one correct sample out of 80) varies from about 43% to over 60%, depending on the seed. This 17-percentage-point swing shows that the identity of the few-shot examples is a much stronger determinant of performance than, for example, the number of test-case asserts included per prompt (which had minimal effect). The paper also notes that the sets of problems solved under different seeds are not simply nested; out of 19 problems solved by one seed but not the best-performing seed, some are solved only by the weaker seed. This complementarity means that ensembling multiple prompts—taking the union of problems solved across seeds—boosts the overall solve rate from 59.6% (best single seed) to 66.4%. Qualitatively, the authors observe that poorly performing prompts often result in repetitive or too-long samples that exceed the 512-token context window, while compact prompts that use external libraries tend to produce the best results. They also find that the model sometimes picks up spurious patterns from the few-shot examples, such as referencing a specific data structure that appeared in the prompt even when not appropriate for the current task. This sensitivity suggests that the model’s behavior in the few-shot setting is heavily influenced by the distribution of examples in the prompt, and that careful prompt design, or methods like prompt-tuning that optimize the prompt automatically, could yield substantial gains. The paper does not investigate a specific causal mechanism but hints that the model’s limited context window and its tendency to reuse patterns from the immediate prompt history are responsible. The finding is important for anyone deploying few-shot code synthesis systems: prompt engineering is not a minor detail but a critical factor. The experiment also shows that simply increasing the number of few-shot examples from 0 to 3 improves performance from 43.2% to 58.4%, so including examples helps, but the choice of which examples is paramount. This provides empirical backing for research into prompt optimization and selection strategies.

### Applicability context
summary: Demonstrates that few-shot program synthesis performance is highly sensitive to which prompt examples are chosen, implying that careful prompt curation or automatic prompt optimization can yield significant gains.
modality: text (Python code and natural language)
scale: 137B parameter model, 500 MBPP test problems, 10 few-shot examples varied across seeds
constraint: observed with a fixed model and task distribution; the sensitivity may differ for other benchmarks or when using instruction-tuned models; the study used only 10 held-out problems as prompt candidates

### Extraction rationale
The claim comes from the author's direct observation and reporting of performance variation across prompt seeds, demonstrated in a dedicated experiment and supported by qualitative analysis of failure modes. No explicit causal mechanism is stated beyond noting that prompts lead to different sample quality and repetition patterns.

## Why It Matters

Claim type: property. Confidence: medium. Demonstrates that few-shot program synthesis performance is highly sensitive to which prompt examples are chosen, implying that careful prompt curation or automatic prompt optimization can yield significant gains. Applies under: observed with a fixed model and task distribution; the sensitivity may differ for other benchmarks or when using instruction-tuned models; the study used only 10 held-out problems as prompt candidates

## Evidence

### Source
arXiv:2108.07732 | 2108.07732__2d6f3c88 | group: t5 | section: results

### Verbatim quote
"Few-shot performance is quite sensitive to the particular examples given in the prompt. We measure this sensitivity in Figure 6, where each seed corresponds to a particular, distinct choice of prompting examples. We find that while one set of examples (seed 14) is able to solve 60% of tasks, many other examples solve far fewer."

### Classification
domain: agent_general
arXiv: cs.CL
Wikipedia: Natural_language_processing
ACM CCS: Computing methodologies → Artificial intelligence → Natural language processing → Natural language generation
CLC: TP391.1

### Keywords
few-shot prompting, prompt sensitivity, program synthesis, large language models, prompt engineering, code generation

```

### 经验 ID: `O-7911d1af45688f84`

```markdown
---
id: O-7911d1af45688f84
created_at: '2026-08-16T15:52:57Z'
summary: '[trend] Applying progressively deeper stages of compilation — from zero‑shot
  to few‑shot, then to bootstrapped demonstration selection, iterated bootstrapping,
  and ensembling — leads to consistent and substantial performance improvements for
  LLM pipelines across distinct tasks and backbone language models.'
memory_type: semantic
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l2
related_observations:
- id: O-3540404f80d70df6
  relation: complements
  reason: 'Both extracted from arXiv:2310.03714: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:40Z'
- id: O-ecfddf48641e1b9a
  relation: complements
  reason: 'Both extracted from arXiv:2310.03714: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:41Z'
---

## Observation

### Declaration
Applying progressively deeper stages of compilation — from zero‑shot to few‑shot, then to bootstrapped demonstration selection, iterated bootstrapping, and ensembling — leads to consistent and substantial performance improvements for LLM pipelines across distinct tasks and backbone language models.

### Narrative
The DSPy compiler enables a hierarchical optimization process where each successive stage refines the program’s prompts or weights, leading to a consistent upward trajectory in performance. The simplest form is adding randomly sampled few‑shot examples from the training set (LabeledFewShot). Beyond that, bootstrapping with random search (BootstrapFewShotWithRandomSearch) simulates the program on training inputs, discards traces that fail the validation metric, and uses the successful traces as candidate demonstrations; a random‑search procedure then selects the subset of demonstrations that maximizes performance on a validation set. This process can be nested: after obtaining a first bootstrap, the compiled program itself can serve as a teacher for a second round of bootstrapping (bootstrap×2), which effectively generates higher‑quality traces because the teacher is already improved. Finally, an ensemble compiler aggregates the predictions of multiple top‑ranked bootstrap variants via majority voting. The authors report that this progression—zero‑shot → few‑shot → bootstrap → bootstrap×2 → ensemble—yields non‑diminishing gains across programs and models. On GSM8K with GPT‑3.5, a ChainOfThought (CoT) program sees accuracy rise from 50.0% (zero‑shot) to 63.0% (few‑shot) and then to 80.3% (bootstrap) on the development set, with an ensemble further lifting it to 88.3%. For the vanilla Predict module, the pattern is starker: from 24.0% (zero‑shot) to 33.1% (few‑shot), 44.0% (bootstrap), and 64.7% (bootstrap×2), after which the ensemble provides a small additional boost to 62.7% (ensembles can sometimes slightly reduce accuracy due to voting errors). For Llama2‑13b‑chat, the same CoT program improves from 26.7% (zero‑shot) to 43.3% (bootstrap), and the reflection module (multi‑chain comparison) reaches 44.3% with bootstrap and 49.0% with ensemble. In the HotPotQA multi‑hop program, the boost from few‑shot to bootstrap is 36.9% → 48.7% for GPT‑3.5 and 34.7% → 42.0% for Llama2‑13b‑chat, with the ensemble reaching 54.7% and 50.0% respectively. This monotonic improvement is not a foregone conclusion—hallucinated or poorly chosen demonstrations could degrade performance—but the metric‑guided filtering and cross‑validation steps implemented by the teleprompters consistently steered selection toward beneficial demonstrations. The underlying mechanism is that each deeper stage provides the modules with demonstrations that more faithfully illustrate the multi‑step reasoning needed to meet the target metric. Bootstrapping filters for entire successful trajectories, not just final answer matches, so intermediate signatures (e.g., search‑query generation, chain‑of‑thought rationales) also receive examples of effective behavior. Iterated bootstrapping further reduces noise, because the teacher program that generates traces is already more competent. Ensembling reduces variance by combining independent reasoning paths. Together, these stages create a self‑reinforcing cycle where each round of optimization builds on the gains of the previous one, allowing a simple program definition to reach a performance level that would otherwise require extensive manual tuning.

### Causal explanation (r)
Bootstrapping selects demonstrations that correspond to successful execution traces (i.e., those that lead to outputs passing the validation metric), thereby providing high‑quality in‑context examples that illustrate effective reasoning patterns. Iterated bootstrapping uses an already‑improved program as teacher, yielding even cleaner traces. Ensembling reduces prediction variance by aggregating independent reasoning chains.

confidence (μ_r): medium
depth: deep
depth rationale: The explanation identifies a specific causal chain: metric‑filtered traces yield high‑quality demonstrations, which improve in‑context learning; using the improved version as a teacher further raises trace quality; ensembling adds diversity. This goes beyond simply restating that performance improves.

### Applicability context
summary: Establishes that each additional optimization pass (bootstrapping, iteration, ensembling) reliably improves a pipeline’s quality, turning a zero‑shot CoT module on GSM8K from 50% to 88% with GPT‑3.5, and allowing a vanilla Predict module to jump from 24% to 62% after two rounds of bootstrapping, with similar trends on Llama2‑13b‑chat and on HotPotQA.
modality: text
scale: 2 benchmarks, 2 LMs (GPT‑3.5 and Llama2‑13b‑chat), 200‑300 training examples each
constraint: the upward trend holds when optimization is guided by a reliable validation metric; gains may plateau if the metric is noisy or if the teacher program fails to produce any passing traces; not evaluated beyond four compilation stages

### Extraction rationale
The author explicitly observes that bootstrapping 'leads to large gains for every program, across both LMs' and the numerical progression across compilation stages in both case studies supports the trend. The causal account is derived from the detailed description of the teleprompter loop (filtering traces, using a teacher, ensemble voting), which the paper presents as the mechanism.

## Why It Matters

Claim type: trend. Confidence: high. Establishes that each additional optimization pass (bootstrapping, iteration, ensembling) reliably improves a pipeline’s quality, turning a zero‑shot CoT module on GSM8K from 50% to 88% with GPT‑3.5, and allowing a vanilla Predict module to jump from 24% to 62% after two rounds of bootstrapping, with similar trends on Llama2‑13b‑chat and on HotPotQA. Applies under: the upward trend holds when optimization is guided by a reliable validation metric; gains may plateau if the metric is noisy or if the teacher program fails to produce any passing traces; not evaluated beyond four compilation stages

## Evidence

### Source
arXiv:2310.03714 | 2310.03714__357266c1 | group: t10 | section: experiment

### Verbatim quote
Overall, the bootstrap compilation procedure leads to large gains for every program, across both LMs.

### Classification
domain: agent_general
arXiv: cs.CL
Wikipedia: Natural_language_processing
ACM CCS: Computing methodologies → Artificial intelligence → Natural language processing
CLC: TP18

### Keywords
prompt compilation, bootstrapping, few‑shot learning, ensemble, teleprompter, chain‑of‑thought, language model optimization, in‑context learning

```

### 经验 ID: `O-dfb69f08a8675576`

```markdown
---
id: O-dfb69f08a8675576
created_at: '2026-08-16T15:52:57Z'
summary: '[medium] Investigates the influence of model scale on few-shot and fine-tuned
  Python program synthesis using the MBPP dataset, showing that accuracy improves
  log-linearly from ~44% (244M) to 59.6% (137B) with few-shot, and fine-tuning on
  a small 374-example set yields a constant ~10 percentage point improvement across
  scales; hand-edited problem specifications further boost few-shot accuracy from
  63% to 79% on a 100-problem subset.'
memory_type: procedural
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l1
related_observations:
- id: O-0f456153fa110e6a
  relation: complements
  reason: 'Both extracted from arXiv:2108.07732: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:07Z'
- id: O-1db62cb56cc3f2da
  relation: complements
  reason: 'Both extracted from arXiv:2108.07732: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:08Z'
---

## Observation

### Narrative
To understand how program synthesis ability scales with model capacity, a sequence of decoder-only Transformer language models was evaluated on the MBPP dataset in both few-shot and fine-tuned modes. The models, ranging from 244 million to 137 billion parameters, were pre-trained on a mixture of web documents that included a small fraction of pages containing code and text (13.8M documents out of 2.97B). From the 974 MBPP tasks, 10 were held out as prompt examples, 374 used for fine-tuning, and 500 reserved as a test set. In the few-shot setting, the prompt consisted of three example task descriptions, each followed by the correct function and its three test assertions; the model was then presented with a new description and asked to generate the function. For each test problem, 80 completions were sampled using temperature 0.5, and any sample that passed all three test cases upon execution was counted as a success. The fraction of problems with at least one successful sample served as the main performance indicator. Results (Figure 3) revealed that few-shot any‑sample accuracy improved log‑linearly with the logarithm of model size, with the 244M model solving about 44% of problems and the 137B model solving 59.6%. The fraction of individual samples that were correct also grew with scale, though less smoothly. Error analysis showed that while the smallest models often produced syntax errors, larger models overcame such basic mistakes: the 137B model’s errors were predominantly semantic failures (failing the test assertions) rather than runtime crashes. Furthermore, when the same models were fine-tuned on the 374‑task training split for a mere 100 steps with a learning rate of 3e‑5, any‑sample accuracy jumped by a roughly constant 10 percentage points across all model sizes. For the largest model, fine-tuned any‑sample accuracy approached 70%. The quality of problem statements also mattered: on a subset of 100 problems that were manually edited for clarity and consistency, the 137B model’s few-shot accuracy rose from 63% (original) to 79% (edited). The edited texts provided more explicit requirements (e.g., specifying “flatten and sum” rather than just “sum”) and corrected issues like comparing floating-point numbers directly versus using approximate equality. This sensitivity to problem clarity and the consistent fine‑tuning gain demonstrate that even tiny amounts of task‑specific data or better specification can substantially enhance synthesis from LLMs. However, the approach is limited to short, entry‑level Python functions, requires dozens of samples to achieve high coverage, and is influenced by the choice of prompt examples.

### Practice environment
MBPP dataset: 974 short Python problems with natural language descriptions and 3 test-case assertions. 500 tasks used for testing, 374 for fine‑tuning, 10 as few‑shot prompt examples. Models: decoder‑only Transformer LMs with 244M, 8B, 68B, 137B parameters, pre‑trained on 2.97B documents. Few‑shot: prompt with 3 example problems, temperature 0.5 sampling, 80 samples per problem. Fine‑tuning: 100 steps, learning rate 3e-5, batch size 8,192 tokens. Edited dataset: 100 hand‑verified problem versions for comparison. Primary metric: fraction of tasks solved by any sample (any‑sample accuracy); secondary metric: fraction of samples solving each task.

### Practice trace
1. Action: Evaluated the few‑shot synthesis performance of Transformer language models across four sizes (244M, 8B, 68B, 137B) on the MBPP test set, measuring any‑sample accuracy and sample accuracy.
   Feedback: Any‑sample accuracy scaled log‑linearly with parameter count, reaching 59.6% for the 137B model. The fraction of correct individual samples also improved, from roughly 3% to 16.8% at the largest size.
2. Action: Fine‑tuned the same models on 374 MBPP training examples for 100 steps and re‑evaluated synthesis performance.
   Feedback: Fine‑tuning added approximately 10 percentage points to any‑sample accuracy across all sizes; the 137B model’s accuracy rose to about 70%. On a hand‑edited subset of 100 problems, few‑shot performance jumped from 63% (original) to 79% (edited), confirming the benefit of precise specifications.

### Task context
modality: natural language descriptions and Python source code
scale: 244M–137B Transformer parameters; MBPP: 974 tasks (500 test, 374 fine‑tune, 10 prompt examples); 80 samples per task in few‑shot experiments.
constraint: models pre‑trained on general web text, not specialized for code; synthesis limited to short, self‑contained Python functions; requires many samples (80) to achieve reliable performance; performance is sensitive to the specific few‑shot examples and the clarity of the natural language prompt.

### Extraction rationale
Practice description drawn from introduction and method overview (Sections 1 and 4). The action sequence follows the paper’s procedural description; numerical feedback values are from Figure 3 (scaling) and Table 2 (edited dataset).

## Why It Matters

Granularity: medium. Scale: 244M–137B Transformer parameters; MBPP: 974 tasks (500 test, 374 fine‑tune, 10 prompt examples); 80 samples per task in few‑shot experiments. Applies under: models pre‑trained on general web text, not specialized for code; synthesis limited to short, self‑contained Python functions; requires many samples (80) to achieve reliable performance; performance is sensitive to the specific few‑shot examples and the clarity of the natural language prompt.

## Evidence

### Source
arXiv:2108.07732 | 2108.07732__2d6f3c88 | group: t5 | section: introduction

### Verbatim quote
"We evaluate a collection of such models (with between 244M and 137B parameters) on two new benchmarks, MBPP and MathQA-Python, in both the few-shot and fine-tuning regimes. [...] Our largest models, even without finetuning on a code dataset, can synthesize solutions to 59.6% of the problems from MBPP using few-shot learning with a well-designed prompt. Fine-tuning on a held-out portion of the dataset improves performance by about 10 percentage points across most model sizes."

### Classification
domain: agent_software_eng
arXiv: cs.SE
Wikipedia: Software_engineering

### Keywords
program synthesis, large language models, Python, few‑shot, fine‑tuning, MBPP, MathQA, human feedback, execution, functional correctness

```

### 经验 ID: `O-fc17597fe9a96925`

```markdown
---
id: O-fc17597fe9a96925
created_at: '2026-08-16T15:52:57Z'
summary: '[relation] Fine-tuned open-source LLMs (achieving 85–88% accuracy) dramatically
  outperform zero-shot and few-shot closed-source MLLMs including GPT-4o, Claude 3
  Opus, and Gemini 1.5 Pro (achieving at best 25–45% accuracy) on domain-specific
  multimodal tool selection tasks, because general-purpose models lack exposure to
  the specific API functional vocabulary and tool corpus ontology, and few-shot prompting
  does not reliably compensate for this domain knowledge deficit.'
memory_type: semantic
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l2
related_observations:
- id: O-c4f8772c435ebd41
  relation: complements
  reason: 'Both extracted from arXiv:2401.10727: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:47Z'
---

## Observation

### Declaration
Fine-tuned open-source LLMs (achieving 85–88% accuracy) dramatically outperform zero-shot and few-shot closed-source MLLMs including GPT-4o, Claude 3 Opus, and Gemini 1.5 Pro (achieving at best 25–45% accuracy) on domain-specific multimodal tool selection tasks, because general-purpose models lack exposure to the specific API functional vocabulary and tool corpus ontology, and few-shot prompting does not reliably compensate for this domain knowledge deficit.

### Narrative
The dominant paradigm for deploying tool-augmented LLM agents currently relies on closed-source, general-purpose multimodal LLMs (MLLMs) such as GPT-4o, Claude 3 Opus, and Gemini 1.5 Pro, which are accessed via API with zero-shot or few-shot prompting. These models are marketed as capable of understanding instructions across modalities and performing tool-use tasks without domain-specific training. However, when applied to specialized tool selection tasks—specifically, recommending the correct machine learning API from a corpus of 932 HuggingFace models based on multimodal user instructions—the performance gap between prompted closed-source MLLMs and fine-tuned open-source LLMs is dramatic and decisive. Fine-tuned open-source models achieve tool selection accuracy in the 85–88% range, while the best closed-source MLLMs in zero-shot and few-shot configurations struggle to exceed 25–45% accuracy, even when provided with streamlined API descriptions and exemplar demonstrations. Specifically, on a subset of 80 evaluation cases (40 text-only, 40 text-plus-image), GPT-4o-128K achieved only 25.0% text accuracy and 5.0% image accuracy in zero-shot mode (where the model receives only the instruction and API descriptions without any examples), improving marginally to 20.0% text and 25.0% image with few-shot prompting (where several example instruction-API pairs are provided in the prompt). Claude 3 Opus-200K performed relatively better in zero-shot text scenarios at 45.0% accuracy but dropped to 17.5% on image inputs, and few-shot prompting actually degraded its text performance to 32.5% while image accuracy remained at 17.5%. Gemini 1.5 Pro achieved 22.5% text and 32.5% image in zero-shot, with few-shot prompting shifting the balance to 35.0% text and 22.5% image. In stark contrast, the fine-tuned Vicuna-13B (an open-source 13B-parameter model trained via Low-Rank Adaptation, or LoRA, on the domain-specific ToolMMBench dataset) achieved 87.5% text accuracy and 85.0% image accuracy on the same evaluation cases, with zero hallucination (no fabricated API names) in both modalities. The closed-source models exhibited notable hallucination rates: GPT-4o hallucinated on 2.5% of text cases and 17.5% of image cases in zero-shot; Claude hallucinated on 7.5% of text cases in few-shot; Gemini hallucinated on 10.0% of image cases in few-shot. The performance pattern reveals several important sub-findings. First, few-shot prompting does not consistently improve performance for closed-source MLLMs on this task—in several cases it degraded accuracy (GPT-4o text dropped from 25% to 20%; Claude text dropped from 45% to 32.5%) while sometimes increasing hallucination, suggesting that in-context examples can confuse rather than guide when the model lacks domain-specific training. Second, image-input scenarios are particularly challenging for zero-shot closed-source models: GPT-4o's 5.0% image accuracy indicates near-total failure to leverage visual information for tool disambiguation without fine-tuning, despite its advertised multimodal capabilities. Third, the consistently high hallucination rates of closed-source models (reaching 17.5–20% in some configurations) contrast sharply with the near-zero hallucination of fine-tuned models, indicating that domain-specific fine-tuning is essential for vocabulary grounding—teaching the model the actual set of valid API names rather than allowing it to generate plausible-sounding but nonexistent tools. The causal explanation for this performance gap, as interpreted by the authors, is that general-purpose MLLMs, while possessing broad world knowledge and reasoning capabilities, lack the specific functional vocabulary and API-to-task mappings required for accurate tool selection in a specialized domain. The ToolMMBench dataset contains 932 APIs organized into a two-level hierarchical classification system with 29 coarse-grained tasks and numerous fine-grained subtasks, many with subtle functional distinctions (e.g., eleven different ControlNet conditioning modalities, multiple image classification models differing only in dataset granularity). General MLLMs have no exposure to this specific API ontology and must rely on semantic similarity between the instruction and API descriptions—a strategy that fails when APIs have similar descriptions but different functional boundaries. Fine-tuning on domain-specific instruction-API pairs allows the model to internalize the precise functional boundaries and vocabulary of the tool corpus, enabling accurate disambiguation. This finding was validated by comparing MLLM-Tool (Vicuna-13B fine-tuned on ToolMMBench) against LLaVA-v1.5 (a leading open-source multimodal LLM fine-tuned on general visual instruction data), where LLaVA achieved 91.05% (7B) and 93.15% (13B) accuracy on image-only subsets—competitive but slightly below MLLM-Tool's 93.68% (Vicuna-7B) and 94.44% (Vicuna-13B), demonstrating that domain-specific fine-tuning of the tool selection capability provides advantage over general visual instruction tuning even when both models use the same parameter scale.

### Causal explanation (r)
General-purpose closed-source MLLMs lack exposure to the specific API functional vocabulary and tool corpus ontology of the target domain, and must rely on semantic similarity matching between instructions and API descriptions—a strategy that fails when APIs have similar descriptions but different functional boundaries—while fine-tuning internalizes the precise functional distinctions and valid API name vocabulary.

confidence (μ_r): medium
depth: deep
depth rationale: Identifies a specific mechanism (domain ontology internalization vs. semantic similarity matching) explaining why prompting fails: general models lack the precise functional vocabulary to distinguish between similar-sounding APIs, while fine-tuning embeds these distinctions directly into the model parameters.

### Applicability context
summary: Demonstrates that fine-tuning open-source LLMs on domain-specific tool corpora is necessary for accurate tool selection, as closed-source MLLMs (GPT-4o, Claude, Gemini) fail to exceed 25–45% accuracy even with few-shot prompting while fine-tuned Vicuna-13B achieves 85–88% with zero hallucination, informing deployment decisions for production tool agent systems.
modality: text, images
scale: 80 evaluation cases (40 text, 40 image), 3 closed-source MLLMs × 2 prompting settings, 1 fine-tuned open-source model, 932 API corpus
constraint: evaluation limited to 80 cases due to API access costs for closed-source models; tested only on HuggingFace ML APIs which may be less represented in general MLLM pre-training; comparison uses Vicuna-13B as the open-source baseline—results may not generalize to all open-source architectures; closed-source model versions (GPT-4o-128K, Claude-3-Opus-200K, Gemini 1.5 Pro) represent snapshots that may change with updates

### Extraction rationale
The author explicitly interprets the comparison with closed-source MLLMs in Section 6.2 as demonstrating 'the necessity of fine-tuning open-source LLMs' (declaration basis). The causal explanation (r) is synthesized from the authors' discussion of why general models fail—they state the performance gap 'underscores the necessity of fine-tuning' and the broader paper context establishes that the tool corpus has specific functional boundaries not present in general pre-training—this attribution is stated by the authors, though the specific mechanism of 'semantic similarity matching vs. vocabulary grounding' is partially elaborated by the authors and partially inferred from the experimental design.

## Why It Matters

Claim type: relation. Confidence: medium. Demonstrates that fine-tuning open-source LLMs on domain-specific tool corpora is necessary for accurate tool selection, as closed-source MLLMs (GPT-4o, Claude, Gemini) fail to exceed 25–45% accuracy even with few-shot prompting while fine-tuned Vicuna-13B achieves 85–88% with zero hallucination, informing deployment decisions for production tool agent systems. Applies under: evaluation limited to 80 cases due to API access costs for closed-source models; tested only on HuggingFace ML APIs which may be less represented in general MLLM pre-training; comparison uses Vicuna-13B as the open-source baseline—results may not generalize to all open-source architectures; closed-source model versions (GPT-4o-128K, Claude-3-Opus-200K, Gemini 1.5 Pro) represent snapshots that may change with updates

## Evidence

### Source
arXiv:2401.10727 | 2401.10727__b58411ce | group: t8 | section: experiment

### Verbatim quote
"When compared with closed-source MLLMs, the significant improvement underscores the necessity of fine-tuning open-source LLMs. [...] We select GPT-4o-128K [17], Claude-3-Opus-200K [1], and Gemini 1.5 Pro [30] for evaluation based on their performance and supported context length. [...] Table 4 shows their zero-shot and few-shot performance. [...] Ours (Vicuna-13B) [achieves] 87.5% text and 85.0% image accuracy with 0.0% hallucination."

### Classification
domain: agent_tool_use
arXiv: cs.CL
Wikipedia: Language_modeling
ACM CCS: Computing methodologies → Artificial intelligence → Natural language processing → Language models
CLC: TP181

### Keywords
fine-tuning vs prompting, open-source vs closed-source LLM, tool selection accuracy, domain-specific training, API vocabulary grounding, hallucination comparison, multimodal agent evaluation

```

### 经验 ID: `O-ecfddf48641e1b9a`

```markdown
---
id: O-ecfddf48641e1b9a
created_at: '2026-08-16T15:52:57Z'
summary: '[medium] Compiles a declarative DSPy program (e.g., ChainOfThought or multi-hop)
  into an optimized pipeline by bootstrapping few-shot demonstrations, achieving accuracy
  gains of 10–43 points across GPT-3.5 and Llama2-13b-chat on GSM8K and HotPotQA without
  any hand-written prompt templates.'
memory_type: procedural
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l1
related_observations:
- id: O-528954be00c79720
  relation: complements
  reason: 'Both extracted from arXiv:2310.03714: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:41Z'
- id: O-7911d1af45688f84
  relation: complements
  reason: 'Both extracted from arXiv:2310.03714: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:41Z'
- id: O-d376ca4a17c00e83
  relation: complements
  reason: 'Both extracted from arXiv:2310.03714: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:41Z'
- id: O-e1da3243c9abe864
  relation: complements
  reason: 'Both extracted from arXiv:2310.03714: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:43Z'
---

## Observation

### Narrative
The central practice of this paper is the creation of a self-improving LM pipeline compiler, DSPy. Researchers define a program as a `dspy.Module` subclass that declares one or more parameterized modules, each carrying a natural-language signature such as `"question -> answer"` or `"context, question -> search query"`. Modules include basic `Predict`, `ChainOfThought` (which prepends a "Reasoning: Let's think step by step" output field), `ReAct` (agent loop with tool use), and `MultiChainComparison` for meta-reasoning. The forward method implements arbitrary control flow, e.g., a multi-hop retrieval loop that iteratively generates queries and retrieves passages. To improve performance, the program is compiled using a teleprompter. The simplest effective teleprompter is `BootstrapFewShot` (with random search or Optuna hyperparameter selection). It takes a metric, a training set (which may only have input and final output labels), and optionally a teacher program. During compilation, the teleprompter simulates the program on training examples, records the input–output trace of every internal `Predict` call, and retains only those traces that lead to a valid final output according to the metric. These traces become few-shot demonstrations for the corresponding modules. Multiple trials (e.g., 16) with different random seeds generate candidate demonstration sets, which are evaluated on a validation set to select the best. The resulting compiled program contains automatically generated, pipeline-adapted prompts with demonstrations, entirely replacing hand-crafted prompt templates. This process was applied to several DSPy programs on GSM8K and HotPotQA. On GSM8K, a `ChainOfThought` program compiled with `BootstrapFewShotWithRandomSearch` using only answer labels (no human reasoning chains) achieved 80.3% dev accuracy with GPT-3.5, versus 63.0% with random few-shot examples and 78.6% when provided with human-written CoT demonstrations. For Llama2-13b-chat, the compiled CoT program reached 43.3%, a 16-point improvement over zero-shot. A more complex `ThoughtReflection` program that samples five reasoning chains and compares them reached 86.7% with an ensemble on GPT-3.5 and 49.0% on Llama2-13b-chat. On HotPotQA multi-hop retrieval, the custom `BasicMultiHop` program compiled with bootstrap achieved 48.7% answer EM on GPT-3.5 (up from 36.9% few-shot) and 42.0% on Llama2-13b-chat (up from 34.7%). An ensemble pushed these to 54.7% and 50.0% on dev. The compiler can also compose a teacher–student pipeline: a compiled large LM program is used to supervise the bootstrapping of demonstrations for finetuning a small LM such as T5-Large. This yielded 39.3% EM on HotPotQA with T5-Large, showing that small, privately fine-tuned models can rival large proprietary LMs for specific tasks. The practice demonstrates that systematic compilation of declarative LM pipelines can eliminate brittle prompt engineering, adapt across LMs, and unlock strong performance from small LMs. However, it requires a reliable automatic metric and a training set of at least a few hundred examples for bootstrapping. The bootstrapped demonstrations are specific to the LM and task, and the compilation process can take minutes to tens of minutes per trial, though parallelism is possible.

### Practice environment
Datasets: GSM8K math word problems (200 train, 300 dev, 1.3k test) and HotPotQA multi-hop QA (200 train, 300 dev hard, 1k test). LMs: GPT-3.5 (gpt-3.5-turbo), Llama2-13b-chat, T5-Large. Retriever: ColBERTv2 on Wikipedia 2017 abstracts. Metrics: GSM8K accuracy on final numerical value; HotPotQA answer exact match and passage accuracy. Teleprompters: `BootstrapFewShotWithRandomSearch` (16 trials, k=8 demos), `Ensemble` (top-7 candidates, majority voting), `BootstrapFinetune` (T5-Large with teacher). Training labels: only final answer labels; no intermediate reasoning or search query labels.

### Practice trace
1. Action: Compiled the `ChainOfThought` program on GSM8K using `BootstrapFewShotWithRandomSearch` with 200 train examples (no reasoning labels)
   Feedback: GPT-3.5 dev accuracy 80.3%, vs. 63.0% random few-shot and 78.6% with human CoT prompts; Llama2-13b-chat 43.3%, vs. 26.7% zero-shot and 27.3% random few-shot
2. Action: Compiled the `BasicMultiHop` program on HotPotQA with bootstrap
   Feedback: GPT-3.5 dev EM 48.7% (vs. 36.9% few-shot); Llama2-13b-chat 42.0% (vs. 34.7% few-shot); ensemble raised GPT-3.5 to 54.7%
3. Action: Compiled T5-Large via teacher–student composition: used compiled multi-hop Llama2-13b-chat as teacher to supervise bootstrap finetuning on 200 labeled + 800 unlabeled HotPotQA questions
   Feedback: T5-Large achieved 39.3% EM and 46.0% passage accuracy, competitive with GPT-3.5 few-shot and far exceeding its size class

### Task context
modality: text (questions, reasoning chains, retrieval passages)
scale: 2 tasks (GSM8K: 200 train/300 dev examples; HotPotQA: 200 train/300 dev hard examples); 2 frozen LMs (GPT-3.5, Llama2-13b-chat); 4 program architectures; 1 fine-tuned T5-Large (770M); bootstrapping trials: 16
constraint: requires a metric that can automatically score outputs; training examples need only final output labels; bootstrapping relies on LM's ability to generate valid multi-step traces; compiled prompts are LM-specific and may not transfer; performance gains taper with very small training sets or noisy metrics

### Extraction rationale
Medium experience drawn from introduction and Section 6–7 overview. The narrative describes the compilation practice and the main experimental outcomes. The action-feedback pairs reflect the high-level steps and the primary numerical gains reported in Tables 1 and 2. All numbers are verbatim from those tables.

## Why It Matters

Granularity: medium. Scale: 2 tasks (GSM8K: 200 train/300 dev examples; HotPotQA: 200 train/300 dev hard examples); 2 frozen LMs (GPT-3.5, Llama2-13b-chat); 4 program architectures; 1 fine-tuned T5-Large (770M); bootstrapping trials: 16 Applies under: requires a metric that can automatically score outputs; training examples need only final output labels; bootstrapping relies on LM's ability to generate valid multi-step traces; compiled prompts are LM-specific and may not transfer; performance gains taper with very small training sets or noisy metrics

## Evidence

### Source
arXiv:2310.03714 | 2310.03714__357266c1 | group: t10 | section: introduction

### Verbatim quote
"We design a compiler that will optimize any DSPy pipeline to maximize a given metric. ... Within minutes of compiling, a few lines of DSPy allow GPT-3.5 and llama2-13b-chat to selfbootstrap pipelines that outperform standard few-shot prompting (generally by over 25% and 65%, respectively) and pipelines with expert-created demonstrations (by up to 5–46% and 16–40%, respectively)."

### Classification
domain: agent_learning
arXiv: cs.CL
Wikipedia: Natural_language_processing

### Keywords
language model pipelines, prompt engineering, declarative programming, in-context learning, few-shot prompting, bootstrapping, self-improving systems, modular neural networks, program optimization, multi-hop reasoning

```

---

# Transformer 位置编码的长度外推

**用户提问：** Transformer 架构中，位置编码的选择如何影响模型在超出训练长度（OOD）数据上的泛化能力？

**检索关键词：** `positional encoding extrapolation length out of distribution sequence`

## 回答

在 Transformer 的**超出训练长度（OOD 长度）泛化**场景中，位置编码的选择会直接决定模型是否具有处理未见更长序列的结构性归纳偏置：

- **固定正弦/余弦位置编码**更有利于长度外推。其位置表示具有代数结构：固定相对偏移的位置关系可以由当前位置表示的线性变换得到。因此，即使推理序列长于训练时见过的长度，模型仍有可能将已学到的位置/相对位置信息延伸到更长序列。
- **可学习的位置嵌入**在训练长度范围内可与正弦编码达到近似相同的任务效果，但经验中没有显示其具备同样的长度外推性质。它们是针对训练中出现的位置直接学习的，因此在超过训练长度时，缺少正弦编码所具有的确定性、可延展的位置结构。
- 因而，若目标是提升 OOD 长度上的泛化能力，应优先选择**固定正弦位置编码**；若只关心训练长度覆盖范围内的性能，则两者在所述机器翻译实验中几乎无差异（开发集 BLEU：正弦编码 25.8、可学习嵌入 25.7）。

需要限定的是，上述等效性能结论来自长度为数百 token 量级的测试；对于截然不同的数据分布或极长序列，经验库仅提示可学习嵌入“可能”有不同表现，未提供直接的 OOD 长度实证比较。

## 思维链 (Reasoning)

1. Transformer 的自注意力本身对排列不敏感，因而非循环序列模型必须显式加入位置信息。（O-5cbdee9f397d0157）
2. 在给定的英德翻译实验中，固定正弦位置编码与可学习位置嵌入的训练长度内性能几乎相同（BLEU 25.8 vs. 25.7），说明选择正弦编码并未牺牲该实验内的常规质量。（O-5cbdee9f397d0157）
3. 正弦编码中，位置 `pos+k` 的表示可由 `pos` 的表示线性表达；这一结构被指出可能支持对训练未见更长序列的外推。（O-5cbdee9f397d0157）
4. 因此，从经验可推导：针对“超过训练长度”的 OOD 泛化，正弦编码提供更明确的长度外推归纳偏置；经验库未提供其他 Transformer 位置编码方案在该问题上的比较证据。（O-5cbdee9f397d0157）

## 召回的原始经验全貌 (5 条)

### 经验 ID: `O-5cbdee9f397d0157`

```markdown
---
id: O-5cbdee9f397d0157
created_at: '2026-08-16T15:52:57Z'
summary: '[relation] Positional encodings are mandatory for non‑recurrent sequence
  models. Fixed sinusoidal encodings and learned positional embeddings yield equivalent
  translation quality, but sinusoidal encodings possess the additional property of
  allowing linear extrapolation to longer sequences, making them preferable when sequence
  length generalisation is desired.'
memory_type: semantic
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l2
related_observations:
- id: O-036453ccd61563a1
  relation: complements
  reason: 'Both extracted from arXiv:1706.03762: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:57:49Z'
---

## Observation

### Declaration
Positional encodings are mandatory for non‑recurrent sequence models. Fixed sinusoidal encodings and learned positional embeddings yield equivalent translation quality, but sinusoidal encodings possess the additional property of allowing linear extrapolation to longer sequences, making them preferable when sequence length generalisation is desired.

### Narrative
A core challenge in non‑recurrent sequence models is encoding positional information, because self‑attention by itself is permutation‑invariant. The Transformer solves this by adding explicit positional encodings to the input embeddings before feeding them into the encoder and decoder stacks. The paper experiments with two approaches: fixed sinusoidal encodings where each dimension follows a sine or cosine of different wavelengths, and learned positional embeddings that are trained as part of the model parameters. Both methods yield nearly identical performance on the English‑German translation task, as shown in Table 3 row (E) where the sinusoidal version achieves a development BLEU of 25.8 and the learned version gives 25.7. Despite this tie in accuracy, the authors advocate for the sinusoidal scheme because of its potential to extrapolate to longer sequences: for any fixed offset k, the representation of position pos+k can be expressed as a linear function of the representation at pos, a property that may help the model generalise to sequence lengths unseen during training. This inductive bias is grounded in the algebraic structure of sinusoids, and it becomes practically important in applications where inference‑time inputs exceed the maximum training length — for instance, in parsing or very long sentence translation. The experience teaches that position information is strictly necessary for a recurrent‑free architecture, but it does not have to be learned from data; a simple deterministic encoding with desirable extrapolation properties can perform equally well. Moreover, the fact that learned embeddings do not offer an advantage suggests that position is a relatively low‑complexity feature that does not require the model to expend capacity on learning it. However, the sinusoidal pattern is only one possible fixed encoding; the key insight is that the choice between fixed and learned can be made based on the desired ability to extrapolate, without sacrificing quality. In the context of large language models that follow, many variants have adopted learned positions, but the sinusoidal embedding paved the way and remains a competitive option, especially when computational efficiency or sequence length flexibility is paramount. The recommendation is therefore: when building a self‑attention model, include positional encodings; using fixed sinusoids is safe, effective, and inherently supports length extrapolation, whereas learned embeddings offer no measurable quality benefit in the tested regime.

### Causal explanation (r)
Sinusoidal encodings are chosen because they can be linearly combined to represent relative positions, enabling the model to generalise beyond the maximum training length.

confidence (μ_r): medium
depth: deep
depth rationale: The mechanism (linear representability of relative positions via trigonometric identities) is a concrete mathematical property that explains extrapolation ability.

### Applicability context
summary: Establishes that sinusoidal and learned positional encodings perform nearly identically (BLEU 25.8 vs 25.7), but sinusoids enable length extrapolation, guiding architectural choice for self‑attention models.
modality: text (token embeddings)
scale: WMT EN‑DE newstest2013 dev set; base Transformer with 6 layers, d_model=512
constraint: the equivalence holds for the tested sequence lengths (up to a few hundred tokens); learned embeddings may be advantageous for very different data distributions or extremely long sequences

### Extraction rationale
The claim is directly supported by an ablation experiment showing nearly identical BLEU, and the choice rationale is explicitly stated in Section 3.5.

## Why It Matters

Claim type: relation. Confidence: high. Establishes that sinusoidal and learned positional encodings perform nearly identically (BLEU 25.8 vs 25.7), but sinusoids enable length extrapolation, guiding architectural choice for self‑attention models. Applies under: the equivalence holds for the tested sequence lengths (up to a few hundred tokens); learned embeddings may be advantageous for very different data distributions or extremely long sequences

## Evidence

### Source
arXiv:1706.03762 | 1706.03762__574207a1 | group: t4 | section: method

### Verbatim quote
"We also experimented with using learned positional embeddings instead, and found that the two versions produced nearly identical results (see Table 3 row (E)). We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training."

### Classification
domain: agent_general
arXiv: cs.CL
Wikipedia: Natural_language_processing
ACM CCS: Computing methodologies → Artificial intelligence → Natural language processing → Machine translation
CLC: TP391.1

### Keywords
positional encoding, sinusoidal, learned embeddings, sequence length extrapolation, transformer

```

### 经验 ID: `O-b71bef4c0317b3e2`

```markdown
---
id: O-b71bef4c0317b3e2
created_at: '2026-08-16T15:52:57Z'
summary: '[trend] As the rendering resolution decreases relative to the resolution
  distribution seen during training, the performance gap between a cone-casting neural
  radiance field using integrated positional encoding (mip-NeRF) and a standard ray-casting
  NeRF using point-sampled positional encoding widens, because IPE attenuates high-frequency
  Fourier features according to the frustum''s spatial variance, thereby resolving
  scale ambiguity and preventing aliasing.'
memory_type: semantic
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l2
related_observations:
- id: O-cd79d3ff4023131c
  relation: complements
  reason: 'Both extracted from arXiv:2103.13415: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:00Z'
---

## Observation

### Declaration
As the rendering resolution decreases relative to the resolution distribution seen during training, the performance gap between a cone-casting neural radiance field using integrated positional encoding (mip-NeRF) and a standard ray-casting NeRF using point-sampled positional encoding widens, because IPE attenuates high-frequency Fourier features according to the frustum's spatial variance, thereby resolving scale ambiguity and preventing aliasing.

### Narrative
When reconstructing a 3D scene from a set of images to synthesize novel views, standard Neural Radiance Fields (NeRF) suffer from severe aliasing and blur when the camera-to-scene distance varies across training or testing images. NeRF renders each pixel by casting a single infinitesimally thin ray, sampling points along it, transforming each point with a sinusoidal positional encoding (PE) and feeding the result into an MLP to predict density and color. Because the PE treats all sampled points identically regardless of the pixel's spatial footprint, the same 3D location observed at different resolutions yields identical encoding features, creating scale ambiguity that leads to aliased renderings. Mip-NeRF addresses this by instead casting a cone through each pixel and dividing the cone into a series of conical frustums (truncated cones). For each frustum, it computes an Integrated Positional Encoding (IPE) that approximates the expected value of the PE over the frustum's volume by modeling the frustum as a 3D Gaussian and analytically integrating the Fourier features. The crucial consequence is that high-frequency components of the encoding are attenuated when the frustum is large (coarse scale) and preserved when the frustum is small (fine scale), effectively acting as a scale-dependent low-pass filter. On a multiscale Blender dataset (constructed by combining the original images with 2×, 4×, and 8× downsampled versions), mip-NeRF reduces the average error metric (geometric mean of MSE, 1−SSIM, and LPIPS) by 60% compared to the JaxNeRF baseline. Moreover, the performance gap widens dramatically as the rendering resolution decreases: at full resolution, mip-NeRF's PSNR is 32.63 dB vs NeRF's 31.20 dB (a 1.43 dB advantage), but at 1/8 resolution the gap explodes to 35.60 dB vs 22.53 dB (a 13.07 dB advantage). This trend demonstrates that cone-casting with IPE not only improves overall accuracy but specifically mitigates the catastrophic aliasing that NeRF exhibits when forced to render at scales different from those seen during training. The authors attribute this behavior to the fact that IPE features explicitly encode the size and shape of the sampled volume: small frustums retain high-frequency detail, while large frustums naturally smooth out high-frequency content that would otherwise alias. This scale-adaptive encoding resolves the ambiguity illustrated in Figure 3 of the paper, where two different camera configurations sampling the same 3D point produce indistinguishable PE features but clearly distinguishable IPE features. The improvement holds across all eight Blender scenes and is consistent across multiple versions of NeRF, including variants with centered pixels and improved training stability, confirming that the cone-casting and IPE are the primary drivers, not auxiliary optimizations.

### Causal explanation (r)
IPE attenuates high-frequency positional encoding components according to the frustum's variance (via the Gaussian attenuation term exp(−½σ²)), so small frustums preserve fine detail while large frustums naturally low-pass filter the signal, preventing aliasing by matching the encoding bandwidth to the pixel footprint.

confidence (μ_r): high
depth: deep
depth rationale: Identifies a specific mathematical mechanism (Gaussian attenuation of Fourier features based on frustum variance) that directly links the encoding to the pixel footprint, not a tautological restatement of the improvement.

### Applicability context
summary: Enables reconstruction of 3D scenes from images with varying camera distance (multi-resolution) for anti-aliased novel view synthesis, reducing average error by 60% on a 4× scale variation dataset relative to standard NeRF, with the PSNR advantage growing from +1.4 dB at full resolution to +13 dB at 1/8 resolution.
modality: images (RGB renders)
scale: 8 scenes, 4 scales each (1×, 2×, 4×, 8× downsampled), 1M iteration training
constraint: requires known camera intrinsics and poses; assumes static scenes; benefits most when training images contain scale variation; tested only on synthetic Blender data and a forward-facing dataset (LLFF) where improvement is negligible due to NDC parameterization

### Extraction rationale
The author explicitly interprets cross-scale results on the multiscale dataset as a widening performance gap with decreasing resolution, and provides a direct attribution of the advantage to IPE's scale-dependent attenuation in both the abstract and the discussion of Figure 3/Figure 4.

## Why It Matters

Claim type: trend. Confidence: high. Enables reconstruction of 3D scenes from images with varying camera distance (multi-resolution) for anti-aliased novel view synthesis, reducing average error by 60% on a 4× scale variation dataset relative to standard NeRF, with the PSNR advantage growing from +1.4 dB at full resolution to +13 dB at 1/8 resolution. Applies under: requires known camera intrinsics and poses; assumes static scenes; benefits most when training images contain scale variation; tested only on synthetic Blender data and a forward-facing dataset (LLFF) where improvement is negligible due to NDC parameterization

## Evidence

### Source
arXiv:2103.13415 | 2103.13415__eae48ad6 | group: t3 | section: abstract

### Verbatim quote
"The rendering procedure used by neural radiance fields (NeRF) samples a scene with a single ray per pixel and may therefore produce renderings that are excessively blurred or aliased when training or testing images observe scene content at different resolutions. [...] By efficiently rendering anti-aliased conical frustums instead of rays, mip-NeRF reduces objectionable aliasing artifacts and significantly improves NeRF’s ability to represent fine details, while also being 7% faster than NeRF and half the size. Compared to NeRF, mip-NeRF reduces average error rates by 17% on the dataset presented with NeRF and by 60% on a challenging multiscale variant of that dataset that we present. [...] IPE features effectively anti-aliased positional encoding features that smoothly encode the size and shape of a volume of space."

### Classification
domain: agent_general
arXiv: cs.CV
Wikipedia: Computer_vision
ACM CCS: Computing methodologies → Artificial intelligence → Computer vision
CLC: TP391.4

### Keywords
neural radiance fields, anti-aliasing, mipmap, cone tracing, integrated positional encoding, multiscale rendering, view synthesis, volumetric rendering, prefiltering, positional encoding

```

### 经验 ID: `O-440e30e45bee7618`

```markdown
---
id: O-440e30e45bee7618
created_at: '2026-08-16T15:52:57Z'
summary: '[property] Integrated positional encoding (IPE) automatically attenuates
  high-frequency Fourier features based on the Gaussian variance of the sampled frustum,
  making the maximum frequency hyperparameter L effectively unnecessary; L can be
  set to an arbitrarily large fixed value without impacting performance, unlike standard
  positional encoding whose performance collapses if L is too large.'
memory_type: semantic
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l2
related_observations:
- id: O-cd79d3ff4023131c
  relation: complements
  reason: 'Both extracted from arXiv:2103.13415: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:57:59Z'
- id: O-e1e60eeb00f4d99c
  relation: complements
  reason: 'Both extracted from arXiv:2103.13415: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:01Z'
- id: O-42f15dfa9f131a56
  relation: complements
  reason: Both classified under agent_general, from arXiv:2012.05420 and arXiv:2103.13415
    respectively — different papers addressing the same problem area.
  linked_at: '2026-08-16T15:59:17Z'
- id: O-2fb51216bba7818d
  relation: complements
  reason: Both classified under agent_general, from arXiv:2103.13415 and arXiv:2105.10327
    respectively — different papers addressing the same problem area.
  linked_at: '2026-08-16T15:59:17Z'
---

## Observation

### Declaration
Integrated positional encoding (IPE) automatically attenuates high-frequency Fourier features based on the Gaussian variance of the sampled frustum, making the maximum frequency hyperparameter L effectively unnecessary; L can be set to an arbitrarily large fixed value without impacting performance, unlike standard positional encoding whose performance collapses if L is too large.

### Narrative
Standard positional encoding (PE) used in NeRF maps a 3D coordinate to a vector of sine and cosine functions at logarithmically spaced frequencies up to a maximum frequency hyperparameter L. This L effectively determines the highest spatial frequency the network can represent, and choosing it requires careful tuning: too small and fine details are lost, too large and the network overfits or produces aliasing because high-frequency features are applied uniformly regardless of the rendering resolution. Mip-NeRF replaces point-sampled PE with Integrated Positional Encoding (IPE), which computes the expected sine and cosine of a Gaussian distribution approximating the conical frustum. The key property is that the expectation of sin(x) for a Gaussian variable x ~ N(μ, σ²) is sin(μ)·exp(−½σ²), and similarly for cosine. This multiplicative Gaussian attenuation factor means that the magnitude of each frequency component is scaled down exponentially with the variance of the frustum. When the frustum is small (fine scale), σ² is near zero, and the encoding is nearly identical to ordinary PE. When the frustum is large (coarse scale), the larger variance heavily attenuates high frequencies, effectively truncating the encoding bandwidth adaptively. As a result, the maximum frequency hyperparameter L becomes irrelevant: one can set L to an arbitrarily large value (e.g., 16, high enough that the finest frusums under the camera geometry produce negligible encoding beyond that) and the IPE will automatically suppress any frequency whose period is smaller than the frustum's spatial extent. The authors demonstrate this by sweeping L from 2 to 14 on a single scene (the LEGO bulldozer) and reporting test-set PSNR. For standard NeRF, PSNR peaks around L=8–10 and then degrades sharply as L increases beyond 10 because the model overfits to high-frequency noise and fails to generalize. For mip-NeRF, PSNR rises monotonically and plateaus after L=10, remaining flat up to L=14, indicating that the IPE's built-in attenuation prevents the large L values from introducing spurious high-frequency content. This property simplifies hyperparameter tuning: instead of searching for the optimal L for each scene or dataset, practitioners can fix L to a large constant and let the encoding adapt based on camera geometry. The authors also note that in all experiments they simply set L=16, a value that causes the last IPE dimension to be smaller than numerical epsilon for all frustums encountered during training, making the parameter essentially inactive.

### Causal explanation (r)
IPE’s Gaussian attenuation factor exp(−½σ²) acts as a continuous low-pass filter whose cutoff frequency is controlled by the frustum’s variance, so any frequency with a period smaller than the frustum’s spatial extent is naturally suppressed, rendering a global maximum frequency hyperparameter redundant.

confidence (μ_r): high
depth: deep
depth rationale: The explanation identifies the mathematical mechanism (Gaussian attenuation proportional to variance) that provides the adaptive bandwidth control, not just restating that L is unnecessary.

### Applicability context
summary: Eliminates the need to tune the positional encoding frequency hyperparameter L when training neural radiance fields, enabling practitioners to set L to a large constant (e.g., 16) and rely on IPE's built-in scale-dependent attenuation to prevent overfitting and aliasing, with PSNR remaining flat across a wide range of L on standard benchmarks.
modality: images (RGB renders)
scale: single scene (lego) for the L sweep; tested across all 8 Blender scenes with L=16
constraint: requires cone-casting or equivalent frustum modeling to provide the variance; the benefit is realized only when IPE is used instead of PE; the choice of L must be high enough to cover the finest frustum’s variance, but any larger value works equally well

### Extraction rationale
The author explicitly states that IPE removes the need for tuning L and provides a side-by-side sweep (Figure 7) showing NeRF’s PSNR collapse vs mip-NeRF’s flat performance as L grows; the causal mechanism is derived in the paper from the closed-form IPE expectation.

## Why It Matters

Claim type: property. Confidence: high. Eliminates the need to tune the positional encoding frequency hyperparameter L when training neural radiance fields, enabling practitioners to set L to a large constant (e.g., 16) and rely on IPE's built-in scale-dependent attenuation to prevent overfitting and aliasing, with PSNR remaining flat across a wide range of L on standard benchmarks. Applies under: requires cone-casting or equivalent frustum modeling to provide the variance; the benefit is realized only when IPE is used instead of PE; the choice of L must be high enough to cover the finest frustum’s variance, but any larger value works equally well

## Evidence

### Source
arXiv:2103.13415 | 2103.13415__eae48ad6 | group: t3 | section: supplement

### Verbatim quote
"Because the ‘soft’ maximum frequency of IPE features is determined entirely by the geometry and intrinsics of the camera, IPE features do not depend on the L hyperparameter, and so using IPE features removes the need for tuning L. This is because in PE the L parameter determines where the high frequencies in the PE are truncated, but in IPE those high frequencies are naturally attenuated by the size of the multivariate Gaussian used as input to the encoding: the smaller the Gaussian, the more high frequencies will be retained. [...] In NeRF, performance decreases due to overfitting for large values of L, but in mip-NeRF this parameter is effectively removed from tuning — it can just be set to a large value and forgotten, because IPE features ‘tune’ their own frequencies automatically."

### Classification
domain: agent_general
arXiv: cs.CV
Wikipedia: Computer_vision
ACM CCS: Computing methodologies → Artificial intelligence → Computer vision
CLC: TP391.4

### Keywords
neural radiance fields, integrated positional encoding, hyperparameter tuning, Fourier features, anti-aliasing, multiscale representation, scale adaptation, positional encoding

```

### 经验 ID: `O-e1e60eeb00f4d99c`

```markdown
---
id: O-e1e60eeb00f4d99c
created_at: '2026-08-16T15:52:57Z'
summary: '[medium] Introduces cone tracing and integrated positional encoding (IPE)
  to extend NeRF to multiscale anti‑aliased rendering, achieving a 60% error reduction
  on a new multiscale Blender benchmark (average error 0.0114 vs NeRF 0.0288) and
  a 17% reduction on the original Blender dataset (0.0161 vs 0.0194), while reducing
  parameters by 50% and training time by 7%.'
memory_type: procedural
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l1
related_observations:
- id: O-440e30e45bee7618
  relation: complements
  reason: 'Both extracted from arXiv:2103.13415: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:01Z'
- id: O-70c1cf83d5e4b74e
  relation: complements
  reason: 'Both extracted from arXiv:2103.13415: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:01Z'
---

## Observation

### Narrative
Neural radiance fields (NeRF) represent a 3D scene as a continuous volumetric function parameterised by a multilayer perceptron (MLP) that maps a 5D coordinate (position and direction) to density and view‑dependent colour. NeRF renders a pixel by casting a single ray, sampling points along it, applying a positional encoding (PE) that lifts coordinates to a high‑frequency Fourier basis (sine and cosine of scaled coordinates up to a maximum frequency hyperparameter L), and integrating the MLP outputs via numerical quadrature. This point‑sampled approach is intrinsically aliasing‑prone when training images observe scene content at different resolutions; for example, if the camera moves closer, the same pixel foot‑print covers a smaller region, but the positional encoding of merely the ray’s centre fails to capture the scale variation, leading to blurry close‑ups and jagged distant views. The straightforward remedy of supersampling (shooting multiple jittered rays per pixel) would multiply the already high cost of NeRF (hundreds of MLP queries per ray). Inspired by mipmapping, a classical graphics technique that pre‑filters a texture at multiple discrete scales and selects the appropriate level during rendering, mip‑NeRF instead performs continuous pre‑filtering. The core idea is to cast a cone through each pixel (the cone’s radius at the image plane matches the pixel’s variance) and divide the cone into conical frustums between successive depth samples. Each frustum is featurized by an integrated positional encoding (IPE): the frustum is approximated as a 3D Gaussian with mean and covariance derived in closed form (Equations 7–8), and the expectation of the sinusoidal PE over that Gaussian is computed. This expectation yields an encoding where frequencies whose period is larger than the frustum’s width are preserved, while those with smaller periods are dampened by a Gaussian factor, effectively creating an anti‑aliased encoding that automatically decreases the effective frequency as the frustum becomes larger (i.e., farther from the camera). Because the IPE inherently encodes scale, NeRF’s hierarchical two‑MLP strategy (one “coarse” and one “fine” network, each operating at a fixed effective scale) can be collapsed into a single multiscale MLP, halving the parameter count. Training minimizes a loss composed of a coarse rendering (from stratified samples) and a fine rendering (from samples drawn from the coarse density distribution), with the coarse loss weighted by λ=0.1. On a newly constructed multiscale Blender benchmark—images downsampled by 2×, 4×, and 8× combined with the originals—mip‑NeRF reduces the average error metric (geometric mean of MSE, 1−SSIM, LPIPS) by 60% relative to NeRF (0.0114 vs 0.0288). For example, at the 1/8‑resolution scale, mip‑NeRF’s PSNR is 35.60 compared to NeRF’s 22.53. On the original single‑scale Blender dataset, mip‑NeRF achieves a 17% error reduction (PSNR 33.09 vs 31.74 for JaxNeRF), is about 7% faster, and uses only 612 K parameters (NeRF: 1.19 M). Moreover, mip‑NeRF roughly matches the accuracy of an oracle supersampled NeRF (trained only on full‑resolution images and post‑rendered with downsampling) while being 22× faster to evaluate. The approach is limited to perspective cameras whose frustum can be approximated by a Gaussian; extreme wide‑angle lenses may violate the approximation, and on forward‑facing real scenes already handled with Normalized Device Coordinates the benefit is marginal.

### Practice environment
Multiscale Blender Dataset: 8 scenes (chair, drums, ficus, hotdog, lego, materials, mic, ship), each with 100 training views at 800×800 plus 2×, 4×, 8× downsampled versions. Single‑scale Blender Dataset: same scenes, no downsampling. Baselines: NeRF (original and JAX reimplementation), NeRF + Area Loss, + Centered Pixels, + Misc. Metrics: PSNR, SSIM, LPIPS; average error = geometric mean of MSE, √(1−SSIM), LPIPS. Hardware: 32‑core TPU v2. Training: Adam, 1 M iterations, batch 4096, lr log‑annealed 5e‑4→5e‑6 with 2500‑step warmup. Mip‑NeRF: 128 coarse + 128 fine samples, λ=0.1, α=0.01; PE degree L=16 (eliminated as hyperparameter).

### Practice trace
1. Action: Replaced ray‑casting and point‑wise PE with cone tracing and integrated positional encoding (IPE), where for each pixel a cone is traced and the expected PE over a Gaussian approximating the conical frustum is computed.
   Feedback: When ablating the single MLP (i.e., retaining two MLPs) but keeping IPE, the multiscale error is 0.0115, nearly matching the full model (0.0114), confirming that IPE is the primary driver of the anti‑aliasing improvement.
2. Action: Merged NeRF’s separate coarse and fine MLPs into a single multiscale MLP that can handle all scales because IPE already encodes the frustum size.
   Feedback: Model size halved to 612 K parameters, training time reduced to 2.84 h (7 % faster), while maintaining comparable accuracy (error 0.0114 vs 0.0115 for two MLPs) and enabling simpler sampling.

### Task context
modality: 3D scenes (RGB images with known camera poses, synthetic Blender data)
scale: 8 scenes, 100 training views per scene; multiscale dataset adds 2×, 4×, 8× downsampled images; 1 M Adam iterations; 612 K parameters (single MLP)
constraint: Conical frustum Gaussian approximation breaks under extreme perspective (fisheye, macro); IPE benefits are diminished on forward‑facing scenes using NDC coordinates; only synthetic scenes tested.

### Extraction rationale
Method description from introduction and Section 3 (cone tracing, IPE derivation, single MLP). Quantitative outcomes from introduction and confirmed by Table 1 and Table 2. The two action‑feedback pairs reflect the core dual contribution: IPE for anti‑aliasing and the single MLP for efficiency, with numerical evidence from the ablations.

## Why It Matters

Granularity: medium. Scale: 8 scenes, 100 training views per scene; multiscale dataset adds 2×, 4×, 8× downsampled images; 1 M Adam iterations; 612 K parameters (single MLP) Applies under: Conical frustum Gaussian approximation breaks under extreme perspective (fisheye, macro); IPE benefits are diminished on forward‑facing scenes using NDC coordinates; only synthetic scenes tested.

## Evidence

### Source
arXiv:2103.13415 | 2103.13415__eae48ad6 | group: t3 | section: introduction

### Verbatim quote
"Our solution, which we call mip‑NeRF (multum in parvo NeRF, as in “mipmap”), extends NeRF to simultaneously represent the prefiltered radiance field for a continuous space of scales. [...] On a challenging multiresolution benchmark we present, mip‑NeRF is able to reduce error rates relative to NeRF by 60% on average."

### Classification
domain: agent_domain_app
arXiv: cs.CV
Wikipedia: Computer_vision

### Keywords
neural radiance fields, view synthesis, anti-aliasing, multiscale representation, integrated positional encoding, cone tracing, volume rendering, 3D scene representation, mipmapping, positional encoding

```

### 经验 ID: `O-92d8e9bf63578683`

```markdown
---
id: O-92d8e9bf63578683
created_at: '2026-08-16T15:52:57Z'
summary: '[conditional] Including a history of previous screen observations and actions
  improves performance of behavioral cloning agents for Android device control, with
  consistent gains across standard and out-of-distribution evaluation settings.'
memory_type: semantic
scope: global
source:
  type: turn
  agent: experience-bank
  session_id: offline-import-l2
related_observations:
- id: O-ef554189c821d9e1
  relation: complements
  reason: 'Both extracted from arXiv:2307.10088: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:33Z'
- id: O-f4036c2c87892960
  relation: complements
  reason: 'Both extracted from arXiv:2307.10088: the L1 practice record and the L2
    inductive claim describe the same work from complementary angles (what was done
    vs. what it generalizes to).'
  linked_at: '2026-08-16T15:58:35Z'
---

## Observation

### Declaration
Including a history of previous screen observations and actions improves performance of behavioral cloning agents for Android device control, with consistent gains across standard and out-of-distribution evaluation settings.

### Narrative
Behavioral cloning (BC) is a standard approach for training device-control agents by imitating human demonstrations. The AITW paper implements a Transformer-based BC model and compares two input configurations: BC-single, which only receives the current screen’s UI features and the task instruction, and BC-history, which additionally ingests a stacked history of the two most recent screen observations and the corresponding actions (touch/lift coordinates and action types). The screen history is embedded by adding the positional encoding of the <x,y> touch and lift points to the element encodings, thereby informing the model about what gestures have been performed and where. The evaluation spans multiple splits—standard in-distribution and out-of-distribution scenarios (unseen Android version, unseen subject/verb templates, unseen apps/websites). Across every split, BC-history achieves a higher partial match score than BC-single. On the standard test set, the gain is from 68.7 to 73.1; on unseen versions, 59.2 to 63.2; on unseen subjects, 64.2 to 68.5; on unseen verbs, 66.4 to 70.4; and on unseen domains, 52.2 to 59.7. The authors themselves note in the architecture description that “we found that including action history improves performance.” This consistent improvement suggests that even a short history buffer provides essential context for the agent: it disambiguates screens that look similar at different stages of a task (e.g., the search results page before and after applying a filter) and helps avoid redundant actions (e.g., re-opening an already opened app). The history embedding also directly carries information about the gesture’s start and end positions, which may assist in predicting subsequent gestures that build on previous interactions, such as choosing a scroll direction after a tap. The gain is modest but meaningful, averaging about 5 percentage points across splits. Notably, the improvement holds even for out-of-distribution domains, indicating that the benefit of history is not merely an overfitting artefact. The BC model was trained on the full AITW dataset (over 700k episodes) using a 4-layer Transformer, and the history variant’s superiority persisted after hyperparameter tuning on validation sets. No causal explanation is offered by the authors beyond the empirical finding, but the architectural design—adding positional embeddings of prior actions to the screen feature vectors—is a direct method to inject temporal state. This experience suggests that for device-control BC agents, incorporating a short interaction history is a reliable, low-cost enhancement. Practitioners should consider including at least the last two steps in their input representation. However, the optimal history length remains an open question; longer histories might improve further but could also increase the complexity. The study’s broad experimental scope (multiple Android versions, hundreds of apps) makes this a trustworthy, generalizable lesson for building robust Android automation agents.

### Applicability context
summary: Enables improved behavioral cloning accuracy for Android device control by incorporating a short history of previous screen observations and actions, yielding +4.4% partial match score on the standard test set and consistent gains across out-of-distribution splits.
modality: screen features and action coordinates
scale: 715k training episodes, 5 dataset splits, 2 prior steps used
constraint: history length fixed at 2 steps; longer histories not tested; model architecture uses Transformer with specific embedding of prior actions

### Extraction rationale
The author reports empirical superiority of BC-history over BC-single across all splits, and explicitly states that history improves performance. No causal explanation is given, only the architectural description; therefore r is null.

## Why It Matters

Claim type: conditional. Confidence: high. Enables improved behavioral cloning accuracy for Android device control by incorporating a short history of previous screen observations and actions, yielding +4.4% partial match score on the standard test set and consistent gains across out-of-distribution splits. Applies under: history length fixed at 2 steps; longer histories not tested; model architecture uses Transformer with specific embedding of prior actions

## Evidence

### Source
arXiv:2307.10088 | 2307.10088__793a79f6 | group: t2 | section: method

### Verbatim quote
We embed the <x,y> positions of the touch and lift actions, which are added to the element encoding, using a dummy value for non-gesture actions. We found that including action history improves performance.

### Classification
domain: agent_web_gui
arXiv: cs.LG
Wikipedia: Supervised_learning
ACM CCS: Computing methodologies → Machine learning → Supervised learning → Classification and regression
CLC: TP181

### Keywords
behavioral cloning, action history, device control, Android, agent performance, step context, Transformer, generalization, AITW dataset

```

---
