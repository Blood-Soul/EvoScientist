#!/usr/bin/env python3
"""Batch generate a 10-query experience answer report.

For each of the 10 daily-development queries:
1. Search the EvoMemory store and keep all hits above a score threshold (>=25).
2. Fetch the full structure of each hit (untruncated).
3. Call the LLM to synthesize an answer and a concise reasoning chain based ONLY on those hits.
4. Format the final output: Query, the Full Experience records, Answer, and Reasoning.

Writes to: docs/report/10_query_experience_showcase.md
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure we're running from the project root in WSL
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from langchain_core.messages import HumanMessage, SystemMessage
from EvoScientist import paths
from EvoScientist.config.settings import load_config
from EvoScientist.llm import get_chat_model
from EvoScientist.memory import read_observation_file, search_observation_files

# The 10 queries covering the requested topics
QUERIES = [
    {
        "title": "代码评测指标的选择",
        "search_q": "code generation functional correctness test cases metric BLEU",
        "display_q": "评估代码生成模型功能正确性时，用 BLEU 分数靠谱吗？应该使用什么指标？"
    },
    {
        "title": "RAG 检索增强的长文本切分",
        "search_q": "retrieval augmented generation chunking context length pipeline",
        "display_q": "在构建 RAG 系统时，长文本应该如何进行切分（Chunking）？切片大小对生成效果有什么影响？"
    },
    {
        "title": "端侧多模态模型部署瓶颈",
        "search_q": "on-device deployment memory latency bottleneck quantization mobile",
        "display_q": "将多模态模型部署到端侧（如手机）时，最主要的性能瓶颈是什么？有哪些工程优化手段？"
    },
    {
        "title": "Multi-agent 多智能体协作分工",
        "search_q": "multi-agent collaboration team topology role assignment",
        "display_q": "设计 Multi-agent 系统时，应该如何划分智能体角色？有哪些被验证有效的协作拓扑结构？"
    },
    {
        "title": "纯视觉 GUI 自动化 Agent",
        "search_q": "GUI agent vision only screenshot grounding click action",
        "display_q": "开发 GUI Agent 时，纯视觉方案（仅依赖截图和坐标点击）可行吗？关键的前置条件和难点是什么？"
    },
    {
        "title": "指令微调的数据质量与数量",
        "search_q": "instruction tuning data quality versus quantity high quality subset fine-tuning",
        "display_q": "在进行指令微调（Instruction Tuning）时，数据质量和数据数量哪个更重要？少量高质量数据足够吗？"
    },
    {
        "title": "LLM 处理超长上下文的机制问题",
        "search_q": "long context window truncation attention bottleneck extrapolation",
        "display_q": "长文本语言模型（Long-context LLMs）在处理超长输入时，会遇到哪些注意力机制或截断带来的结构性问题？"
    },
    {
        "title": "Coding Agent 的执行失败迭代修复",
        "search_q": "agent execution failure recovery root cause repair debugging loop",
        "display_q": "如何让 Coding Agent 在遇到执行报错时能够有效地进行迭代修复，而不是陷入死循环？"
    },
    {
        "title": "Prompt Engineering 的 Few-shot 示例",
        "search_q": "few-shot prompt example selection variance sensitivity",
        "display_q": "在 Prompt Engineering 中，Few-shot 示例的选择对最终效果影响大吗？有何系统性的方法？"
    },
    {
        "title": "Transformer 位置编码的长度外推",
        "search_q": "positional encoding extrapolation length out of distribution sequence",
        "display_q": "Transformer 架构中，位置编码的选择如何影响模型在超出训练长度（OOD）数据上的泛化能力？"
    }
]

SYSTEM_PROMPT = """你是一个基于经验库进行总结的 AI 研究助手。
请**严格只根据提供的经验上下文**来回答用户的问题。如果上下文没有提及相关内容，请明确指出“经验库未覆盖”。

请按照以下结构使用中文输出：
## 回答
<基于经验的直接、综合性回答>

## 思维链 (Reasoning)
<简要的推理链：说明你是如何从提供的经验中推导出上述回答的，标注清楚引用了哪一条经验（如 O-xxxx）>
"""


async def process_query(model, q_obj: dict) -> str:
    print(f"\nProcessing: {q_obj['title']}...")

    # 1. Search observations
    hits = search_observation_files(
        memory_dir=paths.MEMORIES_DIR,
        project_id="",
        query=q_obj["search_q"],
        limit=20  # Fetch a bunch, then filter by score threshold
    )

    # Keep everything with a score > 20
    valid_hits = [h for h in hits if h.get("score", 0) > 20]
    if not valid_hits:
        print("  -> No hits > 20 score.")
        return f"# {q_obj['title']}\n\n**Query:** {q_obj['display_q']}\n\n*经验库中未找到足够相关分数 (>20) 的经验。*\n\n---\n"

    # Limit to top 5 even if many pass the threshold to prevent context blow-up
    valid_hits = valid_hits[:5]
    print(f"  -> Found {len(valid_hits)} valid experiences.")

    # 2. Read full content
    experiences_text = []
    experiences_raw_display = []

    for h in valid_hits:
        doc = read_observation_file(
            memory_dir=paths.MEMORIES_DIR,
            project_id="",
            observation_id=h["observation_id"]
        )
        full_text = doc["text"]

        # Build prompt context
        experiences_text.append(f"### Experience {h['observation_id']}\n{full_text}")

        # Build display text
        experiences_raw_display.append(f"### 经验 ID: `{h['observation_id']}`\n\n```markdown\n{full_text}\n```")

    merged_context = "\n\n".join(experiences_text)

    user_prompt = f"问题：{q_obj['display_q']}\n\n上下文经验：\n{merged_context}"

    # 3. Call LLM
    print(f"  -> Calling LLM ({len(user_prompt)} chars context)...")
    try:
        resp = await asyncio.wait_for(
            model.ainvoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ]),
            timeout=180
        )
        answer = getattr(resp, "content", str(resp))
        print("  -> LLM returned successfully.")
    except Exception as e:
        print(f"  -> LLM generation FAILED: {e}")
        answer = f"*LLM 生成失败: {e}*"

    # 4. Format Output
    output = f"# {q_obj['title']}\n\n"
    output += f"**用户提问：** {q_obj['display_q']}\n\n"
    output += f"**检索关键词：** `{q_obj['search_q']}`\n\n"
    output += f"{answer}\n\n"
    output += f"## 召回的原始经验全貌 ({len(valid_hits)} 条)\n\n"
    output += "\n\n".join(experiences_raw_display)
    output += "\n\n---\n"

    return output


async def main():
    cfg = load_config()
    print(f"Using Model: {cfg.provider} / {cfg.model}")
    model = get_chat_model(model=cfg.model, provider=cfg.provider)
    # Bind a generous output token budget so answers don't get truncated
    if hasattr(model, "bind"):
        model = model.bind(max_tokens=4000)

    out_dir = PROJECT_ROOT / "docs" / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "10_query_experience_showcase.md"

    report_parts = ["# 经验库检索与问答汇报 (10 Queries)\n\n"]
    report_parts.append(
        "> 本报告演示了从 374 条结构化论文经验中检索相关条目，**全量展示其原始字段结构**，"
        "并由大模型基于这些上下文提供综合回答与思维链的端到端能力。\n\n---\n"
    )

    for q_obj in QUERIES:
        part = await process_query(model, q_obj)
        report_parts.append(part)

    out_file.write_text("\n".join(report_parts), encoding="utf-8")
    print(f"\nDone! Report written to {out_file}")

if __name__ == "__main__":
    asyncio.run(main())
