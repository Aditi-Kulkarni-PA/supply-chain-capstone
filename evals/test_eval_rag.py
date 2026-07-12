"""
Eval: RAG quality — RAGAS faithfulness + answer relevancy.

Runs as part of the default eval suite (no --ragas flag needed).

The recommendation agent's own instruction/query never varies, so varying
*that* wouldn't give RAGAS a meaningful spread of samples. Instead: the agent's
OUTPUT naturally cites several distinct SLA topics — one recommendation might
reference weather policy, another partner benchmarks, another distance rules.
We sample a diverse subset of recommended_actions (up to 2 per category:
quick-win / short-term / long-term), issue a SEPARATE retrieval query per
topic (bypassing the tool-output summarizer via query_override), and build one
RAGAS sample per topic. This gives RAGAS a real multi-sample dataset — n > 1 —
instead of one blended sample, so the reported score is a genuine mean across
distinct SLA sections rather than a single noisy LLM judgment.
"""

import re
import pytest
from agents import Runner

from conftest import EVAL_DB
from eval_config import MIN_FAITHFULNESS, MIN_ANSWER_RELEVANCY
from judge import judge_output, get_tool_output

from delivery_agents import recommendation_agent, RecommendedActionsList
from tools.rag_knowledge import retrieve_sla_context
import sys, tools  # noqa: E401
ra_module = sys.modules["tools.recommend_actions"]

_MAX_PER_CATEGORY = 2


def _select_diverse_actions(output: RecommendedActionsList) -> list:
    """Pick up to _MAX_PER_CATEGORY actions per category, skipping empty sla_reference."""
    by_category: dict[str, list] = {}
    for action in output.recommended_actions:
        if not action.sla_reference.strip():
            continue
        by_category.setdefault(action.category, []).append(action)

    selected = []
    for category, actions in by_category.items():
        selected.extend(actions[:_MAX_PER_CATEGORY])
    return selected


def _parse_retrieved_sections(context_str: str) -> list[str]:
    """Split retrieve_sla_context()'s formatted block into individual chunk strings."""
    sections = re.split(r"\n### Retrieved Section \d+\n", context_str)
    return [s.strip() for s in sections[1:] if s.strip()]  # sections[0] is the header line


async def test_rag_faithfulness_and_relevancy(seeded_eval_db):
    """Run recommendation agent once, then score faithfulness + answer_relevancy
    per-topic across a diverse sample of its recommended_actions."""
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import Faithfulness, AnswerRelevancy
        from ragas.llms import LangchainLLMWrapper
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        pytest.skip(f"RAGAS dependencies not installed: {e}. Run: uv sync --extra eval")

    # Patch recommend_actions to read from the eval DB
    original_db = ra_module._DB_PATH
    ra_module._DB_PATH = EVAL_DB
    try:
        result = await Runner.run(
            recommendation_agent,
            "Recommend ways to optimize delivery timelines and reduce delays",
        )
    finally:
        ra_module._DB_PATH = original_db

    output = result.final_output
    assert output is not None, "Recommendation agent returned no output"

    tool_out = get_tool_output(result, "recommend_actions")
    if "--- SLA Knowledge Context" not in tool_out:
        pytest.skip("Could not find SLA context block in recommend_actions output")

    selected = _select_diverse_actions(output)
    if not selected:
        pytest.skip("No recommended_actions with non-empty sla_reference found")

    # One retrieval + one RAGAS sample per topic, each independently retrieved.
    samples = []
    per_topic_rows = []
    for action in selected:
        topic_query = f"What does the SLA say about {action.dimension} — {action.action}?"
        context_block = retrieve_sla_context(tool_output="", query_override=topic_query)
        chunks = _parse_retrieved_sections(context_block)
        if not chunks:
            continue
        samples.append(SingleTurnSample(
            user_input=topic_query,
            retrieved_contexts=chunks,
            response=action.sla_reference,
        ))
        per_topic_rows.append((action.category, action.dimension, action.action, topic_query))

    if not samples:
        pytest.skip("No SLA chunks retrieved for any sampled topic")

    dataset = EvaluationDataset(samples=samples)
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini", temperature=0))

    scores = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(llm=llm), AnswerRelevancy(llm=llm)],
    )
    score_df = scores.to_pandas()
    faithfulness = float(score_df["faithfulness"].mean())
    answer_rel   = float(score_df["answer_relevancy"].mean())

    assert faithfulness >= MIN_FAITHFULNESS, (
        f"RAGAS mean faithfulness {faithfulness:.3f} < threshold {MIN_FAITHFULNESS} "
        f"(n={len(samples)} topics)"
    )
    assert answer_rel >= MIN_ANSWER_RELEVANCY, (
        f"RAGAS mean answer_relevancy {answer_rel:.3f} < threshold {MIN_ANSWER_RELEVANCY} "
        f"(n={len(samples)} topics)"
    )

    # Per-topic breakdown for the markdown report — shows whether retrieval quality
    # varies by SLA section rather than hiding it behind one blended score.
    breakdown_lines = [
        f"Sampled {len(samples)} distinct SLA topics across "
        f"{len({c for c, *_ in per_topic_rows})} categories:\n",
        "| Category | Dimension | Action | Faithfulness | Relevancy |",
        "|---|---|---|---|---|",
    ]
    for i, (category, dimension, action_name, _q) in enumerate(per_topic_rows):
        row = score_df.iloc[i]
        breakdown_lines.append(
            f"| {category} | {dimension} | {action_name[:50]} "
            f"| {row['faithfulness']:.3f} | {row['answer_relevancy']:.3f} |"
        )
    response_text = "\n".join(breakdown_lines)

    judge_output(
        "RAG — SLA Knowledge Retrieval",
        response_text,
        context=(
            "Relevance: does the per-topic breakdown show consistent SLA grounding across "
            "categories? Faithfulness: are sla_reference citations traceable to their own "
            "independently-retrieved SLA context? Safety: no hallucinated SLA thresholds."
        ),
        agent_input="Recommend ways to optimize delivery timelines and reduce delays",
        ragas_scores={"faithfulness": faithfulness, "answer_relevancy": answer_rel},
    )
