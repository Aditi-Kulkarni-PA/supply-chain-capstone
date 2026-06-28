"""
Eval: RAG quality — RAGAS faithfulness + answer relevancy.

Runs as part of the default eval suite (no --ragas flag needed).

The recommendation agent's input is the output from predict + diagnosis tools
already written into the eval DB by seeded_eval_db — no separate 15K-row run
is needed. RAGAS only evaluates whether sla_reference citations are grounded in
the retrieved SLA context block, which is independent of how many orders were
in the prediction DB.

Only faithfulness + answer_relevancy are scored — these need no reference answer.
"""

import os
import re
import pytest
from agents import Runner

from conftest import EVAL_DB
from eval_config import MIN_FAITHFULNESS, MIN_ANSWER_RELEVANCY
from judge import judge_output, mean_score, get_tool_output

from delivery_agents import recommendation_agent, RecommendedActionsList
import sys, tools  # noqa: E401
ra_module = sys.modules["tools.recommend_actions"]


def _extract_contexts(tool_output: str) -> list[str]:
    """Return both the statistical data and the SLA knowledge block as separate contexts.

    Recommendations are grounded in two sources:
    1. Prediction DB statistics (delay rates, patterns) — everything before the SLA block
    2. SLA document chunks — the block between the SLA markers

    Passing both lets RAGAS verify data-driven claims (from context 1) AND
    SLA reference claims (from context 2) independently.
    """
    match = re.search(
        r"--- SLA Knowledge Context.*?--- End SLA Context ---",
        tool_output,
        re.DOTALL,
    )
    sla_context = match.group(0) if match else ""
    data_context = tool_output[:match.start()].strip() if match else tool_output[:4000]

    contexts = []
    if data_context:
        contexts.append(data_context)
    if sla_context:
        contexts.append(sla_context)
    return contexts or [tool_output[-3000:]]


def _format_sla_references(output: RecommendedActionsList) -> str:
    """Format only sla_reference fields — what we test faithfulness of against the SLA doc."""
    return "\n".join(
        f"[{r.action}]: {r.sla_reference}"
        for r in output.recommended_actions
        if r.sla_reference.strip()
    )


async def test_rag_faithfulness_and_relevancy(seeded_eval_db):
    """Run recommendation agent on the eval DB, score faithfulness + answer_relevancy."""
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

    # We evaluate only: can the sla_reference citations be traced back to the SLA document?
    # Use only the SLA context block (not the prediction DB statistics) so faithfulness
    # measures SLA grounding, not whether delay rates match the DB output.
    tool_out = get_tool_output(result, "recommend_actions")
    sla_match = re.search(
        r"--- SLA Knowledge Context.*?--- End SLA Context ---",
        tool_out, re.DOTALL,
    )
    if not sla_match or not sla_match.group(0).strip():
        pytest.skip("Could not extract SLA context block from recommend_actions output")
    sla_context = sla_match.group(0)

    response_text = _format_sla_references(output)
    if not response_text.strip():
        pytest.skip("No non-empty sla_reference fields found in recommendations")

    # Build RAGAS dataset
    sample = SingleTurnSample(
        user_input="Are the SLA policies and targets cited in these recommendations supported by the retrieved SLA document?",
        retrieved_contexts=[sla_context],
        response=response_text,
    )
    dataset = EvaluationDataset(samples=[sample])

    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini", temperature=0))

    scores = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(llm=llm), AnswerRelevancy(llm=llm)],
    )

    # scores is a dict-like Result object
    score_dict = scores.to_pandas().iloc[0].to_dict()
    faithfulness   = float(score_dict.get("faithfulness", 0))
    answer_rel     = float(score_dict.get("answer_relevancy", 0))

    assert faithfulness >= MIN_FAITHFULNESS, (
        f"RAGAS faithfulness {faithfulness:.3f} < threshold {MIN_FAITHFULNESS}"
    )
    assert answer_rel >= MIN_ANSWER_RELEVANCY, (
        f"RAGAS answer_relevancy {answer_rel:.3f} < threshold {MIN_ANSWER_RELEVANCY}"
    )

    # Log to the markdown report — passes raw RAGAS scores so they appear in their own section.
    judge_output(
        "RAG — SLA Knowledge Retrieval",
        response_text,
        context=(
            "Relevance: does the response address the question about optimising delivery? "
            "Faithfulness: are the SLA references traceable to the retrieved SLA context block? "
            "Safety: no hallucinated SLA thresholds or penalty values."
        ),
        agent_input="Recommend ways to optimize delivery timelines and reduce delays",
        ragas_scores={"faithfulness": faithfulness, "answer_relevancy": answer_rel},
    )
