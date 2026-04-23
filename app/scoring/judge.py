"""
LLM-as-judge scorer for open-ended categories (reasoning, summarization).

Why use an LLM to judge LLMs:
  For open-ended tasks, exact-match against an "expected_answer" is too
  brittle — there are many correct ways to explain the 3-switches puzzle
  or summarize a paragraph. A human-quality judgment is needed, and at
  benchmark scale (42 runs × 2 open categories = 84 judgments per sweep)
  that's only practical with an LLM. Claude Sonnet is used here because
  a stronger model than the ones under test should do the judging — the
  judge should not share systematic weaknesses with the models it's scoring.

Rubrics are category-specific:
  REASONING — correctness of the final answer, then reasoning quality.
  SUMMARIZATION — faithfulness (no fabrications), completeness (key points
  preserved), conciseness (appropriate length).

Output format:
  The judge is prompted to return a single integer score 1-5 plus a short
  rationale. We parse that, normalize to 0.0-1.0, and return with the
  rationale as notes.
"""

import os
import re
from anthropic import Anthropic
from dotenv import load_dotenv


load_dotenv()

JUDGE_MODEL = "claude-sonnet-4-5"


REASONING_RUBRIC = """You are evaluating whether a small language model answered a reasoning question correctly.

QUESTION:
{prompt}

EXPECTED ANSWER (reference):
{expected}

MODEL'S ANSWER:
{response}

Score the model's answer on a scale of 1 to 5:
5 = Fully correct. Final answer matches the expected answer, and reasoning (if any) is sound.
4 = Correct final answer, but reasoning has minor issues or is incomplete.
3 = Partially correct. The answer is close but has a meaningful error, OR the answer is right for the wrong reason.
2 = Incorrect, but shows some relevant thinking.
1 = Completely wrong or nonsensical.

Respond in exactly this format, no extra text:
SCORE: <1-5>
REASON: <one short sentence>
"""


SUMMARIZATION_RUBRIC = """You are evaluating a summary produced by a small language model.

SOURCE TEXT:
{source}

INSTRUCTION GIVEN TO THE MODEL:
{prompt}

MODEL'S SUMMARY:
{response}

Score the summary on a scale of 1 to 5 based on three criteria:
- FAITHFULNESS: Does it avoid fabricating information not in the source?
- COMPLETENESS: Does it capture the key points?
- CONCISENESS: Does it respect the length instruction?

5 = Excellent on all three.
4 = Strong, minor issue in one area.
3 = Acceptable but has a clear weakness in at least one area.
2 = Significant problem (e.g., fabricated claim, missing key point, way too long/short).
1 = Unusable (major hallucination, or fails to summarize).

Respond in exactly this format, no extra text:
SCORE: <1-5>
REASON: <one short sentence identifying the main strength or weakness>
"""


def _parse_judge_response(text: str) -> tuple[int, str]:
    """
    Extract SCORE and REASON from the judge's structured response.

    Defensive parsing — the judge occasionally ignores format instructions
    and adds prose before SCORE:. We search anywhere in the text.
    """
    score_match = re.search(r"SCORE:\s*([1-5])", text)
    reason_match = re.search(r"REASON:\s*(.+?)(?:\n|$)", text, re.DOTALL)

    if not score_match:
        raise ValueError(f"Judge did not return a valid score. Got: {text[:200]}")

    score = int(score_match.group(1))
    reason = reason_match.group(1).strip() if reason_match else "(no reason given)"
    return score, reason


def _call_judge(prompt: str) -> str:
    """Single Anthropic API call with the rubric-filled prompt."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not found in environment. Check .env.")

    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def score_reasoning(prompt: str, expected_answer: str, response: str) -> tuple[float, str]:
    """
    Judge a reasoning response against the expected answer.

    Returns (score in [0.0, 1.0], human-readable notes).
    """
    judge_prompt = REASONING_RUBRIC.format(
        prompt=prompt,
        expected=expected_answer,
        response=response,
    )
    try:
        raw_judgment = _call_judge(judge_prompt)
        score_1_5, reason = _parse_judge_response(raw_judgment)
    except Exception as exc:
        return 0.0, f"Judge failed: {type(exc).__name__}: {exc}"

    # Normalize 1-5 → 0.0-1.0
    # 1 → 0.0, 2 → 0.25, 3 → 0.5, 4 → 0.75, 5 → 1.0
    normalized = (score_1_5 - 1) / 4.0
    return normalized, f"Judge score {score_1_5}/5. {reason}"


def score_summarization(prompt: str, source_text: str, response: str) -> tuple[float, str]:
    """
    Judge a summary against its source text and instruction.

    Returns (score in [0.0, 1.0], human-readable notes).
    """
    judge_prompt = SUMMARIZATION_RUBRIC.format(
        source=source_text,
        prompt=prompt,
        response=response,
    )
    try:
        raw_judgment = _call_judge(judge_prompt)
        score_1_5, reason = _parse_judge_response(raw_judgment)
    except Exception as exc:
        return 0.0, f"Judge failed: {type(exc).__name__}: {exc}"

    normalized = (score_1_5 - 1) / 4.0
    return normalized, f"Judge score {score_1_5}/5. {reason}"
