"""
Benchmark prompt suite.

Four task categories, mixed difficulty within each. Each prompt ships
with the scoring metadata the runner needs:

  - REASONING: `expected_answer` — a short canonical answer. Scored by an
    LLM judge (Claude) with a strict rubric; exact-match alone is too
    brittle for natural language.

  - SUMMARIZATION: `source_text` — what to summarize. No fixed answer;
    scored by LLM judge on faithfulness + conciseness.

  - EXTRACTION: `schema_name` — which Pydantic schema from app.models to
    enforce via Instructor. `expected_fields` lists the fields that MUST
    be present and correct. Scored deterministically (did Instructor
    validate? are key fields correct?).

  - CODE: `test_cases` — input/output pairs to exec the generated code
    against. Scored deterministically (did the code run? did it pass?).

Keep prompts short and focused. A benchmark suite lives or dies by its
design, not its size; 4–6 prompts per category is plenty to see a clear
signal across models.
"""

from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class BenchmarkPrompt:
    """A single benchmark prompt with scoring metadata."""
    task_id: str
    category: str                          # reasoning | summarization | extraction | code
    difficulty: str                        # easy | medium | hard
    prompt: str
    system: Optional[str] = None

    # Category-specific scoring fields (only one set populated per prompt)
    expected_answer: Optional[str] = None                  # reasoning
    source_text: Optional[str] = None                      # summarization
    schema_name: Optional[str] = None                      # extraction
    expected_fields: dict[str, Any] = field(default_factory=dict)  # extraction
    test_cases: list[dict[str, Any]] = field(default_factory=list) # code


# =============================================================================
# REASONING — arithmetic, logic, common sense
# =============================================================================

REASONING_PROMPTS: list[BenchmarkPrompt] = [
    BenchmarkPrompt(
        task_id="reasoning_01",
        category="reasoning",
        difficulty="easy",
        prompt="If a train leaves Chicago at 3pm traveling 60 mph, and another "
               "leaves New York at 4pm traveling 80 mph toward Chicago, and "
               "the cities are 790 miles apart, at what time will they meet? "
               "Give the answer as a single time (e.g., '7:30pm').",
        expected_answer="9:00pm",
    ),
    BenchmarkPrompt(
        task_id="reasoning_02",
        category="reasoning",
        difficulty="easy",
        prompt="Alice is taller than Bob. Bob is taller than Carol. "
               "Is Carol taller than Alice? Answer only 'yes' or 'no'.",
        expected_answer="no",
    ),
    BenchmarkPrompt(
        task_id="reasoning_03",
        category="reasoning",
        difficulty="medium",
        prompt="A farmer has 17 sheep. All but 9 die. How many sheep are left? "
               "Answer with just the number.",
        expected_answer="9",
    ),
    BenchmarkPrompt(
        task_id="reasoning_04",
        category="reasoning",
        difficulty="medium",
        prompt="You have a 3-gallon jug and a 5-gallon jug. How can you measure "
               "exactly 4 gallons of water? Describe the steps briefly.",
        expected_answer="Fill the 5-gallon jug. Pour from it into the 3-gallon jug "
                        "until the 3-gallon is full, leaving 2 gallons in the 5-gallon. "
                        "Empty the 3-gallon jug. Pour the 2 gallons from the 5-gallon "
                        "into the 3-gallon. Fill the 5-gallon again. Pour from the "
                        "5-gallon into the 3-gallon (which already has 2) until full, "
                        "using 1 gallon. 4 gallons remain in the 5-gallon jug.",
    ),
    BenchmarkPrompt(
        task_id="reasoning_05",
        category="reasoning",
        difficulty="hard",
        prompt="Three switches outside a room control three bulbs inside. "
               "You can flip switches as much as you want, but you can only "
               "enter the room once. How do you determine which switch "
               "controls which bulb? Answer in 2-3 sentences.",
        expected_answer="Turn on switch 1 and leave it on for several minutes. "
                        "Turn it off, then turn on switch 2, and immediately enter "
                        "the room. The bulb that is on is controlled by switch 2, "
                        "the bulb that is off but warm is switch 1, and the bulb "
                        "that is off and cold is switch 3.",
    ),
]


# =============================================================================
# SUMMARIZATION — condense text faithfully
# =============================================================================

SUMMARIZATION_PROMPTS: list[BenchmarkPrompt] = [
    BenchmarkPrompt(
        task_id="summarization_01",
        category="summarization",
        difficulty="easy",
        prompt="Summarize the following in one sentence:\n\n{source}",
        source_text=(
            "The Python programming language was created by Guido van Rossum and "
            "first released in 1991. Van Rossum designed it as a successor to the "
            "ABC language, emphasizing code readability with its notable use of "
            "significant indentation. Python is dynamically typed and garbage-"
            "collected, and supports multiple programming paradigms including "
            "structured, object-oriented, and functional programming. It is often "
            "described as a 'batteries included' language due to its comprehensive "
            "standard library."
        ),
    ),
    BenchmarkPrompt(
        task_id="summarization_02",
        category="summarization",
        difficulty="medium",
        prompt="Write a 2-3 sentence summary of the following:\n\n{source}",
        source_text=(
            "The 2008 financial crisis was triggered by the collapse of the US "
            "housing bubble, which had been inflated by widespread issuance of "
            "subprime mortgages bundled into complex mortgage-backed securities. "
            "When housing prices began to fall in 2006, borrowers defaulted at "
            "unprecedented rates, and the securities backing those mortgages lost "
            "value rapidly. Major financial institutions including Bear Stearns, "
            "Lehman Brothers, and AIG faced insolvency, with Lehman filing for "
            "bankruptcy in September 2008 in the largest such filing in US history. "
            "The US government responded with the Troubled Asset Relief Program "
            "(TARP), committing up to $700 billion to stabilize the financial "
            "system, while the Federal Reserve lowered interest rates to near zero. "
            "The crisis triggered the Great Recession, which saw unemployment in "
            "the US peak at 10% in 2009 and reshaped financial regulation through "
            "the Dodd-Frank Act passed in 2010."
        ),
    ),
    BenchmarkPrompt(
        task_id="summarization_03",
        category="summarization",
        difficulty="hard",
        prompt="Summarize the following technical text in exactly 2 sentences, "
               "preserving the key technical claims:\n\n{source}",
        source_text=(
            "Transformer architectures rely on self-attention mechanisms that "
            "compute pairwise interactions between all tokens in a sequence, "
            "yielding O(n^2) complexity in sequence length. This quadratic scaling "
            "becomes prohibitive for long contexts, prompting research into "
            "sub-quadratic alternatives such as linear attention, state-space "
            "models, and sparse attention patterns. Linear attention approximates "
            "the softmax attention matrix using kernel methods, reducing complexity "
            "to O(n) but often at some cost to modeling fidelity. State-space "
            "models like Mamba replace attention entirely with selective recurrence, "
            "achieving linear scaling and strong long-context performance. Sparse "
            "attention methods, such as those used in Longformer and BigBird, "
            "maintain the quadratic formulation but restrict attention to local "
            "windows or learned global tokens, making the effective cost tractable "
            "for sequences into the tens of thousands of tokens."
        ),
    ),
]


# =============================================================================
# EXTRACTION — structured output via Instructor + Pydantic
# =============================================================================

EXTRACTION_PROMPTS: list[BenchmarkPrompt] = [
    BenchmarkPrompt(
        task_id="extraction_01",
        category="extraction",
        difficulty="easy",
        prompt="Extract information about the person from this text:\n\n"
               "Sarah Chen is a 34-year-old software engineer living in Seattle. "
               "She works at a mid-sized startup focused on climate tech.",
        schema_name="PersonInfo",
        expected_fields={
            "name": "Sarah Chen",
            "age": 34,
            "occupation_contains": "engineer",
            "location_contains": "Seattle",
        },
    ),
    BenchmarkPrompt(
        task_id="extraction_02",
        category="extraction",
        difficulty="medium",
        prompt="Extract information about the person from this text:\n\n"
               "Dr. Rajesh Patel has spent the last two decades as a cardiologist "
               "at Massachusetts General Hospital. Originally from Mumbai, he now "
               "calls Boston home. He hasn't shared his exact age publicly.",
        schema_name="PersonInfo",
        expected_fields={
            "name_contains": "Patel",
            "age": None,  # must be None — tests whether model hallucinates
            "occupation_contains": "cardiologist",
            "location_contains": "Boston",
        },
    ),
    BenchmarkPrompt(
        task_id="extraction_03",
        category="extraction",
        difficulty="hard",
        prompt="Extract structured action items from these meeting notes:\n\n"
               "Q4 Planning Meeting — Attendees: Maya, Jordan, Priya, Sam\n\n"
               "Main topics: launch timeline for the new dashboard feature and "
               "customer feedback from Q3.\n\n"
               "Maya will finalize the dashboard wireframes by November 8th — "
               "this is blocking engineering so it's top priority. Jordan "
               "volunteered to reach out to the three enterprise customers who "
               "filed critical bugs last quarter; no hard deadline but ideally "
               "done before Thanksgiving. Priya should schedule a follow-up with "
               "the design team, probably sometime in the next couple weeks, "
               "low urgency. We'll reconvene on November 15th.",
        schema_name="MeetingExtraction",
        expected_fields={
            "meeting_topic_contains": "Q4",
            "action_item_count_min": 3,
            "has_high_priority": True,
            "next_meeting_date_present": True,
        },
    ),
]


# =============================================================================
# CODE — generate Python, evaluated by execution against test cases
# =============================================================================

CODE_PROMPTS: list[BenchmarkPrompt] = [
    BenchmarkPrompt(
        task_id="code_01",
        category="code",
        difficulty="easy",
        prompt="Write a Python function named `is_palindrome` that takes a string "
               "and returns True if it is a palindrome (ignoring case and spaces), "
               "False otherwise. Return ONLY the function definition, no examples, "
               "no explanation, no markdown fences.",
        test_cases=[
            {"args": ["racecar"], "expected": True},
            {"args": ["A man a plan a canal Panama"], "expected": True},
            {"args": ["hello"], "expected": False},
            {"args": [""], "expected": True},
        ],
    ),
    BenchmarkPrompt(
        task_id="code_02",
        category="code",
        difficulty="medium",
        prompt="Write a Python function named `two_sum` that takes a list of "
               "integers `nums` and an integer `target`, and returns a list of "
               "the two indices whose values sum to target. Assume exactly one "
               "solution exists. Return ONLY the function definition, no examples, "
               "no explanation, no markdown fences.",
        test_cases=[
            {"args": [[2, 7, 11, 15], 9], "expected": [0, 1]},
            {"args": [[3, 2, 4], 6], "expected": [1, 2]},
            {"args": [[3, 3], 6], "expected": [0, 1]},
        ],
    ),
    BenchmarkPrompt(
        task_id="code_03",
        category="code",
        difficulty="hard",
        prompt="Write a Python function named `longest_substring_without_repeating` "
               "that takes a string and returns the length of the longest "
               "substring that contains no repeating characters. Return ONLY the "
               "function definition, no examples, no explanation, no markdown fences.",
        test_cases=[
            {"args": ["abcabcbb"], "expected": 3},
            {"args": ["bbbbb"], "expected": 1},
            {"args": ["pwwkew"], "expected": 3},
            {"args": [""], "expected": 0},
        ],
    ),
]


# =============================================================================
# Aggregate
# =============================================================================

ALL_PROMPTS: list[BenchmarkPrompt] = (
    REASONING_PROMPTS
    + SUMMARIZATION_PROMPTS
    + EXTRACTION_PROMPTS
    + CODE_PROMPTS
)


def get_prompts_by_category(category: str) -> list[BenchmarkPrompt]:
    """Filter helper for when you want to run one category at a time."""
    return [p for p in ALL_PROMPTS if p.category == category]
