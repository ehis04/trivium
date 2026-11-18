"""Claude-powered query answering: function selection via tool-use, execution in Python,
and plain-English explanation generation.

Only aggregated team-level and position-group-level numbers are ever sent to Claude.
Falls back to the offline matcher when credentials are missing, API calls fail,
refusals occur, or offline mode is requested.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import anthropic

from trivium.config import Config
from trivium.query.fallback import offline_answer
from trivium.query.functions import TOOLS, QueryError, Result, ensure_aggregated, run_function

MODEL = "claude-opus-5-5"
EFFORT = "low"
TIMEOUT_S = 15.0
FALLBACK_BETA = "server-side-fallback-2026-07-01"

CHOOSER_SYSTEM = """You are a football analytics assistant for the FIFA U-17 World Cup 2025.
A coach has asked a question. You must call exactly ONE function from the provided tools catalogue
with appropriate arguments to retrieve the data needed to answer the question.
Do not answer with text alone; you must choose a tool. Default tournament to 'men' unless
women/girls/female is mentioned."""

WRITER_SYSTEM = """Now write the answer. You are writing for a non-technical coach.
Write 2 to 3 concise, plain-English sentences answering the coach's question using ONLY the numbers
and facts provided in the tool result. Do not call any tool.
- Reference metric labels with appropriate units where applicable.
- Do not make unsupported causal claims.
- If an over-performance result is within the luck band, explain that it is within expected variation.
- If a table is not built yet, state clearly that the necessary data is not available.
- If the data is synthetic demo data, briefly note that the figures reflect the synthetic demo build.
- Never make up numbers not present in the payload."""
# The writer instructions go in as an appended system message, and both calls send the same
# top-level system prompt and tools: rebuilding either between the two calls would invalidate
# the first call's thinking blocks (a 400 on accounts enforced for preserved thinking).


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    result: Result | None
    mode: str  # "claude" or "offline"
    function: str | None
    args: dict[str, object]
    error: str  # why the offline path was used, empty when Claude answered


def answer(
    cfg: Config,
    question: str,
    client: anthropic.Anthropic | None = None,
    use_claude: bool = True,
    previous_context: dict[str, Any] | None = None,
) -> Answer:
    """Answer a user question using Claude tool-use or the offline matcher."""
    if not use_claude:
        return offline_answer(cfg, question, error="Claude disabled by user", previous_context=previous_context)

    try:
        api_client = client or anthropic.Anthropic()
    except Exception as e:
        return offline_answer(
            cfg,
            question,
            error=f"Anthropic client initialization failed: {e}",
            previous_context=previous_context,
        )

    api = api_client.with_options(timeout=TIMEOUT_S, max_retries=1)

    chooser_sys = CHOOSER_SYSTEM
    if previous_context:
        t_hint = previous_context.get("team", "")
        tn_hint = previous_context.get("tournament", "")
        chooser_sys += f"\nPrevious question context: Team {t_hint}, tournament {tn_hint}."

    try:
        first = api.beta.messages.create(
            model=MODEL,
            max_tokens=2_000,
            betas=[FALLBACK_BETA],
            fallbacks="default",
            output_config={"effort": EFFORT},
            system=chooser_sys,
            tools=TOOLS,
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
            messages=[{"role": "user", "content": question}],
        )
    except anthropic.AuthenticationError as e:
        return offline_answer(cfg, question, error=f"Authentication failed: {e}", previous_context=previous_context)
    except anthropic.RateLimitError as e:
        return offline_answer(cfg, question, error=f"Rate limit exceeded: {e}", previous_context=previous_context)
    except anthropic.APITimeoutError as e:
        return offline_answer(cfg, question, error=f"API timeout: {e}", previous_context=previous_context)
    except anthropic.APIConnectionError as e:
        return offline_answer(cfg, question, error=f"Connection error: {e}", previous_context=previous_context)
    except anthropic.APIStatusError as e:
        return offline_answer(cfg, question, error=f"API status error ({e.status_code}): {e}", previous_context=previous_context)
    except anthropic.AnthropicError as e:
        return offline_answer(cfg, question, error=f"Anthropic error: {e}", previous_context=previous_context)
    except Exception as e:
        return offline_answer(cfg, question, error=f"Unexpected error calling Claude: {e}", previous_context=previous_context)

    # Check for refusal
    if getattr(first, "stop_reason", None) == "refusal":
        return offline_answer(cfg, question, error="Model refused the request", previous_context=previous_context)

    # Find tool call block
    tool_blocks = [b for b in first.content if getattr(b, "type", None) == "tool_use"]
    if not tool_blocks:
        return offline_answer(cfg, question, error="No tool selected by Claude", previous_context=previous_context)

    block = tool_blocks[0]
    func_name = str(block.name)
    func_args = dict(block.input) if isinstance(block.input, dict) else {}

    # Run the function locally
    result: Result | None = None
    try:
        result = run_function(cfg, func_name, func_args)
        is_error = False
        payload_dict: dict[str, Any] = {
            "title": result.title,
            "available": result.available,
            "note": result.note,
            "numbers": result.numbers,
            "rows": result.table.to_dict(orient="records"),
        }
    except QueryError as qe:
        is_error = True
        payload_dict = {"error": str(qe)}
    except Exception as e:
        is_error = True
        payload_dict = {"error": f"Failed to execute {func_name}: {e}"}

    payload = json.dumps(payload_dict, default=str)

    # Second call: generate explanation using tool result
    try:
        tool_res_item: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": getattr(block, "id", "tool_call_1"),
            "content": payload,
        }
        if is_error:
            tool_res_item["is_error"] = True

        second = api.beta.messages.create(
            model=MODEL,
            max_tokens=2_000,
            betas=[FALLBACK_BETA],
            fallbacks="default",
            output_config={"effort": EFFORT},
            system=chooser_sys,
            tools=TOOLS,
            tool_choice={"type": "none"},
            messages=[
                {"role": "user", "content": question},
                {"role": "assistant", "content": first.content},
                {"role": "user", "content": [tool_res_item]},
                {"role": "system", "content": WRITER_SYSTEM},
            ],
        )
    except Exception as e:
        return offline_answer(cfg, question, error=f"Explanation generation failed: {e}",
                              previous_context=previous_context)
    if getattr(second, "stop_reason", None) == "refusal":
        return offline_answer(cfg, question, error="Model refused to write the answer",
                              previous_context=previous_context)
    text = "".join(b.text for b in second.content if getattr(b, "type", None) == "text").strip()
    if not text:
        return offline_answer(cfg, question, error="Claude returned no answer text",
                              previous_context=previous_context)
    return Answer(
        question=question,
        text=text,
        result=result,
        mode="claude",
        function=func_name,
        args=func_args,
        error="",
    )
