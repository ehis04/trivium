"""Ask the Data — interactive query box for coaches and analysts.

Allows natural language questions about team over-performance, match reviews,
rankings, physical outputs, and tactical styles. Answers are powered by
Claude Opus 5.5 tool-use or the offline rapidfuzz matcher.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from components.layout import page_setup
from components.sidebar import select_context
from three_trees.config import load_config
from three_trees.query.llm import answer


def main() -> None:
    page_setup("Ask a football question")
    tournament, _, _ = select_context()

    cfg = load_config()

    # Initialize session state for follow-ups
    if "last_context" not in st.session_state:
        st.session_state.last_context = {}
    if "current_question" not in st.session_state:
        st.session_state.current_question = ""

    # Controls row
    col_input, col_toggle = st.columns([4, 1])

    with col_toggle:
        has_key = bool(os.getenv("ANTHROPIC_API_KEY"))
        use_claude = st.toggle(
            "Use Claude",
            value=has_key,
            help="Let Claude choose an analysis and explain its numbers. Turn this off to use the offline answer tool.",
        )

    # 5 Example questions
    st.markdown("**Example questions:**")
    examples = [
        "Which men's teams beat their strength most?",
        "Compare Team A1 and Team B2",
        "Why did Team C3 over-perform?",
        "What is the running profile of Team A1?",
        "Show the zone territory map for Team A1",
    ]

    cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        if cols[i].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.current_question = ex

    with col_input:
        user_query = st.text_input(
            "Ask a question about teams, matches, or statistics:",
            value=st.session_state.current_question,
            placeholder="e.g. Compare Team A1 and Team B2, or Why did Team A1 over-perform?",
        )

    st.caption("Only team totals and position-group totals are shared with Claude, never individual actions or tracking details.")

    active_query = user_query.strip()
    if not active_query:
        return

    # Call query system
    context = st.session_state.last_context or {"tournament": tournament}
    with st.spinner("Analyzing data..."):
        ans = answer(cfg, active_query, use_claude=use_claude, previous_context=context)

    # Display mode caption
    if ans.mode == "claude":
        st.caption("Answer prepared with Claude.")
    else:
        reason = f" ({ans.error})" if ans.error and ans.error != "Claude disabled by user" else ""
        st.caption(f"Answer prepared with the offline tool{reason}.")

    # Display Answer Text
    st.markdown(f"### Answer\n{ans.text}")

    # Function details
    if ans.function:
        st.caption("The answer uses the teams, tournament and measures named in your question.")

        # Update last context for follow-ups
        team = ans.args.get("team") or ans.args.get("team_a")
        tourney = ans.args.get("tournament", "men")
        if team:
            st.session_state.last_context["team"] = team
        st.session_state.last_context["tournament"] = tourney
        st.session_state.last_context["function"] = ans.function

    # Display Table and Chart if available
    if ans.result is not None and ans.result.available:
        if not ans.result.table.empty:
            st.markdown("#### Data")
            st.dataframe(ans.result.table, hide_index=True)

        if ans.result.chart is not None:
            st.markdown("#### Chart")
            st.plotly_chart(ans.result.chart, width="stretch")
            st.caption(
                "This chart shows the numbers behind the answer above; hover over a mark to see its details."
            )
    elif ans.result is not None and not ans.result.available:
        st.info(f"ℹ️ {ans.result.note}")


if __name__ == "__main__":
    main()
