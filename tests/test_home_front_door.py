"""Tests for Home page as the front door and the plain-English glossary."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "app"))

import pytest
from streamlit.testing.v1 import AppTest

from components.glossary import TERMS, glossary
from three_trees.definitions import POINT_DEFINITIONS, RESEARCH_QUESTION


REQUIRED_TERMS = [
    "xG",
    "field tilt",
    "Elo",
    "points above expectation",
    "luck band",
    "rank correlation",
    "95% interval",
    "standard deviations",
    "dominance",
    "resourcefulness",
    "efficiency",
    "conversion",
    "security",
    "threat spread",
    "high-speed running",
    "filled-in positions",
    "formation lines",
]


def test_glossary_terms_and_filtering() -> None:
    """Every required term is defined in TERMS and filtering works correctly."""
    # Check all required terms are defined and non-empty
    for term in REQUIRED_TERMS:
        assert term in TERMS, f"Missing required term: {term}"
        assert isinstance(TERMS[term], str) and len(TERMS[term].strip()) > 0

    # The three points must match definitions.POINT_DEFINITIONS
    assert TERMS["dominance"] == POINT_DEFINITIONS["Dominance"]
    assert TERMS["resourcefulness"] == POINT_DEFINITIONS["Resourcefulness"]
    assert TERMS["efficiency"] == POINT_DEFINITIONS["Efficiency"]

    # Test filtering behavior with an AppTest script snippet
    def app_with_filtered_glossary() -> None:
        import streamlit as st
        from components.glossary import glossary

        glossary(terms=["xG", "Elo"])

    app = AppTest.from_function(app_with_filtered_glossary).run()
    assert not app.exception
    expander = app.expander[0]
    markdown_text = " ".join(item.value for item in expander.markdown)
    assert "**xG**:" in markdown_text
    assert "**Elo**:" in markdown_text
    assert "**field tilt**:" not in markdown_text

    # Test unfiltered glossary shows all terms
    def app_with_all_glossary() -> None:
        import streamlit as st
        from components.glossary import glossary

        glossary()

    app_all = AppTest.from_function(app_with_all_glossary).run()
    assert not app_all.exception
    expander_all = app_all.expander[0]
    all_text = " ".join(item.value for item in expander_all.markdown)
    for term in REQUIRED_TERMS:
        assert f"**{term}**:" in all_text


def test_home_front_door_and_captions() -> None:
    """Home acts as the front door with research question, verified links, and chart captions."""
    home_path = REPO_ROOT / "app" / "Home.py"
    app = AppTest.from_file(str(home_path), default_timeout=45).run()
    assert not app.exception

    # Front door title
    assert any("Tournament pathways" in item.value for item in app.title)

    # Research question from definitions
    full_markdown = " ".join(item.value for item in app.markdown if isinstance(item.value, str))
    assert RESEARCH_QUESTION in full_markdown

    # "Start here" section heading
    assert any("Start here" in item.value for item in app.subheader)

    # Links present only for existing pages
    pages_dir = REPO_ROOT / "app" / "pages"
    existing_page_stems = {p.stem for p in pages_dir.glob("*.py")}
    # Streamlit page names may strip the leading number prefix (e.g. 1_Tactical_profiles -> Tactical_profiles)
    existing_normalized = {
        stem.split("_", 1)[-1] if stem[0].isdigit() and "_" in stem else stem
        for stem in existing_page_stems
    }

    for pl in app.page_link:
        page_name = str(pl.page)
        assert page_name in existing_normalized or page_name in existing_page_stems, (
            f"Page link {pl.page} does not correspond to an existing file in {pages_dir}"
        )

    # Verify that existing standard pages have links on Home
    linked_names = {str(pl.page) for pl in app.page_link}
    for norm in existing_normalized:
        assert norm in linked_names, f"Existing page {norm} missing from Home start here links"

    # Captions at least as many as charts
    charts = app.get("plotly_chart")
    captions = app.caption
    assert len(captions) >= len(charts), (
        f"Expected at least as many captions ({len(captions)}) as charts ({len(charts)})"
    )

    # Glossary expander is present at the bottom
    assert any("What the words mean" in e.label for e in app.expander)
