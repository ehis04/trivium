"""Slide figures export (1600 x 900 px, fonts >= 18 px).

Exports PNG images from the build without embedded titles, along with a manifest CSV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

from three_trees.charts.expectation import overperformance_dumbbell
from three_trees.charts.match import dominance_bars, xg_timeline
from three_trees.charts.pitch import zone_figure
from three_trees.charts.profiles import profile_comparison_chart
from three_trees.charts.answer import point_evidence_chart
from three_trees.config import Config, repo_root
from three_trees.definitions import POINTS
from three_trees.models.answer import point_evidence
from three_trees.storage import read_table, table_exists

EXPORT_WIDTH = 1600
EXPORT_HEIGHT = 900
EXPORT_FONT_SIZE = 18


def _check_chrome() -> str:
    """Ensure a Chrome binary is available for Kaleido export."""
    try:
        import kaleido

        chrome_path = kaleido.get_chrome_sync()
        if not chrome_path or not Path(chrome_path).is_file():
            raise RuntimeError("Chrome binary not found")
        return str(chrome_path)
    except Exception as exc:
        raise RuntimeError(
            "Chrome binary required by Kaleido was not found. "
            "Please install Google Chrome or Chromium "
            "(e.g. `sudo apt-get install chromium-browser` or `google-chrome-stable`)."
        ) from exc


def _style_plotly(fig: go.Figure) -> go.Figure:
    """Ensure font sizes are at least 18 px on exported figures."""
    fig.update_layout(font={"size": EXPORT_FONT_SIZE})
    fig.update_xaxes(title_font={"size": EXPORT_FONT_SIZE}, tickfont={"size": EXPORT_FONT_SIZE})
    fig.update_yaxes(title_font={"size": EXPORT_FONT_SIZE}, tickfont={"size": EXPORT_FONT_SIZE})
    return fig


def write_figures(
    cfg: Config,
    out: str | Path,
    team: str | None = None,
    match_id: int | None = None,
) -> pd.DataFrame:
    """Write PNG slide figures (1600 x 900) and figures_manifest.csv outside the repo."""
    out_dir = Path(out).expanduser().resolve()
    root = repo_root()
    if root is not None and out_dir.is_relative_to(root):
        raise ValueError(f"Output folder must be outside the repo ({root}): {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    _check_chrome()

    if not table_exists(cfg, "matches"):
        raise ValueError("Build is missing the required matches table.")
    matches = read_table(cfg, "matches")

    # Select default team and its tournament
    team_tournament = "men"
    if team is None:
        all_teams = set(matches["home_team"]).union(set(matches["away_team"]))
        if "Republic of Ireland" in all_teams:
            team = "Republic of Ireland"
            team_tournament = "men"
        else:
            men_matches = matches[matches["tournament"] == "men"]
            if not men_matches.empty:
                team = str(men_matches.sort_values(["match_date", "match_id"])["home_team"].iloc[0])
                team_tournament = "men"
            else:
                team = str(matches.sort_values(["match_date", "match_id"])["home_team"].iloc[0])
                team_tournament = str(matches.sort_values(["match_date", "match_id"])["tournament"].iloc[0])
    else:
        men_teams = set(matches.loc[matches["tournament"] == "men", "home_team"]).union(
            set(matches.loc[matches["tournament"] == "men", "away_team"])
        )
        if team in men_teams:
            team_tournament = "men"
        else:
            team_tournament = "women"

    # Select match for the team (default: team's first match in date order)
    team_matches = matches[
        ((matches["home_team"] == team) | (matches["away_team"] == team))
        & (matches["tournament"] == team_tournament)
    ]
    if team_matches.empty:
        team_matches = matches[(matches["home_team"] == team) | (matches["away_team"] == team)]
    if team_matches.empty:
        raise ValueError(f"No matches found for team {team}.")

    if match_id is not None:
        sel = team_matches[team_matches["match_id"] == match_id]
        if sel.empty:
            sel = matches[matches["match_id"] == match_id]
        if sel.empty:
            raise ValueError(f"Match ID {match_id} not found.")
        selected_match = sel.iloc[0]
    else:
        sorted_m = team_matches.sort_values(["match_date", "match_id"])
        selected_match = sorted_m.iloc[0]

    match_id = int(selected_match["match_id"])
    match_tournament = str(selected_match["tournament"])
    opponent = str(
        selected_match["away_team"]
        if selected_match["home_team"] == team
        else selected_match["home_team"]
    )

    manifest_rows: list[dict[str, Any]] = []

    # 1. Dumbbell charts per tournament
    if table_exists(cfg, "team_summary"):
        summary = read_table(cfg, "team_summary")
        tournaments = sorted(summary["tournament"].dropna().unique())
        for tourney in tournaments:
            sub = summary[summary["tournament"] == tourney]
            highlight_team = team if (team in sub["team"].values and tourney == team_tournament) else None
            fig = overperformance_dumbbell(sub, tourney, highlight=highlight_team, top=5, bottom=5)
            _style_plotly(fig)
            fname = f"dumbbell_{tourney}.png"
            fig.write_image(out_dir / fname, width=EXPORT_WIDTH, height=EXPORT_HEIGHT)
            manifest_rows.append(
                {
                    "file": fname,
                    "chart": "overperformance_dumbbell",
                    "tournament": tourney,
                    "team": highlight_team,
                    "match_id": pd.NA,
                    "source tables": "team_summary",
                }
            )

    # 2. Profiles comparison per tournament
    if table_exists(cfg, "profile_comparison"):
        profiles = read_table(cfg, "profile_comparison")
        tournaments = sorted(profiles["tournament"].dropna().unique())
        for tourney in tournaments:
            sub = profiles[profiles["tournament"] == tourney]
            if not sub.empty:
                fig = profile_comparison_chart(profiles, tourney)
                _style_plotly(fig)
                fname = f"profiles_{tourney}.png"
                fig.write_image(out_dir / fname, width=EXPORT_WIDTH, height=EXPORT_HEIGHT)
                manifest_rows.append(
                    {
                        "file": fname,
                        "chart": "profile_comparison_chart",
                        "tournament": tourney,
                        "team": None,
                        "match_id": pd.NA,
                        "source tables": "profile_comparison",
                    }
                )

        evidence = point_evidence(profiles)
        if not evidence.empty:
            filenames = {
                "Dominance": "point_dominance.png",
                "Resourcefulness": "point_resourcefulness.png",
                "Efficiency": "point_efficiency.png",
            }
            charts = [("three_points.png", None), *[(filenames[name], name) for name in POINTS]]
            for fname, selected_point in charts:
                fig = point_evidence_chart(evidence, selected_point)
                _style_plotly(fig)
                fig.write_image(out_dir / fname, width=EXPORT_WIDTH, height=EXPORT_HEIGHT)
                manifest_rows.append(
                    {
                        "file": fname,
                        "chart": "point_evidence_chart",
                        "tournament": "both",
                        "team": None,
                        "match_id": pd.NA,
                        "source tables": "profile_comparison",
                    }
                )

    # 2b. Tactical profiles: the profile map and profile results per tournament, score associations
    if table_exists(cfg, "tactical_profile_team") and table_exists(cfg, "tactical_profile_summary"):
        from three_trees.charts.tactical import profile_map, profile_outcomes, score_associations

        teams = read_table(cfg, "tactical_profile_team")
        profile_summary = read_table(cfg, "tactical_profile_summary")
        for tourney in sorted(teams["tournament"].dropna().unique()):
            sub = teams[teams["tournament"] == tourney]
            highlight_team = team if team in set(sub["team"]) else None
            own = sub.loc[sub["team"] == highlight_team, "profile"]
            charts = (("tactical_map", profile_map(teams, tourney, highlight_team), "tactical_profile_team"),
                      ("tactical_outcomes", profile_outcomes(profile_summary, tourney,
                                                             own.iloc[0] if len(own) else None),
                       "tactical_profile_summary"))
            for name, fig, source in charts:
                _style_plotly(fig)
                fname = f"{name}_{tourney}.png"
                fig.write_image(out_dir / fname, width=EXPORT_WIDTH, height=EXPORT_HEIGHT)
                manifest_rows.append({"file": fname, "chart": name, "tournament": tourney, "team": highlight_team,
                                      "match_id": pd.NA, "source tables": source})
        if table_exists(cfg, "tactical_profile_axes"):
            fig = score_associations(read_table(cfg, "tactical_profile_axes"))
            _style_plotly(fig)
            fig.write_image(out_dir / "tactical_scores.png", width=EXPORT_WIDTH, height=EXPORT_HEIGHT)
            manifest_rows.append({"file": "tactical_scores.png", "chart": "score_associations", "tournament": "both",
                                  "team": None, "match_id": pd.NA, "source tables": "tactical_profile_axes"})

    # 3. Match dominance bars
    if table_exists(cfg, "team_match"):
        tm = read_table(cfg, "team_match")
        tm_m = tm[(tm["tournament"] == match_tournament) & (tm["match_id"] == match_id)]
        if not tm_m.empty:
            fig = dominance_bars(tm_m, team)
            _style_plotly(fig)
            fname = "match_dominance.png"
            fig.write_image(out_dir / fname, width=EXPORT_WIDTH, height=EXPORT_HEIGHT)
            manifest_rows.append(
                {
                    "file": fname,
                    "chart": "dominance_bars",
                    "tournament": match_tournament,
                    "team": team,
                    "match_id": match_id,
                    "source tables": "team_match",
                }
            )

    # 4. Match xG timeline
    if table_exists(cfg, "shot_timeline"):
        shots = read_table(cfg, "shot_timeline")
        shots_m = shots[(shots["tournament"] == match_tournament) & (shots["match_id"] == match_id)]
        fig = xg_timeline(shots_m, team, opponent)
        _style_plotly(fig)
        fname = "match_xg.png"
        fig.write_image(out_dir / fname, width=EXPORT_WIDTH, height=EXPORT_HEIGHT)
        manifest_rows.append(
            {
                "file": fname,
                "chart": "xg_timeline",
                "tournament": match_tournament,
                "team": team,
                "match_id": match_id,
                "source tables": "shot_timeline",
            }
        )

    # 5. Match zones beside tournament zones
    if table_exists(cfg, "team_match_zone") and table_exists(cfg, "team_tournament_zone"):
        tmz = read_table(cfg, "team_match_zone")
        ttz = read_table(cfg, "team_tournament_zone")
        m_zones = tmz[
            (tmz["tournament"] == match_tournament)
            & (tmz["match_id"] == match_id)
            & (tmz["team"] == team)
        ]
        t_zones = ttz[(ttz["tournament"] == match_tournament) & (ttz["team"] == team)]
        panels = [(f"{team} v {opponent}", m_zones), (f"{team}, whole tournament", t_zones)]
        fig = zone_figure(panels, measure="territory_share", font_size=EXPORT_FONT_SIZE)
        fig.set_size_inches(EXPORT_WIDTH / 100, EXPORT_HEIGHT / 100)
        fname = "match_zones.png"
        fig.savefig(out_dir / fname, dpi=100)
        plt.close(fig)
        manifest_rows.append(
            {
                "file": fname,
                "chart": "zone_figure",
                "tournament": match_tournament,
                "team": team,
                "match_id": match_id,
                "source tables": "team_match_zone, team_tournament_zone",
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    if "match_id" in manifest:
        manifest["match_id"] = manifest["match_id"].astype("Int64")
    manifest.to_csv(out_dir / "figures_manifest.csv", index=False)
    return manifest
