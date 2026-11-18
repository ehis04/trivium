# Trivium

Trivium looks at why some teams go further in a tournament than their strength says they should, and what those teams do differently on the pitch.

It uses match events, tracking data and match reports from the FIFA U-17 World Cups 2025, men's and women's. Every team is compared with what a pre-tournament rating predicted for it. The gap between the result and that prediction is the thing to explain.

## The idea

A team's path through a tournament depends on more than its rating. The project describes how each team played using three scores, each measured against the other teams in the same tournament:

- **Dominant**: how much of the territory and the chances the team controlled.
- **Efficient**: how well it turned that control into chances while limiting the opponent's.
- **Resourceful**: how much control it achieved at a low defensive and running cost.

Being above or below average on each score gives eight tactical profiles. Each profile is then compared with how far its teams went beyond expectation, to see which ways of playing go with beating your strength.

## How it works

1. **Read the data.** Match events, match lists, per-match physical reports, tracking files and public results are read from a data folder kept outside the repository.
2. **Set expectations.** A rating based on Elo gives each team an expected number of points and an expected stage. Results are measured against it.
3. **Build team-per-match measures.** Events and tracking positions become measures such as territory, chances by phase of play, conversion, running load and team shape.
4. **Find the profiles.** The three scores are built from those measures and checked for how repeatable they are from match to match, and whether early matches predict later ones.
5. **Show the results.** A dashboard shows each team's profile and match-by-match story, and a query box answers questions from team-level results.

Penalty shootouts are left out of every performance measure. Players are shown only by position group or by team.

## Running it

Python 3.11 or later.

```sh
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp config.example.yaml config.yaml    # set data_root to the folder holding the data
.venv/bin/trivium build           # build every table
.venv/bin/streamlit run app/Home.py   # open the dashboard
```

`trivium demo-data --out DIR` builds a small synthetic data set, so the dashboard can be tried without the real data.

## What is in the repository

```
src/trivium/   the pipeline: ingest, tracking, features, models, charts, query box
app/               the Streamlit dashboard
tests/             automated tests on small synthetic data
```

The repository holds code only. The data, local settings and API keys stay outside it.
