"""Repo hygiene: no data, local config or secrets in git, in the index or in any commit."""

from __future__ import annotations

import subprocess
from pathlib import Path, PurePosixPath

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_SUFFIXES = frozenset(
    {
        ".csv", ".tsv", ".parquet", ".feather", ".arrow",
        ".json", ".jsonl", ".ndjson",
        ".zip", ".gz", ".bz2", ".xz", ".7z", ".tar", ".rar",
        ".duckdb", ".wal", ".db", ".sqlite", ".sqlite3",
        ".xlsx", ".xlsm", ".xls",
        ".pdf",
        ".pkl", ".pickle", ".joblib", ".h5", ".hdf5", ".npy", ".npz",
        ".ipynb",
    }
)
DATA_DIRS = frozenset({"data", "build", "raw", "downloads"})
ALLOWED_DATA = {PurePosixPath("src/three_trees/lookup"): frozenset({".csv"})}
LOCAL_FILES = frozenset({"config.yaml", ".env", "secrets.toml"})
MAX_BYTES = 1_000_000
SIZE_EXEMPT_SUFFIXES = frozenset({".pptx"})


def _git(*args: str, stdin: str | None = None) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, input=stdin, capture_output=True, text=True, check=True
    ).stdout


@pytest.fixture(scope="module", autouse=True)
def _git_work_tree() -> None:
    try:
        top = _git("rev-parse", "--show-toplevel").strip()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git work tree")
    if Path(top).resolve() != REPO_ROOT:
        pytest.skip(f"tests are not at the root of the git work tree ({top})")


def _blob_sizes(listing: str) -> list[tuple[str, int]]:
    """(path, size) for each blob in 'object-id path' lines."""
    out = _git("cat-file", "--batch-check=%(objecttype) %(objectsize) %(rest)", stdin=listing)
    files = []
    for line in out.splitlines():
        kind, size, path = line.split(" ", 2)
        if kind == "blob":
            files.append((path, int(size)))
    return files


def _index_files() -> list[tuple[str, int]]:
    entries = [e for e in _git("ls-files", "--stage", "-z").split("\0") if e]
    listing = []
    for entry in entries:
        meta, path = entry.split("\t", 1)
        mode, blob, _ = meta.split()
        if mode != "160000":
            listing.append(f"{blob} {path}\n")
    return _blob_sizes("".join(listing))


def _history_files() -> list[tuple[str, int]]:
    return _blob_sizes(_git("rev-list", "--objects", "--branches", "--tags", "--remotes"))


def _problems(files: list[tuple[str, int]]) -> list[str]:
    problems = []
    for path, size in files:
        p = PurePosixPath(path)
        suffix = p.suffix.lower()
        if p.parts[0] in DATA_DIRS and p.name != ".gitkeep":
            problems.append(f"{path}: inside data folder {p.parts[0]}/")
        elif suffix in DATA_SUFFIXES and suffix not in ALLOWED_DATA.get(p.parent, ()):
            problems.append(f"{path}: data file type {suffix}")
        elif p.name in LOCAL_FILES or p.name.startswith(".env."):
            problems.append(f"{path}: local config or secrets")
        elif size > MAX_BYTES and suffix not in SIZE_EXEMPT_SUFFIXES:
            problems.append(f"{path}: {size:,} bytes, over the {MAX_BYTES:,} byte limit")
    return sorted(set(problems))


def test_no_data_tracked() -> None:
    problems = _problems(_index_files())
    assert not problems, "Tracked by git (remove with git rm --cached):\n" + "\n".join(problems)


def test_no_data_in_history() -> None:
    problems = _problems(_history_files())
    assert not problems, "Present in git history (rewrite history before pushing):\n" + "\n".join(problems)


@pytest.mark.parametrize(
    "path",
    [
        "data/anything.txt",
        "u17_world_cup_2025_events_part001.csv",
        "statsbomb/u17_womens_world_cup_2025_matches.csv",
        "build/team_match.parquet",
        "build/three_trees.duckdb",
        "M1 Qatar - Italy/RSV_Broadcast_Tracking_Data_Tactical_Wide.zip",
        "M1 Qatar - Italy/tracking_data.jsonl",
        "M1 Qatar - Italy/metadata.json",
        "M1 Qatar - Italy/M01 QAT V ITA POST MATCH REPORT.pdf",
        "outputs/team_tables.xlsx",
        "notebooks/exploration.ipynb",
        "config.yaml",
        ".env",
    ],
)
def test_gitignore_blocks(path: str) -> None:
    result = subprocess.run(["git", "check-ignore", "--no-index", "-q", path], cwd=REPO_ROOT)
    assert result.returncode == 0, f"{path} is not ignored by .gitignore"


def test_gitignore_allows_lookups() -> None:
    path = "src/three_trees/lookup/countries.csv"
    result = subprocess.run(["git", "check-ignore", "--no-index", "-q", path], cwd=REPO_ROOT)
    assert result.returncode == 1, f"{path} is ignored by .gitignore"
