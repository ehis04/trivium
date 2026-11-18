"""Command line: three-trees build | validate | senior-check | check-config."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence
from pathlib import Path

from three_trees import __version__
from three_trees.config import Config, ConfigError, load_config

# Pipeline stages in run order. Each module exposes run(cfg).
STAGES: tuple[tuple[str, str], ...] = (
    ("statsbomb", "three_trees.ingest.statsbomb"),
    ("match_folders", "three_trees.ingest.match_folders"),
    ("reports", "three_trees.ingest.reports"),
    ("tracking", "three_trees.tracking.stream"),
    ("positions", "three_trees.tracking.positions"),
    ("external", "three_trees.ingest.external"),
    ("expectation", "three_trees.models.expectation"),
    ("features", "three_trees.features.tables"),
    ("reliability", "three_trees.models.reliability"),
    ("drivers", "three_trees.models.drivers"),
    ("styles", "three_trees.models.styles"),
    ("profiles", "three_trees.models.profiles"),
    ("sustainability", "three_trees.models.sustainability"),
)
STAGE_NAMES = tuple(name for name, _ in STAGES)
VALIDATION = "three_trees.validation"
SENIOR_CHECK = "three_trees.models.senior"


def _run(module: str, cfg: Config) -> None:
    importlib.import_module(module).run(cfg)


def cmd_build(cfg: Config, args: argparse.Namespace) -> int:
    cfg.build_dir.mkdir(parents=True, exist_ok=True)
    selected = set(args.only or STAGE_NAMES)
    unbuilt = []
    for name, module in STAGES:
        if name in selected:
            print(f"[build] {name}", flush=True)
            try:
                _run(module, cfg)
            except NotImplementedError as exc:
                print(f"Not built yet: {exc} ({name} stage skipped)", file=sys.stderr, flush=True)
                unbuilt.append(name)
    if not args.only:
        cmd_validate(cfg, args)
    return 1 if unbuilt else 0


def cmd_validate(cfg: Config, args: argparse.Namespace) -> int:
    print("[validate]", flush=True)
    _run(VALIDATION, cfg)
    return 0


def cmd_senior_check(cfg: Config, args: argparse.Namespace) -> int:
    print("[senior-check]", flush=True)
    _run(SENIOR_CHECK, cfg)
    return 0


def cmd_check_contracts(cfg: Config, args: argparse.Namespace) -> int:
    from three_trees import contracts

    if args.table:
        report = contracts.check_contracts(cfg, tables=args.table)
    else:
        contracts.run(cfg)
        report = contracts.check_contracts(cfg)
    failed = contracts.contract_failures(report)
    not_built = report[report["check"].str.contains("not built yet", regex=False)]
    for row in (*failed.itertuples(index=False), *not_built.itertuples(index=False)):
        print(f"{row.check}: target {row.target}, value {row.value}")
    tables = report["check"].str.split(":").str[0].nunique()
    print(f"contracts: {tables} tables, {len(failed)} failed checks, {len(not_built)} not built yet")
    return 1 if len(failed) else 0


def cmd_status(cfg: Config, args: argparse.Namespace) -> int:
    from three_trees import contracts

    build = contracts.build_status(cfg)
    coverage = contracts.measure_coverage(cfg)
    if build.empty:
        print("No Parquet tables in the build folder.")
    else:
        print(build.to_string(index=False))
    print()
    for tournament in ("men", "women"):
        rows = coverage[coverage["tournament"] == tournament]
        live = int((rows["status"] == "live").sum())
        partial = int((rows["status"] == "partial").sum())
        missing = int((rows["status"] == "missing").sum())
        print(f"{tournament}: {live} measures live, {partial} partial, {missing} missing")
        for status in ("partial", "missing"):
            names = rows.loc[rows["status"] == status, "measure"].tolist()
            if names:
                print(f"  {status}: {', '.join(names)}")
    return 0


def cmd_check_config(cfg: Config, args: argparse.Namespace) -> int:
    def found(path: Path) -> str:
        return "found" if path.exists() else "missing"

    rows = [
        ("data root", cfg.data_root, found(cfg.data_root)),
        ("build dir", cfg.build_dir, found(cfg.build_dir)),
    ]
    missing = 0
    for tournament, sb in cfg.statsbomb.items():
        parts = sb.event_files()
        rows.append((f"statsbomb {tournament} events", sb.events, f"{len(parts)} event files"))
        rows.append((f"statsbomb {tournament} matches", sb.matches, found(sb.matches)))
        missing += (not parts) + (not sb.matches.is_file())
    folders = [p for p in cfg.match_folders_men.iterdir() if p.is_dir()] if cfg.match_folders_men.is_dir() else []
    rows.append(("match folders men", cfg.match_folders_men, f"{len(folders)} match folders"))
    pdfs = list(cfg.fifa_reports_women.glob("*.pdf")) if cfg.fifa_reports_women.is_dir() else []
    rows.append(("fifa reports women", cfg.fifa_reports_women, f"{len(pdfs)} PDF files"))
    rows.append(("external dir", cfg.external_dir, found(cfg.external_dir)))

    width = max(len(label) for label, _, _ in rows) + 2
    print(f"{'config file':<{width}}{cfg.source}")
    for label, path, status in rows:
        print(f"{label:<{width}}{path}  [{status}]")
    return 1 if missing else 0


def cmd_figures(cfg: Config, args: argparse.Namespace) -> int:
    from three_trees.charts.export import write_figures

    try:
        manifest = write_figures(cfg, args.out, team=args.team, match_id=args.match_id)
        print(f"Wrote {len(manifest)} figures to {args.out}")
        return 0
    except (ValueError, RuntimeError) as exc:
        print(f"Figures error: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        type=Path,
        help="config file (default: $THREE_TREES_CONFIG, else ./config.yaml)",
    )

    parser = argparse.ArgumentParser(prog="three-trees", description="Tournament pathways.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser(
        "build",
        parents=[common],
        help="build all tables and models, then validate",
        description="Run every pipeline stage in order, then validate. "
        "With --only, run just the named stages (in pipeline order) and skip validation.",
    )
    build.add_argument("--only", nargs="+", choices=STAGE_NAMES, metavar="STAGE", help=", ".join(STAGE_NAMES))
    build.set_defaults(func=cmd_build)

    sub.add_parser("validate", parents=[common], help="run checks and write the check report").set_defaults(
        func=cmd_validate
    )
    sub.add_parser(
        "senior-check", parents=[common], help="rerun models on StatsBomb senior open data"
    ).set_defaults(func=cmd_senior_check)
    check_contracts = sub.add_parser(
        "check-contracts",
        parents=[common],
        help="check each built table against the registry contract",
    )
    check_contracts.add_argument("--table", nargs="+", default=None, help="optional table names to check")
    check_contracts.set_defaults(func=cmd_check_contracts)

    status = sub.add_parser("status", parents=[common], help="show build status and measure coverage")
    status.set_defaults(func=cmd_status)

    figures = sub.add_parser(
        "figures",
        parents=[common],
        help="write slide PNG figures and manifest to a folder outside the repo",
    )
    figures.add_argument("--out", type=Path, required=True, help="output folder outside the repo")
    figures.add_argument("--team", type=str, default=None, help="team name (default: Republic of Ireland or first men's team)")
    figures.add_argument("--match-id", type=int, default=None, help="match ID for match charts")
    figures.set_defaults(func=cmd_figures)

    sub.add_parser(
        "check-config", parents=[common], help="show configured paths and which inputs are found"
    ).set_defaults(func=cmd_check_config)
    demo = sub.add_parser(
        "demo-data",
        parents=[common],
        help="write a synthetic demo build (fake teams) to a separate folder",
        description="Write synthetic tables in the shape of the real ones to <out>/build and run the "
        "expectation stage on them. Point the dashboard at it with THREE_TREES_DATA_ROOT=<out>.",
    )
    demo.add_argument("--out", type=Path, required=True, help="demo data root, outside the repo and the real data")
    demo.add_argument("--small", action="store_true", help="smaller tournaments (faster, for tests)")
    demo.add_argument("--no-expectation", action="store_true", help="input tables only (no stages run)")
    demo.set_defaults(func=None)
    return parser


def cmd_demo_data(args: argparse.Namespace) -> int:
    from three_trees import synthetic

    try:
        real_root = load_config(args.config).data_root
    except ConfigError:
        real_root = None
    cfg = synthetic.write_demo(
        args.out,
        scale="small" if args.small else "full",
        run_expectation=not args.no_expectation,
        real_data_root=real_root,
        run_models=not args.no_expectation,
    )
    print(f"Synthetic demo build written to {cfg.build_dir}")
    print(f"Use it with: THREE_TREES_DATA_ROOT={cfg.data_root}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo-data":
        try:
            return cmd_demo_data(args)
        except ValueError as exc:
            print(f"Demo data: {exc}", file=sys.stderr)
            return 2
    try:
        cfg = load_config(args.config)
        return args.func(cfg, args)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2
    except NotImplementedError as exc:
        print(f"Not built yet: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
