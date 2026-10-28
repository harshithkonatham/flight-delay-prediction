"""Command line entry point: `flightdelay <stage>`."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from flightdelay import features, ingest, quality, train
from flightdelay.config import Paths

log = logging.getLogger("flightdelay")


def _bronze_glob(paths: Paths) -> str:
    return str(paths.bronze / "**" / "*.parquet")


def cmd_ingest(paths: Paths, year: int, months: Sequence[int]) -> None:
    for month in months:
        ingest.ingest_month(year, month, paths.raw, paths.bronze)


def cmd_validate(paths: Paths) -> None:
    report = quality.validate(_bronze_glob(paths), paths.reports / "data_quality.json")
    for check in report.failures:
        log.warning("Check %s did not pass: %s", check.name, check.detail)
    log.info("Data quality OK on %s rows", f"{report.row_count:,}")


def cmd_features(paths: Paths) -> None:
    features.build_features(_bronze_glob(paths), paths.features)


def cmd_train(paths: Paths, max_train_rows: int | None) -> None:
    metadata = train.train(paths.features, paths.models, max_train_rows=max_train_rows)
    print(f"\nBest model: {metadata['model_name']}  (test from {metadata['split']['cutoff_date']})")
    print(f"{'model':<24}{'ROC-AUC':>9}{'PR-AUC':>9}{'F1':>8}")
    for name, m in metadata["metrics"].items():
        print(f"{name:<24}{m['roc_auc']:>9.3f}{m['pr_auc']:>9.3f}{m['f1']:>8.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flightdelay", description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="project folder")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_period(p: argparse.ArgumentParser) -> None:
        p.add_argument("--year", type=int, required=True)
        p.add_argument("--months", type=int, nargs="+", required=True)

    add_period(sub.add_parser("ingest", help="download and convert BTS months"))
    sub.add_parser("validate", help="run data-quality checks on ingested data")
    sub.add_parser("features", help="build the feature table")
    train_p = sub.add_parser("train", help="train and compare models")
    run_p = sub.add_parser("run", help="ingest, validate, build features and train")
    add_period(run_p)
    for p in (train_p, run_p):
        p.add_argument("--max-train-rows", type=int, default=1_000_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    paths = Paths(args.root)
    paths.ensure()
    try:
        if args.command in ("ingest", "run"):
            for month in args.months:
                ingest.validate_period(args.year, month)
            cmd_ingest(paths, args.year, args.months)
        if args.command in ("validate", "run"):
            cmd_validate(paths)
        if args.command in ("features", "run"):
            cmd_features(paths)
        if args.command in ("train", "run"):
            cmd_train(paths, args.max_train_rows)
    except (ValueError, quality.DataQualityError) as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
