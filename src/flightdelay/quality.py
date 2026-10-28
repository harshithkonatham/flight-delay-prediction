"""Data-quality checks run on the bronze layer before anything downstream uses it."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

import duckdb

from flightdelay.config import RAW_SCHEMA

KEY_COLUMNS = ("FlightDate", "Reporting_Airline", "Origin", "Dest", "CRSDepTime")


class Severity(StrEnum):
    ERROR = "error"  # pipeline stops
    WARNING = "warning"  # logged in the report, pipeline continues


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    severity: Severity
    detail: str


@dataclass
class QualityReport:
    source: str
    row_count: int
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks if c.severity is Severity.ERROR)

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]

    def to_json(self) -> str:
        return json.dumps({**asdict(self), "ok": self.ok}, indent=2, default=str)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())


class DataQualityError(RuntimeError):
    def __init__(self, report: QualityReport) -> None:
        names = ", ".join(c.name for c in report.failures if c.severity is Severity.ERROR)
        super().__init__(f"Data-quality checks failed for {report.source}: {names}")
        self.report = report


def _scalar(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    row = con.execute(sql).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def run_checks(parquet_glob: str, *, max_missing_arrival_ratio: float = 0.05) -> QualityReport:
    """Run every check against the Parquet file(s) matching parquet_glob."""
    con = duckdb.connect()
    try:
        con.read_parquet(parquet_glob).create_view("flights")
        rows = _scalar(con, "SELECT count(*) FROM flights")
        report = QualityReport(source=parquet_glob, row_count=rows)
        add = report.checks.append

        columns = {r[0] for r in con.execute("DESCRIBE flights").fetchall()}
        missing = sorted(set(RAW_SCHEMA) - columns)
        add(CheckResult("schema", not missing, Severity.ERROR, f"missing columns: {missing}"))
        if missing:
            return report

        add(CheckResult("not_empty", rows > 0, Severity.ERROR, f"{rows} rows"))
        if rows == 0:
            return report

        for col in KEY_COLUMNS:
            nulls = _scalar(con, f'SELECT count(*) FROM flights WHERE "{col}" IS NULL')  # noqa: S608
            add(CheckResult(f"no_nulls:{col}", nulls == 0, Severity.ERROR, f"{nulls} nulls"))

        bad_time = _scalar(
            con,
            "SELECT count(*) FROM flights WHERE CRSDepTime NOT BETWEEN 0 AND 2400 "
            "OR CRSDepTime % 100 >= 60",
        )
        add(CheckResult("valid_dep_time", bad_time == 0, Severity.ERROR, f"{bad_time} invalid"))

        bad_codes = _scalar(
            con,
            "SELECT count(*) FROM flights WHERE NOT regexp_full_match(Origin, '[A-Z0-9]{3}') "
            "OR NOT regexp_full_match(Dest, '[A-Z0-9]{3}')",
        )
        add(CheckResult("airport_codes", bad_codes == 0, Severity.ERROR, f"{bad_codes} invalid"))

        same = _scalar(con, "SELECT count(*) FROM flights WHERE Origin = Dest")
        add(CheckResult("origin_ne_dest", same == 0, Severity.ERROR, f"{same} rows"))

        bad_distance = _scalar(con, "SELECT count(*) FROM flights WHERE Distance <= 0")
        add(CheckResult("positive_distance", bad_distance == 0, Severity.ERROR, f"{bad_distance}"))

        negative = _scalar(con, "SELECT count(*) FROM flights WHERE ArrDelayMinutes < 0")
        add(CheckResult("non_negative_delay", negative == 0, Severity.ERROR, f"{negative} rows"))

        # Arrival delay is legitimately missing for cancelled/diverted flights only.
        unexplained = _scalar(
            con,
            "SELECT count(*) FROM flights WHERE ArrDelayMinutes IS NULL "
            "AND coalesce(Cancelled, 0) = 0 AND coalesce(Diverted, 0) = 0",
        )
        ratio = unexplained / rows
        add(
            CheckResult(
                "arrival_delay_present",
                ratio <= max_missing_arrival_ratio,
                Severity.ERROR,
                f"{unexplained} flown flights without arrival delay ({ratio:.2%})",
            )
        )

        dupes = _scalar(
            con,
            "SELECT count(*) - count(DISTINCT (FlightDate, Reporting_Airline, "
            "Flight_Number_Reporting_Airline, Origin, CRSDepTime)) FROM flights",
        )
        add(CheckResult("no_duplicates", dupes == 0, Severity.WARNING, f"{dupes} duplicate rows"))
        return report
    finally:
        con.close()


def validate(parquet_glob: str, report_path: Path | None = None) -> QualityReport:
    """Run checks, save the report, and raise DataQualityError if any error check failed."""
    report = run_checks(parquet_glob)
    if report_path is not None:
        report.save(report_path)
    if not report.ok:
        raise DataQualityError(report)
    return report
