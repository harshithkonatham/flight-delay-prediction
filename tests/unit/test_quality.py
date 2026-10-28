import json
from pathlib import Path

import duckdb
import pytest

from flightdelay import quality
from flightdelay.ingest import zip_to_parquet
from tests.factories import flight_rows, write_zip


def _parquet(tmp_path: Path, rows: list[dict[str, str]]) -> str:
    out = tmp_path / "flights.parquet"
    zip_to_parquet(write_zip(tmp_path / "m.zip", rows), out)
    return str(out)


def _failed(report: quality.QualityReport) -> set[str]:
    return {c.name for c in report.failures}


def test_clean_data_passes_every_check(tmp_path: Path) -> None:
    report = quality.validate(_parquet(tmp_path, flight_rows(200)))
    assert report.ok
    assert report.row_count == 200
    assert not report.failures


def test_invalid_departure_time_fails(tmp_path: Path) -> None:
    rows = flight_rows(20)
    rows[0]["CRSDepTime"] = "1275"  # 75 minutes past the hour
    report = quality.run_checks(_parquet(tmp_path, rows))
    assert "valid_dep_time" in _failed(report)
    assert not report.ok


def test_bad_airport_code_and_same_origin_dest_fail(tmp_path: Path) -> None:
    rows = flight_rows(20)
    rows[0]["Origin"] = "atl"
    rows[1]["Dest"] = rows[1]["Origin"]
    report = quality.run_checks(_parquet(tmp_path, rows))
    assert {"airport_codes", "origin_ne_dest"} <= _failed(report)


def test_missing_key_value_fails(tmp_path: Path) -> None:
    rows = flight_rows(20)
    rows[3]["Reporting_Airline"] = ""
    report = quality.run_checks(_parquet(tmp_path, rows))
    assert "no_nulls:Reporting_Airline" in _failed(report)


def test_unexplained_missing_arrival_delay_fails(tmp_path: Path) -> None:
    rows = flight_rows(20)
    for row in rows[:5]:
        row["ArrDelayMinutes"] = ""
        row["Cancelled"] = "0.00"
    report = quality.run_checks(_parquet(tmp_path, rows))
    assert "arrival_delay_present" in _failed(report)


def test_duplicates_are_only_a_warning(tmp_path: Path) -> None:
    rows = flight_rows(20)
    rows.append(dict(rows[0]))
    report = quality.run_checks(_parquet(tmp_path, rows))
    assert _failed(report) == {"no_duplicates"}
    assert report.ok


def test_empty_file_fails(tmp_path: Path) -> None:
    report = quality.run_checks(_parquet(tmp_path, []))
    assert "not_empty" in _failed(report)
    assert not report.ok


def test_missing_columns_fails_schema_check(tmp_path: Path) -> None:
    path = tmp_path / "partial.parquet"
    duckdb.sql(f"COPY (SELECT 2024 AS Year) TO '{path}' (FORMAT parquet)")
    report = quality.run_checks(str(path))
    assert _failed(report) == {"schema"}


def test_validate_raises_and_still_saves_report(tmp_path: Path) -> None:
    rows = flight_rows(10)
    rows[0]["Distance"] = "-5"
    report_path = tmp_path / "reports" / "quality.json"

    with pytest.raises(quality.DataQualityError, match="positive_distance"):
        quality.validate(_parquet(tmp_path, rows), report_path)

    saved = json.loads(report_path.read_text())
    assert saved["ok"] is False
    assert any(c["name"] == "positive_distance" and not c["passed"] for c in saved["checks"])
