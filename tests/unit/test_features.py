from pathlib import Path

import pandas as pd
import pytest

from flightdelay import features
from flightdelay.ingest import zip_to_parquet
from tests.factories import flight_rows, write_zip


def _build(tmp_path: Path, rows: list[dict[str, str]]) -> pd.DataFrame:
    bronze = tmp_path / "bronze.parquet"
    zip_to_parquet(write_zip(tmp_path / "m.zip", rows), bronze)
    out = tmp_path / "features.parquet"
    features.build_features(str(bronze), out)
    return pd.read_parquet(out)


def test_feature_table_has_expected_columns(tmp_path: Path) -> None:
    df = _build(tmp_path, flight_rows(100))
    assert set(df.columns) == {"flight_date", *features.FEATURES, features.TARGET}


def test_feature_list_contains_no_post_departure_columns() -> None:
    features.assert_no_leakage(features.FEATURES)


def test_leakage_guard_catches_post_departure_column() -> None:
    with pytest.raises(ValueError, match="arrdelayminutes"):
        features.assert_no_leakage(["airline", "arrdelayminutes"])


def test_cancelled_and_diverted_flights_are_dropped(tmp_path: Path) -> None:
    rows = flight_rows(10)
    for row in rows:
        row.update(Cancelled="0.00", Diverted="0.00", ArrDelayMinutes="5")
    rows[0].update(Cancelled="1.00", ArrDelayMinutes="")
    rows[1].update(Diverted="1.00")

    df = _build(tmp_path, rows)

    assert len(df) == 8


def test_target_uses_15_minute_threshold(tmp_path: Path) -> None:
    rows = flight_rows(3)
    for row, delay in zip(rows, ["14", "15", "90"], strict=True):
        row["ArrDelayMinutes"], row["Cancelled"] = delay, "0.00"
    df = _build(tmp_path, rows).sort_values("flight_date")
    by_delay = dict(zip(["14", "15", "90"], [False, True, True], strict=True))
    assert sorted(df["delayed"].tolist()) == sorted(by_delay.values())


def test_hours_are_derived_from_scheduled_times(tmp_path: Path) -> None:
    rows = flight_rows(1)
    rows[0].update(CRSDepTime="2400", CRSArrTime="0905", Cancelled="0.00", ArrDelayMinutes="3")
    df = _build(tmp_path, rows)
    assert df.loc[0, "dep_hour"] == 0  # 24:00 is midnight
    assert df.loc[0, "arr_hour"] == 9


def test_origin_hourly_departures_counts_same_airport_and_hour(tmp_path: Path) -> None:
    rows = flight_rows(3)
    for i, row in enumerate(rows):
        row.update(
            FlightDate="2024-01-05",
            Origin="ATL",
            Dest="LAX",
            CRSDepTime="0815" if i < 2 else "1300",
            Cancelled="0.00",
            ArrDelayMinutes="0",
        )
    df = _build(tmp_path, rows)
    assert sorted(df["origin_hourly_departures"].tolist()) == [1, 2, 2]
