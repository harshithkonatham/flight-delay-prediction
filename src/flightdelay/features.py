"""Turn bronze flight records into a model-ready feature table using DuckDB."""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb

from flightdelay.config import DELAY_THRESHOLD_MINUTES, POST_DEPARTURE_COLUMNS

log = logging.getLogger(__name__)

# Everything below is known when the flight is scheduled, before it departs.
CATEGORICAL_FEATURES = ["airline", "origin", "dest"]
NUMERIC_FEATURES = [
    "month",
    "day_of_month",
    "day_of_week",
    "dep_hour",
    "arr_hour",
    "scheduled_minutes",
    "distance",
    # How busy the origin airport is in that departure hour on that day.
    "origin_hourly_departures",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "delayed"

# No values are formatted into this SQL: the threshold is a bound parameter and
# the input files are registered as the "bronze" view through DuckDB's API.
_FEATURE_SQL = """
SELECT
    FlightDate                                   AS flight_date,
    Reporting_Airline                            AS airline,
    Origin                                       AS origin,
    Dest                                         AS dest,
    Month                                        AS month,
    DayofMonth                                   AS day_of_month,
    DayOfWeek                                    AS day_of_week,
    (CRSDepTime // 100) % 24                     AS dep_hour,
    (CRSArrTime // 100) % 24                     AS arr_hour,
    CRSElapsedTime                               AS scheduled_minutes,
    Distance                                     AS distance,
    count(*) OVER (
        PARTITION BY FlightDate, Origin, (CRSDepTime // 100) % 24
    )                                            AS origin_hourly_departures,
    ArrDelayMinutes >= $threshold                AS delayed
FROM bronze
-- Cancelled and diverted flights have no arrival delay to learn from.
WHERE coalesce(Cancelled, 0) = 0
  AND coalesce(Diverted, 0) = 0
  AND ArrDelayMinutes IS NOT NULL
  AND CRSElapsedTime > 0
"""


def assert_no_leakage(features: list[str]) -> None:
    """Fail loudly if a post-departure column sneaks into the feature list."""
    lowered = {c.lower() for c in POST_DEPARTURE_COLUMNS}
    leaked = [f for f in features if f.lower() in lowered]
    if leaked:
        raise ValueError(f"Post-departure columns used as features: {leaked}")


def build_features(bronze_glob: str, out_path: Path) -> int:
    """Read all bronze Parquet files matching bronze_glob and write the feature table."""
    assert_no_leakage(FEATURES)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_path.with_suffix(".parquet.part")
    con = duckdb.connect()
    try:
        con.read_parquet(bronze_glob).create_view("bronze")
        table = con.sql(_FEATURE_SQL, params={"threshold": DELAY_THRESHOLD_MINUTES})
        table.order("flight_date").write_parquet(str(tmp_out), compression="zstd")
        tmp_out.replace(out_path)
        row = con.execute("SELECT count(*) FROM read_parquet(?)", [str(out_path)]).fetchone()
    finally:
        con.close()
    rows = int(row[0]) if row else 0
    log.info("Built %s feature rows -> %s", f"{rows:,}", out_path)
    return rows
