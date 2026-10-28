"""Shared settings: data source, column schema and on-disk layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

BTS_URL_TEMPLATE = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)

# First and last years the BTS on-time dataset covers that we accept as input.
MIN_YEAR = 1987
MAX_YEAR = 2100

# Columns kept from the ~110 in the raw BTS file, with the type we store them as.
RAW_SCHEMA: dict[str, str] = {
    "Year": "SMALLINT",
    "Month": "TINYINT",
    "DayofMonth": "TINYINT",
    "DayOfWeek": "TINYINT",
    "FlightDate": "DATE",
    "Reporting_Airline": "VARCHAR",
    "Tail_Number": "VARCHAR",
    "Flight_Number_Reporting_Airline": "INTEGER",
    "Origin": "VARCHAR",
    "Dest": "VARCHAR",
    "CRSDepTime": "SMALLINT",
    "CRSArrTime": "SMALLINT",
    "CRSElapsedTime": "DOUBLE",
    "Distance": "DOUBLE",
    "DepDelayMinutes": "DOUBLE",
    "ArrDelayMinutes": "DOUBLE",
    "Cancelled": "DOUBLE",
    "Diverted": "DOUBLE",
    "CarrierDelay": "DOUBLE",
    "WeatherDelay": "DOUBLE",
    "NASDelay": "DOUBLE",
    "SecurityDelay": "DOUBLE",
    "LateAircraftDelay": "DOUBLE",
}

# Columns only known after the flight has left or landed. Using any of them as a
# model input leaks the answer (the original notebook did this and scored 100%).
POST_DEPARTURE_COLUMNS: frozenset[str] = frozenset(
    {
        "DepDelayMinutes",
        "ArrDelayMinutes",
        "CarrierDelay",
        "WeatherDelay",
        "NASDelay",
        "SecurityDelay",
        "LateAircraftDelay",
        "Cancelled",
        "Diverted",
    }
)

# A flight counts as delayed when it arrives 15+ minutes late (the FAA definition).
DELAY_THRESHOLD_MINUTES = 15


@dataclass(frozen=True)
class Paths:
    """Where each pipeline stage reads and writes, relative to one root folder."""

    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def bronze(self) -> Path:
        return self.root / "data" / "bronze"

    @property
    def features(self) -> Path:
        return self.root / "data" / "features" / "flights.parquet"

    @property
    def reports(self) -> Path:
        return self.root / "artifacts" / "reports"

    @property
    def models(self) -> Path:
        return self.root / "artifacts" / "models"

    def ensure(self) -> None:
        for folder in (self.raw, self.bronze, self.features.parent, self.reports, self.models):
            folder.mkdir(parents=True, exist_ok=True)
