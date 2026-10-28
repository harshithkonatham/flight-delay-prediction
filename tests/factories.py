"""Builds small fake BTS files so tests never need the network or the real 30 MB downloads."""

from __future__ import annotations

import csv
import io
import random
import zipfile
from datetime import date, timedelta
from pathlib import Path

from flightdelay.config import RAW_SCHEMA

AIRLINES = ["AA", "DL", "UA", "WN", "B6"]
AIRPORTS = ["ATL", "LAX", "ORD", "DFW", "JFK", "SFO", "SEA", "DEN"]
# Columns the real file has but we drop, to prove ingestion ignores them.
EXTRA_COLUMNS = ["Quarter", "OriginCityName", "TaxiOut"]


def flight_rows(n: int, *, year: int = 2024, month: int = 1, seed: int = 7) -> list[dict[str, str]]:
    """Fake flights where late-evening departures and some airlines are delayed more often,
    so a model trained on them has a real signal to find."""
    rng = random.Random(seed)
    start = date(year, month, 1)
    rows = []
    for i in range(n):
        day = start + timedelta(days=rng.randrange(28))
        origin, dest = rng.sample(AIRPORTS, 2)
        airline = rng.choice(AIRLINES)
        hour = rng.randrange(5, 24)
        dep = hour * 100 + rng.choice([0, 15, 30, 45])
        elapsed = rng.randrange(60, 360)
        arr = ((hour + elapsed // 60) % 24) * 100 + 5
        p_delay = 0.08 + 0.025 * max(0, hour - 12) + (0.15 if airline in ("B6", "WN") else 0)
        cancelled = rng.random() < 0.01
        delayed = rng.random() < p_delay
        arr_delay = (
            "" if cancelled else str(rng.randrange(15, 180) if delayed else rng.randrange(0, 14))
        )
        rows.append(
            {
                "Year": str(day.year),
                "Quarter": "1",
                "Month": str(day.month),
                "DayofMonth": str(day.day),
                "DayOfWeek": str(day.isoweekday()),
                "FlightDate": day.isoformat(),
                "Reporting_Airline": airline,
                "Tail_Number": f"N{rng.randrange(100, 999)}XX",
                "Flight_Number_Reporting_Airline": str(1000 + i),
                "Origin": origin,
                "OriginCityName": "Somewhere, XX",
                "Dest": dest,
                "CRSDepTime": f"{dep:04d}",
                "CRSArrTime": f"{arr:04d}",
                "CRSElapsedTime": f"{elapsed}.00",
                "Distance": f"{elapsed * 7}.00",
                "DepDelayMinutes": "" if cancelled else arr_delay,
                "ArrDelayMinutes": arr_delay,
                "Cancelled": "1.00" if cancelled else "0.00",
                "Diverted": "0.00",
                "TaxiOut": "12.00",
                "CarrierDelay": "",
                "WeatherDelay": "",
                "NASDelay": "",
                "SecurityDelay": "",
                "LateAircraftDelay": "",
            }
        )
    return rows


def csv_bytes(rows: list[dict[str, str]]) -> bytes:
    header = list(RAW_SCHEMA) + EXTRA_COLUMNS
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=header, quoting=csv.QUOTE_NONNUMERIC)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode()


def zip_bytes(rows: list[dict[str, str]], *, member: str = "On_Time_2024_1.csv") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, csv_bytes(rows))
        archive.writestr("readme.html", "<html></html>")
    return buf.getvalue()


def write_zip(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(zip_bytes(rows))
    return path


class FakeOpener:
    """Stands in for urllib: returns canned bytes and records requested URLs."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.urls: list[str] = []

    def __call__(self, url: str, timeout: float) -> io.BytesIO:
        self.urls.append(url)
        return io.BytesIO(self.payload)
