import io
import zipfile
from pathlib import Path

import duckdb
import pytest

from flightdelay import ingest
from flightdelay.config import RAW_SCHEMA
from tests.factories import FakeOpener, flight_rows, write_zip, zip_bytes


@pytest.mark.parametrize(("year", "month"), [(1986, 1), (2024, 0), (2024, 13), (2101, 6)])
def test_rejects_out_of_range_periods(year: int, month: int) -> None:
    with pytest.raises(ValueError):
        ingest.source_url(year, month)


def test_source_url_points_at_bts_over_https() -> None:
    url = ingest.source_url(2024, 3)
    assert url.startswith("https://transtats.bts.gov/")
    assert url.endswith("_2024_3.zip")


def test_default_opener_refuses_plain_http() -> None:
    with pytest.raises(ValueError, match="non-HTTPS"):
        ingest._default_opener("http://example.com/file.zip", timeout=1)


def test_download_saves_zip_and_skips_when_present(tmp_path: Path) -> None:
    opener = FakeOpener(zip_bytes(flight_rows(10)))

    first = ingest.download_month(2024, 1, tmp_path, opener=opener)
    second = ingest.download_month(2024, 1, tmp_path, opener=opener)

    assert first == second == tmp_path / "bts_2024_01.zip"
    assert zipfile.is_zipfile(first)
    assert len(opener.urls) == 1
    assert not list(tmp_path.glob("*.part"))


def test_download_rejects_non_zip_response(tmp_path: Path) -> None:
    opener = FakeOpener(b"<html>Service unavailable</html>")
    with pytest.raises(ValueError, match="not a zip"):
        ingest.download_month(2024, 1, tmp_path, opener=opener)
    assert not list(tmp_path.iterdir())


def test_download_stops_when_response_is_too_large(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ingest, "MAX_DOWNLOAD_BYTES", 10)
    opener = FakeOpener(zip_bytes(flight_rows(5)))
    with pytest.raises(ValueError, match="larger than"):
        ingest.download_month(2024, 1, tmp_path, opener=opener)
    assert not list(tmp_path.iterdir())


def test_zip_to_parquet_keeps_only_schema_columns_with_types(tmp_path: Path) -> None:
    zip_path = write_zip(tmp_path / "m.zip", flight_rows(50))
    out = tmp_path / "out" / "flights.parquet"

    rows = ingest.zip_to_parquet(zip_path, out)

    assert rows == 50
    described = duckdb.sql(f"DESCRIBE SELECT * FROM '{out}'").fetchall()
    types = {name: dtype for name, dtype, *_ in described}
    assert list(types) == list(RAW_SCHEMA)
    assert types["FlightDate"] == "DATE"
    assert types["CRSDepTime"] == "SMALLINT"


def test_zip_to_parquet_rejects_archive_without_single_csv(tmp_path: Path) -> None:
    zip_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("readme.html", "x")
    with pytest.raises(ValueError, match="exactly one CSV"):
        ingest.zip_to_parquet(zip_path, tmp_path / "o.parquet")


def test_zip_to_parquet_rejects_missing_columns(tmp_path: Path) -> None:
    zip_path = tmp_path / "m.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("data.csv", "Year,Month\n2024,1\n")
    with pytest.raises(ValueError, match="missing columns"):
        ingest.zip_to_parquet(zip_path, tmp_path / "o.parquet")


def test_zip_slip_member_name_cannot_escape(tmp_path: Path) -> None:
    buf = io.BytesIO(zip_bytes(flight_rows(5), member="../../escape.csv"))
    zip_path = tmp_path / "work" / "evil.zip"
    zip_path.parent.mkdir()
    zip_path.write_bytes(buf.getvalue())

    ingest.zip_to_parquet(zip_path, tmp_path / "work" / "o.parquet")

    assert not (tmp_path / "escape.csv").exists()
    assert not (tmp_path.parent / "escape.csv").exists()


def test_ingest_month_writes_partitioned_parquet(tmp_path: Path) -> None:
    opener = FakeOpener(zip_bytes(flight_rows(20)))
    out = ingest.ingest_month(2024, 2, tmp_path / "raw", tmp_path / "bronze", opener=opener)
    assert out == tmp_path / "bronze" / "year=2024" / "month=02" / "flights.parquet"
    assert out.exists()
