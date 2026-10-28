"""Download monthly BTS on-time files and store them as typed Parquet ("bronze" layer)."""

from __future__ import annotations

import logging
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import BinaryIO, Protocol

import duckdb

from flightdelay.config import BTS_URL_TEMPLATE, MAX_YEAR, MIN_YEAR, RAW_SCHEMA

log = logging.getLogger(__name__)

DOWNLOAD_TIMEOUT_SECONDS = 120
# A real monthly file is ~25-35 MB zipped / ~250 MB unzipped. Refuse anything far
# larger so a bad or hostile response cannot fill the disk (zip bomb protection).
MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024
MAX_UNZIPPED_BYTES = 2 * 1024 * 1024 * 1024


class Opener(Protocol):
    def __call__(self, url: str, timeout: float) -> BinaryIO: ...


def _default_opener(url: str, timeout: float) -> BinaryIO:
    if not url.startswith("https://"):
        raise ValueError(f"Refusing non-HTTPS URL: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "flightdelay/0.1"})  # noqa: S310
    # Scheme is checked above, so file:// and other schemes can never reach urlopen.
    return urllib.request.urlopen(request, timeout=timeout)  # type: ignore[no-any-return]  # noqa: S310  # nosec B310


def validate_period(year: int, month: int) -> None:
    """Reject months outside the range the dataset can contain."""
    if not MIN_YEAR <= year <= MAX_YEAR:
        raise ValueError(f"year must be between {MIN_YEAR} and {MAX_YEAR}, got {year}")
    if not 1 <= month <= 12:
        raise ValueError(f"month must be between 1 and 12, got {month}")


def source_url(year: int, month: int) -> str:
    validate_period(year, month)
    return BTS_URL_TEMPLATE.format(year=year, month=month)


def download_month(
    year: int,
    month: int,
    dest_dir: Path,
    *,
    opener: Opener = _default_opener,
    force: bool = False,
) -> Path:
    """Download one month's zip into dest_dir and return its path.

    Writes to a temporary file first and renames at the end, so an interrupted
    download never leaves a half-written file that looks complete.
    """
    url = source_url(year, month)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / f"bts_{year}_{month:02d}.zip"
    if target.exists() and not force:
        log.info("Already downloaded %s", target.name)
        return target

    log.info("Downloading %s", url)
    written = 0
    with (
        opener(url, DOWNLOAD_TIMEOUT_SECONDS) as response,
        tempfile.NamedTemporaryFile(dir=dest_dir, suffix=".part", delete=False) as tmp,
    ):
        tmp_path = Path(tmp.name)
        try:
            while chunk := response.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_DOWNLOAD_BYTES:
                    raise ValueError(f"Download larger than {MAX_DOWNLOAD_BYTES} bytes: {url}")
                tmp.write(chunk)
        except BaseException:
            tmp.close()
            tmp_path.unlink(missing_ok=True)
            raise

    if not zipfile.is_zipfile(tmp_path):
        tmp_path.unlink(missing_ok=True)
        raise ValueError(f"Response from {url} is not a zip file")
    tmp_path.replace(target)
    log.info("Saved %s (%.1f MB)", target.name, written / 1e6)
    return target


def _find_csv_member(archive: zipfile.ZipFile) -> zipfile.ZipInfo:
    members = [m for m in archive.infolist() if m.filename.lower().endswith(".csv")]
    if len(members) != 1:
        raise ValueError(f"Expected exactly one CSV in archive, found {len(members)}")
    member = members[0]
    if member.file_size > MAX_UNZIPPED_BYTES:
        raise ValueError(f"CSV is {member.file_size} bytes, over the {MAX_UNZIPPED_BYTES} limit")
    return member


def zip_to_parquet(zip_path: Path, out_path: Path) -> int:
    """Convert a BTS zip into a Parquet file with only RAW_SCHEMA columns.

    Returns the number of rows written.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "month.csv"
        with zipfile.ZipFile(zip_path) as archive:
            member = _find_csv_member(archive)
            # Copy the member to a fixed name instead of archive.extract(), so a
            # crafted member name like "../../x" cannot write outside tmp_dir.
            with archive.open(member) as src, csv_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)

        con = duckdb.connect()
        try:
            relation = con.read_csv(str(csv_path), header=True, all_varchar=True)
            missing = set(RAW_SCHEMA) - set(relation.columns)
            if missing:
                raise ValueError(f"{zip_path.name} is missing columns: {sorted(missing)}")
            casts = ", ".join(
                f'TRY_CAST(NULLIF("{name}", \'\') AS {dtype}) AS "{name}"'
                for name, dtype in RAW_SCHEMA.items()
            )
            typed = relation.select(casts)
            tmp_out = out_path.with_suffix(".parquet.part")
            typed.write_parquet(str(tmp_out), compression="zstd")
            tmp_out.replace(out_path)
            row = con.execute("SELECT count(*) FROM read_parquet(?)", [str(out_path)]).fetchone()
        finally:
            con.close()

    rows = int(row[0]) if row else 0
    log.info("Wrote %s rows to %s", f"{rows:,}", out_path.name)
    return rows


def ingest_month(
    year: int,
    month: int,
    raw_dir: Path,
    bronze_dir: Path,
    *,
    opener: Opener = _default_opener,
) -> Path:
    """Download (if needed) and convert one month. Returns the Parquet path."""
    zip_path = download_month(year, month, raw_dir, opener=opener)
    out_path = bronze_dir / f"year={year}" / f"month={month:02d}" / "flights.parquet"
    zip_to_parquet(zip_path, out_path)
    return out_path
