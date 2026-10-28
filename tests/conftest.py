from pathlib import Path

import pytest

from flightdelay import features, train
from flightdelay.config import Paths
from flightdelay.ingest import zip_to_parquet
from tests.factories import flight_rows, write_zip


def seed_raw_months(paths: Paths, months: list[int], rows_per_month: int = 1500) -> None:
    """Place fake BTS zips where the downloader looks, so no network is needed."""
    for month in months:
        rows = flight_rows(rows_per_month, month=month, seed=month)
        write_zip(paths.raw / f"bts_2024_{month:02d}.zip", rows)


@pytest.fixture(scope="session")
def trained_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A model trained once on synthetic data and shared by tests that need one."""
    root = tmp_path_factory.mktemp("trained")
    paths = Paths(root)
    paths.ensure()
    for month in (1, 2):
        rows = flight_rows(2000, month=month, seed=month)
        zip_to_parquet(
            write_zip(paths.raw / f"m{month}.zip", rows),
            paths.bronze / f"month={month}" / "flights.parquet",
        )
    features.build_features(str(paths.bronze / "**" / "*.parquet"), paths.features)
    train.train(paths.features, paths.models)
    return paths.models
