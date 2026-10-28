import json
from pathlib import Path

import pytest

from flightdelay import cli
from flightdelay.config import Paths
from tests.conftest import seed_raw_months
from tests.factories import flight_rows, write_zip


@pytest.mark.e2e
def test_run_command_builds_every_stage(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    paths = Paths(tmp_path)
    seed_raw_months(paths, [1, 2])

    code = cli.main(["--root", str(tmp_path), "run", "--year", "2024", "--months", "1", "2"])

    assert code == 0
    assert (paths.bronze / "year=2024" / "month=02" / "flights.parquet").exists()
    assert json.loads((paths.reports / "data_quality.json").read_text())["ok"] is True
    assert paths.features.exists()
    assert (paths.models / "model.joblib").exists()
    assert "Best model:" in capsys.readouterr().out


def test_stages_can_run_one_at_a_time(tmp_path: Path) -> None:
    paths = Paths(tmp_path)
    seed_raw_months(paths, [3, 4], rows_per_month=800)
    root = ["--root", str(tmp_path)]

    assert cli.main([*root, "ingest", "--year", "2024", "--months", "3", "4"]) == 0
    assert cli.main([*root, "validate"]) == 0
    assert cli.main([*root, "features"]) == 0
    assert cli.main([*root, "train", "--max-train-rows", "500"]) == 0
    meta = json.loads((paths.models / "model.json").read_text())
    assert meta["split"]["train_rows"] == 500


def test_invalid_month_exits_with_error(tmp_path: Path) -> None:
    assert cli.main(["--root", str(tmp_path), "ingest", "--year", "2024", "--months", "13"]) == 1


def test_bad_data_stops_the_pipeline(tmp_path: Path) -> None:
    paths = Paths(tmp_path)
    rows = flight_rows(50)
    rows[0]["Origin"] = "not-an-airport"
    write_zip(paths.raw / "bts_2024_01.zip", rows)

    code = cli.main(["--root", str(tmp_path), "run", "--year", "2024", "--months", "1"])

    assert code == 1
    assert not paths.features.exists()
