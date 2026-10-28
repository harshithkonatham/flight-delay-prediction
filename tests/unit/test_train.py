import json
from pathlib import Path

import pandas as pd
import pytest

from flightdelay import train


def _frame(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"flight_date": pd.to_datetime(dates), "delayed": [True, False] * 5})


def test_time_split_puts_later_dates_in_test() -> None:
    df = _frame([f"2024-01-{d:02d}" for d in range(1, 11)])
    split = train.time_split(df, test_fraction=0.2)
    assert split.train["flight_date"].max() < split.test["flight_date"].min()
    assert split.cutoff == "2024-01-09"
    assert len(split.train) + len(split.test) == len(df)


@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.5])
def test_time_split_rejects_bad_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        train.time_split(_frame(["2024-01-01"] * 5 + ["2024-01-02"] * 5), fraction)


def test_time_split_needs_two_dates() -> None:
    with pytest.raises(ValueError, match="two dates"):
        train.time_split(_frame(["2024-01-01"] * 10))


def test_candidates_include_a_baseline_to_beat() -> None:
    assert "baseline_prior" in train.candidate_models()


def test_trained_model_beats_baseline(trained_model_dir: Path) -> None:
    meta = json.loads((trained_model_dir / train.METADATA_FILE).read_text())
    metrics = meta["metrics"]
    assert meta["model_name"] != "baseline_prior"
    assert metrics[meta["model_name"]]["roc_auc"] > metrics["baseline_prior"]["roc_auc"] + 0.05


def test_metadata_records_split_and_features(trained_model_dir: Path) -> None:
    meta = json.loads((trained_model_dir / train.METADATA_FILE).read_text())
    assert meta["split"]["train_rows"] > 0
    assert meta["split"]["test_rows"] > 0
    assert "arrdelayminutes" not in [f.lower() for f in meta["features"]]


def test_load_model_verifies_hash(trained_model_dir: Path) -> None:
    loaded = train.load_model(trained_model_dir)
    assert loaded.version.startswith(loaded.metadata["model_name"] + "@")


def test_load_model_refuses_tampered_file(trained_model_dir: Path, tmp_path: Path) -> None:
    for name in (train.MODEL_FILE, train.METADATA_FILE):
        (tmp_path / name).write_bytes((trained_model_dir / name).read_bytes())
    with (tmp_path / train.MODEL_FILE).open("ab") as fh:
        fh.write(b"tampered")
    with pytest.raises(ValueError, match="does not match"):
        train.load_model(tmp_path)


def test_train_rejects_single_class(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "flight_date": pd.to_datetime(["2024-01-01", "2024-01-02"] * 10),
            "airline": "AA",
            "origin": "ATL",
            "dest": "LAX",
            **{c: 1 for c in train.NUMERIC_FEATURES},
            "delayed": False,
        }
    )
    path = tmp_path / "f.parquet"
    df.to_parquet(path)
    with pytest.raises(ValueError, match="one class"):
        train.train(path, tmp_path / "models")
