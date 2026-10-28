"""Train a small model on synthetic flights: `python -m tests.demo_model <out_dir>`.

CI uses it to give the container a real model to load without downloading data.
"""

import sys
import tempfile
from pathlib import Path

from flightdelay import features, train
from flightdelay.ingest import zip_to_parquet
from tests.factories import flight_rows, write_zip


def main(out_dir: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for month in (1, 2):
            zip_path = write_zip(work / f"m{month}.zip", flight_rows(3000, month=month, seed=month))
            zip_to_parquet(zip_path, work / "bronze" / f"m{month}.parquet")
        features.build_features(str(work / "bronze" / "*.parquet"), work / "features.parquet")
        train.train(work / "features.parquet", out_dir)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
