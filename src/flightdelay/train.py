"""Train and compare delay classifiers, then save the best one with its metadata."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from flightdelay import __version__
from flightdelay.features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET

log = logging.getLogger(__name__)

MODEL_FILE = "model.joblib"
METADATA_FILE = "model.json"
RANDOM_STATE = 42


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    test: pd.DataFrame
    cutoff: str


def time_split(df: pd.DataFrame, test_fraction: float = 0.2) -> Split:
    """Train on earlier dates and test on later ones.

    A random split would let the model see flights from the same days it is
    tested on, which overstates how well it predicts the future.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    dates = np.sort(df["flight_date"].unique())
    if len(dates) < 2:
        raise ValueError("Need flights on at least two dates for a time-based split")
    cutoff_index = min(len(dates) - 1, max(1, int(len(dates) * (1 - test_fraction))))
    cutoff = dates[cutoff_index]
    train = df[df["flight_date"] < cutoff]
    test = df[df["flight_date"] >= cutoff]
    return Split(train=train, test=test, cutoff=str(pd.Timestamp(cutoff).date()))


def candidate_models() -> dict[str, Pipeline]:
    one_hot = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50),
                CATEGORICAL_FEATURES,
            ),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )
    # Gradient boosting handles categories natively but allows at most 255 per
    # column; rarer airports are grouped and unseen ones treated as missing.
    ordinal = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                    max_categories=250,
                ),
                CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )
    categorical_mask = [True] * len(CATEGORICAL_FEATURES) + [False] * len(NUMERIC_FEATURES)
    return {
        "baseline_prior": Pipeline([("model", DummyClassifier(strategy="prior"))]),
        "logistic_regression": Pipeline(
            [
                ("prep", one_hot),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("prep", ordinal),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        categorical_features=categorical_mask,
                        max_iter=300,
                        learning_rate=0.1,
                        class_weight="balanced",
                        early_stopping=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def evaluate(model: Pipeline, test: pd.DataFrame) -> dict[str, float]:
    y_true = test[TARGET].astype(int)
    proba = model.predict_proba(test[FEATURES])[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def train(
    features_path: Path,
    model_dir: Path,
    *,
    test_fraction: float = 0.2,
    max_train_rows: int | None = 1_000_000,
) -> dict[str, Any]:
    """Train every candidate, keep the one with the best PR-AUC, and save it."""
    df = pd.read_parquet(features_path)
    split = time_split(df, test_fraction)
    train_df = split.train
    if max_train_rows and len(train_df) > max_train_rows:
        train_df = train_df.sample(max_train_rows, random_state=RANDOM_STATE)
    if train_df[TARGET].nunique() < 2:
        raise ValueError("Training data contains only one class")

    results: dict[str, dict[str, float]] = {}
    fitted: dict[str, Pipeline] = {}
    for name, model in candidate_models().items():
        log.info("Training %s on %s rows", name, f"{len(train_df):,}")
        model.fit(train_df[FEATURES], train_df[TARGET].astype(int))
        results[name] = evaluate(model, split.test)
        fitted[name] = model
        log.info("%s: %s", name, json.dumps({k: round(v, 4) for k, v in results[name].items()}))

    best = max(results, key=lambda n: results[n]["pr_auc"])
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / MODEL_FILE
    joblib.dump(fitted[best], model_path)

    metadata: dict[str, Any] = {
        "model_name": best,
        "package_version": __version__,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "features": FEATURES,
        "split": {
            "cutoff_date": split.cutoff,
            "train_rows": len(train_df),
            "test_rows": len(split.test),
            "test_delay_rate": float(split.test[TARGET].mean()),
        },
        "metrics": results,
        "sha256": _sha256(model_path),
    }
    (model_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2))
    log.info("Saved %s to %s", best, model_path)
    return metadata


@dataclass(frozen=True)
class LoadedModel:
    pipeline: Pipeline
    metadata: dict[str, Any]

    @property
    def version(self) -> str:
        return f"{self.metadata['model_name']}@{self.metadata['sha256'][:12]}"


def load_model(model_dir: Path) -> LoadedModel:
    """Load a saved model after checking it matches the hash recorded at training.

    joblib files can run code when loaded, so a model whose bytes changed since
    training is refused rather than trusted.
    """
    metadata = json.loads((model_dir / METADATA_FILE).read_text())
    model_path = model_dir / MODEL_FILE
    actual = _sha256(model_path)
    if actual != metadata.get("sha256"):
        raise ValueError(f"Model file hash {actual[:12]} does not match its metadata")
    pipeline = joblib.load(model_path)  # hash verified above
    return LoadedModel(pipeline=pipeline, metadata=metadata)
