"""Draw the README charts from the feature table and trained-model metadata.

uv run --extra report python scripts/make_figures.py
"""

import json
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "features" / "flights.parquet"
MODEL_META = ROOT / "artifacts" / "models" / "model.json"
OUT = ROOT / "reports"
FIGURES = OUT / "figures"

INK, MUTED, ACCENT, GRID = "#1f2937", "#6b7280", "#2563eb", "#e5e7eb"


def style(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def query(sql: str) -> list[tuple]:
    con = duckdb.connect()
    con.read_parquet(str(FEATURES)).create_view("f")
    return con.execute(sql).fetchall()


def delay_by_hour() -> None:
    rows = query("SELECT dep_hour, avg(delayed::INT) FROM f GROUP BY 1 ORDER BY 1")
    hours, rates = zip(*rows, strict=True)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(hours, [r * 100 for r in rates], color=ACCENT, width=0.75)
    style(ax, "Share of flights arriving 15+ min late, by scheduled departure hour", "% delayed")
    ax.set_xlabel("Scheduled departure hour (local)", color=MUTED)
    ax.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    fig.savefig(FIGURES / "delay_by_hour.png", dpi=150)


def delay_by_airline() -> None:
    rows = query(
        "SELECT airline, avg(delayed::INT) AS r FROM f GROUP BY 1 HAVING count(*) > 20000 "
        "ORDER BY r"
    )
    airlines, rates = zip(*rows, strict=True)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(airlines, [r * 100 for r in rates], color=ACCENT, width=0.7)
    style(ax, "Share of flights arriving 15+ min late, by airline", "% delayed")
    fig.tight_layout()
    fig.savefig(FIGURES / "delay_by_airline.png", dpi=150)


def model_comparison(meta: dict) -> None:
    names = list(meta["metrics"])
    labels = [n.replace("_", " ") for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    for ax, metric, title in zip(axes, ["roc_auc", "pr_auc"], ["ROC-AUC", "PR-AUC"], strict=True):
        values = [meta["metrics"][n][metric] for n in names]
        colors = [ACCENT if n == meta["model_name"] else "#cbd5e1" for n in names]
        bars = ax.barh(labels, values, color=colors)
        ax.bar_label(bars, fmt="%.3f", padding=3, color=INK, fontsize=9)
        ax.set_xlim(0, max(values) * 1.25)
        style(ax, title, "")
        ax.grid(axis="x", color=GRID)
        ax.grid(axis="y", visible=False)
    fig.suptitle(
        f"Test set: flights from {meta['split']['cutoff_date']} onward (unseen in training)",
        x=0.01,
        ha="left",
        fontsize=10,
        color=MUTED,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "model_comparison.png", dpi=150)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    meta = json.loads(MODEL_META.read_text())
    delay_by_hour()
    delay_by_airline()
    model_comparison(meta)
    summary = {k: meta[k] for k in ("model_name", "features", "split", "metrics")}
    (OUT / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
