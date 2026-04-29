"""Per-epoch metrics plotting helpers."""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from density2sse.utils.logging_utils import setup_logging

LOG = setup_logging(name="density2sse.train.metrics_plot")

METRIC_NAMES = [
    "loss_total",
    "center_error",
    "angle_error",
    "length_error",
    "coverage_ratio",
    "clash_voxels",
]
GEOMETRIC_NAMES = [
    "center_error",
    "angle_error",
    "length_error",
    "coverage_ratio",
    "clash_voxels",
]


def _to_int(v: object, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _to_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _is_sparse_placeholder(row: Dict[str, float], metrics_every_n_epochs: int) -> bool:
    if metrics_every_n_epochs <= 1:
        return False
    epoch = _to_int(row.get("epoch", 0))
    if epoch % metrics_every_n_epochs == 0:
        return False
    return all(abs(_to_float(row.get(k, 0.0))) < 1e-12 for k in GEOMETRIC_NAMES)


def _load_rows(metrics_csv_path: str, upto_epoch: int, metrics_every_n_epochs: int) -> List[Dict[str, float]]:
    if not os.path.isfile(metrics_csv_path):
        return []
    dedup: Dict[Tuple[int, str], Dict[str, float]] = {}
    with open(metrics_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            epoch = _to_int(raw.get("epoch"))
            split = str(raw.get("split", "")).strip().lower()
            if split not in ("train", "val"):
                continue
            if epoch > upto_epoch:
                continue
            row: Dict[str, float] = {"epoch": float(epoch), "split": split}  # type: ignore[assignment]
            for name in METRIC_NAMES:
                row[name] = _to_float(raw.get(name))
            # Keep the last row for duplicate (epoch, split), e.g. final_exact_eval.
            dedup[(epoch, split)] = row
    out = []
    for _, row in sorted(dedup.items(), key=lambda x: (x[0][0], x[0][1])):
        if _is_sparse_placeholder(row, metrics_every_n_epochs):
            continue
        out.append(row)
    return out


def _split_series(rows: List[Dict[str, float]], metric_name: str) -> Dict[str, Tuple[List[int], List[float]]]:
    series = {"train": ([], []), "val": ([], [])}
    for row in rows:
        split = str(row["split"])
        xs, ys = series[split]
        xs.append(_to_int(row["epoch"]))
        ys.append(_to_float(row.get(metric_name)))
    return series


def _save_summary(rows: List[Dict[str, float]], out_path: str, epoch: int) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for idx, name in enumerate(METRIC_NAMES):
        ax = axes[idx // 3][idx % 3]
        series = _split_series(rows, name)
        tx, ty = series["train"]
        vx, vy = series["val"]
        if tx:
            ax.plot(tx, ty, marker="o", linewidth=1.5, label="train")
        if vx:
            ax.plot(vx, vy, marker="o", linewidth=1.5, label="val")
        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(f"Training metrics up to epoch {epoch}")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_per_metric(rows: List[Dict[str, float]], plot_dir: str, epoch: int) -> List[str]:
    files: List[str] = []
    for name in METRIC_NAMES:
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        series = _split_series(rows, name)
        tx, ty = series["train"]
        vx, vy = series["val"]
        if tx:
            ax.plot(tx, ty, marker="o", linewidth=1.8, label="train")
        if vx:
            ax.plot(vx, vy, marker="o", linewidth=1.8, label="val")
        ax.set_title(f"{name} up to epoch {epoch}")
        ax.set_xlabel("epoch")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        out_path = os.path.join(plot_dir, f"epoch_{epoch:04d}_{name}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        files.append(out_path)
    return files


def export_epoch_metric_plots(
    metrics_csv_path: str,
    plot_dir: str,
    epoch: int,
    *,
    per_metric: bool = True,
    metrics_every_n_epochs: int = 1,
) -> List[str]:
    """Export summary and optional per-metric trend plots up to ``epoch``."""
    rows = _load_rows(metrics_csv_path, upto_epoch=epoch, metrics_every_n_epochs=max(1, metrics_every_n_epochs))
    if not rows:
        LOG.info("metrics plot skipped: no usable rows for epoch %d", epoch)
        return []
    os.makedirs(plot_dir, exist_ok=True)
    files: List[str] = []
    summary = os.path.join(plot_dir, f"epoch_{epoch:04d}_summary.png")
    _save_summary(rows, summary, epoch)
    files.append(summary)
    if per_metric:
        files.extend(_save_per_metric(rows, plot_dir, epoch))
    return files
