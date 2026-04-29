import csv
import os
import tempfile

from density2sse.train.metrics_plot import export_epoch_metric_plots


FIELDS = [
    "model_name",
    "run_id",
    "epoch",
    "split",
    "center_error",
    "angle_error",
    "length_error",
    "coverage_ratio",
    "clash_voxels",
    "loss_total",
]


def _write_rows(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _row(epoch, split, loss, center=1.0, angle=1.0, length=1.0, cov=1.0, clash=1.0):
    return {
        "model_name": "m",
        "run_id": "r",
        "epoch": epoch,
        "split": split,
        "center_error": center,
        "angle_error": angle,
        "length_error": length,
        "coverage_ratio": cov,
        "clash_voxels": clash,
        "loss_total": loss,
    }


def test_export_summary_and_per_metric_files():
    with tempfile.TemporaryDirectory() as td:
        metrics = os.path.join(td, "metrics.csv")
        plots = os.path.join(td, "plots")
        _write_rows(
            metrics,
            [
                _row(1, "train", 2.0),
                _row(1, "val", 2.1),
                _row(2, "train", 1.8),
                _row(2, "val", 1.9),
            ],
        )
        files = export_epoch_metric_plots(metrics, plots, 2, per_metric=True, metrics_every_n_epochs=1)
        assert os.path.isfile(os.path.join(plots, "epoch_0002_summary.png"))
        assert len(files) == 7
        assert os.path.isfile(os.path.join(plots, "epoch_0002_loss_total.png"))


def test_duplicate_epoch_split_uses_last_row():
    with tempfile.TemporaryDirectory() as td:
        metrics = os.path.join(td, "metrics.csv")
        plots = os.path.join(td, "plots")
        _write_rows(
            metrics,
            [
                _row(1, "train", 3.0),
                _row(1, "val", 3.0),
                _row(2, "train", 2.0),
                _row(2, "val", 2.0),
                _row(2, "val", 1.0),  # duplicate should win (final exact style)
            ],
        )
        files = export_epoch_metric_plots(metrics, plots, 2, per_metric=False, metrics_every_n_epochs=1)
        assert len(files) == 1
        assert os.path.isfile(os.path.join(plots, "epoch_0002_summary.png"))


def test_sparse_placeholder_rows_ignored():
    with tempfile.TemporaryDirectory() as td:
        metrics = os.path.join(td, "metrics.csv")
        plots = os.path.join(td, "plots")
        _write_rows(
            metrics,
            [
                _row(1, "train", 3.0, center=0.0, angle=0.0, length=0.0, cov=0.0, clash=0.0),
                _row(1, "val", 3.1, center=0.0, angle=0.0, length=0.0, cov=0.0, clash=0.0),
                _row(2, "train", 2.5, center=1.0, angle=1.0, length=1.0, cov=1.0, clash=1.0),
                _row(2, "val", 2.6, center=1.1, angle=1.1, length=1.1, cov=1.1, clash=1.1),
            ],
        )
        files = export_epoch_metric_plots(metrics, plots, 2, per_metric=True, metrics_every_n_epochs=2)
        assert len(files) == 7
        assert os.path.isfile(os.path.join(plots, "epoch_0002_center_error.png"))
