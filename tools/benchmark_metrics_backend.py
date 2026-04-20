"""Benchmark metrics kernel/backend speed and numerical deltas."""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from density2sse.config import resolve_config
from density2sse.data.dataset import HelixNPZDataset, collate_batch
from density2sse.model import registry as model_registry
from density2sse.train import metrics as metrics_mod


def _resolve_path(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(os.getcwd(), p)


def _resolve_device(cfg: Dict[str, Any]) -> torch.device:
    ds = str(cfg["training"].get("device", "auto")).lower().strip()
    if ds in {"auto", ""}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if ds.startswith("cuda") and torch.cuda.is_available():
        return torch.device(ds)
    return torch.device("cpu")


def _load_loader(cfg: Dict[str, Any], split: str) -> DataLoader:
    data_cfg = cfg["data"]
    split_dir = _resolve_path(data_cfg[f"{split}_dir"])
    ds = HelixNPZDataset(split_dir, max_K=int(data_cfg["K_max"]), box_size=int(data_cfg["box_size"]))
    return DataLoader(
        ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["training"]["num_workers"]),
        collate_fn=collate_batch,
    )


def _run_once(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: Dict[str, Any],
    kernel_impl: str,
    backend: str,
    max_batches: int | None,
    stage_label: str,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    out = metrics_mod.aggregate_metrics_loader(
        model,
        loader,
        device,
        cfg,
        max_batches=max_batches,
        compute_coverage=True,
        compute_clash=True,
        log_every_n_batches=0,
        stage_label=stage_label,
        kernel_impl=kernel_impl,
        backend=backend,
        profile_components=True,
    )
    dt = time.perf_counter() - t0
    return {"elapsed_s": dt, "metrics": out}


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark metrics kernel/backend combinations.")
    ap.add_argument("-i", "--config", required=True, help="YAML config path")
    ap.add_argument("--split", default="val", choices=["train", "val"], help="Dataset split for benchmark")
    ap.add_argument("--max-batches", type=int, default=8, help="Cap number of batches for quick profiling")
    ap.add_argument(
        "--pairs",
        nargs="+",
        default=["legacy:numpy", "optimized:numpy", "optimized:auto"],
        help="Kernel/backend pairs, e.g. legacy:numpy optimized:numpy optimized:torch",
    )
    ap.add_argument("--output-json", default="outputs/metrics_backend_benchmark.json", help="Output JSON file")
    args = ap.parse_args()

    cfg = resolve_config(args.config)
    dev = _resolve_device(cfg)
    model = model_registry.build_model(cfg).to(dev)
    if dev.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    loader = _load_loader(cfg, args.split)

    runs: List[Dict[str, Any]] = []
    baseline = None
    for pair in args.pairs:
        if ":" not in pair:
            raise ValueError(f"Bad pair {pair}; expected kernel:backend")
        kernel_impl, backend = pair.split(":", 1)
        res = _run_once(
            model,
            loader,
            dev,
            cfg,
            kernel_impl=kernel_impl.strip(),
            backend=backend.strip(),
            max_batches=args.max_batches,
            stage_label=f"bench {pair}",
        )
        row = {
            "pair": pair,
            "elapsed_s": res["elapsed_s"],
            "metrics": res["metrics"],
            "delta_vs_baseline": {},
        }
        if baseline is None:
            baseline = res["metrics"]
        else:
            for k in baseline:
                row["delta_vs_baseline"][k] = float(res["metrics"][k] - baseline[k])
        runs.append(row)

    out = {
        "config": args.config,
        "split": args.split,
        "device": str(dev),
        "max_batches": args.max_batches,
        "runs": runs,
    }
    out_path = _resolve_path(args.output_json)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
