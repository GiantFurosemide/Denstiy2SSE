"""Console entrypoint for ``density2sse``."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

from density2sse import __version__
from density2sse.config import resolve_config, save_resolved, validate_config
from density2sse.model import registry as model_registry
from density2sse.data.synthetic_generator import SyntheticConfig, generate_dataset_split
from density2sse.export import export_pdb
from density2sse.utils.logging_utils import setup_logging
from density2sse.utils.seed import set_seed

LOG = setup_logging()


def _resolve_runtime_device(device_s: str, cuda_available: bool, command: str) -> str:
    """Map config device to runtime device with cluster-friendly defaults."""
    ds = str(device_s).strip().lower()
    if ds in {"auto", ""}:
        return "cuda" if cuda_available else "cpu"
    if ds == "cpu":
        # For cluster defaults, prefer GPU unless user explicitly forces CPU by env.
        if cuda_available and os.environ.get("DENSITY2SSE_FORCE_CPU", "0") != "1":
            LOG.info("%s: CUDA is available; overriding training.device=cpu -> cuda", command)
            return "cuda"
        return "cpu"
    if ds.startswith("cuda"):
        if cuda_available:
            return ds
        LOG.warning("%s: requested %s but CUDA unavailable, falling back to cpu", command, ds)
        return "cpu"
    LOG.warning("%s: unknown training.device=%s, falling back to auto", command, device_s)
    return "cuda" if cuda_available else "cpu"


def _cmd_init(args: argparse.Namespace) -> int:
    out = os.path.abspath(args.output)
    os.makedirs(out, exist_ok=True)
    for sub in ("configs", "data", "outputs", "logs"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    here = os.path.dirname(__file__)
    root = os.path.dirname(here)
    cfg_src = os.path.join(root, "configs")
    if os.path.isdir(cfg_src):
        for name in os.listdir(cfg_src):
            if name.endswith(".yaml"):
                shutil.copy(os.path.join(cfg_src, name), os.path.join(out, "configs", name))
    readme = os.path.join(out, "README.generated.md")
    with open(readme, "w", encoding="utf-8") as f:
        f.write("# density2sse workspace\n\nRun commands from the project root after `pip install -e .`.\n")
    print(f"Initialized workspace at {out}")
    return 0


def _resolve_path(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(os.getcwd(), p)


def _cmd_generate_data(args: argparse.Namespace) -> int:
    cfg = resolve_config(args.config)
    validate_config(cfg, "generate-data")
    set_seed(int(cfg["project"]["seed"]))
    syn = cfg["synthetic"]
    data = cfg["data"]
    train_dir = _resolve_path(data["train_dir"])
    val_dir = _resolve_path(data["val_dir"])
    test_dir = _resolve_path(data["test_dir"])

    sc = SyntheticConfig(
        num_samples=int(syn["num_samples_train"]),
        seed=int(cfg["project"]["seed"]),
        box_size=int(data["box_size"]),
        voxel_size=float(data["voxel_size"]),
        K_min=int(data["K_min"]),
        K_max=int(data["K_max"]),
        length_min=float(syn["length_min"]),
        length_max=float(syn["length_max"]),
        retry_limit=int(syn["retry_limit"]),
        tube_radius=float(syn.get("tube_radius", 2.5)),
        export_mrc=bool(syn.get("export_mrc", False)),
        export_pdb=bool(syn.get("export_pdb", False)),
        num_workers=int(syn.get("num_workers", 1)),
    )
    generate_dataset_split(train_dir, sc, "train")
    sc_val = SyntheticConfig(
        num_samples=int(syn["num_samples_val"]),
        seed=int(cfg["project"]["seed"]) + 1,
        box_size=int(data["box_size"]),
        voxel_size=float(data["voxel_size"]),
        K_min=int(data["K_min"]),
        K_max=int(data["K_max"]),
        length_min=float(syn["length_min"]),
        length_max=float(syn["length_max"]),
        retry_limit=int(syn["retry_limit"]),
        tube_radius=float(syn.get("tube_radius", 2.5)),
        export_mrc=bool(syn.get("export_mrc", False)),
        export_pdb=bool(syn.get("export_pdb", False)),
        num_workers=int(syn.get("num_workers", 1)),
    )
    generate_dataset_split(val_dir, sc_val, "val")
    sc_te = SyntheticConfig(
        num_samples=int(syn["num_samples_test"]),
        seed=int(cfg["project"]["seed"]) + 2,
        box_size=int(data["box_size"]),
        voxel_size=float(data["voxel_size"]),
        K_min=int(data["K_min"]),
        K_max=int(data["K_max"]),
        length_min=float(syn["length_min"]),
        length_max=float(syn["length_max"]),
        retry_limit=int(syn["retry_limit"]),
        tube_radius=float(syn.get("tube_radius", 2.5)),
        export_mrc=bool(syn.get("export_mrc", False)),
        export_pdb=bool(syn.get("export_pdb", False)),
        num_workers=int(syn.get("num_workers", 1)),
    )
    generate_dataset_split(test_dir, sc_te, "test")
    run_dir = os.path.join(cfg["project"]["output_dir"], "generate_data", uuid.uuid4().hex[:8])
    os.makedirs(run_dir, exist_ok=True)
    save_resolved(cfg, os.path.join(run_dir, "config.resolved.yaml"))
    print("Generated synthetic datasets.")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    import torch

    from density2sse.train import trainer

    cfg = resolve_config(args.config)
    validate_config(cfg, "train")
    set_seed(int(cfg["project"]["seed"]))
    tcfg = cfg["training"]
    device_s = str(tcfg.get("device", "cpu"))
    resolved_device = _resolve_runtime_device(device_s, torch.cuda.is_available(), command="train")
    device = torch.device(resolved_device)
    data = cfg["data"]
    train_dir = _resolve_path(data["train_dir"])
    val_dir = _resolve_path(data["val_dir"])

    if tcfg.get("tiny_overfit"):
        # expect user generated tiny data; optionally subsample is not implemented
        train_dir = train_dir

    model_name = str(cfg["model"].get("name", "baseline_cnn"))
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_name).strip("_") or "model"
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + safe + "_" + uuid.uuid4().hex[:6]
    run_dir = os.path.join(cfg["project"]["output_dir"], "train", run_id)
    os.makedirs(run_dir, exist_ok=True)
    resolved_path = os.path.join(run_dir, "config.resolved.yaml")
    save_resolved(cfg, resolved_path)
    shutil.copy(resolved_path, os.path.join(run_dir, "config.yaml"))
    with open(os.path.join(run_dir, "model.txt"), "w", encoding="utf-8") as f:
        f.write(model_registry.describe_model(cfg))
    trainer.run_training(cfg, train_dir, val_dir, run_dir, device, run_id)
    print(f"Training finished. Run directory: {run_dir}")
    return 0


def _cmd_infer(args: argparse.Namespace) -> int:
    import torch

    from density2sse.infer import predictor

    cfg = resolve_config(args.config)
    validate_config(cfg, "infer")
    set_seed(int(cfg["project"]["seed"]))
    inf = cfg["inference"]
    device_s = str(cfg["training"].get("device", "cpu"))
    resolved_device = _resolve_runtime_device(device_s, torch.cuda.is_available(), command="infer")
    device = torch.device(resolved_device)
    out = predictor.run_inference(cfg, device)
    if inf.get("export_pdb", True):
        npz_path = inf["output_prefix"] + "_pred.npz"
        pdb_path = inf["output_prefix"] + "_pred.pdb"
        export_pdb.export_npz_to_pdb(npz_path, pdb_path)
        print(f"Wrote {pdb_path}")
    print("Inference complete.")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    cfg = resolve_config(args.config)
    validate_config(cfg, "export")
    ex = cfg["export"]
    export_pdb.export_npz_to_pdb(ex["input_npz"], ex["output_pdb"])
    print(f"Wrote {ex['output_pdb']}")
    return 0


def _cmd_validate_config(args: argparse.Namespace) -> int:
    cfg = resolve_config(args.config)
    print(cfg)
    run_dir = os.path.join(cfg["project"]["output_dir"], "validate_config")
    os.makedirs(run_dir, exist_ok=True)
    save_resolved(cfg, os.path.join(run_dir, "config.resolved.yaml"))
    return 0


def _cmd_test(args: argparse.Namespace) -> int:
    try:
        import importlib.util

        if importlib.util.find_spec("pytest") is None:
            raise ImportError
    except ImportError:
        print(
            "pytest is not installed. Run: pip install 'pytest>=7'  or  pip install -e '.[dev]'  or  pip install -e '.[test]'",
            file=sys.stderr,
        )
        return 1
    cmd = [sys.executable, "-m", "pytest", "tests", "-q"]
    if args.extra:
        cmd.extend(args.extra)
    return subprocess.call(cmd)


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = resolve_config(args.config)
    stages: List[str] = list(cfg.get("run", {}).get("stages", ["generate-data", "train"]))
    cfg_path = os.path.abspath(args.config)
    for st in stages:
        if st == "generate-data":
            rc = _cmd_generate_data(argparse.Namespace(config=cfg_path))
            if rc != 0:
                return rc
        elif st == "train":
            rc = _cmd_train(argparse.Namespace(config=cfg_path))
            if rc != 0:
                return rc
        elif st == "infer":
            rc = _cmd_infer(argparse.Namespace(config=cfg_path))
            if rc != 0:
                return rc
        else:
            LOG.warning("Unknown stage %s skipped", st)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="density2sse", description="Density mask to SSE helices")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Create workspace skeleton")
    p_init.add_argument("-o", "--output", required=True, help="Output directory")
    p_init.set_defaults(func=_cmd_init)

    p_g = sub.add_parser("generate-data", help="Generate synthetic NPZ datasets")
    p_g.add_argument("-i", "--config", required=True, help="YAML config")
    p_g.set_defaults(func=_cmd_generate_data)

    p_t = sub.add_parser("train", help="Train baseline model")
    p_t.add_argument("-i", "--config", required=True, help="YAML config")
    p_t.set_defaults(func=_cmd_train)

    p_i = sub.add_parser("infer", help="Run inference on MRC")
    p_i.add_argument("-i", "--config", required=True, help="YAML config")
    p_i.set_defaults(func=_cmd_infer)

    p_e = sub.add_parser("export", help="Export NPZ to PDB")
    p_e.add_argument("-i", "--config", required=True, help="YAML config")
    p_e.set_defaults(func=_cmd_export)

    p_v = sub.add_parser("validate-config", help="Print and save resolved YAML")
    p_v.add_argument("-i", "--config", required=True, help="YAML config")
    p_v.set_defaults(func=_cmd_validate_config)

    p_test = sub.add_parser("test", help="Run pytest suite")
    p_test.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args for pytest")
    p_test.set_defaults(func=_cmd_test)

    p_r = sub.add_parser("run", help="Run pipeline stages from config")
    p_r.add_argument("-i", "--config", required=True, help="YAML config")
    p_r.set_defaults(func=_cmd_run)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
