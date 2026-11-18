#!/usr/bin/env python3
"""Run the v14 pipeline: unsupervised BSE 100, CGCL 100+90, then HCRL.

The dataset, 165-category name catalog, RoBERTa checkpoint, and DashScope
credential are external inputs. This code package contains none of them.

The earlier ACC/NMI values used label-supervised BSE and do not predict the
results of this unsupervised BSE pipeline. New runs are measured afresh.

Example on a GPU server:
    export HCR_LLM_API_KEY="..."
    python -B run_full_baseline.py \
        --csv /path/to/motor_dataset_v14.csv \
        --category-names /path/to/category_names_v14.csv \
        --backbone /path/to/chinese-roberta-wwm-ext \
        --device cuda:4
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shlex
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT / "伪标签可靠性与跨粒度目标分析" / "初始化敏感性R1-1"
OUTPUTS = EXPERIMENT / "outputs"
SEEDS = (42, 43, 44)
DATASET_SHA256 = "7f4d6689ed4484c2c5598ff2ab234452370a9461f64a847353f7df34bec02711"
CATEGORY_SHA256 = "279005887d4b98d7773e7a72088a6f5c18682890b5454d666db6dc747b67c89d"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_environment() -> None:
    modules = (
        "numpy", "pandas", "scipy", "sklearn", "torch", "transformers",
        "matplotlib", "tqdm", "requests", "igraph", "leidenalg", "networkx",
    )
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"当前 Python 环境缺少依赖：{', '.join(missing)}")


def run(script: str, *arguments: str) -> None:
    command = [sys.executable, "-B", str(EXPERIMENT / script), *map(str, arguments)]
    print("+", shlex.join(command), flush=True)
    subprocess.run(command, cwd=EXPERIMENT, check=True)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def validate_run(folder: Path, seed: int, csv: Path) -> None:
    manifest = read_json(folder / "initialization_experiment.json")
    config = read_json(folder / "entry_config.json")
    if (
        manifest.get("case") != "baseline"
        or manifest.get("seed") != seed
        or manifest.get("noise_injected") != 0
        or config.get("seed") != seed
        or config.get("epochs_bse") != 100
        or config.get("fixed_k") != 165
        or config.get("epochs_stage2") != 100
        or config.get("epochs_stage3") != 90
        or config.get("final_embedding_source") != "cgcl"
        or Path(config.get("csv", "")).resolve() != csv
        or read_json(folder / "final_metrics.json").get("method")
        != "unsupervised_instance_bse_then_pseudo_label_cgcl"
    ):
        raise ValueError(f"CGCL 运行与 v14 基线设置不一致：{folder}")
    for filename in (
        "final_metrics.json",
        "entities.json",
        "cgcl_final_unsupervised.npz",
        "bse_anchor_embeddings.npy",
    ):
        if not (folder / filename).is_file():
            raise FileNotFoundError(folder / filename)


def completed_new_runs(before: set[str], csv: Path) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for folder in OUTPUTS.iterdir():
        if not folder.is_dir() or folder.name in before:
            continue
        manifest_path = folder / "initialization_experiment.json"
        if not manifest_path.is_file():
            continue
        manifest = read_json(manifest_path)
        if manifest.get("case") != "baseline" or manifest.get("seed") not in SEEDS:
            continue
        seed = int(manifest["seed"])
        validate_run(folder, seed, csv)
        if seed in found:
            raise RuntimeError(f"本次出现多个 seed {seed} 基线运行")
        found[seed] = folder
    if set(found) != set(SEEDS):
        raise RuntimeError(f"本次只找到这些 seed 的完整输出：{sorted(found)}")
    return found


def report_cgcl(runs: dict[int, Path]) -> None:
    acc = [float(read_json(runs[seed] / "final_metrics.json")["ACC"]) * 100 for seed in SEEDS]
    nmi = [float(read_json(runs[seed] / "final_metrics.json")["NMI"]) * 100 for seed in SEEDS]
    print("CGCL 基线运行目录：")
    for seed in SEEDS:
        print(f"  seed {seed}: {runs[seed]}")
    print(
        f"ACC = {statistics.mean(acc):.2f} ± {statistics.stdev(acc):.2f}%"
    )
    print(
        f"NMI = {statistics.mean(nmi):.2f} ± {statistics.stdev(nmi):.2f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--category-names", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument(
        "--existing-runs", nargs=3, metavar=("SEED42", "SEED43", "SEED44"),
        help="复用本代码包 outputs 下现有的三个 CGCL 目录名，跳过 BSE/CGCL",
    )
    parser.add_argument("--hcrl-subdir", default="unsupervised_bse100_hcrl_v11")
    parser.add_argument(
        "--allow-input-changes", action="store_true",
        help="允许输入文件与指定 v14 版本的哈希不同",
    )
    args = parser.parse_args()

    check_environment()
    csv = args.csv.resolve(strict=True)
    catalog = args.category_names.resolve(strict=True)
    backbone = args.backbone.resolve(strict=True)
    if csv.name != "motor_dataset_v14.csv":
        parser.error("数据文件须为最新版 motor_dataset_v14.csv")
    if catalog.name != "category_names_v14.csv" or catalog.parent != csv.parent:
        parser.error("category_names_v14.csv 须与 motor_dataset_v14.csv 位于同一目录")
    actual_hashes = (sha256_file(csv), sha256_file(catalog))
    if actual_hashes != (DATASET_SHA256, CATEGORY_SHA256):
        if not args.allow_input_changes:
            parser.error(
                "输入文件与指定的 v14 版本不同；"
                "若确认要跑新版本，请加 --allow-input-changes"
            )
        print("输入文件哈希与指定的 v14 版本不同。", flush=True)
    if not os.environ.get("HCR_LLM_API_KEY"):
        parser.error("运行前请设置 HCR_LLM_API_KEY 环境变量")
    if not args.hcrl_subdir or Path(args.hcrl_subdir).name != args.hcrl_subdir:
        parser.error("--hcrl-subdir 须为单个目录名")

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    if args.existing_runs:
        runs = {}
        for seed, name in zip(SEEDS, args.existing_runs):
            if Path(name).name != name or name in {".", ".."}:
                parser.error("--existing-runs 只接受 outputs 下的目录名")
            folder = OUTPUTS / name
            validate_run(folder, seed, csv)
            runs[seed] = folder
    else:
        before = {path.name for path in OUTPUTS.iterdir() if path.is_dir()}
        run(
            "run_initialization.py",
            "--execute", "--cases", "baseline", "--seeds", *map(str, SEEDS),
            "--device", args.device, "--csv", str(csv),
            "--backbone", str(backbone), "--batch-size", "48",
            "--epochs-bse", "100", "--fixed-k", "165",
            "--epochs-stage2", "100",
            "--epochs-stage3", "90",
        )
        runs = completed_new_runs(before, csv)

    report_cgcl(runs)
    names = [runs[seed].name for seed in SEEDS]
    run(
        "run_hcrl.py",
        "--execute", "--cases", "baseline", "--seeds", *map(str, SEEDS),
        "--run-outputs", *names, "--category-names", str(catalog),
        "--output-subdir", args.hcrl_subdir,
    )
    report_path = OUTPUTS / f"{args.hcrl_subdir}_total_root_metrics.json"
    run(
        "report_hierarchy_metrics_total_root.py",
        "--cases", "baseline", "--seeds", *map(str, SEEDS),
        "--run-outputs", *names, "--hcrl-subdir", args.hcrl_subdir,
        "--save-json", str(report_path),
    )
    print(f"HCRL 层次结构：各运行目录下的 {args.hcrl_subdir}/ontology_structure.json")
    print(f"带总根的 DP、ECS、概念深度汇总：{report_path}")


if __name__ == "__main__":
    main()
