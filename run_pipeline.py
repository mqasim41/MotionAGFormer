"""
MotionAGFormer — Full Pipeline: Download Data → Train → Evaluate → Compute Metrics

Recommended usage (via the wrapper scripts):
    Linux/macOS:  ./setup_and_run.sh
    Windows:      setup_and_run.bat

Direct usage (if conda env is already active):
    python run_pipeline.py                          # run everything
    python run_pipeline.py --skip-download          # skip dataset download
    python run_pipeline.py --eval-only              # skip training, only evaluate
    python run_pipeline.py --sizes xsmall base      # only run specific model sizes
    python run_pipeline.py --datasets h36m          # only run on Human3.6M
    python run_pipeline.py --num-cpus 8             # override CPU workers
"""

import argparse
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path


# ─── Configuration ────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent

# Model sizes and their corresponding n_frames for preprocessing
MODEL_CONFIGS = {
    "xsmall": {"h36m_config": "configs/h36m/MotionAGFormer-xsmall.yaml", "mpi_config": "configs/mpi/MotionAGFormer-xsmall.yaml", "n_frames": 27},
    "small":  {"h36m_config": "configs/h36m/MotionAGFormer-small.yaml",  "mpi_config": "configs/mpi/MotionAGFormer-small.yaml",  "n_frames": 81},
    "base":   {"h36m_config": "configs/h36m/MotionAGFormer-base.yaml",   "mpi_config": "configs/mpi/MotionAGFormer-base.yaml",   "n_frames": 243},
    "large":  {"h36m_config": "configs/h36m/MotionAGFormer-large.yaml",  "mpi_config": "configs/mpi/MotionAGFormer-large.yaml",  "n_frames": 243},
}


def run_cmd(cmd, cwd=None, check=True):
    """Run a shell command and stream output in real time."""
    print(f"\n{'='*80}")
    print(f"[CMD] {cmd}")
    print(f"{'='*80}\n")
    result = subprocess.run(
        cmd, shell=True, cwd=cwd or str(ROOT_DIR),
        check=check
    )
    return result.returncode


def extract_zip(zip_path, dest_dir):
    """Cross-platform zip extraction."""
    print(f"[INFO] Extracting {zip_path} → {dest_dir}")
    with zipfile.ZipFile(str(zip_path), 'r') as zf:
        zf.extractall(str(dest_dir))


# ─── Step 1: Download Datasets ───────────────────────────────────────────────

def download_datasets():
    """Download Human3.6M and MPI-INF-3DHP datasets from Google Drive."""
    print("\n" + "="*80)
    print("STEP 1: Downloading Datasets")
    print("="*80)

    data_dir = ROOT_DIR / "data" / "motion3d"
    data_dir.mkdir(parents=True, exist_ok=True)

    h36m_pkl = data_dir / "h36m_sh_conf_cam_source_final.pkl"
    mpi_train = data_dir / "data_train_3dhp.npz"
    mpi_test = data_dir / "data_test_3dhp.npz"

    # Install gdown if needed
    run_cmd(f"{sys.executable} -m pip install --quiet gdown", check=False)

    # Human3.6M dataset
    if not h36m_pkl.exists():
        print("[INFO] Downloading Human3.6M dataset...")
        zip_path = ROOT_DIR / "h36m_sh_conf_cam_source_final.pkl.zip"
        run_cmd(f"gdown https://drive.google.com/uc?id=1hoVBwNi-P-4JIW2VMlK_4dur1U1Y52EC -O \"{zip_path}\"")
        extract_zip(zip_path, data_dir)
        zip_path.unlink(missing_ok=True)
    else:
        print("[INFO] Human3.6M dataset already exists, skipping download.")

    # MPI-INF-3DHP dataset
    if not mpi_train.exists():
        print("[INFO] Downloading MPI-INF-3DHP training data...")
        run_cmd(f"gdown https://drive.google.com/uc?id=1kof4TizGYYzcMDF8VouSV3D1c1DakA79 -O \"{mpi_train}\"")
    else:
        print("[INFO] MPI training data already exists, skipping.")

    if not mpi_test.exists():
        print("[INFO] Downloading MPI-INF-3DHP test data...")
        run_cmd(f"gdown https://drive.google.com/uc?id=1MQUWmD0PtcXzHQ3jTYuofs--dh6b6WyY -O \"{mpi_test}\"")
    else:
        print("[INFO] MPI test data already exists, skipping.")

    print("[OK] All datasets ready.")


# ─── Step 2: Preprocess Data ─────────────────────────────────────────────────

def preprocess_h36m(n_frames_list):
    """Preprocess Human3.6M data for all required frame counts."""
    print("\n" + "="*80)
    print("STEP 2: Preprocessing Human3.6M Data")
    print("="*80)

    preprocess_dir = ROOT_DIR / "data" / "preprocess"
    unique_frames = sorted(set(n_frames_list))

    for n_frames in unique_frames:
        output_dir = ROOT_DIR / "data" / "motion3d" / f"H36M-{n_frames}"
        if output_dir.exists() and len(list(output_dir.glob("**/*.pkl"))) > 0:
            print(f"[INFO] H36M-{n_frames} already preprocessed, skipping.")
            continue
        print(f"[INFO] Preprocessing Human3.6M with n_frames={n_frames}...")
        run_cmd(f"{sys.executable} h36m.py --n-frames {n_frames}", cwd=str(preprocess_dir))

    print("[OK] Human3.6M preprocessing complete.")


# ─── Step 3: Train Models ────────────────────────────────────────────────────

def train_model(size, dataset, config_path, checkpoint_dir, num_cpus):
    """Train a single model configuration."""
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    best_checkpoint = checkpoint_dir_path / "best_epoch.pth.tr"
    if best_checkpoint.exists():
        print(f"[INFO] Checkpoint already exists at {best_checkpoint}, skipping training.")
        return

    if dataset == "h36m":
        run_cmd(
            f"{sys.executable} train.py "
            f"--config {config_path} "
            f"--new-checkpoint {checkpoint_dir} "
            f"--num-cpus {num_cpus}"
        )
    elif dataset == "mpi":
        run_cmd(
            f"{sys.executable} train_3dhp.py "
            f"--config {config_path} "
            f"--new-checkpoint {checkpoint_dir} "
            f"--num-cpus {num_cpus}"
        )


def train_all(sizes, datasets, num_cpus):
    """Train all requested model sizes on all requested datasets."""
    print("\n" + "="*80)
    print("STEP 3: Training Models")
    print("="*80)

    for size in sizes:
        cfg = MODEL_CONFIGS[size]
        for dataset in datasets:
            config_key = f"{dataset}_config"
            config_path = cfg[config_key]
            checkpoint_dir = f"results/checkpoints/{size}-{dataset}"

            print(f"\n{'─'*60}")
            print(f"Training MotionAGFormer-{size} on {dataset.upper()}")
            print(f"Config: {config_path}")
            print(f"Checkpoint: {checkpoint_dir}")
            print(f"{'─'*60}")

            train_model(size, dataset, config_path, checkpoint_dir, num_cpus)

    print("\n[OK] All training complete.")


# ─── Step 4: Evaluate (existing metrics: MPJPE, P-MPJPE, AccErr) ─────────────

def evaluate_model(size, dataset, config_path, checkpoint_dir, num_cpus):
    """Evaluate a single trained model and return captured metrics."""
    best_checkpoint = Path(checkpoint_dir) / "best_epoch.pth.tr"
    if not best_checkpoint.exists():
        print(f"[WARN] No checkpoint found at {best_checkpoint}, skipping evaluation.")
        return None

    if dataset == "h36m":
        cmd = (
            f"{sys.executable} train.py "
            f"--eval-only "
            f"--checkpoint {checkpoint_dir} "
            f"--checkpoint-file best_epoch.pth.tr "
            f"--config {config_path} "
            f"--num-cpus {num_cpus}"
        )
    elif dataset == "mpi":
        cmd = (
            f"{sys.executable} train_3dhp.py "
            f"--eval-only "
            f"--checkpoint {checkpoint_dir} "
            f"--checkpoint-file best_epoch.pth.tr "
            f"--config {config_path} "
            f"--new-checkpoint {checkpoint_dir} "
            f"--num-cpus {num_cpus}"
        )

    # Capture output to parse metrics
    print(f"\n[INFO] Evaluating {size}-{dataset}...")
    result = subprocess.run(
        cmd, shell=True, cwd=str(ROOT_DIR),
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    # Parse standard metrics from output
    metrics = {}
    for line in result.stdout.split('\n'):
        if 'Protocol #1 Error (MPJPE):' in line:
            metrics['MPJPE'] = float(line.split(':')[1].strip().replace('mm', '').strip())
        if 'Protocol #2 Error (P-MPJPE):' in line:
            metrics['P-MPJPE'] = float(line.split(':')[1].strip().replace('mm', '').strip())
        if 'Acceleration error:' in line:
            metrics['AccErr'] = float(line.split(':')[1].strip().replace('mm/s^2', '').strip())

    return metrics


def evaluate_all(sizes, datasets, num_cpus):
    """Evaluate all models and return results dict."""
    print("\n" + "="*80)
    print("STEP 4: Evaluating Models (Standard Metrics)")
    print("="*80)

    all_results = {}
    for size in sizes:
        cfg = MODEL_CONFIGS[size]
        for dataset in datasets:
            config_path = cfg[f"{dataset}_config"]
            checkpoint_dir = f"results/checkpoints/{size}-{dataset}"

            metrics = evaluate_model(size, dataset, config_path, checkpoint_dir, num_cpus)
            if metrics:
                all_results[f"{size}-{dataset}"] = metrics
                print(f"  {size}-{dataset}: {metrics}")

    return all_results


# ─── Step 5: Compute Temporal Metrics ─────────────────────────────────────────

def compute_temporal_metrics(sizes, datasets, num_cpus):
    """Compute the 6 temporal evaluation metrics on all trained models."""
    print("\n" + "="*80)
    print("STEP 5: Computing Temporal Metrics (MPJVE, MPJAE, MPJJE, DTW, FID, FVD)")
    print("="*80)

    all_temporal = {}
    for size in sizes:
        cfg = MODEL_CONFIGS[size]
        for dataset in datasets:
            config_path = cfg[f"{dataset}_config"]
            checkpoint_dir = f"results/checkpoints/{size}-{dataset}"
            best_ckpt = Path(checkpoint_dir) / "best_epoch.pth.tr"

            if not best_ckpt.exists():
                print(f"[WARN] No checkpoint at {best_ckpt}, skipping temporal metrics.")
                continue

            output_file = f"results/temporal_metrics_{size}_{dataset}.json"
            print(f"\n[INFO] Computing temporal metrics for {size}-{dataset}...")

            run_cmd(
                f"{sys.executable} compute_temporal_metrics.py "
                f"--config {config_path} "
                f"--dataset {dataset} "
                f"--checkpoint-dir {checkpoint_dir} "
                f"--checkpoint-file best_epoch.pth.tr "
                f"--output {output_file} "
                f"--num-cpus {num_cpus}"
            )

            if os.path.exists(output_file):
                with open(output_file) as f:
                    temporal = json.load(f)
                all_temporal[f"{size}-{dataset}"] = temporal
                print(f"  {size}-{dataset}: {temporal}")

    return all_temporal


# ─── Step 6: Generate Final Report ───────────────────────────────────────────

def generate_report(standard_results, temporal_results, output_path):
    """Generate a consolidated results report."""
    print("\n" + "="*80)
    print("STEP 6: Generating Final Report")
    print("="*80)

    report = {
        "standard_metrics": standard_results,
        "temporal_metrics": temporal_results,
    }

    # Save JSON
    json_path = Path(output_path) / "full_results.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Print human-readable tables
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)

    # Group by dataset
    for dataset in ["h36m", "mpi"]:
        dataset_label = "Human3.6M" if dataset == "h36m" else "MPI-INF-3DHP"
        print(f"\n{'─'*70}")
        print(f"  {dataset_label} Results")
        print(f"{'─'*70}")

        header_parts = ["Model".ljust(25)]
        # Standard metrics
        for m in ["MPJPE", "P-MPJPE", "AccErr"]:
            header_parts.append(m.center(12))
        # Temporal metrics
        for m in ["MPJVE", "MPJAE", "MPJJE", "DTW", "FID", "FVD"]:
            header_parts.append(m.center(12))
        print(" | ".join(header_parts))
        print("-" * (len(" | ".join(header_parts))))

        for size in MODEL_CONFIGS:
            key = f"{size}-{dataset}"
            row = [f"MotionAGFormer-{size}".ljust(25)]

            std = standard_results.get(key, {})
            for m in ["MPJPE", "P-MPJPE", "AccErr"]:
                val = std.get(m, None)
                row.append(f"{val:.4f}".center(12) if val is not None else "N/A".center(12))

            tmp = temporal_results.get(key, {})
            for m in ["MPJVE", "MPJAE", "MPJJE", "DTW", "FID", "FVD"]:
                val = tmp.get(m, None)
                row.append(f"{val:.4f}".center(12) if val is not None else "N/A".center(12))

            print(" | ".join(row))

    print(f"\n[OK] Full results saved to {json_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="MotionAGFormer Full Training & Evaluation Pipeline")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip dataset download (assumes data is already present)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, only run evaluation and metrics")
    parser.add_argument("--skip-temporal-metrics", action="store_true",
                        help="Skip computing temporal metrics (MPJVE, MPJAE, MPJJE, DTW, FID, FVD)")
    parser.add_argument("--sizes", nargs="+", default=["xsmall", "small", "base", "large"],
                        choices=["xsmall", "small", "base", "large"],
                        help="Model sizes to train/evaluate")
    parser.add_argument("--datasets", nargs="+", default=["h36m", "mpi"],
                        choices=["h36m", "mpi"],
                        help="Datasets to use")
    parser.add_argument("--num-cpus", type=int, default=min(os.cpu_count() or 4, 16),
                        help="Number of CPU workers for data loading")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory for all outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    print("="*80)
    print("  MotionAGFormer — Full Pipeline")
    print(f"  Sizes:    {args.sizes}")
    print(f"  Datasets: {args.datasets}")
    print(f"  CPUs:     {args.num_cpus}")
    print(f"  Output:   {args.output_dir}")
    print("="*80)

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Download datasets
    if not args.skip_download and not args.eval_only:
        download_datasets()

    # Step 2: Preprocess H36M
    if "h36m" in args.datasets and not args.eval_only:
        n_frames_needed = [MODEL_CONFIGS[s]["n_frames"] for s in args.sizes]
        preprocess_h36m(n_frames_needed)

    # Step 3: Train
    if not args.eval_only:
        train_all(args.sizes, args.datasets, args.num_cpus)

    # Step 4: Standard evaluation
    standard_results = evaluate_all(args.sizes, args.datasets, args.num_cpus)

    # Step 5: Temporal metrics
    temporal_results = {}
    if not args.skip_temporal_metrics:
        temporal_results = compute_temporal_metrics(args.sizes, args.datasets, args.num_cpus)

    # Step 6: Report
    generate_report(standard_results, temporal_results, args.output_dir)

    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    print(f"\n[DONE] Total time: {int(hours)}h {int(mins)}m {int(secs)}s")


if __name__ == "__main__":
    main()
