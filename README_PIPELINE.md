# MotionAGFormer — Training & Evaluation Pipeline

## Quick Start (One Command)

### Linux / macOS
```bash
chmod +x setup_and_run.sh
./setup_and_run.sh
```

### Windows
```cmd
setup_and_run.bat
```

That's it. The script will:
1. Create a conda environment (`motionagformer`) with Python 3.8 and all dependencies
2. Download the Human3.6M and MPI-INF-3DHP datasets (~2 GB)
3. Preprocess the data
4. Train all 4 model sizes (xsmall, small, base, large) on both datasets
5. Evaluate all models (MPJPE, P-MPJPE, Acceleration Error)
6. Compute 6 temporal metrics (MPJVE, MPJAE, MPJJE, DTW, FID, FVD)
7. Save a consolidated report to `results/full_results.json`

---

## Prerequisites

| Requirement | Details |
|---|---|
| **conda** | Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (lightweight) or Anaconda |
| **NVIDIA GPU** | GPU with CUDA support + drivers installed (`nvidia-smi` should work) |
| **Disk space** | ~50 GB (datasets + checkpoints + preprocessed data) |
| **RAM** | 16 GB+ recommended |

> **Note:** No Docker or nvidia-container-toolkit needed. Just conda + GPU drivers.

---

## Options

Pass these flags to customize the run:

```bash
# Only train specific model sizes
./setup_and_run.sh --sizes xsmall base

# Only train on one dataset
./setup_and_run.sh --datasets h36m

# Skip dataset download (if already present)
./setup_and_run.sh --skip-download

# Only evaluate pre-trained models (skip training)
./setup_and_run.sh --eval-only

# Skip temporal metric computation
./setup_and_run.sh --skip-temporal-metrics

# Adjust CPU workers
./setup_and_run.sh --num-cpus 8

# Combine options
./setup_and_run.sh --sizes base large --datasets h36m --num-cpus 8
```

---

## Output

All results are saved to the `results/` directory:

```
results/
├── full_results.json                      # consolidated report (all metrics)
├── temporal_metrics_xsmall_h36m.json      # per-model temporal metrics
├── temporal_metrics_xsmall_mpi.json
├── temporal_metrics_small_h36m.json
├── ...
└── checkpoints/
    ├── xsmall-h36m/
    │   ├── best_epoch.pth.tr
    │   └── latest_epoch.pth.tr
    ├── xsmall-mpi/
    ├── small-h36m/
    ├── small-mpi/
    ├── base-h36m/
    ├── base-mpi/
    ├── large-h36m/
    └── large-mpi/
```

---

## Metrics Computed

### Standard Metrics (from original codebase)
| Metric | Description | Unit |
|---|---|---|
| **MPJPE** | Mean Per Joint Position Error (Protocol #1) | mm |
| **P-MPJPE** | Procrustes-aligned MPJPE (Protocol #2) | mm |
| **AccErr** | Acceleration Error | mm/s² |

### Temporal Metrics (newly added)
| Metric | Description | Unit | Ground Truth? |
|---|---|---|---|
| **MPJVE** | Mean Per Joint Velocity Error | mm/frame | Required |
| **MPJAE** | Mean Per Joint Acceleration Error | mm/frame² | Required |
| **MPJJE** | Mean Per Joint Jerk Error | mm/frame³ | Required |
| **DTW** | Dynamic Time Warping | mm | Required |
| **FID** | Fréchet Inception Distance (pose-space) | — | Not needed |
| **FVD** | Fréchet Video Distance (pose-space) | — | Not needed |

---

## Estimated Training Time

| Model Size | H36M Epochs | MPI Epochs | Approx. Time (1× A100) |
|---|---|---|---|
| xsmall | 60 | 90 | ~4 hours |
| small | 60 | 90 | ~8 hours |
| base | 60 | 90 | ~14 hours |
| large | 60 | 90 | ~22 hours |

**Total (all sizes, both datasets): ~2–4 days on a single GPU.**

---

## Troubleshooting

### conda: command not found
Install Miniconda: https://docs.conda.io/en/latest/miniconda.html

### CUDA out of memory
Reduce batch size in the config YAML files (e.g., `configs/h36m/MotionAGFormer-base.yaml`):
```yaml
batch_size: 8  # reduce from 16
```

### gdown download fails
Google Drive may rate-limit. Wait a few minutes and re-run. The script will skip already-downloaded files.

### Training resumes from where it stopped
If training is interrupted, re-run the same command. The script detects existing checkpoints and skips completed training runs.
