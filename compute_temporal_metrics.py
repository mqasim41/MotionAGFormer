"""
Compute 6 Temporal Evaluation Metrics for MotionAGFormer
========================================================

Metrics computed:
  1. MPJVE  — Mean Per Joint Velocity Error         (mm/frame)     [requires GT]
  2. MPJAE  — Mean Per Joint Acceleration Error      (mm/frame²)   [requires GT]
  3. MPJJE  — Mean Per Joint Jerk Error              (mm/frame³)   [requires GT]
  4. DTW    — Dynamic Time Warping distance          (mm)           [requires GT]
  5. FID    — Fréchet Inception Distance (pose-space)(unitless)     [no GT needed]
  6. FVD    — Fréchet Video Distance (pose-space)    (unitless)     [no GT needed]

Usage:
    python compute_temporal_metrics.py \\
        --config configs/h36m/MotionAGFormer-base.yaml \\
        --dataset h36m \\
        --checkpoint-dir results/checkpoints/base-h36m \\
        --checkpoint-file best_epoch.pth.tr \\
        --output results/temporal_metrics_base_h36m.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

# ─── Project imports ──────────────────────────────────────────────────────────
from utils.tools import get_config, set_random_seed
from utils.learning import load_model
from utils.data import flip_data
from data.reader.h36m import DataReaderH36M
from data.reader.motion_dataset import MotionDataset3D, Fusion
from torch.utils.data import DataLoader


# =============================================================================
#  Metric Implementations
# =============================================================================

def compute_mpjve(predicted, target):
    """
    Mean Per Joint Velocity Error.

    vel = pos(t+1) - pos(t)   →  1st-order difference
    MPJVE = (1 / ((T-1) × J)) × Σ || vel_pred - vel_gt ||₂

    Args:
        predicted: np.ndarray of shape (T, J, 3)
        target:    np.ndarray of shape (T, J, 3)

    Returns:
        float: MPJVE in mm/frame
    """
    vel_pred = predicted[1:] - predicted[:-1]   # (T-1, J, 3)
    vel_gt   = target[1:]   - target[:-1]       # (T-1, J, 3)
    err = np.linalg.norm(vel_pred - vel_gt, axis=-1)  # (T-1, J)
    return float(np.mean(err))


def compute_mpjae(predicted, target):
    """
    Mean Per Joint Acceleration Error.

    accel = vel(t+1) - vel(t) = pos(t+2) - 2*pos(t+1) + pos(t)   →  2nd-order difference
    MPJAE = (1 / ((T-2) × J)) × Σ || accel_pred - accel_gt ||₂

    Args:
        predicted: np.ndarray of shape (T, J, 3)
        target:    np.ndarray of shape (T, J, 3)

    Returns:
        float: MPJAE in mm/frame²
    """
    accel_pred = predicted[:-2] - 2 * predicted[1:-1] + predicted[2:]  # (T-2, J, 3)
    accel_gt   = target[:-2]   - 2 * target[1:-1]   + target[2:]      # (T-2, J, 3)
    err = np.linalg.norm(accel_pred - accel_gt, axis=-1)  # (T-2, J)
    return float(np.mean(err))


def compute_mpjje(predicted, target):
    """
    Mean Per Joint Jerk Error.

    jerk = accel(t+1) - accel(t)   →  3rd-order difference
    MPJJE = (1 / ((T-3) × J)) × Σ || jerk_pred - jerk_gt ||₂

    Args:
        predicted: np.ndarray of shape (T, J, 3)
        target:    np.ndarray of shape (T, J, 3)

    Returns:
        float: MPJJE in mm/frame³
    """
    if predicted.shape[0] < 4:
        return 0.0
    # 3rd-order finite difference
    accel_pred = predicted[:-2] - 2 * predicted[1:-1] + predicted[2:]
    accel_gt   = target[:-2]   - 2 * target[1:-1]   + target[2:]
    jerk_pred = accel_pred[1:] - accel_pred[:-1]  # (T-3, J, 3)
    jerk_gt   = accel_gt[1:]   - accel_gt[:-1]    # (T-3, J, 3)
    err = np.linalg.norm(jerk_pred - jerk_gt, axis=-1)  # (T-3, J)
    return float(np.mean(err))


def compute_dtw(predicted, target):
    """
    Dynamic Time Warping distance between predicted and ground truth sequences.

    Flattens each frame to a vector (J*3,) and computes DTW with Euclidean distance.

    Args:
        predicted: np.ndarray of shape (T, J, 3)
        target:    np.ndarray of shape (T, J, 3)

    Returns:
        float: DTW distance (mm), lower is better
    """
    try:
        from dtaidistance import dtw_ndim
        pred_flat = predicted.reshape(predicted.shape[0], -1).astype(np.double)
        gt_flat   = target.reshape(target.shape[0], -1).astype(np.double)
        distance = dtw_ndim.distance(pred_flat, gt_flat)
        return float(distance)
    except ImportError:
        # Fallback: simple DTW with scipy
        from scipy.spatial.distance import cdist
        T_pred = predicted.shape[0]
        T_gt   = target.shape[0]
        pred_flat = predicted.reshape(T_pred, -1)
        gt_flat   = target.reshape(T_gt, -1)
        cost_matrix = cdist(pred_flat, gt_flat, metric='euclidean')
        # Compute DTW with dynamic programming
        dtw_matrix = np.full((T_pred + 1, T_gt + 1), np.inf)
        dtw_matrix[0, 0] = 0
        for i in range(1, T_pred + 1):
            for j in range(1, T_gt + 1):
                dtw_matrix[i, j] = cost_matrix[i-1, j-1] + min(
                    dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]
                )
        return float(dtw_matrix[T_pred, T_gt])


def compute_frechet_distance(mu1, sigma1, mu2, sigma2):
    """
    Compute the Fréchet distance between two multivariate Gaussians.

    FD = ||mu1 - mu2||² + Tr(Σ1 + Σ2 - 2*(Σ1·Σ2)^½)

    Used by both FID and FVD computations.
    """
    from scipy.linalg import sqrtm

    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)

    # Numerical stability: remove imaginary components
    if np.iscomplexobj(covmean):
        if not np.allclose(np.imag(covmean), 0, atol=1e-3):
            raise ValueError("Imaginary component too large in sqrtm")
        covmean = np.real(covmean)

    fd = np.sum(diff ** 2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fd)


def compute_fid(predicted_sequences, target_sequences):
    """
    Fréchet Inception Distance (pose-space).

    Instead of image features from InceptionV3, we use flattened per-frame
    pose vectors (J*3 dimensions) as the feature representation.
    Computes FD between the distributions of predicted vs real frames.

    Args:
        predicted_sequences: list of np.ndarray, each (T_i, J, 3)
        target_sequences:    list of np.ndarray, each (T_i, J, 3)

    Returns:
        float: FID score (lower is better)
    """
    # Collect all frames
    pred_frames = np.concatenate([s.reshape(s.shape[0], -1) for s in predicted_sequences], axis=0)
    gt_frames   = np.concatenate([s.reshape(s.shape[0], -1) for s in target_sequences], axis=0)

    mu_pred = np.mean(pred_frames, axis=0)
    mu_gt   = np.mean(gt_frames, axis=0)
    sigma_pred = np.cov(pred_frames, rowvar=False)
    sigma_gt   = np.cov(gt_frames, rowvar=False)

    return compute_frechet_distance(mu_pred, sigma_pred, mu_gt, sigma_gt)


def compute_fvd(predicted_sequences, target_sequences, window_size=16):
    """
    Fréchet Video Distance (pose-space).

    Extends FID to capture temporal coherence by using spatiotemporal features.
    Each feature vector is a flattened window of consecutive frames:
      feature = [frame_t, frame_{t+1}, ..., frame_{t+W-1}]  →  (W * J * 3,)

    Args:
        predicted_sequences: list of np.ndarray, each (T_i, J, 3)
        target_sequences:    list of np.ndarray, each (T_i, J, 3)
        window_size: number of frames per spatiotemporal window

    Returns:
        float: FVD score (lower is better)
    """
    def extract_st_features(sequences, w):
        """Extract sliding-window spatiotemporal features from a list of sequences."""
        features = []
        for seq in sequences:
            T = seq.shape[0]
            if T < w:
                # Pad by repeating last frame
                pad = np.tile(seq[-1:], (w - T, 1, 1))
                seq = np.concatenate([seq, pad], axis=0)
                T = w
            flat = seq.reshape(T, -1)  # (T, J*3)
            for t in range(T - w + 1):
                window = flat[t:t+w].flatten()  # (W * J * 3,)
                features.append(window)
        return np.array(features) if features else np.zeros((1, w * sequences[0].shape[1] * 3))

    pred_features = extract_st_features(predicted_sequences, window_size)
    gt_features   = extract_st_features(target_sequences, window_size)

    mu_pred = np.mean(pred_features, axis=0)
    mu_gt   = np.mean(gt_features, axis=0)
    sigma_pred = np.cov(pred_features, rowvar=False)
    sigma_gt   = np.cov(gt_features, rowvar=False)

    # Regularize covariance if needed (high-dimensional, possibly singular)
    reg = 1e-6 * np.eye(sigma_pred.shape[0])
    sigma_pred += reg
    sigma_gt += reg

    return compute_frechet_distance(mu_pred, sigma_pred, mu_gt, sigma_gt)


# =============================================================================
#  Inference — Collect paired (predicted, ground_truth) sequences
# =============================================================================

def run_inference_h36m(args, model, device, num_cpus):
    """
    Run model inference on H36M test set and return paired pred/gt clips.

    Returns:
        list of tuples: [(pred_array, gt_array), ...] each of shape (T, J, 3)
    """
    test_dataset = MotionDataset3D(args, args.subset_list, 'test')
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size,
                             num_workers=max(1, num_cpus - 1), pin_memory=True,
                             prefetch_factor=max(1, (num_cpus - 1) // 3),
                             persistent_workers=True)

    datareader = DataReaderH36M(n_frames=args.n_frames, sample_stride=1,
                                data_stride_train=args.n_frames // 3, data_stride_test=args.n_frames,
                                dt_root='data/motion3d', dt_file=args.dt_file)

    print("[INFO] Running H36M inference for temporal metrics...")
    results_all = []
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Inference"):
            x, y = x.to(device), y.to(device)
            if hasattr(args, 'flip') and args.flip:
                batch_input_flip = flip_data(x)
                pred1 = model(x)
                pred_flip = model(batch_input_flip)
                pred2 = flip_data(pred_flip)
                predicted = (pred1 + pred2) / 2
            else:
                predicted = model(x)
            if hasattr(args, 'root_rel') and args.root_rel:
                predicted[:, :, 0, :] = 0
            results_all.append(predicted.cpu().numpy())

    results_all = np.concatenate(results_all)
    results_all = datareader.denormalize(results_all)

    _, split_id_test = datareader.get_split_id()
    factors = np.array(datareader.dt_dataset['test']['2.5d_factor'])
    gts = np.array(datareader.dt_dataset['test']['joints_2.5d_image'])
    sources = np.array(datareader.dt_dataset['test']['source'])

    factor_clips = factors[split_id_test]
    source_clips = sources[split_id_test]
    gt_clips = gts[split_id_test]

    block_list = ['s_09_act_05_subact_02', 's_09_act_10_subact_02', 's_09_act_13_subact_01']

    paired_sequences = []
    for idx in range(len(results_all)):
        source = source_clips[idx][0][:-6]
        if source in block_list:
            continue
        factor = factor_clips[idx][:, None, None]
        gt = gt_clips[idx]
        pred = results_all[idx] * factor

        # Root-relative
        pred = pred - pred[:, 0:1, :]
        gt = gt - gt[:, 0:1, :]
        paired_sequences.append((pred, gt))

    return paired_sequences


def run_inference_mpi(args, model, device, num_cpus):
    """
    Run model inference on MPI-INF-3DHP test set and return paired pred/gt clips.

    Returns:
        list of tuples: [(pred_array, gt_array), ...] each of shape (T, J, 3)
    """
    from utils.utils_3dhp import get_variable
    from utils.data import denormalize

    test_dataset = Fusion(args, train=False)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.test_batch_size,
                             num_workers=max(1, num_cpus - 1), pin_memory=True,
                             prefetch_factor=max(1, (num_cpus - 1) // 3),
                             persistent_workers=True)

    joints_left = [5, 6, 7, 11, 12, 13]
    joints_right = [2, 3, 4, 8, 9, 10]

    print("[INFO] Running MPI inference for temporal metrics...")
    paired_sequences = []
    model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Inference"):
            batch_cam, gt_3D, input_2D, seq, scale, bb_box = data
            [input_2D, gt_3D, batch_cam, scale, bb_box] = get_variable('test',
                [input_2D, gt_3D, batch_cam, scale, bb_box])
            N = input_2D.size(0)

            out_target = gt_3D.clone().view(N, -1, 17, 3)
            out_target[:, :, 14] = 0

            # Input augmentation
            input_2D_flip = input_2D[:, 1]
            input_2D_non_flip = input_2D[:, 0]
            output_flip = model(input_2D_flip)
            output_flip[..., 0] *= -1
            output_flip[:, :, joints_left + joints_right, :] = output_flip[:, :, joints_right + joints_left, :]
            output_non_flip = model(input_2D_non_flip)
            output_3D = (output_non_flip + output_flip) / 2

            output_3D = output_3D * scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, output_3D.size(1), 17, 3)
            pad = (args.n_frames - 1) // 2
            pred_out = output_3D[:, pad].unsqueeze(1)
            pred_out[..., 14, :] = 0
            pred_out = denormalize(pred_out, seq)
            pred_out = pred_out - pred_out[..., 14:15, :]
            out_target = out_target - out_target[..., 14:15, :]

            for i in range(N):
                pred_np = pred_out[i].cpu().numpy()  # (1, 17, 3)
                gt_np = out_target[i].cpu().numpy()   # (T, 17, 3)
                # For MPI, each sample is a single center frame — accumulate them
                paired_sequences.append((pred_np, gt_np[:pred_np.shape[0]]))

    # MPI gives per-frame predictions, group into longer sequences for temporal metrics
    # Concatenate all into one long sequence
    if paired_sequences:
        all_pred = np.concatenate([p for p, g in paired_sequences], axis=0)
        all_gt   = np.concatenate([g for p, g in paired_sequences], axis=0)
        # Split into chunks for metrics that need sequences
        chunk_size = max(args.n_frames, 27)
        n_chunks = max(1, len(all_pred) // chunk_size)
        chunked_pairs = []
        for i in range(n_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, len(all_pred))
            if end - start >= 4:  # Need at least 4 frames for jerk
                chunked_pairs.append((all_pred[start:end], all_gt[start:end]))
        return chunked_pairs

    return paired_sequences


# =============================================================================
#  Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Compute temporal evaluation metrics")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["h36m", "mpi"])
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--checkpoint-file", type=str, default="best_epoch.pth.tr")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num-cpus", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fvd-window", type=int, default=16,
                        help="Window size for FVD spatiotemporal features")
    return parser.parse_args()


def main():
    opts = parse_args()
    set_random_seed(opts.seed)

    args = get_config(opts.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model = load_model(args)
    checkpoint_path = os.path.join(opts.checkpoint_dir, opts.checkpoint_file)
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    # Handle DataParallel state dicts
    state_dict = checkpoint['model']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()

    print(f"[INFO] Loaded checkpoint from {checkpoint_path}")
    print(f"[INFO] Dataset: {opts.dataset}")
    print(f"[INFO] Device: {device}")

    # Run inference
    if opts.dataset == "h36m":
        paired_sequences = run_inference_h36m(args, model, device, opts.num_cpus)
    else:
        paired_sequences = run_inference_mpi(args, model, device, opts.num_cpus)

    if not paired_sequences:
        print("[ERROR] No sequences collected. Cannot compute metrics.")
        sys.exit(1)

    print(f"[INFO] Collected {len(paired_sequences)} sequence pairs for metric computation.")

    # ─── Compute per-sequence metrics ─────────────────────────────────────
    mpjve_list = []
    mpjae_list = []
    mpjje_list = []
    dtw_list = []

    for pred, gt in tqdm(paired_sequences, desc="Computing per-sequence metrics"):
        T = pred.shape[0]
        if T >= 2:
            mpjve_list.append(compute_mpjve(pred, gt))
        if T >= 3:
            mpjae_list.append(compute_mpjae(pred, gt))
        if T >= 4:
            mpjje_list.append(compute_mpjje(pred, gt))
        dtw_list.append(compute_dtw(pred, gt))

    # ─── Compute distribution-level metrics ───────────────────────────────
    pred_sequences = [p for p, g in paired_sequences]
    gt_sequences   = [g for p, g in paired_sequences]

    print("[INFO] Computing FID...")
    fid_score = compute_fid(pred_sequences, gt_sequences)

    # FVD window size: use smaller of specified window or shortest sequence
    min_seq_len = min(s.shape[0] for s in pred_sequences)
    fvd_window = min(opts.fvd_window, min_seq_len)
    print(f"[INFO] Computing FVD (window={fvd_window})...")
    fvd_score = compute_fvd(pred_sequences, gt_sequences, window_size=fvd_window)

    # ─── Aggregate results ────────────────────────────────────────────────
    results = {
        "MPJVE": float(np.mean(mpjve_list)) if mpjve_list else None,
        "MPJAE": float(np.mean(mpjae_list)) if mpjae_list else None,
        "MPJJE": float(np.mean(mpjje_list)) if mpjje_list else None,
        "DTW":   float(np.mean(dtw_list))   if dtw_list   else None,
        "FID":   fid_score,
        "FVD":   fvd_score,
    }

    # Print results
    print("\n" + "="*60)
    print(f"  Temporal Metrics — {opts.dataset.upper()}")
    print("="*60)
    for name, val in results.items():
        unit = {"MPJVE": "mm/frame", "MPJAE": "mm/frame²", "MPJJE": "mm/frame³",
                "DTW": "mm", "FID": "", "FVD": ""}
        if val is not None:
            print(f"  {name:8s}: {val:.4f} {unit.get(name, '')}")
        else:
            print(f"  {name:8s}: N/A")

    # Save to file
    os.makedirs(os.path.dirname(opts.output), exist_ok=True)
    with open(opts.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results saved to {opts.output}")


if __name__ == "__main__":
    main()
