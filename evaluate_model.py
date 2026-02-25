"""
Evaluate a single trained MotionAGFormer model — all 9 metrics.
================================================================

Computes both the standard metrics (MPJPE, P-MPJPE, Acceleration Error)
AND the 6 temporal metrics (MPJVE, MPJAE, MPJJE, DTW, FID, FVD)
for a single model checkpoint on a single dataset.

Usage Examples:
    # Evaluate xsmall on Human3.6M using a pre-trained checkpoint
    python evaluate_model.py --size xsmall --dataset h36m --checkpoint checkpoint/motionagformer-xs-h36m.pth.tr

    # Evaluate base on MPI using a checkpoint you trained
    python evaluate_model.py --size base --dataset mpi --checkpoint results/checkpoints/base-mpi/best_epoch.pth.tr

    # Evaluate large on Human3.6M, save results to a custom path
    python evaluate_model.py --size large --dataset h36m --checkpoint checkpoint/motionagformer-large-h36m.pth.tr --output my_results.json

    # Only compute temporal metrics (skip standard MPJPE/P-MPJPE/AccErr)
    python evaluate_model.py --size base --dataset h36m --checkpoint checkpoint/motionagformer-b-h36m.pth.tr --temporal-only

    # Only compute standard metrics (skip temporal)
    python evaluate_model.py --size base --dataset h36m --checkpoint checkpoint/motionagformer-b-h36m.pth.tr --standard-only
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

from utils.tools import get_config, set_random_seed
from utils.learning import load_model
from utils.data import flip_data
from data.reader.h36m import DataReaderH36M
from data.reader.motion_dataset import MotionDataset3D, Fusion
from torch.utils.data import DataLoader

# Import the temporal metric functions
from compute_temporal_metrics import (
    compute_mpjve, compute_mpjae, compute_mpjje,
    compute_dtw, compute_fid, compute_fvd,
)

# ─── Lookup tables ────────────────────────────────────────────────────────────

SIZE_TO_CONFIG = {
    "xsmall": {"h36m": "configs/h36m/MotionAGFormer-xsmall.yaml", "mpi": "configs/mpi/MotionAGFormer-xsmall.yaml", "n_frames": 27},
    "small":  {"h36m": "configs/h36m/MotionAGFormer-small.yaml",  "mpi": "configs/mpi/MotionAGFormer-small.yaml",  "n_frames": 81},
    "base":   {"h36m": "configs/h36m/MotionAGFormer-base.yaml",   "mpi": "configs/mpi/MotionAGFormer-base.yaml",   "n_frames": 243},
    "large":  {"h36m": "configs/h36m/MotionAGFormer-large.yaml",  "mpi": "configs/mpi/MotionAGFormer-large.yaml",  "n_frames": 243},
}


# =============================================================================
#  Standard Metrics (MPJPE, P-MPJPE, Acceleration Error)
# =============================================================================

def evaluate_h36m_standard(args, model, device, num_cpus):
    """Run standard H36M evaluation — returns (mpjpe, p_mpjpe, accel_err, paired_sequences)."""
    from loss.pose3d import mpjpe as calculate_mpjpe
    from loss.pose3d import p_mpjpe as calculate_p_mpjpe
    from loss.pose3d import acc_error as calculate_acc_err

    test_dataset = MotionDataset3D(args, args.subset_list, 'test')
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size,
                             num_workers=max(1, num_cpus - 1), pin_memory=True,
                             prefetch_factor=max(1, (num_cpus - 1) // 3),
                             persistent_workers=True)
    datareader = DataReaderH36M(n_frames=args.n_frames, sample_stride=1,
                                data_stride_train=args.n_frames // 3, data_stride_test=args.n_frames,
                                dt_root='data/motion3d', dt_file=args.dt_file)

    print("[INFO] Running H36M inference...")
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
    actions = np.array(datareader.dt_dataset['test']['action'])
    factors = np.array(datareader.dt_dataset['test']['2.5d_factor'])
    gts = np.array(datareader.dt_dataset['test']['joints_2.5d_image'])
    sources = np.array(datareader.dt_dataset['test']['source'])

    num_test_frames = len(actions)
    frames = np.array(range(num_test_frames))
    action_clips = actions[split_id_test]
    factor_clips = factors[split_id_test]
    source_clips = sources[split_id_test]
    frame_clips = frames[split_id_test]
    gt_clips = gts[split_id_test]

    if hasattr(args, 'add_velocity') and args.add_velocity:
        action_clips = action_clips[:, :-1]
        factor_clips = factor_clips[:, :-1]
        frame_clips = frame_clips[:, :-1]
        gt_clips = gt_clips[:, :-1]

    e1_all = np.zeros(num_test_frames)
    e2_all = np.zeros(num_test_frames)
    acc_err_all = np.zeros(num_test_frames - 2)
    oc = np.zeros(num_test_frames)

    block_list = ['s_09_act_05_subact_02', 's_09_act_10_subact_02', 's_09_act_13_subact_01']
    action_names = sorted(set(datareader.dt_dataset['test']['action']))

    results = {a: [] for a in action_names}
    results_procrustes = {a: [] for a in action_names}
    results_acceleration = {a: [] for a in action_names}

    paired_sequences = []

    for idx in range(len(action_clips)):
        source = source_clips[idx][0][:-6]
        if source in block_list:
            continue
        frame_list = frame_clips[idx]
        factor = factor_clips[idx][:, None, None]
        gt = gt_clips[idx]
        pred = results_all[idx] * factor

        # Root-relative
        pred = pred - pred[:, 0:1, :]
        gt = gt - gt[:, 0:1, :]

        paired_sequences.append((pred, gt))

        err1 = calculate_mpjpe(pred, gt)
        acc_err = calculate_acc_err(pred, gt)
        acc_err_all[frame_list[:-2]] += acc_err
        e1_all[frame_list] += err1
        err2 = calculate_p_mpjpe(pred, gt)
        e2_all[frame_list] += err2
        oc[frame_list] += 1

    for idx in range(num_test_frames):
        if e1_all[idx] > 0:
            action = actions[idx]
            results[action].append(e1_all[idx] / oc[idx])
            results_procrustes[action].append(e2_all[idx] / oc[idx])
            results_acceleration[action].append(acc_err_all[idx] / oc[idx] if idx < num_test_frames - 2 else 0)

    e1 = np.mean([np.mean(results[a]) for a in action_names])
    e2 = np.mean([np.mean(results_procrustes[a]) for a in action_names])
    acc = np.mean([np.mean(results_acceleration[a]) for a in action_names])

    return e1, e2, acc, paired_sequences


def evaluate_mpi_standard(args, model, device, num_cpus):
    """Run standard MPI evaluation — returns (mpjpe, paired_sequences)."""
    from utils.utils_3dhp import get_variable, AccumLoss
    from utils.data import denormalize

    test_dataset = Fusion(args, train=False)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.test_batch_size,
                             num_workers=max(1, num_cpus - 1), pin_memory=True,
                             prefetch_factor=max(1, (num_cpus - 1) // 3),
                             persistent_workers=True)

    joints_left = [5, 6, 7, 11, 12, 13]
    joints_right = [2, 3, 4, 8, 9, 10]

    print("[INFO] Running MPI inference...")
    error_sum = AccumLoss()
    all_pred_frames = []
    all_gt_frames = []
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

            from utils.utils_3dhp import mpjpe_cal
            joint_error = mpjpe_cal(pred_out, out_target).item()
            error_sum.update(joint_error * N, N)

            for i in range(N):
                all_pred_frames.append(pred_out[i].cpu().numpy())
                all_gt_frames.append(out_target[i, :pred_out.shape[1]].cpu().numpy())

    mpjpe_val = error_sum.avg

    # Group single-frame predictions into sequences for temporal metrics
    paired_sequences = []
    if all_pred_frames:
        all_pred = np.concatenate(all_pred_frames, axis=0)
        all_gt = np.concatenate(all_gt_frames, axis=0)
        chunk_size = max(args.n_frames, 27)
        n_chunks = max(1, len(all_pred) // chunk_size)
        for i in range(n_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, len(all_pred))
            if end - start >= 4:
                paired_sequences.append((all_pred[start:end], all_gt[start:end]))

    return mpjpe_val, paired_sequences


# =============================================================================
#  Temporal Metrics (MPJVE, MPJAE, MPJJE, DTW, FID, FVD)
# =============================================================================

def compute_all_temporal(paired_sequences, fvd_window=16):
    """Compute all 6 temporal metrics from paired (pred, gt) sequences."""
    mpjve_list, mpjae_list, mpjje_list, dtw_list = [], [], [], []

    for pred, gt in tqdm(paired_sequences, desc="Temporal metrics"):
        T = pred.shape[0]
        if T >= 2:
            mpjve_list.append(compute_mpjve(pred, gt))
        if T >= 3:
            mpjae_list.append(compute_mpjae(pred, gt))
        if T >= 4:
            mpjje_list.append(compute_mpjje(pred, gt))
        dtw_list.append(compute_dtw(pred, gt))

    pred_seqs = [p for p, g in paired_sequences]
    gt_seqs = [g for p, g in paired_sequences]

    print("[INFO] Computing FID...")
    fid_score = compute_fid(pred_seqs, gt_seqs)

    min_seq_len = min(s.shape[0] for s in pred_seqs)
    fvd_win = min(fvd_window, min_seq_len)
    print(f"[INFO] Computing FVD (window={fvd_win})...")
    fvd_score = compute_fvd(pred_seqs, gt_seqs, window_size=fvd_win)

    return {
        "MPJVE": float(np.mean(mpjve_list)) if mpjve_list else None,
        "MPJAE": float(np.mean(mpjae_list)) if mpjae_list else None,
        "MPJJE": float(np.mean(mpjje_list)) if mpjje_list else None,
        "DTW":   float(np.mean(dtw_list))   if dtw_list   else None,
        "FID":   fid_score,
        "FVD":   fvd_score,
    }


# =============================================================================
#  Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MotionAGFormer model (standard + temporal metrics)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_model.py --size xsmall --dataset h36m --checkpoint checkpoint/motionagformer-xs-h36m.pth.tr
  python evaluate_model.py --size base --dataset mpi --checkpoint results/checkpoints/base-mpi/best_epoch.pth.tr
  python evaluate_model.py --size large --dataset h36m --checkpoint checkpoint/motionagformer-large-h36m.pth.tr --output large_h36m.json
  python evaluate_model.py --size base --dataset h36m --checkpoint checkpoint/motionagformer-b-h36m.pth.tr --temporal-only
  python evaluate_model.py --size base --dataset h36m --checkpoint checkpoint/motionagformer-b-h36m.pth.tr --standard-only
        """)
    parser.add_argument("--size", type=str, required=True,
                        choices=["xsmall", "small", "base", "large"],
                        help="Model size")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["h36m", "mpi"],
                        help="Dataset to evaluate on")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint file (.pth.tr)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file (default: results/eval_{size}_{dataset}.json)")
    parser.add_argument("--num-cpus", type=int, default=min(os.cpu_count() or 4, 16),
                        help="Number of CPU workers for data loading")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fvd-window", type=int, default=16,
                        help="Window size for FVD computation")
    parser.add_argument("--temporal-only", action="store_true",
                        help="Only compute temporal metrics (skip standard MPJPE/P-MPJPE/AccErr)")
    parser.add_argument("--standard-only", action="store_true",
                        help="Only compute standard metrics (skip temporal)")
    return parser.parse_args()


def main():
    opts = parse_args()
    set_random_seed(opts.seed)
    start_time = time.time()

    # Resolve config
    cfg = SIZE_TO_CONFIG[opts.size]
    config_path = cfg[opts.dataset]
    args = get_config(config_path)

    # Resolve output path
    if opts.output is None:
        os.makedirs("results", exist_ok=True)
        opts.output = f"results/eval_{opts.size}_{opts.dataset}.json"

    # Validate checkpoint
    if not os.path.exists(opts.checkpoint):
        print(f"[ERROR] Checkpoint not found: {opts.checkpoint}")
        sys.exit(1)

    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    print(f"\n{'='*60}")
    print(f"  Evaluating MotionAGFormer-{opts.size} on {opts.dataset.upper()}")
    print(f"  Checkpoint: {opts.checkpoint}")
    print(f"  Config:     {config_path}")
    print(f"  Device:     {device}")
    print(f"{'='*60}\n")

    model = load_model(args)
    checkpoint = torch.load(opts.checkpoint, map_location='cpu')
    state_dict = checkpoint['model']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()

    # ─── Run evaluation ──────────────────────────────────────────────────
    all_metrics = {}
    paired_sequences = []

    if not opts.temporal_only:
        print("\n── Standard Metrics ──────────────────────────────────────")
        if opts.dataset == "h36m":
            mpjpe, p_mpjpe, acc_err, paired_sequences = evaluate_h36m_standard(
                args, model, device, opts.num_cpus)
            all_metrics["MPJPE (mm)"] = round(float(mpjpe), 4)
            all_metrics["P-MPJPE (mm)"] = round(float(p_mpjpe), 4)
            all_metrics["Acceleration Error (mm/s²)"] = round(float(acc_err), 4)
            print(f"  MPJPE:    {mpjpe:.4f} mm")
            print(f"  P-MPJPE:  {p_mpjpe:.4f} mm")
            print(f"  AccErr:   {acc_err:.4f} mm/s²")

        elif opts.dataset == "mpi":
            mpjpe, paired_sequences = evaluate_mpi_standard(
                args, model, device, opts.num_cpus)
            all_metrics["MPJPE (mm)"] = round(float(mpjpe), 4)
            print(f"  MPJPE:    {mpjpe:.2f} mm")

    if not opts.standard_only:
        # If we skipped standard eval, we still need to run inference for temporal
        if not paired_sequences:
            print("\n[INFO] Running inference to collect sequences for temporal metrics...")
            from compute_temporal_metrics import run_inference_h36m, run_inference_mpi
            if opts.dataset == "h36m":
                paired_sequences = run_inference_h36m(args, model, device, opts.num_cpus)
            else:
                paired_sequences = run_inference_mpi(args, model, device, opts.num_cpus)

        if paired_sequences:
            print(f"\n── Temporal Metrics ({len(paired_sequences)} sequences) ─────────────")
            temporal = compute_all_temporal(paired_sequences, fvd_window=opts.fvd_window)
            for name, val in temporal.items():
                unit = {"MPJVE": "mm/frame", "MPJAE": "mm/frame²", "MPJJE": "mm/frame³",
                        "DTW": "mm", "FID": "", "FVD": ""}
                if val is not None:
                    all_metrics[name] = round(val, 4)
                    print(f"  {name:8s}: {val:.4f} {unit.get(name, '')}")
                else:
                    all_metrics[name] = None
                    print(f"  {name:8s}: N/A")
        else:
            print("[WARN] No sequences available for temporal metrics.")

    # ─── Save results ────────────────────────────────────────────────────
    output_dir = os.path.dirname(opts.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    result = {
        "model": f"MotionAGFormer-{opts.size}",
        "dataset": opts.dataset,
        "checkpoint": opts.checkpoint,
        "config": config_path,
        "metrics": all_metrics,
    }

    with open(opts.output, 'w') as f:
        json.dump(result, f, indent=2)

    elapsed = time.time() - start_time
    mins, secs = divmod(elapsed, 60)

    print(f"\n{'='*60}")
    print(f"  Results saved to: {opts.output}")
    print(f"  Time: {int(mins)}m {int(secs)}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
