"""
Python port of the MPI-INF-3DHP MATLAB evaluation code.

Computes PCK (Percentage of Correct Keypoints) and AUC (Area Under Curve)
metrics from inference_data.mat predictions and test set ground truth.

Ported from:
  - mpii_test_predictions_py.m
  - mpii_evaluate_errors.m
  - mpii_compute_3d_pck.m
  - mpii_get_joints.m
  - mpii_get_pck_auc_joint_groups.m
  - mpii_get_activity_name.m
"""

import os
import argparse
import numpy as np
import scipy.io as scio
import h5py
import csv


# ─── Constants ────────────────────────────────────────────────────────────────

JOINT_NAMES_RELEVANT = [
    'head_top', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist',
    'left_shoulder', 'left_elbow', 'left_wrist',
    'right_hip', 'right_knee', 'right_ankle',
    'left_hip', 'left_knee', 'left_ankle',
    'pelvis', 'spine', 'head'
]

# Joint groups for PCK/AUC (0-indexed into the 17 relevant joints)
JOINT_GROUPS = [
    ('Head',  [0]),          # head_top
    ('Neck',  [1]),          # neck
    ('Shou',  [2, 5]),       # right_shoulder, left_shoulder
    ('Elbow', [3, 6]),       # right_elbow, left_elbow
    ('Wrist', [4, 7]),       # right_wrist, left_wrist
    ('Hip',   [8, 11]),      # right_hip, left_hip
    ('Knee',  [9, 12]),      # right_knee, left_knee
    ('Ankle', [10, 13]),     # right_ankle, left_ankle
]

ACTIVITY_NAMES = {
    1: 'Standing/Walking',
    2: 'Exercising',
    3: 'Sitting',
    4: 'Reaching/Crouching',
    5: 'On The Floor',
    6: 'Sports',
    7: 'Miscellaneous',
}


# ─── Core metric functions ────────────────────────────────────────────────────

def compute_3d_pck(error_data_list, joint_groups, pck_thresh=150, thresh_step=5, thresh_max=150):
    """
    Compute PCK and AUC tables from per-joint errors.

    Args:
        error_data_list: list of dicts with keys 'method' (str) and 'error' (nj x nf ndarray)
        joint_groups: list of (name, joint_indices) tuples
        pck_thresh: threshold in mm for PCK (default 150)
        thresh_step: step size for AUC curve thresholds
        thresh_max: max threshold for AUC curve

    Returns:
        pck_table: dict mapping (method, group_name) -> pck value
        auc_table: dict mapping (method, group_name) -> auc value
        pck_header: list of group names + ['Total']
        method_names: list of method names
    """
    thresholds = np.arange(0, thresh_max + thresh_step, thresh_step)  # 0, 5, 10, ..., 150
    pck_table = {}
    auc_table = {}
    method_names = []

    for entry in error_data_list:
        method = entry['method']
        errors = entry['error']  # shape: (nj, nf)
        method_names.append(method)
        nf = errors.shape[1]

        total_pck_curve = None
        total_pck = 0.0
        total_joint_count = 0

        for group_name, joint_indices in joint_groups:
            group_errors = errors[joint_indices, :]  # (n_joints_in_group, nf)
            n_joints = len(joint_indices)

            # PCK curve for AUC
            pck_curve = np.array([
                np.sum(group_errors < t) / (n_joints * nf)
                for t in thresholds
            ])

            # AUC
            auc_val = 100.0 * np.sum(pck_curve) / len(thresholds)
            auc_table[(method, group_name)] = auc_val

            # PCK at threshold
            pck_val = 100.0 * np.sum(group_errors < pck_thresh) / (n_joints * nf)
            pck_table[(method, group_name)] = pck_val

            # Accumulate for total
            if total_pck_curve is None:
                total_pck_curve = pck_curve * n_joints
            else:
                total_pck_curve += pck_curve * n_joints
            total_pck += pck_val * n_joints
            total_joint_count += n_joints

        # Total (weighted average across groups)
        total_pck_curve /= total_joint_count
        total_auc = 100.0 * np.sum(total_pck_curve) / len(thresholds)
        total_pck /= total_joint_count

        pck_table[(method, 'Total')] = total_pck
        auc_table[(method, 'Total')] = total_auc

    pck_header = [g[0] for g in joint_groups] + ['Total']
    return pck_table, auc_table, pck_header, method_names


def evaluate_errors(sequencewise_errors, sequencewise_activities):
    """
    Full evaluation: sequence-wise and activity-wise MPJPE, PCK, and AUC.

    Args:
        sequencewise_errors: list of ndarrays, each (nj, nf_i) per-joint errors per sequence
        sequencewise_activities: list of ndarrays, each (nf_i,) activity labels per sequence

    Returns:
        sequencewise_results: dict with 'mpjpe', 'pck', 'auc' tables
        activitywise_results: dict with 'mpjpe', 'pck', 'auc' tables
    """
    # ── Sequence-wise ─────────────────────────────────────────────────────
    seq_error_data = []
    all_errors = []
    all_activities = []

    seq_mpjpe = {}
    for i, (errors, activities) in enumerate(zip(sequencewise_errors, sequencewise_activities)):
        seq_name = f'TestSeq{i + 1}'
        seq_error_data.append({'method': seq_name, 'error': errors})

        # Per-joint MPJPE
        mpjpe_per_joint = np.mean(errors, axis=1)  # (nj,)
        seq_mpjpe[seq_name] = {
            'per_joint': mpjpe_per_joint,
            'average': np.mean(mpjpe_per_joint),
        }

        all_errors.append(errors)
        all_activities.append(activities)

    all_errors_cat = np.concatenate(all_errors, axis=1)     # (nj, total_nf)
    all_activities_cat = np.concatenate(all_activities)      # (total_nf,)

    seq_pck, seq_auc, pck_header, seq_methods = compute_3d_pck(seq_error_data, JOINT_GROUPS)

    # ── Activity-wise ─────────────────────────────────────────────────────
    act_error_data = []
    act_mpjpe = {}
    for act_id in range(1, 8):
        act_name = ACTIVITY_NAMES[act_id]
        mask = all_activities_cat == act_id
        if np.sum(mask) == 0:
            continue
        act_errors = all_errors_cat[:, mask]
        act_error_data.append({'method': act_name, 'error': act_errors})

        mpjpe_per_joint = np.mean(act_errors, axis=1)
        act_mpjpe[act_name] = {
            'per_joint': mpjpe_per_joint,
            'average': np.mean(mpjpe_per_joint),
        }

    # Overall
    overall_mpjpe_per_joint = np.mean(all_errors_cat, axis=1)
    act_mpjpe['All'] = {
        'per_joint': overall_mpjpe_per_joint,
        'average': np.mean(overall_mpjpe_per_joint),
    }
    act_error_data.append({'method': 'All', 'error': all_errors_cat})

    act_pck, act_auc, _, act_methods = compute_3d_pck(act_error_data, JOINT_GROUPS)

    sequencewise_results = {
        'mpjpe': seq_mpjpe,
        'pck': seq_pck,
        'auc': seq_auc,
        'header': pck_header,
        'methods': seq_methods,
    }
    activitywise_results = {
        'mpjpe': act_mpjpe,
        'pck': act_pck,
        'auc': act_auc,
        'header': pck_header,
        'methods': act_methods,
    }
    return sequencewise_results, activitywise_results


# ─── Main evaluation pipeline ────────────────────────────────────────────────

def load_test_data(test_data_path, test_subject_ids):
    """
    Load ground truth annotations for each test sequence.

    Args:
        test_data_path: path to the directory containing TS1/, TS2/, ... folders
        test_subject_ids: list of subject IDs (e.g. [1, 2, 3, 4, 5, 6])

    Returns:
        gt_data: dict mapping seq_name -> {'univ_annot3': ..., 'valid_frame': ..., 'activity': ...}
    """
    gt_data = {}
    for sid in test_subject_ids:
        seq_name = f'TS{sid}'
        annot_path = os.path.join(test_data_path, seq_name, 'annot_data.mat')
        if not os.path.exists(annot_path):
            print(f"[WARN] {annot_path} not found, skipping {seq_name}")
            continue

        try:
            # Try h5py first (newer MATLAB v7.3 format)
            data = h5py.File(annot_path, 'r')
            valid_frame = np.squeeze(np.array(data['valid_frame'][()]))
            univ_annot3 = np.array(data['univ_annot3'][()])
            activity = np.squeeze(np.array(data['activity_annotation'][()]))
            data.close()
        except Exception:
            # Fall back to scipy for older .mat files
            data = scio.loadmat(annot_path)
            valid_frame = np.squeeze(data['valid_frame'])
            univ_annot3 = data['univ_annot3']
            activity = np.squeeze(data['activity_annotation'])

        gt_data[seq_name] = {
            'univ_annot3': univ_annot3,
            'valid_frame': valid_frame,
            'activity': activity,
        }
        print(f"[INFO] Loaded GT for {seq_name}: {int(np.sum(valid_frame))} valid frames")

    return gt_data


def compute_per_joint_errors(inference_mat_path, gt_data, test_subject_ids):
    """
    Compute per-joint errors between predictions and ground truth.

    Args:
        inference_mat_path: path to the inference_data.mat file
        gt_data: dict from load_test_data
        test_subject_ids: list of subject IDs

    Returns:
        sequencewise_errors: list of (17, nf) arrays
        sequencewise_activities: list of (nf,) arrays
    """
    pred_data = scio.loadmat(inference_mat_path)

    sequencewise_errors = []
    sequencewise_activities = []

    for sid in test_subject_ids:
        seq_name = f'TS{sid}'
        if seq_name not in gt_data:
            continue

        gt = gt_data[seq_name]
        valid_frame = gt['valid_frame'].astype(bool)
        univ_annot3 = gt['univ_annot3']  # Can be (3, 17, 1, nframes) or (nframes, 1, 17, 3) depending on h5py/scipy
        activity = gt['activity']

        pred_seq = pred_data[seq_name]  # (3, 17, 1, nf_valid)

        # Handle different data formats (h5py vs scipy.io.loadmat)
        # h5py loads as (nframes, 1, 17, 3), scipy loads as (3, 17, 1, nframes)
        if univ_annot3.shape[0] > 100:  # First dimension is likely frames (h5py format)
            # Transpose from (nframes, 1, 17, 3) to (3, 17, 1, nframes)
            univ_annot3 = np.transpose(univ_annot3, (3, 2, 1, 0))
        
        # Extract activity labels for valid frames
        valid_activities = activity[valid_frame].astype(int)
        sequencewise_activities.append(valid_activities)

        n_valid = int(np.sum(valid_frame))
        per_joint_error = np.zeros((17, n_valid))

        pje_idx = 0
        for j in range(len(valid_frame)):
            if valid_frame[j]:
                # GT: root-relative (subtract pelvis = joint 14, 0-indexed)
                gt_pose = univ_annot3[:, :, 0, j]  # (3, 17)
                gt_pose = gt_pose - gt_pose[:, 14:15]  # root-relative

                # Prediction
                pred_pose = pred_seq[:, :, 0, pje_idx]  # (3, 17)

                # Euclidean error per joint
                error = np.sqrt(np.sum((pred_pose - gt_pose) ** 2, axis=0))  # (17,)
                per_joint_error[:, pje_idx] = error
                pje_idx += 1

        sequencewise_errors.append(per_joint_error)

    return sequencewise_errors, sequencewise_activities


# ─── Output formatting ───────────────────────────────────────────────────────

def print_results(sequencewise_results, activitywise_results):
    """Print evaluation results in a readable format."""

    header = sequencewise_results['header']

    print("\n" + "=" * 80)
    print("SEQUENCE-WISE RESULTS")
    print("=" * 80)

    # MPJPE
    print(f"\n{'MPJPE (mm)':<25}", end="")
    for name in JOINT_NAMES_RELEVANT:
        print(f"{name[:8]:>10}", end="")
    print(f"{'Average':>10}")
    for method in sequencewise_results['methods']:
        mpjpe = sequencewise_results['mpjpe'][method]
        print(f"{method:<25}", end="")
        for val in mpjpe['per_joint']:
            print(f"{val:>10.1f}", end="")
        print(f"{mpjpe['average']:>10.1f}")

    # PCK
    print(f"\n{'PCK (%)':>25}", end="")
    for h in header:
        print(f"{h:>10}", end="")
    print()
    for method in sequencewise_results['methods']:
        print(f"{method:<25}", end="")
        for h in header:
            print(f"{sequencewise_results['pck'][(method, h)]:>10.1f}", end="")
        print()

    # AUC
    print(f"\n{'AUC (%)':>25}", end="")
    for h in header:
        print(f"{h:>10}", end="")
    print()
    for method in sequencewise_results['methods']:
        print(f"{method:<25}", end="")
        for h in header:
            print(f"{sequencewise_results['auc'][(method, h)]:>10.1f}", end="")
        print()

    print("\n" + "=" * 80)
    print("ACTIVITY-WISE RESULTS")
    print("=" * 80)

    # MPJPE
    print(f"\n{'MPJPE (mm)':<25} {'Average':>10}")
    for method in activitywise_results['methods']:
        mpjpe = activitywise_results['mpjpe'][method]
        print(f"{method:<25} {mpjpe['average']:>10.1f}")

    # PCK
    print(f"\n{'PCK (%)':>25}", end="")
    for h in header:
        print(f"{h:>10}", end="")
    print()
    for method in activitywise_results['methods']:
        print(f"{method:<25}", end="")
        for h in header:
            print(f"{activitywise_results['pck'][(method, h)]:>10.1f}", end="")
        print()

    # AUC
    print(f"\n{'AUC (%)':>25}", end="")
    for h in header:
        print(f"{h:>10}", end="")
    print()
    for method in activitywise_results['methods']:
        print(f"{method:<25}", end="")
        for h in header:
            print(f"{activitywise_results['auc'][(method, h)]:>10.1f}", end="")
        print()

    # Summary line
    print("\n" + "=" * 80)
    overall_pck = activitywise_results['pck'][('All', 'Total')]
    overall_auc = activitywise_results['auc'][('All', 'Total')]
    overall_mpjpe = activitywise_results['mpjpe']['All']['average']
    print(f"Overall => MPJPE: {overall_mpjpe:.1f} mm | PCK@150mm: {overall_pck:.1f}% | AUC: {overall_auc:.1f}%")
    print("=" * 80)


def save_results_csv(output_path, sequencewise_results, activitywise_results):
    """Save results to CSV files matching the MATLAB output format."""
    header = sequencewise_results['header']

    # Sequence-wise CSV
    seq_csv = os.path.join(output_path, 'mpii_3dhp_evaluation_sequencewise.csv')
    with open(seq_csv, 'w', newline='') as f:
        writer = csv.writer(f)

        # MPJPE section
        writer.writerow(['MPJPE'] + JOINT_NAMES_RELEVANT + ['Average'])
        for method in sequencewise_results['methods']:
            mpjpe = sequencewise_results['mpjpe'][method]
            writer.writerow([method] + [f'{v:.2f}' for v in mpjpe['per_joint']] + [f'{mpjpe["average"]:.2f}'])
        writer.writerow([])

        # PCK section
        writer.writerow(['PCK'] + header)
        for method in sequencewise_results['methods']:
            writer.writerow([method] + [f'{sequencewise_results["pck"][(method, h)]:.2f}' for h in header])
        writer.writerow([])

        # AUC section
        writer.writerow(['AUC'] + header)
        for method in sequencewise_results['methods']:
            writer.writerow([method] + [f'{sequencewise_results["auc"][(method, h)]:.2f}' for h in header])

    print(f"[INFO] Saved sequence-wise results to {seq_csv}")

    # Activity-wise CSV
    act_csv = os.path.join(output_path, 'mpii_3dhp_evaluation_activitywise.csv')
    with open(act_csv, 'w', newline='') as f:
        writer = csv.writer(f)

        # MPJPE section
        writer.writerow(['MPJPE'] + JOINT_NAMES_RELEVANT + ['Average'])
        for method in activitywise_results['methods']:
            mpjpe = activitywise_results['mpjpe'][method]
            writer.writerow([method] + [f'{v:.2f}' for v in mpjpe['per_joint']] + [f'{mpjpe["average"]:.2f}'])
        writer.writerow([])

        # PCK section
        writer.writerow(['PCK'] + header)
        for method in activitywise_results['methods']:
            writer.writerow([method] + [f'{activitywise_results["pck"][(method, h)]:.2f}' for h in header])
        writer.writerow([])

        # AUC section
        writer.writerow(['AUC'] + header)
        for method in activitywise_results['methods']:
            writer.writerow([method] + [f'{activitywise_results["auc"][(method, h)]:.2f}' for h in header])

    print(f"[INFO] Saved activity-wise results to {act_csv}")


# ─── CLI entry point ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='MPI-INF-3DHP PCK/AUC Evaluation (Python port)')
    parser.add_argument('--inference-mat', type=str, required=True,
                        help='Path to inference_data.mat (model predictions)')
    parser.add_argument('--test-data-path', type=str, required=True,
                        help='Path to test set root containing TS1/ ... TS6/ folders')
    parser.add_argument('--output-path', type=str, default=None,
                        help='Directory to save CSV results (default: same as inference mat dir)')
    parser.add_argument('--subjects', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6],
                        help='Test subject IDs (default: 1 2 3 4 5 6)')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.output_path is None:
        args.output_path = os.path.dirname(args.inference_mat)

    print(f"[INFO] Loading test ground truth from: {args.test_data_path}")
    gt_data = load_test_data(args.test_data_path, args.subjects)

    print(f"[INFO] Loading predictions from: {args.inference_mat}")
    seq_errors, seq_activities = compute_per_joint_errors(args.inference_mat, gt_data, args.subjects)

    print("[INFO] Computing PCK and AUC metrics...")
    seq_results, act_results = evaluate_errors(seq_errors, seq_activities)

    print_results(seq_results, act_results)
    save_results_csv(args.output_path, seq_results, act_results)


if __name__ == '__main__':
    main()
