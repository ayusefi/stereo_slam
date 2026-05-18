#!/usr/bin/env python3
"""
Generate top-down (X-Z) trajectory comparison plots for KITTI benchmark results.

Usage:
    python3 scripts/plot_trajectories.py --bench bench/49dca96 \
        --gt data/kitti/dataset/poses --seqs 00 07 02
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def load_kitti_poses(path: str) -> np.ndarray:
    poses = []
    for line in Path(path).read_text().splitlines():
        vals = [float(x) for x in line.split()]
        if len(vals) == 12:
            T = np.eye(4)
            T[:3, :] = np.array(vals).reshape(3, 4)
            poses.append(T)
    return np.array(poses)


def align_umeyama(est: np.ndarray, gt: np.ndarray, with_scale: bool = False):
    """Umeyama alignment: returns R, t, s such that s*(R @ est_pts.T) + t ≈ gt_pts."""
    mu_e = est.mean(0)
    mu_g = gt.mean(0)
    de = est - mu_e
    dg = gt - mu_g
    n = est.shape[0]
    cov = (dg.T @ de) / n
    U, sig, Vt = np.linalg.svd(cov)
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1
    R = U @ W @ Vt
    var_e = (np.linalg.norm(de, axis=1) ** 2).mean()
    s = (np.trace(np.diag(sig) @ W) / var_e) if with_scale else 1.0
    t = mu_g - s * (R @ mu_e)
    return R, t, s


def apply_align(pts: np.ndarray, R, t, s) -> np.ndarray:
    return (s * (R @ pts.T)).T + t


def plot_seq(ax, est_poses, gt_poses, seq: str, loop_kfs=None):
    est_pts = est_poses[:, :3, 3]
    gt_pts  = gt_poses[:, :3, 3]

    # Align estimated to GT
    R, t, s = align_umeyama(est_pts, gt_pts, with_scale=True)
    est_aligned = apply_align(est_pts, R, t, s)

    ax.plot(gt_pts[:, 0],          gt_pts[:, 2],          color="#888888",
            linewidth=1.2, label="Ground truth")
    ax.plot(est_aligned[:, 0],     est_aligned[:, 2],     color="#e05c2a",
            linewidth=1.2, label="Estimated (aligned)")

    ax.plot(gt_pts[0, 0],  gt_pts[0, 2],  "g^", markersize=7, label="Start")
    ax.plot(gt_pts[-1, 0], gt_pts[-1, 2], "rs", markersize=7, label="End")

    # Mark loop closure KFs if provided
    if loop_kfs:
        lx = [est_aligned[k, 0] for k in loop_kfs if k < len(est_aligned)]
        lz = [est_aligned[k, 2] for k in loop_kfs if k < len(est_aligned)]
        if lx:
            ax.scatter(lx, lz, marker="*", s=120, color="#2a7ae0", zorder=5,
                       label=f"Loop closure ({len(lx)})")

    ax.set_title(f"KITTI seq {seq}", fontsize=11, fontweight="bold")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, linewidth=0.4, alpha=0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", required=True, help="Bench output dir, e.g. bench/49dca96")
    parser.add_argument("--gt",    required=True, help="GT poses dir, e.g. data/kitti/dataset/poses")
    parser.add_argument("--seqs",  nargs="+", default=["00", "07"])
    parser.add_argument("--out",   default="results", help="Output directory for PNG files")
    args = parser.parse_args()

    bench_dir = Path(args.bench)
    gt_dir    = Path(args.gt)
    out_dir   = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect valid sequences
    seqs_to_plot = []
    for seq in args.seqs:
        traj_path = bench_dir / seq / "traj.txt"
        gt_path   = gt_dir / f"{seq}.txt"
        if not traj_path.exists():
            print(f"[skip] {seq}: no traj file at {traj_path}")
            continue
        if not gt_path.exists():
            print(f"[skip] {seq}: no GT at {gt_path}")
            continue
        seqs_to_plot.append((seq, traj_path, gt_path))

    if not seqs_to_plot:
        print("No sequences to plot.")
        return

    n = len(seqs_to_plot)
    fig_w = max(6 * n, 8)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 6))
    if n == 1:
        axes = [axes]

    for ax, (seq, traj_path, gt_path) in zip(axes, seqs_to_plot):
        est_poses = load_kitti_poses(str(traj_path))
        gt_poses  = load_kitti_poses(str(gt_path))

        # Trim to same length
        L = min(len(est_poses), len(gt_poses))
        est_poses = est_poses[:L]
        gt_poses  = gt_poses[:L]

        plot_seq(ax, est_poses, gt_poses, seq)
        print(f"[seq {seq}] plotted {L} poses")

    plt.suptitle("sslam vs Ground Truth  ·  KITTI Odometry", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    seqs_str = "_".join(s for s, *_ in seqs_to_plot)
    out_path = out_dir / f"traj_{seqs_str}.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out_path}")

    # Also save individual plots
    for ax_i, (seq, traj_path, gt_path) in enumerate(seqs_to_plot):
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        est_poses = load_kitti_poses(str(traj_path))
        gt_poses  = load_kitti_poses(str(gt_path))
        L = min(len(est_poses), len(gt_poses))
        plot_seq(ax2, est_poses[:L], gt_poses[:L], seq)
        plt.tight_layout()
        p = out_dir / f"traj_{seq}_xz.png"
        fig2.savefig(str(p), dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved → {p}")

    plt.close(fig)


if __name__ == "__main__":
    main()
