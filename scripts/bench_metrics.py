#!/usr/bin/env python3
"""
Compute benchmark metrics from a sslam trajectory and write metrics.json.

Usage:
    bench_metrics.py --traj EST.txt --out metrics.json \
        [--gt GT.txt] [--log app_log.txt] [--seq 00]

Outputs a JSON file with:
  - trajectory-internal stats (step lengths, second differences)
  - GT-relative stats (raw ATE, evo-aligned ATE) when --gt is given
  - app summary stats (latency, lost frames, KF/MP counts, BA, loops)
    when --log is given
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Trajectory I/O
# ---------------------------------------------------------------------------

def load_kitti_poses(path: str) -> np.ndarray:
    """Load KITTI-format trajectory (12 floats per line) → (N, 4, 4)."""
    poses = []
    for line in Path(path).read_text().splitlines():
        vals = [float(x) for x in line.split()]
        if len(vals) == 12:
            T = np.eye(4)
            T[:3, :] = np.array(vals).reshape(3, 4)
            poses.append(T)
    if not poses:
        raise ValueError(f"No valid poses found in {path}")
    return np.array(poses)


# ---------------------------------------------------------------------------
# Trajectory statistics
# ---------------------------------------------------------------------------

def traj_stats(poses: np.ndarray) -> dict:
    """Step lengths and second-difference (jump) scores."""
    pos = poses[:, :3, 3]
    steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    sec = np.linalg.norm(pos[2:] - 2.0 * pos[1:-1] + pos[:-2], axis=1)

    worst_step_idx = int(np.argmax(steps))
    worst_sec_idx  = int(np.argmax(sec)) + 1  # +1 because sec[0] is pos[1]

    return {
        "n_frames":            len(poses),
        "max_step_m":          round(float(steps.max()), 4),
        "max_step_frame":      worst_step_idx,
        "median_step_m":       round(float(np.median(steps)), 4),
        "p95_step_m":          round(float(np.percentile(steps, 95)), 4),
        "max_second_diff_m":   round(float(sec.max()), 4),
        "max_second_diff_frame": worst_sec_idx,
        "p95_second_diff_m":   round(float(np.percentile(sec, 95)), 4),
    }


# ---------------------------------------------------------------------------
# GT-relative accuracy
# ---------------------------------------------------------------------------

def ate_raw(est: np.ndarray, gt: np.ndarray) -> float:
    """Raw (no alignment) position RMSE in metres."""
    n = min(len(est), len(gt))
    err = np.linalg.norm(est[:n, :3, 3] - gt[:n, :3, 3], axis=1)
    return round(float(np.sqrt(np.mean(err ** 2))), 4)


def umeyama_align(src: np.ndarray, dst: np.ndarray) -> tuple:
    """
    Umeyama rigid-body alignment (no scale).  Minimises ||dst - (R@src + t)||.
    src, dst: (N, 3) position arrays.  Returns (R, t) as np.ndarray.
    """
    n = len(src)
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c  = src - mu_src
    dst_c  = dst - mu_dst
    Sigma  = (dst_c.T @ src_c) / n
    U, _, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    t = mu_dst - R @ mu_src
    return R, t


def ate_aligned(est: np.ndarray, gt: np.ndarray) -> float:
    """
    Umeyama-aligned translational ATE (no scale correction).
    Trims to the shorter of the two trajectories.
    """
    n = min(len(est), len(gt))
    ep = est[:n, :3, 3]
    gp = gt[:n, :3, 3]
    R, t = umeyama_align(ep, gp)
    aligned = (R @ ep.T).T + t
    err = np.linalg.norm(aligned - gp, axis=1)
    return round(float(np.sqrt(np.mean(err ** 2))), 4)


# ---------------------------------------------------------------------------
# App log parsing
# ---------------------------------------------------------------------------

def parse_app_log(log_path: str) -> dict:
    """
    Parse the kitti_stereo stdout summary block for structured fields.

    Expected summary format:
        avg latency  : 50.57 ms/frame
        avg stereo % : 71%
        lost frames  : 0 / 4541
        keyframes    : 412 active / 560 total
        map points   : 19342 active / 45000 total
        local BA     : 412 runs, avg 18.3 ms, max 95.1 ms
        loop closures: 1
        ATE RMSE     : 28.40 m  (raw, no alignment)     ← only if --gt given
    """
    text = Path(log_path).read_text()
    out = {}

    def _float(pat):
        m = re.search(pat, text)
        return round(float(m.group(1)), 4) if m else None

    def _int(pat):
        m = re.search(pat, text)
        return int(m.group(1)) if m else None

    lost = re.search(r"lost frames\s*:\s*(\d+)\s*/\s*(\d+)", text)
    kf   = re.search(r"keyframes\s*:\s*(\d+)\s*active\s*/\s*(\d+)\s*total", text)
    mp   = re.search(r"map points\s*:\s*(\d+)\s*active\s*/\s*(\d+)\s*total", text)
    ba   = re.search(
        r"local BA\s*:\s*(\d+)\s*runs,\s*avg\s*([\d.]+)\s*ms,\s*max\s*([\d.]+)\s*ms",
        text,
    )

    out["avg_latency_ms"] = _float(r"avg latency\s*:\s*([\d.]+)\s*ms")
    out["avg_stereo_pct"] = _int(r"avg stereo %\s*:\s*(\d+)%")

    if lost:
        out["lost_frames"] = int(lost.group(1))
        out["frames_run"]  = int(lost.group(2))

    if kf:
        out["kf_active"] = int(kf.group(1))
        out["kf_total"]  = int(kf.group(2))

    if mp:
        out["mp_active"] = int(mp.group(1))
        out["mp_total"]  = int(mp.group(2))

    if ba:
        out["ba_runs"]   = int(ba.group(1))
        out["ba_avg_ms"] = round(float(ba.group(2)), 2)
        out["ba_max_ms"] = round(float(ba.group(3)), 2)

    lc = _int(r"loop closures\s*:\s*(\d+)")
    if lc is not None:
        out["loop_closures"] = lc

    ate_app = _float(r"ATE RMSE\s*:\s*([\d.]+)\s*m")
    if ate_app is not None:
        out["ate_raw_app_m"] = ate_app

    return {k: v for k, v in out.items() if v is not None}


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

_LABELS = {
    "seq":                   "Sequence",
    "n_frames":              "Frames",
    "frames_run":            "Frames run",
    "ate_raw_m":             "ATE raw (no align) [m]",
    "ate_aligned_m":         "ATE aligned (evo)  [m]",
    "max_step_m":            "Max step            [m]",
    "max_step_frame":        "Max step @ frame",
    "median_step_m":         "Median step         [m]",
    "p95_step_m":            "p95 step            [m]",
    "max_second_diff_m":     "Max 2nd-diff        [m]",
    "max_second_diff_frame": "Max 2nd-diff @ frame",
    "p95_second_diff_m":     "p95 2nd-diff        [m]",
    "lost_frames":           "Lost frames",
    "kf_active":             "KFs active",
    "kf_total":              "KFs total",
    "mp_active":             "MPs active",
    "mp_total":              "MPs total",
    "ba_runs":               "Local BA runs",
    "ba_avg_ms":             "Local BA avg        [ms]",
    "ba_max_ms":             "Local BA max        [ms]",
    "loop_closures":         "Loop closures",
    "avg_latency_ms":        "Avg latency         [ms]",
    "avg_stereo_pct":        "Avg stereo match    [%]",
}

_ORDER = list(_LABELS.keys())


def print_table(metrics: dict, seq: str) -> None:
    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  Benchmark — seq {seq or metrics.get('seq', '?')}")
    print(sep)
    for key in _ORDER:
        if key in metrics:
            label = _LABELS.get(key, key)
            print(f"  {label:<36s} {metrics[key]}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", required=True,  help="Estimated trajectory (KITTI format)")
    ap.add_argument("--out",  required=True,  help="Output metrics JSON file")
    ap.add_argument("--gt",   default="",     help="Ground-truth poses (KITTI format)")
    ap.add_argument("--log",  default="",     help="kitti_stereo stdout log")
    ap.add_argument("--seq",  default="",     help="Sequence label for display")
    args = ap.parse_args()

    metrics: dict = {}
    if args.seq:
        metrics["seq"] = args.seq

    # ---- Trajectory internal stats ----------------------------------------
    try:
        est = load_kitti_poses(args.traj)
        metrics.update(traj_stats(est))
    except Exception as exc:
        print(f"[bench_metrics] WARNING: trajectory stats failed: {exc}", file=sys.stderr)

    # ---- GT-relative accuracy ---------------------------------------------
    if args.gt:
        try:
            gt = load_kitti_poses(args.gt)
            metrics["ate_raw_m"]     = ate_raw(est, gt)
            metrics["ate_aligned_m"] = ate_aligned(est, gt)
        except Exception as exc:
            print(f"[bench_metrics] WARNING: ATE computation failed: {exc}", file=sys.stderr)

    # ---- App log ----------------------------------------------------------
    if args.log:
        try:
            metrics.update(parse_app_log(args.log))
        except Exception as exc:
            print(f"[bench_metrics] WARNING: log parsing failed: {exc}", file=sys.stderr)

    # ---- Write JSON -------------------------------------------------------
    Path(args.out).write_text(json.dumps(metrics, indent=2) + "\n")

    # ---- Print table ------------------------------------------------------
    print_table(metrics, args.seq)


if __name__ == "__main__":
    main()
