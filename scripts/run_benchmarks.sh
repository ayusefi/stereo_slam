#!/usr/bin/env bash
# Run sslam on one or more KITTI sequences and emit metrics.json.
#
# Usage:
#   ./scripts/run_benchmarks.sh [options]
#
# Options:
#   --dataset-root DIR    KITTI dataset root   (default: data/kitti/dataset)
#   --seqs LIST           Comma-separated ids  (default: 00,07)
#   --out DIR             Output directory     (default: bench/<git-hash>)
#   --max-frames N        Limit frames/seq     (default: all)
#   --baseline FILE       Compare against this metrics.json; exit non-zero
#                         if any tracked metric regresses beyond tolerance.
#
# Output layout:
#   bench/<run>/
#     <seq>/
#       traj.txt        resolved trajectory (KITTI format)
#       app_log.txt     kitti_stereo stdout
#       stderr.txt      kitti_stereo stderr (loop/LC diagnostics)
#       metrics.json    per-sequence metrics
#     metrics.json      combined metrics for all sequences
#
# Examples:
#   # Quick smoke test (200 frames each)
#   ./scripts/run_benchmarks.sh --seqs 00,07 --max-frames 200
#
#   # Full run, compare to saved baseline
#   ./scripts/run_benchmarks.sh --seqs 00,07 \
#       --baseline bench/baseline/metrics.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- Defaults --------------------------------------------------------------
DATASET_ROOT="$REPO_ROOT/data/kitti/dataset"
SEQS="00,07"
OUT_DIR=""
MAX_FRAMES=""
BASELINE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
        --seqs)         SEQS="$2";         shift 2 ;;
        --out)          OUT_DIR="$2";      shift 2 ;;
        --max-frames)   MAX_FRAMES="$2";   shift 2 ;;
        --baseline)     BASELINE="$2";     shift 2 ;;
        -h|--help)
            sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ---- Derived paths ---------------------------------------------------------
GIT_HASH=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo "unknown")
BUILD_TYPE=$(grep -i 'cmake_build_type' "$REPO_ROOT/build/CMakeCache.txt" 2>/dev/null \
             | awk -F= '{print $2}' || echo "unknown")
[[ -z "$OUT_DIR" ]] && OUT_DIR="$REPO_ROOT/bench/$GIT_HASH"

APP="$REPO_ROOT/build/apps/kitti_stereo"
METRICS_PY="$SCRIPT_DIR/bench_metrics.py"

# ---- Preflight checks ------------------------------------------------------
if [[ ! -x "$APP" ]]; then
    echo "error: $APP not found — build first with: cmake --build build -j4" >&2
    exit 1
fi
if [[ ! -f "$METRICS_PY" ]]; then
    echo "error: $METRICS_PY not found" >&2
    exit 1
fi
if ! python3 -c "import numpy" 2>/dev/null; then
    echo "error: numpy not found — pip install numpy" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "==================================================================="
echo "  sslam benchmark"
echo "  git      : $GIT_HASH  ($BUILD_TYPE)"
echo "  seqs     : $SEQS"
echo "  out      : $OUT_DIR"
[[ -n "$MAX_FRAMES" ]] && echo "  max-frames: $MAX_FRAMES"
echo "==================================================================="

FAIL=0
PROCESSED_SEQS=()

IFS=',' read -ra SEQ_LIST <<< "$SEQS"

for SEQ in "${SEQ_LIST[@]}"; do
    SEQ_DIR="$DATASET_ROOT/sequences/$SEQ"
    GT_FILE="$DATASET_ROOT/poses/$SEQ.txt"
    OUT_SEQ="$OUT_DIR/$SEQ"

    if [[ ! -d "$SEQ_DIR" ]]; then
        echo ""
        echo "[SKIP] seq $SEQ — not found at $SEQ_DIR"
        continue
    fi

    mkdir -p "$OUT_SEQ"

    # Build kitti_stereo command
    CMD=("$APP" "$SEQ_DIR"
         --no-display
         --output "$OUT_SEQ/traj.txt")
    [[ -f "$REPO_ROOT/configs/kitti.yaml" ]] && CMD+=(--config "$REPO_ROOT/configs/kitti.yaml")
    [[ -f "$GT_FILE"      ]] && CMD+=(--gt "$GT_FILE")
    [[ -n "$MAX_FRAMES"   ]] && CMD+=(--max-frames "$MAX_FRAMES")

    echo ""
    echo "--- seq $SEQ ---"
    echo "  cmd: ${CMD[*]}"
    echo ""

    # Run and tee stdout; stderr goes to its own file for LC diagnostics
    if "${CMD[@]}" \
            2>"$OUT_SEQ/stderr.txt" \
            | tee "$OUT_SEQ/app_log.txt"; then
        echo "[OK] seq $SEQ finished"
    else
        echo "[FAIL] kitti_stereo exited with error for seq $SEQ" >&2
        FAIL=1
        continue
    fi

    # Compute metrics JSON
    MARGS=(--traj "$OUT_SEQ/traj.txt"
           --out  "$OUT_SEQ/metrics.json"
           --seq  "$SEQ"
           --log  "$OUT_SEQ/app_log.txt")
    [[ -f "$GT_FILE" ]] && MARGS+=(--gt "$GT_FILE")

    if python3 "$METRICS_PY" "${MARGS[@]}"; then
        echo "[OK] metrics → $OUT_SEQ/metrics.json"
        PROCESSED_SEQS+=("$SEQ")
    else
        echo "[FAIL] bench_metrics.py failed for seq $SEQ" >&2
        FAIL=1
    fi
done

# ---- Combine per-sequence JSONs into one top-level file --------------------
if [[ ${#PROCESSED_SEQS[@]} -gt 0 ]]; then
    python3 - "$GIT_HASH" "$BUILD_TYPE" "$OUT_DIR" "${PROCESSED_SEQS[@]}" << 'PYEOF'
import sys, json
from pathlib import Path

git_hash, build_type, out_dir_s = sys.argv[1], sys.argv[2], sys.argv[3]
seqs = sys.argv[4:]
out_dir = Path(out_dir_s)

combined = {
    "git_commit":  git_hash,
    "build_type":  build_type,
    "sequences":   {},
}
for s in seqs:
    p = out_dir / s / "metrics.json"
    if p.exists():
        combined["sequences"][s] = json.loads(p.read_text())

(out_dir / "metrics.json").write_text(json.dumps(combined, indent=2) + "\n")
print(f"\nCombined metrics → {out_dir}/metrics.json")
PYEOF
fi

# ---- Optional baseline comparison ------------------------------------------
if [[ -n "$BASELINE" && -f "$BASELINE" && -f "$OUT_DIR/metrics.json" ]]; then
    echo ""
    echo "--- Baseline comparison ---"
    python3 - "$BASELINE" "$OUT_DIR/metrics.json" << 'PYEOF'
import sys, json

TRACKED = {
    "ate_raw_m":        ("ATE raw    [m]",  "lower_is_better", 0.05),
    "ate_aligned_m":    ("ATE aligned[m]",  "lower_is_better", 0.05),
    "max_step_m":       ("Max step   [m]",  "lower_is_better", 0.10),
    "lost_frames":      ("Lost frames",      "lower_is_better", 2),
    "loop_closures":    ("Loops",            "higher_is_better", 0),
}

baseline = json.loads(open(sys.argv[1]).read())
current  = json.loads(open(sys.argv[2]).read())

fail = False
for seq, cur_seq in current.get("sequences", {}).items():
    base_seq = baseline.get("sequences", {}).get(seq, {})
    print(f"\n  seq {seq}:")
    for key, (label, direction, tol) in TRACKED.items():
        if key not in cur_seq or key not in base_seq:
            continue
        b, c = base_seq[key], cur_seq[key]
        delta = c - b
        if direction == "lower_is_better":
            regressed = delta > tol
        else:
            regressed = delta < -tol
        status = "REGRESS" if regressed else "ok"
        if regressed:
            fail = True
        print(f"    {label:<20s}  baseline={b:>10}  now={c:>10}  delta={delta:>+10.4f}  [{status}]")

sys.exit(1 if fail else 0)
PYEOF
    COMPARE_EXIT=$?
    if [[ $COMPARE_EXIT -ne 0 ]]; then
        echo ""
        echo "[FAIL] Baseline regression detected." >&2
        FAIL=1
    fi
fi

# ---- Final status ----------------------------------------------------------
echo ""
if [[ $FAIL -eq 0 ]]; then
    echo "=== Benchmark PASSED ==="
else
    echo "=== Benchmark FAILED ===" >&2
fi
exit $FAIL
