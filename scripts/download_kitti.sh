#!/usr/bin/env bash
# Download KITTI Odometry grayscale stereo (~22 GB) and ground-truth poses.
# Usage:  scripts/download_kitti.sh <dest-dir>
#
# We download only what stereo SLAM needs:
#   - data_odometry_gray.zip    (image_0 + image_1, all 22 sequences)
#   - data_odometry_calib.zip   (calib.txt + times.txt per sequence)
#   - data_odometry_poses.zip   (ground truth for sequences 00..10)
#
# After extraction the layout is:
#   <dest>/sequences/00/{image_0,image_1,calib.txt,times.txt}
#   <dest>/poses/00.txt ...

set -euo pipefail

DEST="${1:-./data/kitti}"
mkdir -p "$DEST"
cd "$DEST"

BASE="https://s3.eu-central-1.amazonaws.com/avg-kitti"

fetch() {
    local f="$1"
    if [[ -f "$f" ]]; then
        echo "[skip] $f already present"
    else
        echo "[get ] $f"
        wget --no-verbose --show-progress -c "$BASE/$f"
    fi
}

fetch data_odometry_calib.zip
fetch data_odometry_poses.zip
fetch data_odometry_gray.zip   # large

for z in data_odometry_calib.zip data_odometry_poses.zip data_odometry_gray.zip; do
    echo "[unzip] $z"
    unzip -n -q "$z"
done

echo
echo "Done. Try:"
echo "  export SSLAM_KITTI_SEQ=$PWD/dataset/sequences/00"
echo "  ./build/apps/kitti_stereo \"\$SSLAM_KITTI_SEQ\""
