#!/usr/bin/env bash
# Download the ORB-SLAM3 vocabulary (ORB-SLAM2 text format, gzip-compressed).
# The file is ~12 MB compressed / ~65 MB uncompressed.
# Never commit vocab files to git — they are listed in .gitignore.

set -euo pipefail

DEST_DIR="$(cd "$(dirname "$0")/.." && pwd)/thirdparty/vocab"
DEST_FILE="$DEST_DIR/ORBvoc.txt"
TAR_FILE="$DEST_DIR/ORBvoc.txt.tar.gz"
# ORB-SLAM2 repository — vocabulary is identical to ORB-SLAM3
URL="https://github.com/raulmur/ORB_SLAM2/raw/master/Vocabulary/ORBvoc.txt.tar.gz"

mkdir -p "$DEST_DIR"

if [[ -f "$DEST_FILE" ]]; then
    echo "Vocabulary already present: $DEST_FILE"
    exit 0
fi

echo "Downloading ORB vocabulary from $URL ..."
if command -v wget &>/dev/null; then
    wget -q --show-progress -O "$TAR_FILE" "$URL"
elif command -v curl &>/dev/null; then
    curl -L --progress-bar -o "$TAR_FILE" "$URL"
else
    echo "Error: neither wget nor curl is available." >&2
    exit 1
fi

tar xzf "$TAR_FILE" -C "$DEST_DIR" && rm "$TAR_FILE"
echo "Saved to $DEST_FILE"
