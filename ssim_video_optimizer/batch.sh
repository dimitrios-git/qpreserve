#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"

find "$ROOT" -type f | while read -r f; do
    # Skip non-videos using MIME type
    if ! file --mime-type "$f" | grep -q 'video/'; then
        continue
    fi

    dir="$(dirname "$f")"
    base="$(basename "$f")"

    ext="${base##*.}"
    name="${base%.*}"

    # Detect converted versions:
    # Original:  Some Video.mp4
    # Converted: Some Video [h264_nvenc qp XX].mp4
    already_done=$(find "$dir" -maxdepth 1 -type f -regex ".*/$name \[.*\]\.$ext" | head -n 1)

    if [ -n "$already_done" ]; then
        echo "SKIP (already converted): $f"
        continue
    fi

    echo "PROCESS: $f"
    ssim-video-optimizer "$f"
done
