#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

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
    # Use -name with escaped brackets to avoid regex errors on braces/regex chars in $name.
    # Look for a converted file that shares the same base name.
    # Check for an existing converted file with the same base.
    converted=( "$dir/$name [h264_nvenc qp "*"].$ext" )
    already_done=""
    if ((${#converted[@]} > 0)); then
        already_done="${converted[0]}"
    fi

    if [ -n "$already_done" ]; then
        echo "SKIP (already converted): $f"
        continue
    fi

    echo "PROCESS: $f"
    ssim-video-optimizer "$f"
done
