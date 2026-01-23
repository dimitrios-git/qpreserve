#!/usr/bin/env bash
# Batch runner:
# - Streams optimizer output
# - Skips files that already have a tagged conversion `[h264_nvenc XXXpYY qp ZZ]`
# - Uses container logic from the Python code (mp4 inputs → mp4 outputs, otherwise mkv)
# - Continues on individual failures

set -uo pipefail

ROOT="${1:-.}"
video_exts="mp4 mkv mov m4v avi ts mpg mpeg wmv"
found_any=0

find_converted() {
    local dir="$1" name="$2" ext="$3"
    python3 - <<'PY' "$dir" "$name" "$ext"
import os, sys, re
dir_path, src_name, out_ext = sys.argv[1:]
# Updated pattern to match: [h264_nvenc 1080p60 qp 21]
tag_pattern = re.compile(r'\[h264_nvenc \d+p\d+ qp \d+\]')
best = None
for fn in os.listdir(dir_path):
    if not tag_pattern.search(fn):
        continue
    base, ext = os.path.splitext(fn)
    if ext.lower() != f".{out_ext.lower()}":
        continue
    # Check if filename starts with the source name
    if not base.startswith(src_name + " [h264_nvenc "):
        continue
    path = os.path.join(dir_path, fn)
    try:
        size = os.path.getsize(path)
    except OSError:
        continue
    if size < 1048576:
        os.remove(path)
        print(f"remove_tiny::{path}")
        continue
    best = path
    break

if best:
    print(f"found::{best}")
PY
}

# Collect files first to avoid pipeline issues
mapfile -d '' FILES < <(find "$ROOT" -type f -print0)

for f in "${FILES[@]}"; do
    dir="$(dirname "$f")"
    base="$(basename "$f")"
    ext="${base##*.}"
    name="${base%.*}"

    # Quick extension filter
    is_video_ext=1
    for e in $video_exts; do
        if [[ "${ext,,}" == "$e" ]]; then
            is_video_ext=0
            break
        fi
    done
    if [[ $is_video_ext -ne 0 ]]; then
        continue
    fi

    # MIME check
    mime=$(file --mime-type -b "$f")
    if [[ "$mime" != video/* && "$mime" != "application/octet-stream" ]]; then
        continue
    fi
# Updated pattern to match: [h264_nvenc 1080p60 qp 21] or [baseline qp 0]
    if [[ "$name" =~ \[h264_nvenc\ [0-9]+p[0-9]+\ qp\ [0-9]+\] ]] || [[ "$name" =~ \[baseline
    # Skip files that are already tagged outputs (baseline or encoded)
    if [[ "$name" =~ \[(h264_nvenc|baseline)\ qp\ [0-9]+\] ]]; then
        echo "SKIP (tagged output): $f"
        continue
    fi

    # Expected output extension (mirrors encoder.py logic)
    out_ext="mkv"
    if [[ "${ext,,}" == "mp4" ]]; then
        out_ext="mp4"
    fi

    status=$(find_converted "$dir" "$name" "$out_ext")
    if [[ "$status" == found::* ]]; then
        echo "SKIP (already converted): $f"
        continue
    fi

    if [[ "$status" == remove_tiny::* ]]; then
        echo "${status#remove_tiny::}"
    fi

    echo "PROCESS: $f"
    found_any=1
    if ! ssim-video-optimizer "$f"; then
        echo "FAIL (optimizer exited non-zero): $f"
        continue
    fi

    # Locate produced file (tagged output with expected ext)
    status=$(find_converted "$dir" "$name" "$out_ext")
    if [[ "$status" == found::* ]]; then
        echo "DONE -> ${status#found::}"
    else
        echo "WARN: could not locate produced file for $f"
    fi
done

if [[ $found_any -eq 0 ]]; then
    echo "No supported video files found under: $ROOT"
fi
