#!/usr/bin/env python3
"""
Analyze a QP ladder by measuring full-file SSIM against a source.

Usage:
  python scripts/analyze_qp_ladder.py \
      --source "/path/to/source.mkv" \
      --dir "/path/to/encoded/files"
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


LADDER_RE = re.compile(
    r"^(?P<base>.+) \[(?P<tag>.+?) qp (?P<qp>\d+)\]\.source(?P<ext>\.[^.]+)$"
)
SSIM_RE = re.compile(r"All:(?P<ssim>\d+\.\d+)")


@dataclass
class LadderItem:
    qp: int
    path: Path
    size_bytes: int
    ssim: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure SSIM and per-step size/quality deltas for encoded QP ladders."
    )
    parser.add_argument("--source", required=True, help="Path to original/source video.")
    parser.add_argument(
        "--dir",
        default=".",
        help="Directory containing encoded QP ladder files. Defaults to current directory.",
    )
    parser.add_argument(
        "--csv",
        default="",
        help="Optional CSV output path. If omitted, only terminal output is printed.",
    )
    return parser.parse_args()


def find_ladder_files(source: Path, root: Path) -> list[tuple[int, Path]]:
    source_base = source.stem
    found: list[tuple[int, Path]] = []
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        m = LADDER_RE.match(path.name)
        if not m:
            continue
        if m.group("base") != source_base:
            continue
        found.append((int(m.group("qp")), path))
    return sorted(found, key=lambda x: x[0])


def measure_ssim(source: Path, encoded: Path) -> float:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(source),
        "-i",
        str(encoded),
        "-lavfi",
        "ssim",
        "-f",
        "null",
        "-",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    text = (res.stderr or "") + "\n" + (res.stdout or "")
    matches = SSIM_RE.findall(text)
    if not matches:
        raise RuntimeError(f"Could not parse SSIM for: {encoded.name}")
    return float(matches[-1])


def build_rows(items: list[LadderItem]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    prev: LadderItem | None = None
    for item in items:
        row = {
            "qp": str(item.qp),
            "size_mb": f"{item.size_bytes / (1024 * 1024):.2f}",
            "ssim": f"{item.ssim:.6f}",
            "size_saved_mb_vs_prev": "",
            "size_saved_pct_vs_prev": "",
            "ssim_drop_vs_prev": "",
            "ssim_drop_per_100mb_saved": "",
        }
        if prev is not None:
            size_saved_bytes = prev.size_bytes - item.size_bytes
            size_saved_mb = size_saved_bytes / (1024 * 1024)
            size_saved_pct = (size_saved_bytes / prev.size_bytes * 100.0) if prev.size_bytes > 0 else 0.0
            ssim_drop = prev.ssim - item.ssim
            drop_per_100mb = (ssim_drop / size_saved_mb * 100.0) if size_saved_mb > 0 else 0.0
            row["size_saved_mb_vs_prev"] = f"{size_saved_mb:.2f}"
            row["size_saved_pct_vs_prev"] = f"{size_saved_pct:.2f}"
            row["ssim_drop_vs_prev"] = f"{ssim_drop:.6f}"
            row["ssim_drop_per_100mb_saved"] = f"{drop_per_100mb:.6f}"
        rows.append(row)
        prev = item
    return rows


def print_table(rows: list[dict[str, str]]) -> None:
    headers = [
        "qp",
        "size_mb",
        "ssim",
        "size_saved_mb_vs_prev",
        "size_saved_pct_vs_prev",
        "ssim_drop_vs_prev",
        "ssim_drop_per_100mb_saved",
    ]
    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(row[h]))

    print(" | ".join(h.ljust(widths[h]) for h in headers))
    print("-+-".join("-" * widths[h] for h in headers))
    for row in rows:
        print(" | ".join(row[h].ljust(widths[h]) for h in headers))


def maybe_write_csv(rows: list[dict[str, str]], csv_path: str) -> None:
    if not csv_path:
        return
    headers = list(rows[0].keys()) if rows else []
    out = Path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV written to: {out}")


def main() -> int:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    root = Path(args.dir).expanduser().resolve()

    if not source.is_file():
        print(f"ERROR: source not found: {source}", file=sys.stderr)
        return 1
    if not root.is_dir():
        print(f"ERROR: directory not found: {root}", file=sys.stderr)
        return 1

    ladder_files = find_ladder_files(source, root)
    if not ladder_files:
        print(
            "ERROR: no ladder files found matching source base name.\n"
            f"Expected pattern: '{source.stem} [ ... qp N].source.ext' in {root}",
            file=sys.stderr,
        )
        return 1

    print(f"Source: {source}")
    print(f"Found {len(ladder_files)} ladder files in {root}\n")

    items: list[LadderItem] = []
    for qp, path in ladder_files:
        print(f"Measuring SSIM for QP={qp}: {path.name}")
        ssim = measure_ssim(source, path)
        items.append(
            LadderItem(
                qp=qp,
                path=path,
                size_bytes=path.stat().st_size,
                ssim=ssim,
            )
        )

    rows = build_rows(items)
    print()
    print_table(rows)
    maybe_write_csv(rows, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
