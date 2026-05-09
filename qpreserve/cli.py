# cli.py
"""
Command-line interface for QPreserve (qpreserve).

Coordinates:
  - baseline generation (QP 0, optional HDR->SDR)
  - sampling (uniform / scene / motion)
  - SSIM-based QP search
  - final encoding + full-file SSIM verification
"""

import argparse
import dataclasses
import json
import sys
import logging
import os
import tempfile
import shutil
import re
from statistics import mean, median
import math
from typing import Any, Dict, List, cast

from .probes import (
    probe_audio_streams,
    probe_video_duration,
    probe_video_framerate,
    probe_video_stream_info,
    probe_video_codec,
    probe_video_bitrate,
    detect_hdr,
)
from .utils import (
    build_audio_options,
    check_output_filename_length,
    run_cmd,
    run_ffmpeg_progress,
    setup_logging,
    has_filter,
    normalize_video_codec,
    nvenc_encoder_for,
    nvenc_pix_fmt_for,
    output_codec_tag,
)
from .sampling import extract_sample_segments
from .ssim_search import measure_ssim_values, _sample_encoded_sizes
from .encoder import encode_final, encode_baseline
from .config import EncodeConfig, config_from_args

_RESIZE_WIDTH_BY_LABEL_WIDESCREEN: Dict[str, int] = {
    "2160p": 3840,
    "1080p": 1920,
    "720p": 1280,
    "576p": 1024,
    "480p": 854,
    "360p": 640,
}

_RESIZE_WIDTH_BY_LABEL_4_3: Dict[str, int] = {
    "2160p": 2880,
    "1080p": 1440,
    "720p": 960,
    "576p": 768,
    "480p": 640,
    "360p": 480,
}

_RESIZE_HEIGHT_BY_LABEL: Dict[str, int] = {
    "2160p": 2160,
    "1080p": 1080,
    "720p": 720,
    "576p": 576,
    "480p": 480,
    "360p": 360,
}

_VIDEO_EXTENSIONS = {
    ".mkv", ".mp4", ".avi", ".mov", ".m4v", ".ts", ".mpg", ".mpeg", ".wmv", ".flv", ".vid",
}

_QP_FOR_TIER: Dict[str, int] = {"ultra": 9, "high": 13, "medium": 21, "low": 25, "lower": 30}
_TIER_ORDER = ["ultra", "high", "medium", "low", "lower"]
_SSIM_FLAT_VELOCITY_FLOOR = 0.00008
_SSIM_FLAT_CUMULATIVE_DROP = 0.0008
_SSIM_FLAT_RANGE = 0.0005
_SSIM_FULL_DISCREPANCY_THRESHOLD = 0.02


class BatchOutputLargerThanSourceError(Exception):
    """Raised by the expected-QP pipeline when the ladder predicts output > source.

    Caught by batch mode to retry the cluster representative at a lower quality tier
    without wasting time on a final encode that is already known to be too large.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.baseline_file: str | None = None
        self.baseline_tmp: str | None = None
        self.segments: list[str] = []
        self.segments_tmpdir: str | None = None
        self.rows: list[dict] = []


def _print_h264_problems(problems: list[str]) -> None:
    print("The source video may not be safely encodable with H.264/NVENC:")
    for problem in problems:
        print("  ", problem)
    print()


def _format_bytes_human(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(max(0, num_bytes))
    idx = 0
    while size >= 1024.0 and idx < len(units) - 1:
        size /= 1024.0
        idx += 1
    if idx == 0:
        return f"{int(size)} {units[idx]}"
    return f"{size:.2f} {units[idx]}"


def _format_duration_hms(seconds: float) -> str:
    total = int(max(0.0, seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _safe_get_file_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _print_source_profile(input_path: str) -> None:
    file_size = _safe_get_file_size(input_path)
    vinfo = probe_video_stream_info(input_path)
    codec = normalize_video_codec(probe_video_codec(input_path) or "unknown")
    fps = probe_video_framerate(input_path)
    duration = probe_video_duration(input_path)
    pix_fmt = str(vinfo.get("pix_fmt", "") or "unknown")
    width = int(vinfo.get("width", 0) or 0)
    height = int(vinfo.get("height", 0) or 0)
    hdr_info = detect_hdr(input_path)
    video_bitrate = probe_video_bitrate(input_path)
    avg_bitrate = int((file_size * 8) / duration) if duration > 0 and file_size > 0 else 0
    bitrate_kbps = (video_bitrate or avg_bitrate) / 1000.0 if (video_bitrate or avg_bitrate) > 0 else 0.0
    pixels_per_sec = float(width) * float(height) * max(fps, 0.0001)
    bppf = ((video_bitrate or avg_bitrate) / pixels_per_sec) if (video_bitrate or avg_bitrate) > 0 else 0.0
    dyn_range = "HDR" if hdr_info.get("is_hdr") else "SDR"

    print("\nSource Profile:")
    print(f"  File: {input_path}")
    print(
        f"  Video: {codec} | {width}x{height} @ {fps:.2f} fps | {pix_fmt} | {dyn_range}"
    )
    print(
        f"  Duration: {_format_duration_hms(duration)} | Size: {_format_bytes_human(file_size)}"
        + (f" | Avg bitrate: ~{bitrate_kbps:.0f} kb/s" if bitrate_kbps > 0 else "")
        + (f" | bppf: {bppf:.4f}" if bppf > 0 else "")
    )


def _resolve_h264_choice(decision: str | None) -> str:
    if decision in ("abort", "continue"):
        return "1" if decision == "abort" else "2"
    if not sys.stdin.isatty():
        logging.warning(
            "Non-interactive shell detected; aborting due to H.264 incompatibilities. "
            "Use --h264-compat continue to override."
        )
        return "1"
    print("Choose an action:")
    print("  [1] Abort (recommended)")
    print("  [2] Continue anyway with H.264 (may fail or produce non-standard output)")
    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice in ('1', '2'):
            return choice

def _same_file(a: str, b: str) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.abspath(a) == os.path.abspath(b)

def _prompt_use_baseline_as_source() -> bool:
    if not sys.stdin.isatty():
        return False
    choice = input(
        "Baseline was required. Use baseline as source for labels/output naming? [y/N]: "
    ).strip().lower()
    return choice in ("y", "yes")

def _confirm_h264_compat(
    width: int,
    height: int,
    pix_fmt: str,
    decision: str | None = None,
) -> bool:
    problems: List[str] = []

    # Conservative H.264/NVENC safe bounds
    if width > 4096 or height > 4096:
        problems.append(f"- Resolution {width}x{height} exceeds 4096 in at least one dimension.")

    # crude but effective: any 10-bit pix_fmt contains "10"
    if '10' in pix_fmt:
        problems.append(
            f"- Pixel format '{pix_fmt}' is 10-bit; H.264/NVENC 8-bit pipeline may be invalid."
        )

    if not problems:
        return True

    _print_h264_problems(problems)
    choice = _resolve_h264_choice(decision)

    if choice == '1':
        print("Aborting due to incompatible source for H.264/NVENC.")
        return False

    print("Continuing with H.264/NVENC despite potential incompatibilities...")
    return True


def _detect_vertical_overscan_crop(width: int, height: int) -> tuple[int, int] | None:
    standards = {
        1280: 720,
        1920: 1080,
        2560: 1440,
        3840: 2160,
        7680: 4320,
    }
    expected = standards.get(width)
    if expected is None:
        return None
    if height <= expected:
        return None
    diff = height - expected
    if diff > 16 or diff % 2 != 0:
        return None
    return expected, diff


def _prompt_vertical_crop(width: int, height: int, expected: int) -> bool:
    if not sys.stdin.isatty():
        return False
    diff = height - expected
    crop_each = diff // 2
    choice = input(
        f"Detected {width}x{height} (expected {width}x{expected}). "
        f"Crop {crop_each}px top/bottom to {width}x{expected}? [y/N]: "
    ).strip().lower()
    return choice in ("y", "yes")


def _resolve_scratch_root(scratch_dir: str | None) -> str:
    scratch_root = scratch_dir or os.environ.get("SSIM_SCRATCH_DIR")
    if scratch_root:
        os.makedirs(scratch_root, exist_ok=True)
    return scratch_root or tempfile.gettempdir()


def _warn_if_scratch_space_low(
    scratch_root: str, input_path: str, skip_baseline: bool
) -> None:
    try:
        input_size = os.path.getsize(input_path)
        if skip_baseline:
            # No baseline: only sample _enc files + final encode ≈ 1.5× input.
            needed = int(input_size * 1.5)
        else:
            # Baseline at QP=6 is typically 5–10× larger than the source
            # (e.g. a 1 GB H.264 source produces a ~10 GB HEVC baseline).
            # Use 10× as a conservative upper bound so the warning fires early.
            needed = int(input_size * 10)
        free = shutil.disk_usage(scratch_root).free
        if free < needed:
            print(
                f"WARNING: scratch dir '{scratch_root}' has {free/1e9:.1f} GB free; "
                f"estimated need ~{needed/1e9:.1f} GB. "
                "Use --scratch-dir (or SSIM_SCRATCH_DIR env var) to point to a "
                "disk-backed location with sufficient space."
            )
    except OSError:
        pass


def _determine_baseline_file(
    input_path: str,
    skip_baseline: bool,
    baseline_tmp: str,
    baseline_qp: int,
    extra_vf: str | None = None,
    video_codec: str = "h264",
    skip_baseline_requested: bool = False,
    allow_skip_with_filters: bool = False,
) -> str:
    if skip_baseline:
        if extra_vf:
            if allow_skip_with_filters:
                print("Applying video filters in final encode path; skipping baseline as requested.")
            else:
                print("Video filters require baseline generation; disabling --skip-baseline.")
                skip_baseline = False
        if skip_baseline:
            src_codec = normalize_video_codec(probe_video_codec(input_path))
            target_codec = normalize_video_codec(video_codec)
            if skip_baseline_requested:
                if src_codec != "h265":
                    print(
                        "WARNING: --skip-baseline was explicitly requested on a non-hevc source "
                        f"('{src_codec}'). Comparing source {src_codec} directly against "
                        f"{target_codec} encodes may impair quality comparison accuracy."
                    )
                else:
                    print("Skipping baseline generation; source codec is h265/hevc.")
                return input_path
            if src_codec == "h265":
                print("Skipping baseline generation; source codec is h265/hevc.")
            else:
                print(f"Skipping baseline generation for {src_codec} source.")
            return input_path

    baseline_file = encode_baseline(
        input_path,
        output_dir=baseline_tmp,
        qp=baseline_qp,
        extra_vf=extra_vf,
        video_codec=video_codec,
    )
    print(f"Baseline file created: {baseline_file}")
    logging.info("Baseline file created: %s", baseline_file)
    return baseline_file


def _maybe_enable_default_skip_baseline(
    config: EncodeConfig,
    input_path: str,
    extra_vf: str | None,
) -> None:
    if config.skip_baseline:
        return
    if extra_vf is not None:
        return
    if config.use_baseline_as_source:
        return
    src_codec = normalize_video_codec(probe_video_codec(input_path))
    if src_codec == "h265":
        config.skip_baseline = True
        print("Skipping baseline generation by default for h265/hevc source.")
        return
    if src_codec == "h264":
        hdr_info = detect_hdr(input_path)
        stream_info = probe_video_stream_info(input_path)
        sar = stream_info.get("sample_aspect_ratio", "") or ""
        pix_fmt = stream_info.get("pix_fmt", "") or ""
        is_sdr = not hdr_info.get("is_hdr", False)
        is_yuv420p = pix_fmt.startswith("yuv420p")
        is_square_sar = sar in ("", "0:1", "1:1")
        if is_sdr and is_yuv420p and is_square_sar:
            config.skip_baseline = True
            print("Skipping baseline generation by default for h264 SDR yuv420p source.")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Optimize video quality via perceptual metric and QP binary search.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('input', help='Source video file or directory.')
    parser.add_argument(
        'output',
        nargs='?',
        default=None,
        help='Output file (single-file mode) or output directory (batch mode). '
             'If omitted in single-file mode, the output is placed alongside the source '
             'with an auto-generated quality suffix. '
             'Batch mode (directory input) requires this argument.'
    )
    parser.add_argument(
        '--video-codec',
        choices=['h264', 'h265', 'hevc'],
        default='h265',
        help='Target video codec used for baseline, samples, and final output.'
    )

    # Quality targets
    parser.add_argument('--ssim', type=float, default=0.986,
                        help='Target SSIM threshold for acceptance.')

    # QP search bounds
    parser.add_argument('--min-qp', type=int, default=6,
                        help='Minimum QP allowed during search.')
    parser.add_argument('--max-qp', type=int, default=40,
                        help='Maximum QP allowed during search.')

    # Sampling configuration
    parser.add_argument(
        '--sample-percent',
        type=str,
        default='auto',
        help="Percentage of video duration used for sampling. Use a number or 'auto'."
    )
    parser.add_argument(
        '--sample-count',
        type=str,
        default='auto',
        help="How many sample clips to extract. Use an integer or 'auto'."
    )
    parser.add_argument('--sample-qp', type=int, default=6,
                        help=argparse.SUPPRESS)
    parser.add_argument(
        '--initial-qp',
        type=int,
        help='Skip sampling and start full-file SSIM descent from this QP.'
    )
    parser.add_argument(
        '--source-quality',
        type=str,
        help='Starting point for expected-QP mode. Accepts a numeric QP start (e.g. 22) '
             'or one of: ultra, high, medium, low, lower. If omitted, interactive runs prompt for it '
             '(non-interactive defaults to medium). '
             'Guide: ultra=near-lossless, high=clean, medium=compressed, low=heavily compressed, lower=very heavily compressed.'
    )
    parser.add_argument(
        '--expected-min-gain',
        type=float,
        default=1.0,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        '--expected-max-steps',
        type=int,
        default=8,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        '--expected-knee-ratio',
        type=float,
        default=1.5,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        '--expected-choice',
        choices=['prompt', 'safe', 'balanced'],
        default='prompt',
        help='Expected-QP mode selection behavior.'
    )
    parser.add_argument(
        '--re-encode-same-codec-video',
        action='store_true',
        dest='re_encode_same_codec_video',
        help='Include sources that already use the target codec. By default, batch mode '
             'excludes files whose source codec matches --video-codec.'
    )
    parser.add_argument(
        '--batch-bppf-tolerance',
        type=float,
        default=0.15,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        '--batch-bitrate-tolerance',
        type=float,
        default=0.20,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        '--batch-dry-run',
        action='store_true',
        help='Batch mode: print clustering and planned actions without encoding.'
    )
    parser.add_argument(
        '--batch-size-guard',
        action='store_true',
        help='Batch mode: if the estimated output at the selected quality tier is larger '
             'than the source, automatically retry with the next lower tier.'
    )
    parser.add_argument(
        '--disable-clustering',
        action='store_true',
        help='Batch mode: skip clustering and run the full quality search independently '
             'on every file. Produces optimal per-file results at the cost of speed.'
    )

    parser.add_argument(
        '--sampling-mode',
        choices=['uniform', 'scene', 'motion'],
        default='motion',
        help='Sampling strategy for selecting sample start times.'
    )

    # SSIM aggregation metric
    parser.add_argument(
        '--metric',
        choices=['avg', 'min', 'max'],
        default='avg',
        help='Which SSIM aggregation metric to use across samples.'
    )

    # Logging
    parser.add_argument('--log-file', help='Optional log file path.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose logging.')
    parser.add_argument(
        '--scratch-dir',
        help='Directory to store temporary baseline/final files (defaults to system temp). '
             'Use a disk-backed path to avoid filling a tmpfs /tmp.'
    )
    parser.add_argument(
        '--no-audio-normalize',
        action='store_false',
        dest='audio_normalize',
        help='Disable audio loudness normalization (loudnorm).'
    )
    parser.add_argument(
        '--add-stereo-downmix',
        action='store_true',
        help='For each multichannel audio stream, add a stereo AAC downmix alongside the original.'
    )
    parser.add_argument(
        '--add-stereo-downmix-copy-video',
        action='store_true',
        help='Copy the video stream and only process audio (normalization/downmix). Implies --add-stereo-downmix.'
    )
    parser.add_argument(
        '--full-ssim-chunk-seconds',
        type=float,
        default=600.0,
        help='Split full-file SSIM into chunks to reduce RAM use (0 disables).'
    )
    parser.add_argument(
        '--target-fps',
        type=float,
        default=None,
        help='Target output framerate for sample/final encodes. Allowed: 24, 30, 60, 120 '
             '(24 encodes at 23.976). Default uses source framerate.'
    )
    parser.add_argument(
        '--skip-baseline',
        action='store_true',
        help='Skip baseline generation and use source directly. This is enabled automatically '
             'for h265/hevc sources (unless crop/resize or baseline-as-source requires baseline).'
    )
    parser.add_argument(
        '--use-baseline-as-source',
        action='store_true',
        help='Treat the generated baseline as the source reference for labels/output naming.'
    )

    parser.add_argument(
        '--h264-compat',
        choices=['abort', 'continue'],
        default=None,
        help='Non-interactive choice when the source may be incompatible with H.264/NVENC (non-TTY defaults to abort).'
    )

    parser.add_argument(
        '--baseline-qp',
        type=int,
        default=6,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        '--resize-resolution',
        choices=list(_RESIZE_WIDTH_BY_LABEL_WIDESCREEN.keys()),
        default=None,
        help='Resize video while preserving aspect ratio using standardized labels mapped to '
             'horizontal widths. Widescreen: 2160p=3840, 1080p=1920, 720p=1280, 576p=1024, '
             '480p=854, 360p=640. For common 4:3 sources: 2160p=2880, 1080p=1440, 720p=960, '
             '576p=768, 480p=640, 360p=480. Aspect class is chosen from display aspect ratio '
             '(DAR), including anamorphic sources.'
    )
    parser.add_argument(
        '--display-ar',
        choices=['auto', 'full', 'wide', '4:3', '16:9'],
        default='auto',
        help='Display aspect class used for resize targets. auto derives from DAR/SAR; '
             'set full (4:3) or wide (16:9) to force exact target geometry '
             '(e.g. 480p -> 640x480 or 854x480).'
    )
    parser.add_argument(
        '--auto-crop',
        choices=['off', 'prompt', 'force'],
        default='prompt',
        help='Detect small vertical overscan (e.g., 1920x1088) and crop to a standard height. '
             '"prompt" asks when a TTY is available; "force" applies automatically.'
    )

    return parser


def _parse_ratio_text(value: str) -> float | None:
    text = str(value or "").strip()
    if not text or text in {"N/A", "0:1", "0/1"}:
        return None
    for sep in (":", "/"):
        if sep in text:
            left, right = text.split(sep, 1)
            try:
                num = float(left)
                den = float(right)
            except ValueError:
                return None
            if den == 0:
                return None
            ratio = num / den
            return ratio if ratio > 0 else None
    try:
        ratio = float(text)
    except ValueError:
        return None
    return ratio if ratio > 0 else None


def _compute_display_aspect_ratio(
    width: int,
    height: int,
    display_aspect_ratio: str,
    sample_aspect_ratio: str,
) -> float:
    dar = _parse_ratio_text(display_aspect_ratio)
    if dar is not None:
        return dar
    sar = _parse_ratio_text(sample_aspect_ratio)
    if sar is not None and width > 0 and height > 0:
        return (float(width) * sar) / float(height)
    if width > 0 and height > 0:
        return float(width) / float(height)
    return 1.0


def _probe_video_basics(input_path: str) -> tuple[int, int, str, float, str, str]:
    vinfo: Dict[str, Any] = probe_video_stream_info(input_path)
    width: int = int(vinfo['width'])
    height: int = int(vinfo['height'])
    pix_fmt: str = str(vinfo['pix_fmt'])
    sar_text = str(vinfo.get("sample_aspect_ratio", "") or "")
    dar_text = str(vinfo.get("display_aspect_ratio", "") or "")
    display_ar = _compute_display_aspect_ratio(width, height, dar_text, sar_text)
    return width, height, pix_fmt, display_ar, sar_text, dar_text


def _prepare_audio_and_framerate(
    input_path: str, audio_normalize: bool, add_stereo_downmix: bool
) -> tuple[List[str], float, bool]:
    streams: List[Dict[str, Any]] = probe_audio_streams(input_path)
    normalize_enabled = audio_normalize
    if normalize_enabled and not has_filter('loudnorm'):
        print("WARNING: loudnorm filter not available; disabling audio normalization.")
        normalize_enabled = False
    audio_opts: List[str] = build_audio_options(
        streams,
        normalize=normalize_enabled,
        add_stereo_downmix=add_stereo_downmix,
    )
    raw_fr: float = probe_video_framerate(input_path)
    return audio_opts, raw_fr, normalize_enabled


def _resolve_target_framerate(source_fr: float, target_fps: float | None) -> float:
    if target_fps is None:
        print(f"Target framerate: source ({source_fr:.2f} fps)")
        return source_fr
    allowed = [24.0, 30.0, 60.0, 120.0]
    selected = None
    for val in allowed:
        if math.isclose(float(target_fps), val, rel_tol=0.0, abs_tol=1e-6):
            selected = val
            break
    if selected is None:
        raise ValueError("--target-fps must be one of: 24, 30, 60, 120.")

    # User-friendly option: 24 maps to true film cadence.
    normalized = 24000.0 / 1001.0 if selected == 24.0 else selected
    if selected == 24.0:
        print(
            f"Target framerate: 24 -> encoding at {normalized:.3f} fps "
            f"(source {source_fr:.2f} fps)"
        )
    else:
        print(f"Target framerate: {normalized:.2f} fps (source {source_fr:.2f} fps)")

    if normalized > source_fr + 1e-6:
        print(
            "WARNING: target fps is higher than source fps. Extra frames will be duplicates, "
            "which can inflate filesize without improving motion quality."
        )
        if not sys.stdin.isatty():
            raise ValueError(
                "Cannot confirm --target-fps > source fps in non-interactive mode. "
                "Use a value <= source fps."
            )
        confirm = input("Continue with higher target fps anyway? [y/N]: ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Cancelled by user.")
            raise RuntimeError("cancelled_target_fps")

    return float(normalized)


def _resolve_output_dest(
    encoded_path: str,
    source_path: str,
    explicit_output: str | None,
    batch_input_dir: str | None = None,
) -> str:
    if explicit_output is None:
        # Auto-name alongside source; encoded_path already carries the quality suffix.
        return os.path.join(os.path.dirname(source_path), os.path.basename(encoded_path))

    if os.path.isdir(explicit_output):
        # Directory destination (batch mode or user passed an existing dir).
        # Reconstruct relative subdirectory structure when inside a batch run.
        if batch_input_dir:
            rel_dir = os.path.relpath(os.path.dirname(source_path), batch_input_dir)
            out_subdir = os.path.normpath(os.path.join(explicit_output, rel_dir))
        else:
            out_subdir = explicit_output
        os.makedirs(out_subdir, exist_ok=True)
        _, enc_ext = os.path.splitext(encoded_path)
        src_base = os.path.splitext(os.path.basename(source_path))[0]
        return os.path.join(out_subdir, f"{src_base}{enc_ext}")

    # Explicit file path: use exactly as given.
    parent = os.path.dirname(explicit_output)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return explicit_output


def _run_audio_only_copy_video(
    input_path: str,
    audio_opts: List[str],
    output_dir: str,
) -> str:
    base, ext = os.path.splitext(os.path.basename(input_path))
    if not ext:
        ext = ".mkv"
    output_name = f"{base}.stereo-downmix-added{ext}"
    output_path = os.path.join(output_dir, output_name)

    cmd = [
        'ffmpeg', '-y',
        '-fflags', '+discardcorrupt',
        '-i', input_path,
        '-map', '0:v?', '-map', '0:a?', '-map', '0:s?', '-map_metadata', '0',
    ] + audio_opts + ['-c:v', 'copy', '-c:s', 'copy', output_path]

    duration = probe_video_duration(input_path)
    if duration > 0:
        run_ffmpeg_progress(cmd, duration, desc="Audio-only processing")
    else:
        run_cmd(cmd)

    return output_path


def _maybe_run_audio_only_path(config: EncodeConfig) -> bool:
    if not config.add_stereo_downmix_copy_video:
        return False
    predicted = _predict_output_path(config)
    if predicted and os.path.exists(predicted):
        print(f"Skipping (output exists): {predicted}")
        logging.info("Skipping (output exists): %s", predicted)
        return True
    config.add_stereo_downmix = True
    normalize_enabled = config.audio_normalize
    if normalize_enabled and not has_filter('loudnorm'):
        print("WARNING: loudnorm filter not available; disabling audio normalization.")
        normalize_enabled = False

    streams = probe_audio_streams(config.input)
    audio_opts = build_audio_options(
        streams,
        normalize=normalize_enabled,
        add_stereo_downmix=True,
    )

    scratch_root = _resolve_scratch_root(config.scratch_dir)
    tmpdir = tempfile.mkdtemp(prefix="audio_copy_", dir=scratch_root)
    print(f"Scratch directory: {scratch_root}")
    logging.info("Scratch directory: %s", scratch_root)
    try:
        output_path = _run_audio_only_copy_video(
            input_path=config.input,
            audio_opts=audio_opts,
            output_dir=tmpdir,
        )

        dest = _resolve_output_dest(
            encoded_path=output_path,
            source_path=config.input,
            explicit_output=config.output,
            batch_input_dir=config.batch_input_dir,
        )
        shutil.move(output_path, dest)
        print(f"Output file created: {dest}")
        logging.info("Output file created: %s", dest)
        return True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _calculate_gop(raw_fr: float) -> int:
    return max(1, int(round(raw_fr / 2)))


def _prompt_initial_qp_rerun(
    sample_metric: float,
    full_metric: float,
    final_qp: int,
    min_qp: int,
    max_qp: int,
    threshold: float,
) -> int | None:
    if not sys.stdin.isatty():
        return None
    diff = abs(full_metric - sample_metric)
    if diff < threshold:
        return None
    print(
        "\nNotice: large SSIM gap between samples and full-file result."
        f"\n  Sample SSIM (QP={final_qp}): {sample_metric:.4f}"
        f"\n  Full-file SSIM (QP={final_qp}): {full_metric:.4f}"
        f"\n  Δ={diff:.4f} (threshold {threshold:.4f})"
    )
    print(
        "You can accept the current result, or rerun full-file descent starting at a new QP."
    )
    while True:
        choice = input(
            f"Enter starting QP [{min_qp}-{max_qp}] to rerun, or press Enter to keep current: "
        ).strip().lower()
        if choice == "":
            return None
        try:
            qp_choice = int(choice)
        except ValueError:
            print("Please enter a valid QP number.")
            continue
        if qp_choice < min_qp:
            print(f"QP must be >= {min_qp}.")
            continue
        return qp_choice


def _suggest_qp_from_ssim(ssim_by_qp: Dict[int, tuple[float, str]]) -> tuple[int, float]:
    avg_ssim = mean(val for val, _ in ssim_by_qp.values())
    best_qp = min(
        ssim_by_qp.items(),
        key=lambda item: (abs(item[1][0] - avg_ssim), -item[0]),
    )[0]
    return best_qp, avg_ssim


def _prompt_qp_choice(
    ssim_by_qp: Dict[int, tuple[float, str]],
    suggested_qp: int,
    avg_ssim: float,
    target_ssim: float,
) -> int:
    if not sys.stdin.isatty():
        return suggested_qp

    print(
        f"\nTarget SSIM {target_ssim:.4f} was not reached. "
        "Full-file SSIM results by QP:"
    )
    for qp in sorted(ssim_by_qp.keys(), reverse=True):
        ssim_val = ssim_by_qp[qp][0]
        tag = " (suggested)" if qp == suggested_qp else ""
        print(f"  QP={qp}: SSIM={ssim_val:.4f}{tag}")

    print(f"Suggested achievable target ~{avg_ssim:.4f} (QP={suggested_qp}).")
    while True:
        choice = input("Pick a QP to use (Enter for suggested): ").strip().lower()
        if choice == "":
            return suggested_qp
        try:
            qp_choice = int(choice)
        except ValueError:
            print("Please enter a valid QP number.")
            continue
        if qp_choice in ssim_by_qp:
            return qp_choice
        print("That QP wasn't measured; choose one from the list above.")


def _resolve_source_ref_path(
    config: EncodeConfig,
    baseline_file: str,
) -> str:
    skip_baseline_forced_off = config.skip_baseline and not _same_file(baseline_file, config.input)
    if skip_baseline_forced_off and not config.use_baseline_as_source:
        if config.batch_mode:
            print("Tip: use --use-baseline-as-source to treat the baseline as the source.")
        elif _prompt_use_baseline_as_source():
            config.use_baseline_as_source = True
        else:
            print("Tip: use --use-baseline-as-source to treat the baseline as the source.")
    return baseline_file if config.use_baseline_as_source else config.input


def _prepare_work_dirs(
    scratch_root: str,
    input_path: str,
    skip_baseline: bool,
) -> tuple[str, str, str | None]:
    _warn_if_scratch_space_low(scratch_root, input_path, skip_baseline)
    baseline_tmp = tempfile.mkdtemp(prefix="ssim_baseline_", dir=scratch_root)
    tmpdir = tempfile.mkdtemp(prefix="ssim_final_", dir=scratch_root)
    print(f"Scratch directory: {scratch_root}")
    logging.info("Scratch directory: %s", scratch_root)
    return baseline_tmp, tmpdir, None


def _estimate_full_size_from_samples(sample_bytes: int, percent: float) -> int:
    if percent <= 0:
        return sample_bytes
    return int(sample_bytes * (100.0 / percent))



def _prepare_size_segments(
    baseline_file: str,
    config: EncodeConfig,
    scratch_root: str,
) -> tuple[list[str], str | None]:
    # For H.264 sources, sample from the original to avoid extracting from the
    # (often much larger) QP=6 HEVC baseline. Cross-codec SSIM between H.264
    # and HEVC is reliable because both codecs share similar block-DCT artifact
    # profiles; quality degrades monotonically with QP as expected.
    #
    # For legacy codecs (MPEG-1, MPEG-2, VC-1, etc.), cross-codec SSIM breaks
    # down: heavy HEVC quantization produces blocking artifacts that SSIM scores
    # as MORE similar to the already-blocky source, causing SSIM to increase
    # with QP and the algorithm to select excessively high QP values. Use the
    # baseline (intra-HEVC comparison) so SSIM degrades properly with QP.
    #
    # Exception: use_baseline_as_source forces baseline regardless of codec
    # (e.g. HDR→SDR tonemapped or scaled sources).
    src_codec = normalize_video_codec(probe_video_codec(config.input))
    use_source = not config.use_baseline_as_source and src_codec == "h264"
    sample_source = config.input if use_source else baseline_file
    segments, segments_tmpdir, _, _ = extract_sample_segments(
        sample_source,
        percent=config.sample_percent,
        count=config.sample_count,
        sampling_mode=config.sampling_mode,
        tmp_root=scratch_root,
    )
    if not segments_tmpdir:
        segments_tmpdir = None
    return segments, segments_tmpdir


def _auto_sample_count_for_duration(duration_sec: float) -> int:
    if duration_sec <= 0:
        return 8
    minutes = duration_sec / 60.0
    # Favor timeline coverage over long clips: about one sample per ~4 minutes.
    return max(4, min(12, int(math.ceil(minutes / 4.0))))


def _auto_sample_percent_for_duration(duration_sec: float) -> float:
    if duration_sec <= 0:
        return 20.0
    minutes = duration_sec / 60.0
    if minutes < 20.0:
        return 40.0
    if minutes < 45.0:
        return 30.0
    if minutes <= 90.0:
        return 20.0
    return 15.0


def _resolve_sample_percent(config: EncodeConfig, input_path: str) -> None:
    raw_value = str(config.sample_percent).strip().lower()
    if raw_value == "auto":
        duration_sec = probe_video_duration(input_path)
        auto_percent = _auto_sample_percent_for_duration(duration_sec)
        config.sample_percent = auto_percent
        print(
            f"Sampling config: --sample-percent auto -> {auto_percent:.1f}% "
            f"(duration {_format_duration_hms(duration_sec)})."
        )
        return
    try:
        percent = float(raw_value)
    except ValueError:
        raise ValueError("--sample-percent must be a number in (0, 100] or 'auto'.")
    if percent <= 0.0 or percent > 100.0:
        raise ValueError("--sample-percent must be in (0, 100].")
    config.sample_percent = percent


def _resolve_sample_count(config: EncodeConfig, input_path: str) -> None:
    raw_value = str(config.sample_count).strip().lower()
    if raw_value == "auto":
        duration_sec = probe_video_duration(input_path)
        auto_count = _auto_sample_count_for_duration(duration_sec)
        config.sample_count = auto_count
        print(
            f"Sampling config: --sample-count auto -> {auto_count} "
            f"(duration {_format_duration_hms(duration_sec)})."
        )
        return
    try:
        count = int(raw_value)
    except ValueError:
        raise ValueError("--sample-count must be an integer >= 1 or 'auto'.")
    if count < 1:
        raise ValueError("--sample-count must be >= 1.")
    config.sample_count = count


def _safe_file_size(path: str) -> int | None:
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def _resolve_crop_filter(config: EncodeConfig, width: int, height: int) -> str | None:
    crop_candidate = _detect_vertical_overscan_crop(width, height)
    if not crop_candidate:
        return None
    expected = crop_candidate[0]
    apply_crop = False
    if config.auto_crop == "force":
        apply_crop = True
    elif config.auto_crop == "prompt":
        if sys.stdin.isatty():
            apply_crop = _prompt_vertical_crop(width, height, expected)
        else:
            print(
                f"Auto-crop candidate detected ({width}x{height} → {width}x{expected}). "
                "Use --auto-crop force to apply in non-interactive runs."
            )
    if not apply_crop:
        return None
    crop_filter = f"crop=iw:{expected}:0:(ih-{expected})/2"
    print(f"Auto-crop enabled: {width}x{height} → {width}x{expected} (centered).")
    if config.skip_baseline:
        print("Auto-crop requires baseline generation; disabling --skip-baseline.")
        config.skip_baseline = False
    return crop_filter


def _round_to_even(value: float) -> int:
    v = int(round(value))
    if v < 2:
        return 2
    if v % 2 != 0:
        v += 1
    return v


def _aspect_class_from_dar(display_ar: float) -> str:
    if abs(display_ar - (4.0 / 3.0)) <= 0.03:
        return "4:3"
    if abs(display_ar - (16.0 / 9.0)) <= 0.03:
        return "16:9"
    return "custom"


def _is_square_sar(sample_aspect_ratio: str) -> bool:
    sar = _parse_ratio_text(sample_aspect_ratio)
    return sar is None or abs(sar - 1.0) <= 0.01


def _resolve_resize_filter(
    config: EncodeConfig,
    width: int,
    height: int,
    display_ar: float,
    sample_aspect_ratio: str,
) -> str | None:
    if not config.resize_resolution:
        return None
    label = str(config.resize_resolution).lower()
    aspect_mode_raw = str(config.display_ar or "auto").lower()
    aspect_alias = {
        "4:3": "full",
        "16:9": "wide",
    }
    aspect_mode = aspect_alias.get(aspect_mode_raw, aspect_mode_raw)
    if aspect_mode == "full":
        aspect_class = "4:3"
    elif aspect_mode == "wide":
        aspect_class = "16:9"
    else:
        aspect_class = _aspect_class_from_dar(display_ar)
    target_height = _RESIZE_HEIGHT_BY_LABEL[label]
    if aspect_class == "4:3":
        target_width = _RESIZE_WIDTH_BY_LABEL_4_3[label]
        profile = "4:3"
    elif aspect_class == "16:9":
        target_width = _RESIZE_WIDTH_BY_LABEL_WIDESCREEN[label]
        profile = "widescreen"
    else:
        target_width = _round_to_even(float(target_height) * display_ar)
        profile = f"custom DAR {display_ar:.3f}"

    source_display_width = float(height) * display_ar
    source_display_height = float(height)
    if source_display_width > target_width + 0.5:
        print(
            "WARNING: source display resolution is bigger than requested "
            f"(~{source_display_width:.0f}x{source_display_height:.0f} -> width {target_width}, "
            f"{profile}); resolution will be decreased, "
            "some data might be lost."
        )
    elif source_display_width < target_width - 0.5:
        print(
            "WARNING: source display resolution is smaller than requested "
            f"(~{source_display_width:.0f}x{source_display_height:.0f} -> width {target_width}, "
            f"{profile}); resolution will be increased, "
            "data will be upscaled."
        )
    else:
        print(
            f"WARNING: source already matches {label} by display geometry "
            f"({target_width}px width, {profile}), "
            "ignoring resize flag."
        )
        return None
    anamorphic = not _is_square_sar(sample_aspect_ratio)
    if anamorphic:
        print(
            f"Anamorphic source detected (SAR={sample_aspect_ratio}); "
            "output will be normalized to square pixels (SAR=1)."
        )
    if aspect_mode in ("full", "wide"):
        print(
            "Display aspect override: forcing "
            f"{aspect_mode} ({'4:3' if aspect_mode == 'full' else '16:9'}) target geometry."
        )
    print(
        f"Resize enabled: target {target_width}x{target_height} "
        f"({label} label, {profile}, DAR~{display_ar:.3f})."
    )
    if anamorphic:
        return (
            "scale=trunc(iw*sar/2)*2:ih,setsar=1,"
            f"scale={target_width}:{target_height}:flags=lanczos,setsar=1"
        )
    return f"scale={target_width}:{target_height}:flags=lanczos,setsar=1"


def _merge_video_filters(*filters: str | None) -> str | None:
    parts = [f for f in filters if f]
    if not parts:
        return None
    return ",".join(parts)


def _resolve_quality_mode(
    config: EncodeConfig,
    input_path: str,
) -> str | None:
    codec = probe_video_codec(input_path)
    if config.batch_force_fixed_qp_mode:
        print("Quality metric in use: fixed QP (batch peer mode)")
        return codec
    if not has_filter('ssim'):
        print("ERROR: ssim filter not available in ffmpeg. Cannot measure quality.")
        return None
    print("Quality metric in use: SSIM (expected-QP mode)")
    return codec


def _tier_for_bppf(codec: str, bppf: float) -> str:
    c = normalize_video_codec(codec)
    if c == "h265":
        if bppf >= 0.120:
            return "ultra"
        if bppf >= 0.060:
            return "high"
        if bppf >= 0.024:
            return "medium"
        if bppf >= 0.012:
            return "low"
        return "lower"
    # h264 and other codecs: generally need more bppf for similar perceptual quality.
    if bppf >= 0.200:
        return "ultra"
    if bppf >= 0.120:
        return "high"
    if bppf >= 0.060:
        return "medium"
    if bppf >= 0.030:
        return "low"
    return "lower"


def _next_lower_tier(tier: str) -> str | None:
    """Return the next lower quality tier, or None if already at the lowest."""
    try:
        idx = _TIER_ORDER.index(tier.lower())
    except ValueError:
        return None
    return _TIER_ORDER[idx + 1] if idx + 1 < len(_TIER_ORDER) else None


def _selected_quality_label(config: EncodeConfig) -> str | None:
    if not config.source_quality:
        return None
    text = str(config.source_quality).strip().lower()
    if _quality_label_to_qp(text) is not None:
        return text.replace("_", "-").replace(" ", "-")
    return None


def _compute_source_advice(
    input_path: str,
    width: int,
    height: int,
    raw_fr: float,
) -> tuple[str, str, float, float]:
    """
    Returns (codec, advised_tier, bitrate_kbps, bppf)
    """
    duration = probe_video_duration(input_path)
    try:
        size_bytes = os.path.getsize(input_path)
    except OSError:
        size_bytes = 0
    src_codec = normalize_video_codec(probe_video_codec(input_path))
    stream_br = probe_video_bitrate(input_path)
    avg_br = int((size_bytes * 8) / duration) if duration > 0 and size_bytes > 0 else 0
    bitrate_bps = stream_br if stream_br and stream_br > 0 else avg_br
    pixels_per_sec = max(1.0, float(width) * float(height) * max(raw_fr, 0.0001))
    bppf = float(bitrate_bps) / pixels_per_sec
    advised = _tier_for_bppf(src_codec, bppf)
    return src_codec, advised, bitrate_bps / 1000.0 if bitrate_bps > 0 else 0.0, bppf


def _maybe_print_expected_mode_advice(
    config: EncodeConfig,
    input_path: str,
    width: int,
    height: int,
    raw_fr: float,
) -> None:
    if not _expected_mode_requested(config):
        return
    if raw_fr <= 0:
        return
    src_codec, advised, bitrate_kbps, bppf = _compute_source_advice(
        input_path=input_path,
        width=width,
        height=height,
        raw_fr=raw_fr,
    )
    selected = _selected_quality_label(config)

    print(
        "Source profile: "
        f"{src_codec} {width}x{height}@{raw_fr:.2f}, "
        f"avg bitrate ~{bitrate_kbps:.0f} kb/s, bppf={bppf:.4f}"
    )
    print(f"Heuristic suggested source-quality tier: {advised}.")
    if selected and selected != advised:
        tier_rank = {"lower": 0, "low": 1, "medium": 2, "high": 3, "ultra": 4}
        sel_rank = tier_rank.get(selected, 0)
        adv_rank = tier_rank.get(advised, 0)
        if sel_rank < adv_rank:
            guidance = "Quality may degrade. Rerun with a higher tier if needed."
        elif sel_rank > adv_rank:
            guidance = "Output may inflate. Rerun with a lower tier if needed."
        else:
            guidance = "Selected tier differs from heuristic recommendation."
        print(
            f"Note: you selected '{selected}'. This source looks closer to '{advised}'. "
            f"{guidance}"
        )


def _expected_mode_requested(config: EncodeConfig) -> bool:
    return bool(config.source_quality)


def _default_choice_for_tier(tier: str) -> str:
    return {
        "ultra": "1",
        "high": "2",
        "medium": "3",
        "low": "4",
        "lower": "5",
    }.get(tier, "3")


def _print_source_quality_menu(default_choice: str) -> None:
    rec_label = {
        "1": "ultra",
        "2": "high",
        "3": "medium",
        "4": "low",
        "5": "lower",
    }.get(default_choice, "medium")
    print("Select source quality tier for Expected-QP mode:")
    print(f"  [1] ultra{' (recommended)' if rec_label == 'ultra' else ''}")
    print(f"  [2] high{' (recommended)' if rec_label == 'high' else ''}")
    print(f"  [3] medium{' (recommended)' if rec_label == 'medium' else ''}")
    print(f"  [4] low{' (recommended)' if rec_label == 'low' else ''}")
    print(f"  [5] lower{' (recommended)' if rec_label == 'lower' else ''}")
    print("  [6] Enter numeric QP")
    print("  [q] Quit")
    print("Note: recommendation is estimated from bitrate per pixel per frame.")
    print("It is usually accurate, but can be misleading for low-quality sources with inflated bitrate.")


def _resolve_source_quality_choice(default_choice: str) -> str | None:
    tier_by_choice = {
        "1": "ultra",
        "2": "high",
        "3": "medium",
        "4": "low",
        "5": "lower",
    }

    def _prompt_numeric_qp() -> str | None:
        raw = input("Enter expected starting QP: ").strip()
        try:
            int(raw)
        except ValueError:
            print("Please enter a valid integer QP.")
            return None
        return raw

    while True:
        choice = input(f"Choose 1-6 or q [{default_choice}]: ").strip().lower()
        if choice == "":
            choice = default_choice
        if choice in {"q", "quit"}:
            print("Cancelled by user.")
            return None
        tier = tier_by_choice.get(choice)
        if tier:
            return tier
        if choice == "6":
            raw = _prompt_numeric_qp()
            if raw is None:
                continue
            return raw
        print("Please choose 1, 2, 3, 4, 5, 6, or q.")


def _ensure_source_quality_for_default_pipeline(config: EncodeConfig) -> bool:
    if _expected_mode_requested(config):
        return True
    if config.batch_mode:
        return True
    if not sys.stdin.isatty():
        config.source_quality = "medium"
        print("No --source-quality provided; defaulting to 'medium' in non-interactive mode.")
        return True

    vinfo = probe_video_stream_info(config.input)
    width = int(vinfo.get("width", 0) or 0)
    height = int(vinfo.get("height", 0) or 0)
    fps = probe_video_framerate(config.input)
    _codec, advised, _bitrate_kbps, _bppf = _compute_source_advice(
        input_path=config.input,
        width=width,
        height=height,
        raw_fr=fps,
    )

    default_choice = _default_choice_for_tier(advised)
    _print_source_quality_menu(default_choice)
    selected = _resolve_source_quality_choice(default_choice)
    if selected is None:
        return False
    config.source_quality = selected
    return True


def _quality_label_to_qp(label: str) -> int | None:
    key = label.strip().lower().replace("_", "-").replace(" ", "-")
    return _QP_FOR_TIER.get(key)


def _resolve_expected_start_qp(config: EncodeConfig) -> int:
    token = config.source_quality
    if token is None:
        raise ValueError("Expected mode requested without source quality or expected qp.")

    text = str(token).strip().lower()
    qp_from_label = _quality_label_to_qp(text)
    if qp_from_label is not None:
        start_qp = qp_from_label
        print(f"Expected-QP mode: start QP={start_qp} (source-quality={text})")
    else:
        try:
            start_qp = int(text)
        except ValueError as exc:
            raise ValueError(
                "Invalid --source-quality value. Use QP integer or one of: ultra, high, medium, low, lower."
            ) from exc
    if start_qp < config.min_qp:
        print(
            f"WARNING: expected start QP {start_qp} is below --min-qp {config.min_qp}; "
            "lowering min-qp to match."
        )
        config.min_qp = start_qp
    if start_qp > config.max_qp:
        print(
            f"WARNING: expected start QP {start_qp} is above --max-qp {config.max_qp}; "
            "raising max-qp to match."
        )
        config.max_qp = start_qp
    return start_qp


def _safe_pct_delta(prev: int, cur: int) -> float:
    if prev <= 0:
        return 0.0
    return max(0.0, (prev - cur) * 100.0 / prev)


def _transition_velocity(
    prev_ssim: float,
    cur_ssim: float,
    size_saved_mb: float,
) -> float:
    if size_saved_mb <= 0:
        return 0.0
    ssim_drop = max(0.0, prev_ssim - cur_ssim)
    return ssim_drop / (size_saved_mb / 100.0)


def _tail_drop_worst_fraction(
    base_vals: List[float],
    cur_vals: List[float],
    fraction: float = 0.5,
) -> float:
    if not base_vals or not cur_vals:
        return 0.0
    n = min(len(base_vals), len(cur_vals))
    if n <= 0:
        return 0.0
    drops = [max(0.0, base_vals[i] - cur_vals[i]) for i in range(n)]
    k = max(1, int(math.ceil(n * fraction)))
    worst = sorted(drops, reverse=True)[:k]
    return mean(worst) if worst else 0.0


def _print_expected_qp_ladder(rows: List[Dict[str, Any]]) -> None:
    print("\nExpected-QP ladder summary:")
    print("  qp  size_mb   ssim      size_saved_mb  size_saved_pct  ssim_drop  drop_per_100mb")
    prev = None
    for row in rows:
        qp = row["qp"]
        size_mb = row["size_bytes"] / 1_000_000.0
        ssim = row["ssim"]
        if prev is None:
            print(f"  {qp:2d}  {size_mb:7.2f}  {ssim:0.6f}")
            prev = row
            continue
        prev_size_mb = prev["size_bytes"] / 1_000_000.0
        size_saved_mb = max(0.0, prev_size_mb - size_mb)
        size_saved_pct = _safe_pct_delta(prev["size_bytes"], row["size_bytes"])
        ssim_drop = max(0.0, prev["ssim"] - ssim)
        velocity = _transition_velocity(prev["ssim"], ssim, size_saved_mb)
        print(
            f"  {qp:2d}  {size_mb:7.2f}  {ssim:0.6f}"
            f"  {size_saved_mb:13.2f}  {size_saved_pct:13.2f}%"
            f"  {ssim_drop:9.6f}  {velocity:14.6f}"
        )
        prev = row


def _row_ssim_metric(row: Dict[str, Any], use_raw_metric: bool) -> float:
    if use_raw_metric:
        raw = row.get("raw_ssim")
        if isinstance(raw, (float, int)):
            return float(raw)
    return float(row.get("ssim", 0.0))


def _effective_flat_stats(rows: List[Dict[str, Any]]) -> tuple[float, float, float]:
    if len(rows) <= 1:
        val = float(rows[0]["ssim"]) if rows else 0.0
        return 0.0, 0.0, val
    effective_vals = [float(r["ssim"]) for r in rows]
    effective_range = max(effective_vals) - min(effective_vals)
    cumulative_drop = 0.0
    drop_per_100mb_values: list[float] = []
    for i in range(1, len(rows)):
        prev = rows[i - 1]
        cur = rows[i]
        prev_ssim = float(prev["ssim"])
        cur_ssim = float(cur["ssim"])
        cumulative_drop += max(0.0, prev_ssim - cur_ssim)
        size_saved_mb = max(0.0, (int(prev["size_bytes"]) - int(cur["size_bytes"])) / 1_000_000.0)
        if size_saved_mb > 0:
            drop_per_100mb_values.append(_transition_velocity(prev_ssim, cur_ssim, size_saved_mb))
    median_drop_per_100mb = median(drop_per_100mb_values) if drop_per_100mb_values else 0.0
    return effective_range, cumulative_drop, median_drop_per_100mb


def _is_flat_effective_regime(rows: List[Dict[str, Any]]) -> bool:
    effective_range, cumulative_drop, median_drop_per_100mb = _effective_flat_stats(rows)
    return (
        effective_range <= 0.0005
        and cumulative_drop <= 0.0008
        and median_drop_per_100mb <= 0.00008
    )


def _find_adaptive_cutoff(rows: List[Dict[str, Any]]) -> tuple[int, str | None]:
    if len(rows) <= 1:
        return len(rows) - 1, None
    baseline_metric = _row_ssim_metric(rows[0], use_raw_metric=True)
    low_gain_streak = 0
    for i in range(1, len(rows)):
        prev = rows[i - 1]
        cur = rows[i]
        gain_pct = _safe_pct_delta(int(prev["size_bytes"]), int(cur["size_bytes"]))
        if gain_pct < 2.0:
            low_gain_streak += 1
        else:
            low_gain_streak = 0
        if low_gain_streak >= 2:
            return i, "size gain fell below 2% for two consecutive steps"
        cur_metric = _row_ssim_metric(cur, use_raw_metric=True)
        if (baseline_metric - cur_metric) > 0.015:
            keep_idx = max(1, i - 1)
            return keep_idx, "raw SSIM cumulative drop exceeded 0.015"
    return len(rows) - 1, None


def _suggest_expected_qps(
    rows: List[Dict[str, Any]],
    min_gain_pct: float,
    knee_ratio: float,
    use_raw_metric: bool = False,
) -> tuple[int, int]:
    flat_drop_per_100mb_threshold = _SSIM_FLAT_VELOCITY_FLOOR
    flat_cumulative_drop_threshold = _SSIM_FLAT_CUMULATIVE_DROP
    flat_range_threshold = _SSIM_FLAT_RANGE

    if len(rows) <= 1:
        qp = rows[0]["qp"]
        return qp, qp

    transitions: list[tuple[int, float, float]] = []
    drop_per_100mb_values: list[float] = []
    for i in range(1, len(rows)):
        prev = rows[i - 1]
        cur = rows[i]
        size_saved_mb = max(0.0, (prev["size_bytes"] - cur["size_bytes"]) / 1_000_000.0)
        size_saved_pct = _safe_pct_delta(prev["size_bytes"], cur["size_bytes"])
        prev_metric = _row_ssim_metric(prev, use_raw_metric=use_raw_metric)
        cur_metric = _row_ssim_metric(cur, use_raw_metric=use_raw_metric)
        velocity = _transition_velocity(prev_metric, cur_metric, size_saved_mb)
        transitions.append((i, size_saved_pct, velocity))
        if size_saved_mb > 0:
            drop_per_100mb_values.append(velocity)

    metric_vals = [_row_ssim_metric(r, use_raw_metric=use_raw_metric) for r in rows]
    metric_range = max(metric_vals) - min(metric_vals)
    cumulative_drop = sum(
        max(0.0, _row_ssim_metric(rows[i - 1], use_raw_metric=use_raw_metric) - _row_ssim_metric(rows[i], use_raw_metric=use_raw_metric))
        for i in range(1, len(rows))
    )
    median_drop_per_100mb = median(drop_per_100mb_values) if drop_per_100mb_values else 0.0
    if (
        use_raw_metric
        and cumulative_drop <= flat_cumulative_drop_threshold
        and median_drop_per_100mb <= flat_drop_per_100mb_threshold
    ):
        safe_qp = rows[-2]["qp"]
        balanced_qp = rows[-1]["qp"]
        print(
            "Expected-QP: non-degrading raw SSIM ladder detected "
            f"(range={metric_range:.6f}, cumulative_drop={cumulative_drop:.6f}, "
            f"median_drop_per_100mb={median_drop_per_100mb:.6f}); "
            f"preferring higher-QP tail ({safe_qp}/{balanced_qp})."
        )
        return safe_qp, balanced_qp
    flat_ladder = (
        metric_range <= flat_range_threshold
        and cumulative_drop <= flat_cumulative_drop_threshold
        and median_drop_per_100mb <= flat_drop_per_100mb_threshold
    )
    if flat_ladder:
        safe_qp = rows[-2]["qp"]
        balanced_qp = rows[-1]["qp"]
        metric_name = "raw SSIM" if use_raw_metric else "effective SSIM"
        print(
            f"Expected-QP: flat {metric_name}/size ladder detected "
            f"(range={metric_range:.6f}, cumulative_drop={cumulative_drop:.6f}, "
            f"median_drop_per_100mb={median_drop_per_100mb:.6f}); "
            f"preferring higher-QP tail ({safe_qp}/{balanced_qp})."
        )
        return safe_qp, balanced_qp

    valid = [t for t in transitions if t[1] >= min_gain_pct]
    if not valid:
        return rows[-2]["qp"], rows[-1]["qp"]

    # Rolling-window knee: compare local mean velocity before/after each candidate.
    # This is more robust than single-step ratio checks on noisy sample ladders.
    knee_index = None
    window = 2
    velocity_floor = _SSIM_FLAT_VELOCITY_FLOOR
    if len(valid) >= window * 2:
        idxs = [t[0] for t in valid]
        vels = [t[2] for t in valid]
        for j in range(window, len(valid) - window + 1):
            pre = mean(vels[j - window:j])
            post = mean(vels[j:j + window])
            if pre <= 0:
                continue
            if post >= velocity_floor and post >= pre * knee_ratio:
                knee_index = idxs[j]
                break

    # Fallback to the legacy step ratio when rolling-window cannot decide.
    if knee_index is None:
        prev_vel = None
        for i, _gain, vel in valid:
            if (
                prev_vel is not None
                and prev_vel > 0
                and vel >= velocity_floor
                and vel >= prev_vel * knee_ratio
            ):
                knee_index = i
                break
            prev_vel = vel

    if knee_index is None:
        mid = valid[len(valid) // 2][0]
        knee_index = mid

    balanced_idx = min(knee_index, len(rows) - 1)
    safe_idx = max(0, balanced_idx - 1)
    return rows[safe_idx]["qp"], rows[balanced_idx]["qp"]


def _parse_expected_custom_qps(raw: str, available_qps: set[int]) -> list[int] | None:
    parts = [p.strip() for p in re.split(r"[,/]", raw)]
    if not parts or all(not p for p in parts):
        return None
    parsed: list[int] = []
    seen: set[int] = set()
    for part in parts:
        if not part:
            continue
        range_match = re.fullmatch(r"(-?\d+)\s*-\s*(-?\d+)", part)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            step = 1 if end >= start else -1
            for qp in range(start, end + step, step):
                if qp not in available_qps:
                    valid = ", ".join(str(v) for v in sorted(available_qps))
                    print(f"QP {qp} is not in the explored ladder. Available: {valid}")
                    return None
                if qp in seen:
                    continue
                parsed.append(qp)
                seen.add(qp)
            continue
        try:
            qp = int(part)
        except ValueError:
            print(
                f"Invalid QP token '{part}'. Use single QPs and/or ranges "
                "(example: 21,22,24-26 or 21/22/24-26)."
            )
            return None
        if qp not in available_qps:
            valid = ", ".join(str(v) for v in sorted(available_qps))
            print(f"QP {qp} is not in the explored ladder. Available: {valid}")
            return None
        if qp in seen:
            continue
        parsed.append(qp)
        seen.add(qp)
    return parsed or None


def _choose_expected_qps(
    config: EncodeConfig,
    rows: List[Dict[str, Any]],
    safe_qp: int,
    balanced_qp: int,
    source_codec: str,
    source_size_bytes: int,
    size_is_estimate: bool = False,
) -> list[int]:
    by_qp = {row["qp"]: row for row in rows}
    safe_size = by_qp[safe_qp]["size_bytes"] / 1_000_000.0
    balanced_size = by_qp[balanced_qp]["size_bytes"] / 1_000_000.0
    size_label = "est. size" if size_is_estimate else "size"
    print("\nExpected-QP suggestions:")
    print(f"  Source codec: {source_codec}, size: {source_size_bytes/1_000_000.0:.2f} MB")
    print(f"  [1] Safe (recommended): QP={safe_qp}, {size_label}={safe_size:.2f} MB")
    print(f"  [2] Balanced: QP={balanced_qp}, {size_label}={balanced_size:.2f} MB")
    print("  [3] Custom QP(s): enter ladder QPs and/or ranges (comma or slash separated)")

    if config.expected_choice == "safe":
        return [safe_qp]
    if config.expected_choice == "balanced":
        return [balanced_qp]
    if not sys.stdin.isatty():
        return [safe_qp]

    while True:
        choice = input("Choose 1, 2, or 3 [1]: ").strip()
        if choice in ("", "1"):
            return [safe_qp]
        if choice == "2":
            return [balanced_qp]
        if choice == "3":
            raw = input("Enter QP list/range (e.g. 21,22,24-26 or 21/22/24-26): ").strip()
            selected = _parse_expected_custom_qps(raw, set(by_qp.keys()))
            if selected is not None:
                return selected
            continue
        print("Please enter 1, 2, or 3.")


def _run_expected_qp_mode(
    config: EncodeConfig,
    baseline_file: str,
    source_ref_path: str,
    audio_opts: List[str],
    raw_fr: float,
    gop: int,
    tmpdir: str,
    video_codec: str,
    scratch_root: str,
) -> tuple[str, int, float | None, list[str]]:
    start_qp = _resolve_expected_start_qp(config)
    print(
        "Expected-QP mode: "
        f"start QP={start_qp}, min_gain={config.expected_min_gain:.2f}%, "
        f"max_steps={config.expected_max_steps}, knee_ratio={config.expected_knee_ratio:.2f}"
    )
    source_base, source_ext = os.path.splitext(os.path.basename(source_ref_path))
    max_qp = min(config.max_qp, start_qp + max(0, config.expected_max_steps))
    max_qp_hard = min(config.max_qp, 40)

    # Rows and segments are preserved across tier retries via BatchOutputLargerThanSourceError.
    precomputed_rows: list[dict] | None = config.batch_precomputed_rows
    precomputed_segs: list[str] | None = config.batch_precomputed_segments
    precomputed_segs_tmpdir: str | None = config.batch_precomputed_segments_tmpdir
    if precomputed_segs is not None:
        segments: list[str] = precomputed_segs
        segments_tmpdir: str | None = precomputed_segs_tmpdir
        own_segments = False
    else:
        segments, segments_tmpdir = _prepare_size_segments(
            baseline_file=baseline_file,
            config=config,
            scratch_root=scratch_root,
        )
        own_segments = True

    try:
        rows: List[Dict[str, Any]] = _build_expected_rows_sample(
            config=config,
            baseline_file=baseline_file,
            raw_fr=raw_fr,
            gop=gop,
            audio_opts=audio_opts,
            video_codec=video_codec,
            scratch_root=scratch_root,
            start_qp=start_qp,
            max_qp=max_qp,
            segments=segments,
            segments_tmpdir=segments_tmpdir,
            precomputed_rows=precomputed_rows,
        )

        if not rows:
            raise RuntimeError("Expected-QP mode could not produce any full-file encodes.")

        use_raw_metric = False
        if _is_flat_effective_regime(rows):
            use_raw_metric = True
            print(
                "Adaptive ladder enabled: effective SSIM is flat "
                "(range<=0.0005, cumulative<=0.0008, median drop/100MB<=0.00008)."
            )
            current_max_qp = int(rows[-1]["qp"])
            if current_max_qp >= max_qp_hard:
                print(f"Adaptive ladder stop: reached hard QP cap ({max_qp_hard}).")
            else:
                adaptive_max_qp = max_qp_hard
                print(f"Adaptive ladder extending: QP {current_max_qp + 1}..{adaptive_max_qp}")
                rows = _extend_expected_rows_sample_raw(
                    config=config,
                    baseline_file=baseline_file,
                    raw_fr=raw_fr,
                    gop=gop,
                    audio_opts=audio_opts,
                    video_codec=video_codec,
                    scratch_root=scratch_root,
                    rows=rows,
                    start_qp=current_max_qp + 1,
                    max_qp=adaptive_max_qp,
                    segments=segments,
                    segments_tmpdir=segments_tmpdir,
                )
                cutoff_idx, stop_reason = _find_adaptive_cutoff(rows)
                if cutoff_idx < (len(rows) - 1):
                    rows = rows[:cutoff_idx + 1]
                if stop_reason:
                    print(f"Adaptive ladder stop: {stop_reason}.")
                elif int(rows[-1]["qp"]) >= max_qp_hard:
                    print(f"Adaptive ladder stop: reached hard QP cap ({max_qp_hard}).")
                else:
                    print(f"Adaptive ladder stop: reached extension limit at QP={rows[-1]['qp']}.")

        _print_expected_qp_ladder(rows)
        safe_qp, balanced_qp = _suggest_expected_qps(
            rows=rows,
            min_gain_pct=config.expected_min_gain,
            knee_ratio=config.expected_knee_ratio,
            use_raw_metric=use_raw_metric,
        )
        src_codec = probe_video_codec(config.input)
        src_size = _safe_file_size(config.input) or 0
        selected_qps = _choose_expected_qps(
            config=config,
            rows=rows,
            safe_qp=safe_qp,
            balanced_qp=balanced_qp,
            source_codec=src_codec,
            source_size_bytes=src_size,
            size_is_estimate=True,
        )
        selected_qp = selected_qps[0]
        print(
            f"Expected-QP mode suggestions: safe={safe_qp}, balanced={balanced_qp}. "
            f"Selected QP(s)={selected_qps}."
        )

        # In batch mode with a heuristic tier, check the ladder estimate before encoding.
        # If the selected QP predicts an output larger than the source, first scan the
        # already-measured rows for a higher QP that fits — avoiding a full tier-drop
        # retry that would re-measure all the same QP levels from scratch.
        # Only raise BatchOutputLargerThanSourceError (triggering a tier drop) when no
        # row in the current ladder can produce an output small enough.
        batch_size_limit: int | None = config.batch_heuristic_size_limit
        if batch_size_limit is not None:
            selected_row = next((r for r in rows if r["qp"] == selected_qp), None)
            if selected_row is not None:
                est_size = int(selected_row.get("size_bytes", 0))
                if est_size > batch_size_limit:
                    fitting = next(
                        (
                            r for r in rows
                            if int(r["qp"]) > selected_qp
                            and int(r.get("size_bytes", 0)) > 0
                            and int(r.get("size_bytes", 0)) <= batch_size_limit
                        ),
                        None,
                    )
                    if fitting is not None:
                        fallback_qp = int(fitting["qp"])
                        fallback_size = int(fitting.get("size_bytes", 0))
                        print(
                            f"  Selected QP={selected_qp} estimated {est_size/1e6:.1f} MB > "
                            f"source {batch_size_limit/1e6:.1f} MB; "
                            f"using QP={fallback_qp} (est. {fallback_size/1e6:.1f} MB) "
                            f"from existing ladder — no tier drop needed."
                        )
                        selected_qp = fallback_qp
                        selected_qps = [fallback_qp]
                    else:
                        err = BatchOutputLargerThanSourceError(
                            f"estimated {est_size / 1e6:.1f} MB > source "
                            f"{batch_size_limit / 1e6:.1f} MB at QP={selected_qp}"
                        )
                        err.segments = segments
                        err.segments_tmpdir = segments_tmpdir
                        err.rows = rows
                        own_segments = False  # exception takes ownership of cleanup
                        raise err

        extra_paths: list[str] = []
        print("Expected-QP sample mode: skipping post-selection full-file SSIM measurement.")
        output_path: str | None = None
        for idx, qp in enumerate(selected_qps):
            result = encode_final(
                input_file=baseline_file,
                qp=qp,
                audio_opts=audio_opts,
                raw_fr=raw_fr,
                gop=gop,
                video_codec=video_codec,
                return_ssim=False,
                ssim_chunk_seconds=config.full_ssim_chunk_seconds,
                output_dir=tmpdir,
                output_base=source_base,
                output_ext=source_ext,
            )
            if idx == 0:
                output_path = cast(str, result)
            else:
                extra_paths.append(cast(str, result))
        if output_path is None:
            raise RuntimeError("Expected-QP mode failed to encode selected output(s).")
        selected_row = next((row for row in rows if row["qp"] == selected_qp), rows[0])
        resolved_ssim = selected_row["ssim"]
        return output_path, selected_qp, resolved_ssim, extra_paths
    finally:
        if own_segments and segments_tmpdir:
            shutil.rmtree(segments_tmpdir, ignore_errors=True)


def _build_expected_rows_sample(
    config: EncodeConfig,
    baseline_file: str,
    raw_fr: float,
    gop: int,
    audio_opts: List[str],
    video_codec: str,
    scratch_root: str,
    start_qp: int,
    max_qp: int,
    segments: list[str] | None = None,
    segments_tmpdir: str | None = None,
    precomputed_rows: list[dict] | None = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    baseline_vals: List[float] | None = None
    baseline_metric = 0.0
    prev_tail_drop = 0.0
    effective_start = start_qp

    # Reuse the row for start_qp from a previous tier run if available, so we skip
    # re-measuring a QP that was already measured and use its per-sample SSIM values
    # as the baseline for tail_drop continuity on subsequent QPs.
    if precomputed_rows:
        precomputed_by_qp = {int(r["qp"]): r for r in precomputed_rows}
        if start_qp in precomputed_by_qp:
            carried = precomputed_by_qp[start_qp]
            prior_sample_vals: list[float] = carried.get("raw_sample_vals") or []
            if prior_sample_vals:
                baseline_vals = prior_sample_vals
                baseline_metric = mean(baseline_vals)
                rows.append({
                    "qp": start_qp,
                    "ssim": float(baseline_metric),
                    "raw_ssim": float(baseline_metric),
                    "size_bytes": int(carried.get("size_bytes", 0)),
                    "raw_sample_vals": prior_sample_vals,
                })
                effective_start = start_qp + 1
                print(
                    f"  [Row reuse] QP={start_qp} carried from previous tier "
                    f"(raw_avg={baseline_metric:.6f}); starting ladder at QP={effective_start}."
                )

    own_segments = segments is None
    if own_segments:
        segments, segments_tmpdir = _prepare_size_segments(
            baseline_file=baseline_file,
            config=config,
            scratch_root=scratch_root,
        )
    if not segments:
        raise RuntimeError("Expected-QP sample mode could not extract sample segments.")
    try:
        for qp in range(effective_start, max_qp + 1):
            sample_vals = measure_ssim_values(
                qp=qp,
                samples=segments,
                raw_fr=raw_fr,
                gop=gop,
                audio_opts=audio_opts,
                video_codec=video_codec,
            )
            sample_metric = mean(sample_vals) if sample_vals else 0.0
            if baseline_vals is None:
                baseline_vals = sample_vals
                baseline_metric = sample_metric
                effective_metric = sample_metric
            else:
                tail_drop = _tail_drop_worst_fraction(
                    baseline_vals,
                    sample_vals,
                    fraction=0.5,
                )
                # Enforce monotonic degradation to reduce SSIM jitter effects.
                tail_drop = max(prev_tail_drop, tail_drop)
                prev_tail_drop = tail_drop
                effective_metric = max(0.0, baseline_metric - tail_drop)
            sample_bytes = _sample_encoded_sizes(segments)
            size_bytes = _estimate_full_size_from_samples(sample_bytes, config.sample_percent)
            rows.append({
                "qp": qp,
                "ssim": float(effective_metric),
                "raw_ssim": float(sample_metric),
                "size_bytes": int(size_bytes),
                "raw_sample_vals": sample_vals,
            })
            tqdm_msg = (
                f"Expected-QP sample metric at QP={qp}: "
                f"raw_avg={sample_metric:.6f}, effective={effective_metric:.6f}"
            )
            print(tqdm_msg)
            _print_expected_step_progress(rows, estimate=True)
    finally:
        if own_segments and segments_tmpdir:
            shutil.rmtree(segments_tmpdir, ignore_errors=True)
    return rows


def _extend_expected_rows_sample_raw(
    config: EncodeConfig,
    baseline_file: str,
    raw_fr: float,
    gop: int,
    audio_opts: List[str],
    video_codec: str,
    scratch_root: str,
    rows: List[Dict[str, Any]],
    start_qp: int,
    max_qp: int,
    segments: list[str] | None = None,
    segments_tmpdir: str | None = None,
) -> List[Dict[str, Any]]:
    if start_qp > max_qp:
        return rows
    own_segments = segments is None
    if own_segments:
        segments, segments_tmpdir = _prepare_size_segments(
            baseline_file=baseline_file,
            config=config,
            scratch_root=scratch_root,
        )
    if not segments:
        raise RuntimeError("Adaptive expected-QP sample extension could not extract sample segments.")
    try:
        for qp in range(start_qp, max_qp + 1):
            sample_vals = measure_ssim_values(
                qp=qp,
                samples=segments,
                raw_fr=raw_fr,
                gop=gop,
                audio_opts=audio_opts,
                video_codec=video_codec,
            )
            sample_metric = mean(sample_vals) if sample_vals else 0.0
            sample_bytes = _sample_encoded_sizes(segments)
            size_bytes = _estimate_full_size_from_samples(sample_bytes, config.sample_percent)
            rows.append({
                "qp": qp,
                "ssim": float(sample_metric),
                "raw_ssim": float(sample_metric),
                "size_bytes": int(size_bytes),
            })
            print(
                f"Expected-QP adaptive metric at QP={qp}: "
                f"raw_avg={sample_metric:.6f}, effective(raw-regime)={sample_metric:.6f}"
            )
            _print_expected_step_progress(rows, estimate=True)
    finally:
        if own_segments and segments_tmpdir:
            shutil.rmtree(segments_tmpdir, ignore_errors=True)
    return rows


def _print_expected_step_progress(rows: List[Dict[str, Any]], estimate: bool) -> None:
    if len(rows) < 2:
        return
    prev = rows[-2]
    cur = rows[-1]
    gain = _safe_pct_delta(prev["size_bytes"], cur["size_bytes"])
    prefix = "est size" if estimate else "size"
    print(
        f"  QP {prev['qp']} -> {cur['qp']}: {prefix} {prev['size_bytes']/1_000_000.0:.1f} -> "
        f"{cur['size_bytes']/1_000_000.0:.1f} MB (gain {gain:.2f}%)"
    )


def _prepare_baseline_and_source(
    config: EncodeConfig,
    extra_vf: str | None,
    baseline_tmp: str,
    video_codec: str,
) -> tuple[str, str]:
    baseline_file = _determine_baseline_file(
        input_path=config.input,
        skip_baseline=config.skip_baseline,
        baseline_tmp=baseline_tmp,
        baseline_qp=config.baseline_qp,
        extra_vf=extra_vf,
        video_codec=video_codec,
        skip_baseline_requested=config.skip_baseline_requested,
        allow_skip_with_filters=config.batch_force_fixed_qp_mode,
    )
    source_ref_path = _resolve_source_ref_path(config, baseline_file)
    return baseline_file, source_ref_path


def _prepare_main_context(
    config: EncodeConfig,
) -> tuple[str | None, str, List[str], float]:
    width, height, pix_fmt, display_ar, sar_text, _dar_text = _probe_video_basics(config.input)
    _resolve_sample_percent(config, config.input)
    _resolve_sample_count(config, config.input)

    if config.video_codec == "h264":
        if not _confirm_h264_compat(width, height, pix_fmt, decision=config.h264_compat):
            raise RuntimeError("aborted_h264_compat")

    crop_filter = _resolve_crop_filter(config, width, height)
    resize_filter = _resolve_resize_filter(
        config,
        width,
        height,
        display_ar=display_ar,
        sample_aspect_ratio=sar_text,
    )
    extra_vf = _merge_video_filters(crop_filter, resize_filter)
    _maybe_enable_default_skip_baseline(config, config.input, extra_vf)
    if not _ensure_source_quality_for_default_pipeline(config):
        raise RuntimeError("cancelled")

    codec = _resolve_quality_mode(config, config.input)
    if codec is None:
        raise RuntimeError("invalid_quality_mode")

    audio_opts, source_fr, config.audio_normalize = _prepare_audio_and_framerate(
        config.input, config.audio_normalize, config.add_stereo_downmix
    )
    raw_fr = _resolve_target_framerate(source_fr, config.target_fps)
    _maybe_print_expected_mode_advice(
        config=config,
        input_path=config.input,
        width=width,
        height=height,
        raw_fr=source_fr,
    )
    _stem = os.path.splitext(os.path.basename(config.input))[0]
    _raw_ext = os.path.splitext(config.input)[1].lower()
    _out_ext = _raw_ext if _raw_ext == ".mp4" else ".mkv"
    check_output_filename_length(
        stem=_stem,
        out_ext=_out_ext,
        encoder_tag=output_codec_tag(normalize_video_codec(codec)),
        resolution_label=f"{min(width, height)}p",
        fps_int=int(round(raw_fr)),
    )
    return extra_vf, codec, audio_opts, raw_fr


def _run_transcode_pipeline(
    config: EncodeConfig,
    extra_vf: str | None,
    codec: str,
    audio_opts: List[str],
    raw_fr: float,
) -> tuple[str, int]:
    scratch_root = _resolve_scratch_root(config.scratch_dir)
    baseline_tmp, tmpdir, samples_tmpdir = _prepare_work_dirs(
        scratch_root, config.input, config.skip_baseline
    )
    # When handing off baseline ownership to caller on BatchOutputLargerThanSourceError,
    # skip deletion of baseline_tmp in finally so the next retry can reuse it.
    _baseline_tmp_owned = True
    precomputed_baseline: str | None = config.batch_precomputed_baseline_file
    baseline_file: str = config.input  # safe default; overwritten below before use

    try:
        if precomputed_baseline:
            baseline_file = precomputed_baseline
            source_ref_path = _resolve_source_ref_path(config, baseline_file)
        else:
            baseline_file, source_ref_path = _prepare_baseline_and_source(
                config=config,
                extra_vf=extra_vf,
                baseline_tmp=baseline_tmp,
                video_codec=config.video_codec,
            )

        gop = _calculate_gop(raw_fr)
        extra_final_files: list[str] = []

        if config.batch_force_fixed_qp_mode:
            source_base, source_ext = os.path.splitext(os.path.basename(source_ref_path))
            final_file = cast(str, encode_final(
                input_file=baseline_file,
                qp=config.initial_qp,
                audio_opts=audio_opts,
                raw_fr=raw_fr,
                gop=gop,
                video_codec=config.video_codec,
                return_ssim=False,
                ssim_chunk_seconds=config.full_ssim_chunk_seconds,
                output_dir=tmpdir,
                output_base=source_base,
                output_ext=source_ext,
            ))
            final_qp = config.initial_qp
        else:
            final_file, final_qp, _final_full_ssim, extra_final_files = _run_expected_qp_mode(
                config=config,
                baseline_file=baseline_file,
                source_ref_path=source_ref_path,
                audio_opts=audio_opts,
                raw_fr=raw_fr,
                gop=gop,
                tmpdir=tmpdir,
                video_codec=config.video_codec,
                scratch_root=scratch_root,
            )

        final_path: str = final_file
        dest: str = _resolve_output_dest(
            encoded_path=final_path,
            source_path=config.input,
            explicit_output=config.output,
            batch_input_dir=config.batch_input_dir,
        )

        shutil.move(final_path, dest)
        print(f"Optimized file: {dest} (QP={final_qp})")
        logging.info("Optimized file: %s (QP=%s)", dest, final_qp)
        dest_dir = os.path.dirname(dest) or "."
        for extra_path in extra_final_files:
            extra_dest = os.path.join(dest_dir, os.path.basename(extra_path))
            shutil.move(extra_path, extra_dest)
            print(f"Additional output: {extra_dest}")
            logging.info("Additional output: %s", extra_dest)
        return dest, int(final_qp)

    except BatchOutputLargerThanSourceError as exc:
        # Attach the baseline location so the batch loop can reuse it on the next tier.
        # Only do this when we created the baseline ourselves (not when we reused one).
        if not precomputed_baseline:
            exc.baseline_file = baseline_file
            exc.baseline_tmp = baseline_tmp
            _baseline_tmp_owned = False
        raise

    finally:
        if samples_tmpdir:
            shutil.rmtree(samples_tmpdir, ignore_errors=True)
        shutil.rmtree(tmpdir, ignore_errors=True)
        if _baseline_tmp_owned:
            shutil.rmtree(baseline_tmp, ignore_errors=True)


def _relative_delta(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / denom


def _list_video_files(root: str) -> list[str]:
    found: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext in _VIDEO_EXTENSIONS:
                found.append(os.path.join(dirpath, name))
    found.sort()
    return found


def _build_batch_profile(path: str) -> Dict[str, Any] | None:
    try:
        vinfo = probe_video_stream_info(path)
        width = int(vinfo.get("width", 0) or 0)
        height = int(vinfo.get("height", 0) or 0)
        pix_fmt = str(vinfo.get("pix_fmt", "") or "unknown")
        codec = normalize_video_codec(probe_video_codec(path) or "unknown")
        fps = float(probe_video_framerate(path) or 0.0)
        hdr_info = detect_hdr(path)
        is_hdr = bool(hdr_info.get("is_hdr"))
        duration = float(probe_video_duration(path) or 0.0)
        size_bytes = _safe_get_file_size(path)
        stream_br = int(probe_video_bitrate(path) or 0)
        avg_br = int((size_bytes * 8) / duration) if duration > 0 and size_bytes > 0 else 0
        bitrate_bps = stream_br if stream_br > 0 else avg_br
        pixels_per_sec = float(max(1, width) * max(1, height)) * max(fps, 0.0001)
        bppf = float(bitrate_bps) / pixels_per_sec if bitrate_bps > 0 else 0.0
        return {
            "path": path,
            "codec": codec,
            "width": width,
            "height": height,
            "fps": fps,
            "fps_bucket": round(fps, 3),
            "pix_fmt": pix_fmt,
            "is_hdr": is_hdr,
            "bitrate_bps": bitrate_bps,
            "bppf": bppf,
            "size_bytes": size_bytes,
            "duration": duration,
        }
    except Exception as exc:
        print(f"WARNING: could not profile '{path}': {exc}")
        return None


def _split_profiles_by_quality(
    profiles: list[Dict[str, Any]],
    bppf_tolerance: float,
    bitrate_tolerance: float,
) -> list[list[Dict[str, Any]]]:
    if not profiles:
        return []
    ordered = sorted(profiles, key=lambda p: (float(p["bppf"]), int(p["bitrate_bps"]), str(p["path"])))
    groups: list[list[Dict[str, Any]]] = []
    current: list[Dict[str, Any]] = [ordered[0]]
    for prof in ordered[1:]:
        cur_bppf_med = float(median([float(p["bppf"]) for p in current]))
        cur_br_med = float(median([float(p["bitrate_bps"]) for p in current]))
        bppf_ok = _relative_delta(float(prof["bppf"]), cur_bppf_med) <= max(0.0, bppf_tolerance)
        if cur_br_med <= 0 or float(prof["bitrate_bps"]) <= 0:
            br_ok = True
        else:
            br_ok = _relative_delta(float(prof["bitrate_bps"]), cur_br_med) <= max(0.0, bitrate_tolerance)
        if bppf_ok and br_ok:
            current.append(prof)
            continue
        groups.append(current)
        current = [prof]
    if current:
        groups.append(current)
    return groups


def _cluster_videos_for_batch(
    profiles: list[Dict[str, Any]],
    bppf_tolerance: float,
    bitrate_tolerance: float,
) -> list[Dict[str, Any]]:
    base_groups: Dict[tuple[Any, ...], list[Dict[str, Any]]] = {}
    for p in profiles:
        key = (
            p["codec"],
            int(p["width"]),
            int(p["height"]),
            float(p["fps_bucket"]),
            str(p["pix_fmt"]),
            bool(p["is_hdr"]),
        )
        base_groups.setdefault(key, []).append(p)

    clusters: list[Dict[str, Any]] = []
    cluster_id = 1
    for key in sorted(base_groups.keys(), key=lambda k: str(k)):
        split_groups = _split_profiles_by_quality(
            base_groups[key],
            bppf_tolerance=bppf_tolerance,
            bitrate_tolerance=bitrate_tolerance,
        )
        for members in split_groups:
            ordered = sorted(members, key=lambda p: (float(p["bppf"]), str(p["path"])))
            rep = ordered[len(ordered) // 2]
            clusters.append(
                {
                    "id": cluster_id,
                    "key": key,
                    "representative": rep,
                    "members": sorted(members, key=lambda p: str(p["path"])),
                }
            )
            cluster_id += 1
    clusters.sort(key=lambda c: str(c["representative"]["path"]))
    for idx, cluster in enumerate(clusters, start=1):
        cluster["id"] = idx
    return clusters


def _format_batch_bitrate(bitrate_bps: int) -> str:
    if bitrate_bps <= 0:
        return "unknown"
    if bitrate_bps >= 1_000_000:
        return f"{bitrate_bps / 1_000_000.0:.1f} Mb/s"
    return f"{bitrate_bps / 1000.0:.0f} kb/s"


def _format_batch_resfps(width: int, height: int, fps: float) -> str:
    if width <= 0 or height <= 0 or fps <= 0:
        return "unknown"
    return f"{width}x{height}@{fps:.3f}"


def _print_batch_cluster_summary(clusters: list[Dict[str, Any]], root: str) -> None:
    print(f"\nBatch clustering summary: {len(clusters)} cluster(s)")
    for cl in clusters:
        rep = cl["representative"]
        members = cl["members"]
        bppf_vals = [float(m["bppf"]) for m in members]
        br_vals = [float(m["bitrate_bps"]) / 1000.0 for m in members]
        print(
            f"  Cluster {cl['id']}: {len(members)} file(s), "
            f"{rep['codec']} {rep['width']}x{rep['height']}@{rep['fps_bucket']:.3f}, "
            f"{'HDR' if rep['is_hdr'] else 'SDR'}, {rep['pix_fmt']}"
        )
        print(
            f"    bppf {min(bppf_vals):.4f}..{max(bppf_vals):.4f}, "
            f"bitrate {min(br_vals):.0f}..{max(br_vals):.0f} kb/s"
        )
        print("    files:")
        for member in members:
            full_path = str(member["path"])
            rel_path = os.path.relpath(full_path, root)
            marker = "REP" if _same_file(full_path, str(rep["path"])) else "   "
            codec = str(member["codec"]).upper()
            bitrate = _format_batch_bitrate(int(member["bitrate_bps"]))
            resfps = _format_batch_resfps(
                int(member["width"]),
                int(member["height"]),
                float(member["fps_bucket"]),
            )
            tier = _tier_for_bppf(str(member["codec"]), float(member["bppf"]))
            print(
                f"      [{marker}] {rel_path} | {codec:<4} | {bitrate:<10} | "
                f"{resfps} | bppf={float(member['bppf']):.4f} | tier={tier}"
            )


def _print_batch_excluded_summary(
    excluded: list[Dict[str, Any]],
    root: str,
    reason: str,
) -> None:
    if not excluded:
        return
    print(f"\nExcluded files: {len(excluded)}")
    print(f"  reason: {reason}")
    for member in sorted(excluded, key=lambda p: str(p["path"])):
        full_path = str(member["path"])
        rel_path = os.path.relpath(full_path, root)
        codec = str(member["codec"]).upper()
        bitrate = _format_batch_bitrate(int(member["bitrate_bps"]))
        resfps = _format_batch_resfps(
            int(member["width"]),
            int(member["height"]),
            float(member["fps_bucket"]),
        )
        tier = _tier_for_bppf(str(member["codec"]), float(member["bppf"]))
        print(
            f"      {rel_path} | {codec:<4} | {bitrate:<10} | "
            f"{resfps} | bppf={float(member['bppf']):.4f} | tier={tier}"
        )


def _write_qp_sidecar(dest: str, qp: int) -> None:
    try:
        with open(dest + ".qpreserve.json", "w") as f:
            json.dump({"qp": qp}, f)
    except OSError:
        pass


def _read_qp_sidecar(dest: str) -> int | None:
    try:
        with open(dest + ".qpreserve.json") as f:
            data = json.load(f)
        val = data.get("qp")
        return int(val) if val is not None else None
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _predict_output_path(config: EncodeConfig) -> str | None:
    """Return the expected output path when it can be determined before encoding, else None.

    When config.output is None (auto-naming), the path cannot be predicted because
    it includes the QP discovered during encoding.
    """
    if config.output is None:
        return None
    if os.path.isdir(config.output):
        # Directory destination: construct the path from source basename + extension.
        src_ext = os.path.splitext(config.input)[1].lower()
        enc_ext = src_ext if src_ext in {'.mp4', '.mkv'} else '.mkv'
        dummy_encoded = os.path.splitext(os.path.basename(config.input))[0] + enc_ext
        return _resolve_output_dest(
            encoded_path=dummy_encoded,
            source_path=config.input,
            explicit_output=config.output,
            batch_input_dir=config.batch_input_dir,
        )
    # Explicit file path.
    return config.output


def _run_single_file_pipeline(config: EncodeConfig) -> tuple[str, int]:
    predicted = _predict_output_path(config)
    if predicted and os.path.exists(predicted):
        print(f"Skipping (output exists): {predicted}")
        logging.info("Skipping (output exists): %s", predicted)
        return predicted, -1
    config.skip_baseline_requested = bool(config.skip_baseline)
    extra_vf, codec, audio_opts, raw_fr = _prepare_main_context(config)
    dest, qp = _run_transcode_pipeline(
        config=config,
        extra_vf=extra_vf,
        codec=codec,
        audio_opts=audio_opts,
        raw_fr=raw_fr,
    )
    if qp >= 0:
        _write_qp_sidecar(dest, qp)
    return dest, qp


def _run_batch_auto(config: EncodeConfig) -> None:
    if config.auto_crop == "prompt":
        print("Batch mode: forcing --auto-crop off to avoid interactive prompts.")
        config.auto_crop = "off"
    config.expected_choice = "safe"
    if not config.disable_clustering:
        print(
            "NOTE: Batch mode applies the QP learned from one representative to all peers "
            "in each cluster. Files with inflated bitrates (re-uploads, screen recordings) "
            "may land in the same cluster and receive a suboptimal QP. "
            "Use --disable-clustering for independent per-file quality search."
        )

    all_files = _list_video_files(config.input)
    if not all_files:
        print("No supported video files found.")
        return
    print(f"Batch scan: found {len(all_files)} video file(s).")

    profiles: list[Dict[str, Any]] = []
    for path in all_files:
        prof = _build_batch_profile(path)
        if prof is not None:
            profiles.append(prof)
    if not profiles:
        print("No readable video profiles found.")
        return

    target_codec = normalize_video_codec(config.video_codec)
    included: list[Dict[str, Any]] = []
    excluded: list[Dict[str, Any]] = []
    for p in profiles:
        if not config.re_encode_same_codec_video and p["codec"] == target_codec:
            excluded.append(p)
        else:
            included.append(p)

    if excluded:
        print(
            f"Batch filter: excluded {len(excluded)} source file(s) already using target codec "
            f"'{target_codec}'. Use --re-encode-same-codec-video to include them."
        )
    if not included:
        print("No files left to process after filtering.")
        return

    clusters = _cluster_videos_for_batch(
        included,
        bppf_tolerance=config.batch_bppf_tolerance,
        bitrate_tolerance=config.batch_bitrate_tolerance,
    )
    if not clusters:
        print("No clusters produced from batch scan.")
        return

    if config.disable_clustering:
        clusters = [
            {
                "id": i + 1,
                "key": (p["codec"], int(p["width"]), int(p["height"]),
                        float(p["fps_bucket"]), str(p["pix_fmt"]), bool(p["is_hdr"])),
                "representative": p,
                "members": [p],
            }
            for i, p in enumerate(included)
        ]

    _print_batch_cluster_summary(clusters, config.input)
    if config.batch_dry_run:
        _print_batch_excluded_summary(
            excluded=excluded,
            root=config.input,
            reason=f"source codec matches target codec '{target_codec}'",
        )
        print("\nBatch dry run enabled; no encodes were started.")
        return

    processed = 0
    for cluster in clusters:
        rep = cluster["representative"]
        members = cluster["members"]
        rep_path = str(rep["path"])
        print(f"\n[Cluster {cluster['id']}] Sampling representative: {rep_path}")

        # Track whether the tier was auto-detected so we know whether to retry.
        heuristic_tier = not bool(config.source_quality)
        initial_tier = (
            _tier_for_bppf(str(rep["codec"]), float(rep["bppf"]))
            if heuristic_tier
            else str(config.source_quality)
        )

        cluster_dest: str | None = None
        cluster_qp: int | None = None
        current_tier = initial_tier
        # Baseline, segments, and measured rows preserved across tier retries.
        batch_baseline_file: str | None = None
        batch_baseline_tmp: str | None = None
        batch_segments: list[str] = []
        batch_segments_tmpdir: str | None = None
        batch_rows: list[dict] = []

        while True:
            rep_config = dataclasses.replace(
                config,
                input=rep_path,
                batch_input_dir=config.input,
                source_quality=current_tier,
                batch_precomputed_baseline_file=batch_baseline_file or None,
                batch_precomputed_segments=batch_segments or None,
                batch_precomputed_segments_tmpdir=batch_segments_tmpdir if batch_segments else None,
                batch_precomputed_rows=batch_rows or None,
            )
            if heuristic_tier:
                print(
                    f"[Cluster {cluster['id']}] No --source-quality provided; "
                    f"using heuristic tier '{current_tier}' for representative."
                )
                # Pass source size to the pipeline only when the guard is enabled
                # and a lower tier exists to fall back to.
                if config.batch_size_guard and _next_lower_tier(current_tier) is not None:
                    try:
                        rep_config = dataclasses.replace(
                            rep_config,
                            batch_heuristic_size_limit=os.path.getsize(rep_path),
                        )
                    except OSError:
                        pass

            try:
                cluster_dest, cluster_qp = _run_single_file_pipeline(rep_config)
                # Verify actual output size — the pre-flight estimate may have passed
                # or been skipped (user-specified tier), but the real encode can still
                # be larger than the source.
                if (
                    config.batch_size_guard
                    and cluster_qp is not None
                    and cluster_qp >= 0  # not a skip sentinel
                    and cluster_dest
                    and os.path.exists(cluster_dest)
                ):
                    actual_size = os.path.getsize(cluster_dest)
                    source_size = os.path.getsize(rep_path)
                    if actual_size > source_size:
                        next_tier = _next_lower_tier(current_tier)
                        try:
                            os.remove(cluster_dest)
                        except OSError:
                            pass
                        cluster_dest = None
                        cluster_qp = None
                        if next_tier is not None:
                            print(
                                f"[Cluster {cluster['id']}] Actual output "
                                f"({actual_size/1e6:.1f} MB) > source "
                                f"({source_size/1e6:.1f} MB); "
                                f"retrying at lower tier '{next_tier}'."
                            )
                            current_tier = next_tier
                            continue
                        print(
                            f"[Cluster {cluster['id']}] WARNING: output "
                            f"({actual_size/1e6:.1f} MB) > source "
                            f"({source_size/1e6:.1f} MB) at lowest tier; skipping cluster."
                        )
                        break
            except BatchOutputLargerThanSourceError as exc:
                next_tier = _next_lower_tier(current_tier)
                # next_tier is always non-None here because we only set the size limit
                # when a lower tier exists (see above).
                print(
                    f"[Cluster {cluster['id']}] {exc}; "
                    f"retrying at lower tier '{next_tier}'."
                )
                if exc.baseline_file:
                    batch_baseline_file = exc.baseline_file
                    batch_baseline_tmp = exc.baseline_tmp
                if exc.segments:
                    batch_segments = exc.segments
                    batch_segments_tmpdir = exc.segments_tmpdir
                if exc.rows:
                    batch_rows = exc.rows
                current_tier = next_tier  # type: ignore[assignment]
                continue
            except Exception as exc:
                print(f"[Cluster {cluster['id']}] ERROR: representative failed: {exc}")
                cluster_dest = None
                cluster_qp = None
                break
            break

        if batch_baseline_tmp:
            shutil.rmtree(batch_baseline_tmp, ignore_errors=True)
        if batch_segments_tmpdir:
            shutil.rmtree(batch_segments_tmpdir, ignore_errors=True)

        if cluster_dest is None or cluster_qp is None:
            continue
        processed += 1
        if cluster_qp < 0:
            recovered_qp = _read_qp_sidecar(cluster_dest) if cluster_dest else None
            if recovered_qp is not None:
                print(
                    f"[Cluster {cluster['id']}] Representative output already exists "
                    f"(QP={recovered_qp} from previous run). Applying to peers."
                )
                cluster_qp = recovered_qp
            else:
                print(
                    f"[Cluster {cluster['id']}] Representative output already exists; "
                    "no QP record found. Skipping peers."
                )
                continue
        print(f"[Cluster {cluster['id']}] Learned QP={cluster_qp}. Applying to peers.")

        for member in members:
            member_path = str(member["path"])
            if _same_file(member_path, rep_path):
                continue
            print(f"[Cluster {cluster['id']}] Re-encoding with fixed QP={cluster_qp}: {member_path}")
            member_config = dataclasses.replace(
                config,
                input=member_path,
                batch_input_dir=config.input,
                source_quality=None,
                initial_qp=int(cluster_qp),
                skip_baseline=True,
                batch_force_fixed_qp_mode=True,
            )
            try:
                _member_dest, _member_qp = _run_single_file_pipeline(member_config)
                if (
                    config.batch_size_guard
                    and _member_qp >= 0  # not a skip sentinel
                    and _member_dest
                    and os.path.exists(_member_dest)
                ):
                    actual_size = os.path.getsize(_member_dest)
                    source_size = os.path.getsize(member_path)
                    if actual_size > source_size:
                        print(
                            f"[Cluster {cluster['id']}] WARNING: member output "
                            f"({actual_size/1e6:.1f} MB) > source "
                            f"({source_size/1e6:.1f} MB); discarding: {member_path}"
                        )
                        try:
                            os.remove(_member_dest)
                        except OSError:
                            pass
                        continue
                processed += 1
            except Exception as exc:
                print(f"[Cluster {cluster['id']}] ERROR: file failed: {exc}")

    _print_batch_excluded_summary(
        excluded=excluded,
        root=config.input,
        reason=f"source codec matches target codec '{target_codec}'",
    )
    print(f"\nBatch mode complete. Processed {processed} file(s).")


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    config.video_codec = normalize_video_codec(config.video_codec)
    config.skip_baseline_requested = bool(config.skip_baseline)

    # ------------------------------------------------------------
    # Logging setup
    # ------------------------------------------------------------
    setup_logging(config.verbose, config.log_file)

    if config.use_baseline_as_source and config.skip_baseline:
        print("WARNING: --use-baseline-as-source requires a baseline; ignoring --skip-baseline.")
        config.skip_baseline = False

    def _is_same_or_subpath(child: str, parent: str) -> bool:
        child_r = os.path.realpath(child)
        parent_r = os.path.realpath(parent)
        return child_r == parent_r or child_r.startswith(parent_r + os.sep)

    if os.path.isdir(config.input):
        config.batch_mode = True
        if config.output is None:
            parser.error("batch mode requires an explicit output directory")
        if _is_same_or_subpath(config.output, config.input):
            parser.error("output directory must not be the same as or inside the input directory")
        if _is_same_or_subpath(config.input, config.output):
            parser.error("input directory must not be inside the output directory")
        os.makedirs(config.output, exist_ok=True)
        try:
            _run_batch_auto(config)
        except ValueError as exc:
            print(f"ERROR: {exc}")
        return

    if config.output is not None:
        if os.path.realpath(config.output) == os.path.realpath(config.input):
            parser.error("output path must differ from input path")

    if not os.path.isfile(config.input):
        logging.error("Input file not found: %s", config.input)
        return

    _print_source_profile(config.input)

    # ------------------------------------------------------------
    # Audio-only path: copy video, process audio, skip SSIM pipeline
    # ------------------------------------------------------------
    if _maybe_run_audio_only_path(config):
        return

    try:
        _run_single_file_pipeline(config)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return
    except RuntimeError:
        return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
