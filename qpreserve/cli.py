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
import sys
import logging
import os
import tempfile
import shutil
from typing import Any, Dict, List, cast

from .probes import (
    probe_audio_streams,
    probe_video_framerate,
    probe_video_stream_info,
    detect_hdr,
)
from .utils import (
    build_audio_options,
    setup_logging,
    has_filter,
)
from .sampling import extract_samples
from .ssim_search import find_best_qp
from .encoder import encode_final, encode_baseline

def _print_h264_problems(problems: list[str]) -> None:
    print("The source video may not be safely encodable with H.264/NVENC:")
    for problem in problems:
        print("  ", problem)
    print()


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


def _resolve_scratch_root(scratch_dir: str | None) -> str:
    scratch_root = scratch_dir or os.environ.get("SSIM_SCRATCH_DIR")
    if scratch_root:
        os.makedirs(scratch_root, exist_ok=True)
    return scratch_root or tempfile.gettempdir()


def _warn_if_scratch_space_low(
    scratch_root: str, input_path: str, skip_baseline: bool
) -> None:
    # Warn if scratch space looks too small.
    # Baseline+final+samples ~2.5x input; skip-baseline reduces temp usage.
    try:
        input_size = os.path.getsize(input_path)
        multiplier = 1.5 if skip_baseline else 2.5
        needed = int(input_size * multiplier)
        free = shutil.disk_usage(scratch_root).free
        if free < needed:
            print(
                f"WARNING: scratch dir '{scratch_root}' has {free/1e9:.1f} GB free; "
                f"estimated need ~{needed/1e9:.1f} GB. "
                "Use --scratch-dir to point to a disk-backed location."
            )
    except OSError:
        pass


def _validate_skip_baseline(input_path: str, pix_fmt: str) -> bool:
    hdr_info: Dict[str, Any] = detect_hdr(input_path)
    warnings: List[str] = []

    # Check HDR
    if hdr_info["is_hdr"]:
        warnings.append("HDR detected (requires HDR→SDR tonemapping)")

    # Check color space
    if hdr_info["primaries"] and hdr_info["primaries"] != "bt709":
        warnings.append(f"Color primaries: {hdr_info['primaries']} (expected bt709)")
    if hdr_info["transfer"] and hdr_info["transfer"] != "bt709":
        warnings.append(f"Color transfer: {hdr_info['transfer']} (expected bt709)")
    if hdr_info["matrix"] and hdr_info["matrix"] != "bt709":
        warnings.append(f"Color space: {hdr_info['matrix']} (expected bt709)")

    # Check pixel format
    if pix_fmt != "yuv420p":
        warnings.append(f"Pixel format: {pix_fmt} (expected yuv420p)")

    if not warnings:
        return True

    print("WARNING: --skip-baseline ignored due to source incompatibilities:")
    for warning in warnings:
        print(f"  - {warning}")
    print("Baseline generation required for normalization.")
    return False


def _determine_baseline_file(
    input_path: str,
    skip_baseline: bool,
    pix_fmt: str,
    baseline_tmp: str,
    baseline_qp: int,
) -> str:
    if skip_baseline and _validate_skip_baseline(input_path, pix_fmt):
        print("Skipping baseline generation; using source file directly.")
        return input_path

    baseline_file = encode_baseline(input_path, output_dir=baseline_tmp, qp=baseline_qp)
    print(f"Baseline file created: {baseline_file}")
    return baseline_file


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Optimize video quality via perceptual metric and QP binary search.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('input', help='Source video file')

    # Quality targets
    parser.add_argument('--ssim', type=float, default=0.985,
                        help='Target SSIM threshold for acceptance.')

    # QP search bounds
    parser.add_argument('--min-qp', type=int, default=23,
                        help='Minimum QP allowed during search.')
    parser.add_argument('--max-qp', type=int, default=37,
                        help='Maximum QP allowed during search.')

    # Sampling configuration
    parser.add_argument('--sample-percent', type=float, default=15,
                        help='Percentage of video duration used for sampling.')
    parser.add_argument('--sample-count', type=int, default=3,
                        help='How many sample clips to extract.')
    parser.add_argument('--sample-qp', type=int, default=15,
                        help='QP used to encode sample clips.')
    parser.add_argument(
        '--initial-qp',
        type=int,
        help='Skip sampling and start full-file SSIM descent from this QP.'
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
        '--skip-full-ssim',
        action='store_true',
        help='Skip full-file SSIM verification step (use sample-based QP only).'
    )
    parser.add_argument(
        '--no-audio-normalize',
        action='store_false',
        dest='audio_normalize',
        help='Disable audio loudness normalization (loudnorm).'
    )
    parser.add_argument(
        '--full-ssim-chunk-seconds',
        type=float,
        default=300.0,
        help='Split full-file SSIM into chunks to reduce RAM use (0 disables).'
    )
    parser.add_argument(
        '--skip-baseline',
        action='store_true',
        help='Skip baseline generation and use source directly. Only recommended for sources '
             'that are already SDR, yuv420p, and BT.709. Ignored for HDR sources.'
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
        default=15,
        help='QP used to generate the baseline file.'
    )

    return parser


def _probe_video_basics(input_path: str) -> tuple[int, int, str]:
    vinfo: Dict[str, Any] = probe_video_stream_info(input_path)
    width: int = int(vinfo['width'])
    height: int = int(vinfo['height'])
    pix_fmt: str = str(vinfo['pix_fmt'])
    return width, height, pix_fmt


def _prepare_audio_and_framerate(
    input_path: str, audio_normalize: bool
) -> tuple[List[str], float, bool]:
    streams: List[Dict[str, Any]] = probe_audio_streams(input_path)
    normalize_enabled = audio_normalize
    if normalize_enabled and not has_filter('loudnorm'):
        print("WARNING: loudnorm filter not available; disabling audio normalization.")
        normalize_enabled = False
    audio_opts: List[str] = build_audio_options(streams, normalize=normalize_enabled)
    raw_fr: float = probe_video_framerate(input_path)
    return audio_opts, raw_fr, normalize_enabled


def _calculate_gop(raw_fr: float) -> int:
    return max(1, int(round(raw_fr / 2)))


def _select_best_qp(
    args: argparse.Namespace,
    baseline_file: str,
    audio_opts: List[str],
    raw_fr: float,
    gop: int,
    scratch_root: str,
) -> tuple[int, str | None]:
    if args.initial_qp is not None:
        if args.initial_qp < args.min_qp:
            print(
                f"WARNING: --initial-qp {args.initial_qp} is below --min-qp {args.min_qp}; "
                "lowering min-qp to match."
            )
            args.min_qp = args.initial_qp
        if args.initial_qp > args.max_qp:
            print(
                f"WARNING: --initial-qp {args.initial_qp} is above --max-qp {args.max_qp}; "
                "sampling bounds will be ignored."
            )
        best_qp = args.initial_qp
        print(f"Skipping sampling; starting full-file SSIM descent at QP={best_qp}.")
        return best_qp, None

    samples, samples_tmpdir = extract_samples(
        baseline_file,
        percent=args.sample_percent,
        count=args.sample_count,
        sample_qp=args.sample_qp,
        audio_opts=audio_opts,
        raw_fr=raw_fr,
        sampling_mode=args.sampling_mode,
        tmp_root=scratch_root,
    )
    if not samples_tmpdir:
        samples_tmpdir = None

    # NOTE: metric_effective is always 'ssim' for now.
    try:
        is_source_ref = os.path.samefile(baseline_file, args.input)
    except OSError:
        is_source_ref = baseline_file == args.input
    ref_label = "source" if is_source_ref else "baseline"
    print(f"Starting sample SSIM checks against {ref_label}...")
    best_qp = find_best_qp(
        samples=samples,
        min_qp=args.min_qp,
        max_qp=args.max_qp,
        target_ssim=args.ssim,
        metric=args.metric,
        audio_opts=audio_opts,
        raw_fr=raw_fr,
        gop=gop
    )
    return best_qp, samples_tmpdir


def _run_final_encode(
    args: argparse.Namespace,
    baseline_file: str,
    best_qp: int,
    audio_opts: List[str],
    raw_fr: float,
    gop: int,
    tmpdir: str,
) -> tuple[str, int]:
    source_base, source_ext = os.path.splitext(os.path.basename(args.input))

    if args.skip_full_ssim:
        if args.initial_qp is not None:
            print("WARNING: --skip-full-ssim overrides full-file verification.")
        print("\nSkipping full-file SSIM verification; encoding once at selected QP.")
        final_file = cast(
            str,
            encode_final(
                input_file=baseline_file,
                qp=best_qp,
                audio_opts=audio_opts,
                raw_fr=raw_fr,
                gop=gop,
                return_ssim=False,
                output_dir=tmpdir,
                output_base=source_base,
                output_ext=source_ext,
            ),
        )
        return final_file, best_qp

    return _encode_with_full_ssim(
        baseline_file=baseline_file,
        best_qp=best_qp,
        min_qp=args.min_qp,
        target_ssim=args.ssim,
        audio_opts=audio_opts,
        raw_fr=raw_fr,
        gop=gop,
        ssim_chunk_seconds=args.full_ssim_chunk_seconds,
        tmpdir=tmpdir,
        source_base=source_base,
        source_ext=source_ext,
    )


def _encode_with_full_ssim(
    baseline_file: str,
    best_qp: int,
    min_qp: int,
    target_ssim: float,
    audio_opts: List[str],
    raw_fr: float,
    gop: int,
    ssim_chunk_seconds: float | None,
    tmpdir: str,
    source_base: str,
    source_ext: str,
) -> tuple[str, int]:
    prev_file: str | None = None
    final_file: str | None = None
    for qp in range(best_qp, min_qp - 1, -1):
        print(f"\nChecking full-file quality at QP={qp}...")

        # Clean up older intermediate file
        if prev_file and os.path.exists(prev_file):
            os.remove(prev_file)

        # Encode full file (this has its own real FFmpeg progress bar)
        output_path, ssim_val = cast(
            tuple[str, float | None],
            encode_final(
                input_file=baseline_file,
                qp=qp,
                audio_opts=audio_opts,
                raw_fr=raw_fr,
                gop=gop,
                return_ssim=True,
                ssim_chunk_seconds=ssim_chunk_seconds,
                output_dir=tmpdir,
                output_base=source_base,
                output_ext=source_ext,
            ),
        )
        # output_path is str, ssim_val is Optional[float]
        output_path_str: str = output_path
        ssim_val_opt: float | None = ssim_val
        if ssim_val_opt is None:
            logging.warning("Full-file SSIM unavailable at QP=%d; keeping this encode.", qp)
            return output_path_str, qp

        print(f"  → SSIM={ssim_val_opt:.4f}")

        # Round to 4 decimals before comparison to match displayed value
        ssim_rounded = round(ssim_val_opt, 4)
        if ssim_rounded >= target_ssim:
            print(f"  ✓ Meets target SSIM ≥ {target_ssim}; accepting QP={qp}")
            return output_path_str, qp

        print("  ✗ Below target; trying lower QP...")
        prev_file = output_path_str

    logging.warning("Could not meet SSIM target; using sample-based QP=%d", best_qp)
    final_file = prev_file or cast(
        str,
        encode_final(
            input_file=baseline_file,
            qp=best_qp,
            audio_opts=audio_opts,
            raw_fr=raw_fr,
            gop=gop,
            return_ssim=False,
            output_dir=tmpdir,
            output_base=source_base,
            output_ext=source_ext,
        ),
    )
    return final_file, best_qp


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # ------------------------------------------------------------
    # Logging setup
    # ------------------------------------------------------------
    setup_logging(args.verbose, args.log_file)

    if not os.path.isfile(args.input):
        logging.error("Input file not found: %s", args.input)
        return

    # ------------------------------------------------------------
    # Probe basic video info (for impossible-encode checks)
    # ------------------------------------------------------------
    width, height, pix_fmt = _probe_video_basics(args.input)

    if not _confirm_h264_compat(width, height, pix_fmt, decision=args.h264_compat):
        return

    # ------------------------------------------------------------
    # Check for SSIM filter availability
    # ------------------------------------------------------------
    if not has_filter('ssim'):
        print("ERROR: ssim filter not available in ffmpeg. Cannot measure quality.")
        return

    print("Quality metric in use: SSIM")

    # ------------------------------------------------------------
    # Probe audio/video from original file
    # ------------------------------------------------------------
    audio_opts, raw_fr, args.audio_normalize = _prepare_audio_and_framerate(
        args.input, args.audio_normalize
    )

    # ------------------------------------------------------------
    # Prepare temp directories
    # ------------------------------------------------------------
    scratch_root = _resolve_scratch_root(args.scratch_dir)
    _warn_if_scratch_space_low(scratch_root, args.input, args.skip_baseline)

    baseline_tmp = tempfile.mkdtemp(prefix="ssim_baseline_", dir=scratch_root)
    tmpdir = tempfile.mkdtemp(prefix="ssim_final_", dir=scratch_root)
    print(f"Scratch directory: {scratch_root}")
    samples_tmpdir: str | None = None

    try:
        # ------------------------------------------------------------
        # STEP 1 — BASELINE (QP=baseline) ENCODE with PROGRESS (+ optional HDR->SDR)
        # ------------------------------------------------------------
        # Check if we should skip baseline (only if source meets requirements)
        baseline_file = _determine_baseline_file(
            input_path=args.input,
            skip_baseline=args.skip_baseline,
            pix_fmt=pix_fmt,
            baseline_tmp=baseline_tmp,
            baseline_qp=args.baseline_qp,
        )

        # ------------------------------------------------------------
        # GOP ~ half framerate
        gop = _calculate_gop(raw_fr)

        # ------------------------------------------------------------
        # STEP 2 — SAMPLE CLIP EXTRACTION (optional)
        # ------------------------------------------------------------
        best_qp, samples_tmpdir = _select_best_qp(
            args=args,
            baseline_file=baseline_file,
            audio_opts=audio_opts,
            raw_fr=raw_fr,
            gop=gop,
            scratch_root=scratch_root,
        )

        # ------------------------------------------------------------
        # STEP 4 — FULL-FILE FINAL ENCODE DESCENT (CLEANER OUTPUT)
        # ------------------------------------------------------------

        final_file, final_qp = _run_final_encode(
            args=args,
            baseline_file=baseline_file,
            best_qp=best_qp,
            audio_opts=audio_opts,
            raw_fr=raw_fr,
            gop=gop,
            tmpdir=tmpdir,
        )

        # ------------------------------------------------------------
        # STEP 5 — MOVE FINAL RESULT TO SOURCE DIRECTORY
        # ------------------------------------------------------------
        final_path: str = final_file
        dest_name: str = os.path.basename(final_path)
        dest: str = os.path.join(os.path.dirname(args.input), dest_name)

        shutil.move(final_path, dest)
        print(f"Optimized file: {dest} (QP={final_qp})")

    finally:
        # Cleanup temporary dirs
        if samples_tmpdir:
            shutil.rmtree(samples_tmpdir, ignore_errors=True)
        shutil.rmtree(tmpdir, ignore_errors=True)
        shutil.rmtree(baseline_tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
