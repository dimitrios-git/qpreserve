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
from statistics import mean
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
    run_cmd,
    run_ffmpeg_progress,
    setup_logging,
    has_filter,
)
from .sampling import extract_samples, extract_sample_segments
from .ssim_search import find_best_qp, measure_ssim
from .encoder import encode_final, encode_baseline

def _normalize_video_codec(video_codec: str) -> str:
    codec = (video_codec or "h264").lower()
    if codec == "hevc":
        return "h265"
    return codec


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
    extra_vf: str | None = None,
    video_codec: str = "h264",
) -> str:
    if skip_baseline and _validate_skip_baseline(input_path, pix_fmt):
        print("Skipping baseline generation; using source file directly.")
        return input_path

    baseline_file = encode_baseline(
        input_path,
        output_dir=baseline_tmp,
        qp=baseline_qp,
        extra_vf=extra_vf,
        video_codec=video_codec,
    )
    print(f"Baseline file created: {baseline_file}")
    return baseline_file


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Optimize video quality via perceptual metric and QP binary search.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('input', help='Source video file')
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
    parser.add_argument('--min-qp', type=int, default=13,
                        help='Minimum QP allowed during search.')
    parser.add_argument('--max-qp', type=int, default=40,
                        help='Maximum QP allowed during search.')

    # Sampling configuration
    parser.add_argument('--sample-percent', type=float, default=15,
                        help='Percentage of video duration used for sampling.')
    parser.add_argument('--sample-count', type=int, default=4,
                        help='How many sample clips to extract.')
    parser.add_argument('--sample-qp', type=int, default=13,
                        help='QP used to encode sample clips.')
    parser.add_argument(
        '--initial-qp',
        type=int,
        help='Skip sampling and start full-file SSIM descent from this QP.'
    )
    parser.add_argument(
        '--source-quality',
        type=str,
        help='Enable expected-QP mode. Accepts a numeric QP start (e.g. 22) '
             'or one of: ultra, high, medium, low. '
             'Guide: ultra=near-lossless, high=clean, medium=compressed, low=heavily compressed.'
    )
    parser.add_argument(
        '--expected-qp',
        type=int,
        dest='expected_qp_alias',
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        '--expected-min-gain',
        type=float,
        default=1.0,
        help='Expected-QP mode: minimum per-step size gain percent considered meaningful.'
    )
    parser.add_argument(
        '--expected-max-steps',
        type=int,
        default=10,
        help='Expected-QP mode: maximum number of QP steps to explore upward.'
    )
    parser.add_argument(
        '--expected-knee-ratio',
        type=float,
        default=1.5,
        help='Expected-QP mode: velocity jump ratio used to detect the knee point.'
    )
    parser.add_argument(
        '--expected-choice',
        choices=['prompt', 'safe', 'balanced'],
        default='prompt',
        help='Expected-QP mode selection behavior.'
    )

    parser.add_argument(
        '--sampling-mode',
        choices=['uniform', 'scene', 'motion'],
        default='motion',
        help='Sampling strategy for selecting sample start times.'
    )

    # Size-targeted defaults for modern codecs (HEVC/VP9/AV1)
    parser.add_argument('--size-delta-hevc', type=float, default=0.50,
                        help='Target size delta for HEVC/H.265 sources (example: 0.50 = +50 pct).')
    parser.add_argument('--size-delta-vp9', type=float, default=0.50,
                        help='Target size delta for VP9 sources (example: 0.50 = +50 pct).')
    parser.add_argument('--size-delta-av1', type=float, default=0.60,
                        help='Target size delta for AV1 sources (example: 0.60 = +60 pct).')
    parser.add_argument('--size-tolerance', type=float, default=0.03,
                        help='Allowed size error around target (example: 0.03 = +/-3 pct).')
    parser.add_argument('--h264-compat-ssim', type=float, default=0.996,
                        help='SSIM target used after size-targeting for modern codecs.')

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
        '--skip-baseline',
        action='store_true',
        help='Skip baseline generation and use source directly. Only recommended for sources '
             'that are already SDR, yuv420p, and BT.709. Ignored for HDR sources.'
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
        default=13,
        help='QP used to generate the baseline file.'
    )
    parser.add_argument(
        '--auto-crop',
        choices=['off', 'prompt', 'force'],
        default='prompt',
        help='Detect small vertical overscan (e.g., 1920x1088) and crop to a standard height. '
             '"prompt" asks when a TTY is available; "force" applies automatically.'
    )

    return parser


def _probe_video_basics(input_path: str) -> tuple[int, int, str]:
    vinfo: Dict[str, Any] = probe_video_stream_info(input_path)
    width: int = int(vinfo['width'])
    height: int = int(vinfo['height'])
    pix_fmt: str = str(vinfo['pix_fmt'])
    return width, height, pix_fmt


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
        '-i', input_path,
        '-map', '0:v?', '-map', '0:a?', '-map', '0:s?', '-map_metadata', '0',
    ] + audio_opts + ['-c:v', 'copy', '-c:s', 'copy', output_path]

    duration = probe_video_duration(input_path)
    if duration > 0:
        run_ffmpeg_progress(cmd, duration, desc="Audio-only processing")
    else:
        run_cmd(cmd)

    return output_path


def _maybe_run_audio_only_path(args: argparse.Namespace) -> bool:
    if not args.add_stereo_downmix_copy_video:
        return False
    args.add_stereo_downmix = True
    normalize_enabled = args.audio_normalize
    if normalize_enabled and not has_filter('loudnorm'):
        print("WARNING: loudnorm filter not available; disabling audio normalization.")
        normalize_enabled = False

    streams = probe_audio_streams(args.input)
    audio_opts = build_audio_options(
        streams,
        normalize=normalize_enabled,
        add_stereo_downmix=True,
    )

    scratch_root = _resolve_scratch_root(args.scratch_dir)
    tmpdir = tempfile.mkdtemp(prefix="audio_copy_", dir=scratch_root)
    print(f"Scratch directory: {scratch_root}")

    output_path = _run_audio_only_copy_video(
        input_path=args.input,
        audio_opts=audio_opts,
        output_dir=tmpdir,
    )

    dest_name = os.path.basename(output_path)
    dest = os.path.join(os.path.dirname(args.input), dest_name)
    shutil.move(output_path, dest)
    print(f"Output file created: {dest}")
    return True


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
    args: argparse.Namespace,
    baseline_file: str,
) -> str:
    skip_baseline_forced_off = args.skip_baseline and not _same_file(baseline_file, args.input)
    if skip_baseline_forced_off and not args.use_baseline_as_source:
        if _prompt_use_baseline_as_source():
            args.use_baseline_as_source = True
        else:
            print("Tip: use --use-baseline-as-source to treat the baseline as the source.")
    return baseline_file if args.use_baseline_as_source else args.input


def _prepare_work_dirs(
    scratch_root: str,
    input_path: str,
    skip_baseline: bool,
) -> tuple[str, str, str | None]:
    _warn_if_scratch_space_low(scratch_root, input_path, skip_baseline)
    baseline_tmp = tempfile.mkdtemp(prefix="ssim_baseline_", dir=scratch_root)
    tmpdir = tempfile.mkdtemp(prefix="ssim_final_", dir=scratch_root)
    print(f"Scratch directory: {scratch_root}")
    return baseline_tmp, tmpdir, None


def _is_modern_codec(codec: str) -> bool:
    return codec in {"hevc", "h265", "vp9", "av1"}


def _resolve_size_delta(args: argparse.Namespace, codec: str) -> float:
    if codec in {"hevc", "h265"}:
        return args.size_delta_hevc
    if codec == "vp9":
        return args.size_delta_vp9
    if codec == "av1":
        return args.size_delta_av1
    return 0.0


def _estimate_full_size_from_samples(sample_bytes: int, percent: float) -> int:
    if percent <= 0:
        return sample_bytes
    return int(sample_bytes * (100.0 / percent))


def _nvenc_encoder_for(video_codec: str) -> str:
    codec = _normalize_video_codec(video_codec)
    return "hevc_nvenc" if codec == "h265" else "h264_nvenc"


def _nvenc_pix_fmt_for(video_codec: str) -> str:
    codec = _normalize_video_codec(video_codec)
    return "p010le" if codec == "h265" else "yuv420p"


def _encode_samples_for_qp(
    segments: list[str],
    qp: int,
    audio_opts: List[str],
    raw_fr: float,
    gop: int,
    tmp_root: str,
    video_codec: str,
) -> int:
    if not segments:
        return 0

    tmpdir = tempfile.mkdtemp(prefix=f"size_qp{qp}_", dir=tmp_root)
    total_bytes = 0
    nvenc_encoder = _nvenc_encoder_for(video_codec)
    pix_fmt = _nvenc_pix_fmt_for(video_codec)
    try:
        for idx, seg in enumerate(segments):
            ext = os.path.splitext(seg)[1]
            sample_file = os.path.join(tmpdir, f"sample_{idx}{ext}")
            run_cmd([
                'ffmpeg', '-y', '-hwaccel', 'cuda', '-i', seg,
                '-map', '0:v', '-map', '0:a?', '-map', '0:s?', '-map_metadata', '0',
                '-r', str(raw_fr), '-g', str(gop),
                '-bf', '2', '-pix_fmt', pix_fmt, '-c:v', nvenc_encoder,
                '-preset', 'p7', '-rc', 'constqp', '-qp', str(qp)
            ] + audio_opts + ['-c:s', 'copy', sample_file])
            try:
                total_bytes += os.path.getsize(sample_file)
            except OSError:
                pass
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return total_bytes


def _maybe_use_initial_qp_size(args: argparse.Namespace) -> int | None:
    if args.initial_qp is None:
        return None
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
    print(f"Skipping sampling; starting size targeting at QP={args.initial_qp}.")
    return args.initial_qp


def _prepare_size_segments(
    baseline_file: str,
    args: argparse.Namespace,
    scratch_root: str,
) -> tuple[list[str], str | None]:
    segments, segments_tmpdir, _, _ = extract_sample_segments(
        baseline_file,
        percent=args.sample_percent,
        count=args.sample_count,
        sampling_mode=args.sampling_mode,
        tmp_root=scratch_root,
    )
    if not segments_tmpdir:
        segments_tmpdir = None
    return segments, segments_tmpdir


def _estimate_full_size_for_qp(
    qp: int,
    sizes_by_qp: Dict[int, int],
    segments: list[str],
    audio_opts: List[str],
    raw_fr: float,
    gop: int,
    scratch_root: str,
    sample_percent: float,
    video_codec: str,
) -> int:
    if qp in sizes_by_qp:
        return sizes_by_qp[qp]
    sample_bytes = _encode_samples_for_qp(
        segments=segments,
        qp=qp,
        audio_opts=audio_opts,
        raw_fr=raw_fr,
        gop=gop,
        tmp_root=scratch_root,
        video_codec=video_codec,
    )
    est = _estimate_full_size_from_samples(sample_bytes, sample_percent)
    sizes_by_qp[qp] = est
    print(f"  Sample size @ QP={qp}: ~{est/1e6:.1f} MB (estimated)")
    return est


def _select_best_qp_size(
    args: argparse.Namespace,
    baseline_file: str,
    audio_opts: List[str],
    raw_fr: float,
    gop: int,
    scratch_root: str,
    target_bytes: int,
    tolerance: float,
    video_codec: str,
) -> tuple[int, str | None, list[str]]:
    initial_qp = _maybe_use_initial_qp_size(args)
    if initial_qp is not None:
        return initial_qp, None, []

    segments, segments_tmpdir = _prepare_size_segments(
        baseline_file=baseline_file,
        args=args,
        scratch_root=scratch_root,
    )

    if not segments:
        print("WARNING: could not extract samples; using min QP.")
        return args.min_qp, segments_tmpdir, []

    sizes_by_qp: Dict[int, int] = {}

    size_min = _estimate_full_size_for_qp(
        qp=args.min_qp,
        sizes_by_qp=sizes_by_qp,
        segments=segments,
        audio_opts=audio_opts,
        raw_fr=raw_fr,
        gop=gop,
        scratch_root=scratch_root,
        sample_percent=args.sample_percent,
        video_codec=video_codec,
    )
    size_max = _estimate_full_size_for_qp(
        qp=args.max_qp,
        sizes_by_qp=sizes_by_qp,
        segments=segments,
        audio_opts=audio_opts,
        raw_fr=raw_fr,
        gop=gop,
        scratch_root=scratch_root,
        sample_percent=args.sample_percent,
        video_codec=video_codec,
    )
    print(f"Target size: {target_bytes/1e6:.1f} MB")

    if target_bytes >= size_min:
        return args.min_qp, segments_tmpdir, []
    if target_bytes <= size_max:
        return args.max_qp, segments_tmpdir, []

    low, high = args.min_qp, args.max_qp
    best_qp = low
    best_diff = abs(size_min - target_bytes)

    while low <= high:
        mid = (low + high) // 2
        est = _estimate_full_size_for_qp(
            qp=mid,
            sizes_by_qp=sizes_by_qp,
            segments=segments,
            audio_opts=audio_opts,
            raw_fr=raw_fr,
            gop=gop,
            scratch_root=scratch_root,
            sample_percent=args.sample_percent,
            video_codec=video_codec,
        )
        diff = abs(est - target_bytes)
        print(
            f"  Sample check QP={mid}: est ~{est/1e6:.1f} MB "
            f"(target {target_bytes/1e6:.1f} MB)"
        )
        if diff < best_diff:
            best_qp = mid
            best_diff = diff
        if target_bytes > 0 and (diff / target_bytes) <= tolerance:
            return mid, segments_tmpdir, []
        if est > target_bytes:
            low = mid + 1
        else:
            high = mid - 1

    return best_qp, segments_tmpdir, []


def _remove_if_not_best(path: str, best_path: str) -> None:
    if not path or path == best_path:
        return
    try:
        os.remove(path)
    except OSError:
        pass


def _encode_final_for_size(
    baseline_file: str,
    qp: int,
    audio_opts: List[str],
    raw_fr: float,
    gop: int,
    tmpdir: str,
    source_base: str,
    source_ext: str,
    video_codec: str,
) -> str:
    return cast(
        str,
        encode_final(
            input_file=baseline_file,
            qp=qp,
            audio_opts=audio_opts,
            raw_fr=raw_fr,
            gop=gop,
            video_codec=video_codec,
            return_ssim=False,
            output_dir=tmpdir,
            output_base=source_base,
            output_ext=source_ext,
        ),
    )


def _safe_file_size(path: str) -> int | None:
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def _update_best_size_candidate(
    output_path: str,
    current_qp: int,
    last_size: int,
    target_bytes: int,
    best_file: str,
    best_qp: int,
    best_diff: float | None,
) -> tuple[str, int, float]:
    diff = abs(last_size - target_bytes)
    if best_diff is None or diff < best_diff:
        _remove_if_not_best(best_file, output_path)
        return output_path, current_qp, diff
    return best_file, best_qp, best_diff


def _next_qp_for_size(current_qp: int, last_size: int, target_bytes: int) -> int:
    if last_size > target_bytes:
        print("  Result too large → increasing QP")
        return current_qp + 1
    print("  Result too small → decreasing QP")
    return current_qp - 1


def _size_search_exhausted(
    current_qp: int,
    args: argparse.Namespace,
    last_size: int,
    target_bytes: int,
) -> bool:
    if target_bytes <= 0:
        return True
    if last_size > target_bytes and current_qp < args.min_qp:
        return True
    if last_size < target_bytes and current_qp > args.max_qp:
        return True
    return False


def _run_final_encode_size(
    args: argparse.Namespace,
    baseline_file: str,
    source_ref_path: str,
    best_qp: int,
    audio_opts: List[str],
    raw_fr: float,
    gop: int,
    tmpdir: str,
    target_bytes: int,
    tolerance: float,
    video_codec: str,
) -> tuple[str, int, None]:
    source_base, source_ext = os.path.splitext(os.path.basename(source_ref_path))

    current_qp = best_qp
    last_file = ""
    last_size = None
    last_qp = current_qp
    best_file = ""
    best_qp = current_qp
    best_diff = None
    seen_qps: set[int] = set()

    while args.min_qp <= current_qp <= args.max_qp:
        if current_qp in seen_qps:
            break
        seen_qps.add(current_qp)

        _remove_if_not_best(last_file, best_file)

        output_path = _encode_final_for_size(
            baseline_file=baseline_file,
            qp=current_qp,
            audio_opts=audio_opts,
            raw_fr=raw_fr,
            gop=gop,
            tmpdir=tmpdir,
            source_base=source_base,
            source_ext=source_ext,
            video_codec=video_codec,
        )
        last_file = output_path
        last_qp = current_qp
        last_size = _safe_file_size(output_path)

        if last_size is not None:
            print(
                f"  Full-file size @ QP={current_qp}: {last_size/1e6:.1f} MB "
                f"(target {target_bytes/1e6:.1f} MB)"
            )

        if last_size is None or target_bytes <= 0:
            return output_path, current_qp, None

        best_file, best_qp, best_diff = _update_best_size_candidate(
            output_path=output_path,
            current_qp=current_qp,
            last_size=last_size,
            target_bytes=target_bytes,
            best_file=best_file,
            best_qp=best_qp,
            best_diff=best_diff,
        )

        if (best_diff / target_bytes) <= tolerance:
            return output_path, current_qp, None

        current_qp = _next_qp_for_size(current_qp, last_size, target_bytes)

        if _size_search_exhausted(current_qp, args, last_size, target_bytes):
            break

    if best_file:
        return best_file, best_qp, None
    return last_file, last_qp, None


def _select_best_qp(
    args: argparse.Namespace,
    baseline_file: str,
    source_ref_path: str,
    audio_opts: List[str],
    raw_fr: float,
    gop: int,
    scratch_root: str,
) -> tuple[int, str | None, list[str]]:
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
        return best_qp, None, []

    samples, samples_tmpdir = extract_samples(
        baseline_file,
        percent=args.sample_percent,
        count=args.sample_count,
        sample_qp=args.sample_qp,
        audio_opts=audio_opts,
        raw_fr=raw_fr,
        video_codec=args.video_codec,
        sampling_mode=args.sampling_mode,
        tmp_root=scratch_root,
    )
    if not samples_tmpdir:
        samples_tmpdir = None

    # NOTE: metric_effective is always 'ssim' for now.
    is_source_ref = _same_file(baseline_file, source_ref_path)
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
        gop=gop,
        video_codec=args.video_codec,
    )
    return best_qp, samples_tmpdir, samples


def _run_final_encode(
    args: argparse.Namespace,
    baseline_file: str,
    source_ref_path: str,
    best_qp: int,
    audio_opts: List[str],
    raw_fr: float,
    gop: int,
    tmpdir: str,
    video_codec: str,
) -> tuple[str, int, float | None]:
    source_base, source_ext = os.path.splitext(os.path.basename(source_ref_path))

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
                video_codec=video_codec,
                return_ssim=False,
                output_dir=tmpdir,
                output_base=source_base,
                output_ext=source_ext,
            ),
        )
        return final_file, best_qp, None

    return _encode_with_full_ssim(
        baseline_file=baseline_file,
        best_qp=best_qp,
        min_qp=args.min_qp,
        target_ssim=args.ssim,
        audio_opts=audio_opts,
        raw_fr=raw_fr,
        gop=gop,
        video_codec=video_codec,
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
    video_codec: str,
    ssim_chunk_seconds: float | None,
    tmpdir: str,
    source_base: str,
    source_ext: str,
) -> tuple[str, int, float | None]:
    ssim_by_qp: Dict[int, tuple[float, str]] = {}
    final_file: str | None = None
    for qp in range(best_qp, min_qp - 1, -1):
        print(f"\nChecking full-file quality at QP={qp}...")

        # Encode full file (this has its own real FFmpeg progress bar)
        output_path, ssim_val = cast(
            tuple[str, float | None],
            encode_final(
                input_file=baseline_file,
                qp=qp,
                audio_opts=audio_opts,
                raw_fr=raw_fr,
                gop=gop,
                video_codec=video_codec,
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
            return output_path_str, qp, None

        print(f"  → SSIM={ssim_val_opt:.4f}")
        ssim_by_qp[qp] = (ssim_val_opt, output_path_str)

        # Round to 4 decimals before comparison to match displayed value
        ssim_rounded = round(ssim_val_opt, 4)
        if ssim_rounded >= target_ssim:
            print(f"  ✓ Meets target SSIM ≥ {target_ssim}; accepting QP={qp}")
            return output_path_str, qp, ssim_val_opt

        print("  ✗ Below target; trying lower QP...")

    if ssim_by_qp:
        suggested_qp, avg_ssim = _suggest_qp_from_ssim(ssim_by_qp)
        if not sys.stdin.isatty():
            logging.warning(
                "Could not meet SSIM target; using QP=%d near average SSIM %.4f",
                suggested_qp,
                avg_ssim,
            )
        chosen_qp = _prompt_qp_choice(ssim_by_qp, suggested_qp, avg_ssim, target_ssim)
        chosen_file = ssim_by_qp.get(chosen_qp, (None, None))[1]
        if chosen_file and os.path.exists(chosen_file):
            return chosen_file, chosen_qp, ssim_by_qp[chosen_qp][0]
        final_file = cast(
            str,
            encode_final(
                input_file=baseline_file,
                qp=chosen_qp,
                audio_opts=audio_opts,
                raw_fr=raw_fr,
                gop=gop,
                video_codec=video_codec,
                return_ssim=False,
                output_dir=tmpdir,
                output_base=source_base,
                output_ext=source_ext,
            ),
        )
        return final_file, chosen_qp, None

    logging.warning("Could not meet SSIM target; using sample-based QP=%d", best_qp)
    final_file = cast(
        str,
        encode_final(
            input_file=baseline_file,
            qp=best_qp,
            audio_opts=audio_opts,
            raw_fr=raw_fr,
            gop=gop,
            video_codec=video_codec,
            return_ssim=False,
            output_dir=tmpdir,
            output_base=source_base,
            output_ext=source_ext,
        ),
    )
    return final_file, best_qp, None


def _resolve_crop_filter(args: argparse.Namespace, width: int, height: int) -> str | None:
    crop_candidate = _detect_vertical_overscan_crop(width, height)
    if not crop_candidate:
        return None
    expected = crop_candidate[0]
    apply_crop = False
    if args.auto_crop == "force":
        apply_crop = True
    elif args.auto_crop == "prompt":
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
    if args.skip_baseline:
        print("Auto-crop requires baseline generation; disabling --skip-baseline.")
        args.skip_baseline = False
    return crop_filter


def _resolve_quality_mode(
    args: argparse.Namespace,
    input_path: str,
    force_ssim: bool = False,
) -> tuple[str, bool] | None:
    codec = probe_video_codec(input_path)
    if force_ssim:
        if not has_filter('ssim'):
            print("ERROR: ssim filter not available in ffmpeg. Cannot measure quality.")
            return None
        print("Quality metric in use: SSIM (expected-QP mode)")
        return codec, False
    use_size_target = _is_modern_codec(codec)
    if use_size_target:
        delta = _resolve_size_delta(args, codec)
        print(
            "Quality metric in use: size target "
            f"(codec={codec}, delta={delta:.2f}, tolerance={args.size_tolerance:.2%})"
        )
        return codec, True
    if not has_filter('ssim'):
        print("ERROR: ssim filter not available in ffmpeg. Cannot measure quality.")
        return None
    print("Quality metric in use: SSIM")
    return codec, False


def _resolve_target_bytes(
    args: argparse.Namespace,
    input_path: str,
    codec: str,
) -> int:
    try:
        source_size = os.path.getsize(input_path)
    except OSError:
        source_size = 0
    delta = _resolve_size_delta(args, codec)
    return int(source_size * (1.0 + delta))


def _tier_for_bppf(codec: str, bppf: float) -> str:
    c = _normalize_video_codec(codec)
    if c == "h265":
        if bppf >= 0.120:
            return "ultra"
        if bppf >= 0.070:
            return "high"
        if bppf >= 0.024:
            return "medium"
        return "low"
    # h264 and other codecs: generally need more bppf for similar perceptual quality.
    if bppf >= 0.200:
        return "ultra"
    if bppf >= 0.120:
        return "high"
    if bppf >= 0.060:
        return "medium"
    return "low"


def _selected_quality_label(args: argparse.Namespace) -> str | None:
    if not args.source_quality:
        return None
    text = str(args.source_quality).strip().lower()
    if _quality_label_to_qp(text) is not None:
        return text.replace("_", "-").replace(" ", "-")
    return None


def _maybe_print_expected_mode_advice(
    args: argparse.Namespace,
    input_path: str,
    width: int,
    height: int,
    raw_fr: float,
) -> None:
    if not _expected_mode_requested(args):
        return
    duration = probe_video_duration(input_path)
    if duration <= 0 or raw_fr <= 0:
        return
    try:
        size_bytes = os.path.getsize(input_path)
    except OSError:
        size_bytes = 0
    if size_bytes <= 0:
        return

    src_codec = _normalize_video_codec(probe_video_codec(input_path))
    stream_br = probe_video_bitrate(input_path)
    avg_br = int((size_bytes * 8) / duration)
    bitrate_bps = stream_br if stream_br and stream_br > 0 else avg_br
    pixels_per_sec = max(1.0, float(width) * float(height) * raw_fr)
    bppf = float(bitrate_bps) / pixels_per_sec
    advised = _tier_for_bppf(src_codec, bppf)
    selected = _selected_quality_label(args)

    print(
        "Source profile: "
        f"{src_codec} {width}x{height}@{raw_fr:.2f}, "
        f"avg bitrate ~{bitrate_bps/1000:.0f} kb/s, bppf={bppf:.4f}"
    )
    print(f"Heuristic suggested source-quality tier: {advised}.")
    if selected and selected != advised:
        print(
            f"Note: you selected '{selected}'. This source looks closer to '{advised}'. "
            "If output inflates, rerun with a lower tier."
        )


def _expected_mode_requested(args: argparse.Namespace) -> bool:
    return bool(args.source_quality) or (args.expected_qp_alias is not None)


def _quality_label_to_qp(label: str) -> int | None:
    key = label.strip().lower().replace("_", "-").replace(" ", "-")
    mapping = {
        # Canonical tiers
        "ultra": 13,
        "high": 17,
        "medium": 21,
        "low": 25,
    }
    return mapping.get(key)


def _resolve_expected_start_qp(args: argparse.Namespace) -> int:
    token = args.source_quality
    if token is None and args.expected_qp_alias is not None:
        token = str(args.expected_qp_alias)
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
                "Invalid --source-quality value. Use QP integer or one of: ultra, high, medium, low."
            ) from exc
    if start_qp < args.min_qp:
        print(
            f"WARNING: expected start QP {start_qp} is below --min-qp {args.min_qp}; "
            "lowering min-qp to match."
        )
        args.min_qp = start_qp
    if start_qp > args.max_qp:
        print(
            f"WARNING: expected start QP {start_qp} is above --max-qp {args.max_qp}; "
            "raising max-qp to match."
        )
        args.max_qp = start_qp
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


def _suggest_expected_qps(
    rows: List[Dict[str, Any]],
    min_gain_pct: float,
    knee_ratio: float,
) -> tuple[int, int]:
    if len(rows) <= 1:
        qp = rows[0]["qp"]
        return qp, qp

    transitions: list[tuple[int, float, float]] = []
    for i in range(1, len(rows)):
        prev = rows[i - 1]
        cur = rows[i]
        size_saved_mb = max(0.0, (prev["size_bytes"] - cur["size_bytes"]) / 1_000_000.0)
        size_saved_pct = _safe_pct_delta(prev["size_bytes"], cur["size_bytes"])
        velocity = _transition_velocity(prev["ssim"], cur["ssim"], size_saved_mb)
        transitions.append((i, size_saved_pct, velocity))

    valid = [t for t in transitions if t[1] >= min_gain_pct]
    if not valid:
        return rows[-2]["qp"], rows[-1]["qp"]

    knee_index = None
    prev_vel = None
    for i, _gain, vel in valid:
        if prev_vel is not None and prev_vel > 0 and vel >= prev_vel * knee_ratio:
            knee_index = i
            break
        prev_vel = vel

    if knee_index is None:
        mid = valid[len(valid) // 2][0]
        knee_index = mid

    balanced_idx = min(knee_index, len(rows) - 1)
    safe_idx = max(0, balanced_idx - 1)
    return rows[safe_idx]["qp"], rows[balanced_idx]["qp"]


def _choose_expected_qp(
    args: argparse.Namespace,
    rows: List[Dict[str, Any]],
    safe_qp: int,
    balanced_qp: int,
    source_codec: str,
    source_size_bytes: int,
) -> int:
    by_qp = {row["qp"]: row for row in rows}
    safe_size = by_qp[safe_qp]["size_bytes"] / 1_000_000.0
    balanced_size = by_qp[balanced_qp]["size_bytes"] / 1_000_000.0
    print("\nExpected-QP suggestions:")
    print(f"  Source codec: {source_codec}, size: {source_size_bytes/1_000_000.0:.2f} MB")
    print(f"  [1] Safe (recommended): QP={safe_qp}, size={safe_size:.2f} MB")
    print(f"  [2] Balanced: QP={balanced_qp}, size={balanced_size:.2f} MB")

    if args.expected_choice == "safe":
        return safe_qp
    if args.expected_choice == "balanced":
        return balanced_qp
    if not sys.stdin.isatty():
        return safe_qp

    while True:
        choice = input("Choose 1 or 2 [1]: ").strip()
        if choice in ("", "1"):
            return safe_qp
        if choice == "2":
            return balanced_qp
        print("Please enter 1 or 2.")


def _run_expected_qp_mode(
    args: argparse.Namespace,
    baseline_file: str,
    source_ref_path: str,
    audio_opts: List[str],
    raw_fr: float,
    gop: int,
    tmpdir: str,
    video_codec: str,
) -> tuple[str, int, float | None]:
    start_qp = _resolve_expected_start_qp(args)
    print(
        "Expected-QP mode: "
        f"start QP={start_qp}, min_gain={args.expected_min_gain:.2f}%, "
        f"max_steps={args.expected_max_steps}, knee_ratio={args.expected_knee_ratio:.2f}"
    )
    source_base, source_ext = os.path.splitext(os.path.basename(source_ref_path))
    max_qp = min(args.max_qp, start_qp + max(0, args.expected_max_steps))

    rows: List[Dict[str, Any]] = []
    for qp in range(start_qp, max_qp + 1):
        output_path, ssim_val = cast(
            tuple[str, float | None],
            encode_final(
                input_file=baseline_file,
                qp=qp,
                audio_opts=audio_opts,
                raw_fr=raw_fr,
                gop=gop,
                video_codec=video_codec,
                return_ssim=True,
                ssim_chunk_seconds=args.full_ssim_chunk_seconds,
                output_dir=tmpdir,
                output_base=source_base,
                output_ext=source_ext,
            ),
        )
        if ssim_val is None:
            logging.warning("Full-file SSIM unavailable at QP=%d; stopping expected mode ladder.", qp)
            break
        size_bytes = _safe_file_size(output_path) or 0
        rows.append({
            "qp": qp,
            "ssim": float(ssim_val),
            "size_bytes": int(size_bytes),
            "path": output_path,
        })
        if len(rows) >= 2:
            prev = rows[-2]
            gain = _safe_pct_delta(prev["size_bytes"], size_bytes)
            print(
                f"  QP {prev['qp']} -> {qp}: size {prev['size_bytes']/1_000_000.0:.1f} -> "
                f"{size_bytes/1_000_000.0:.1f} MB (gain {gain:.2f}%)"
            )

    if not rows:
        raise RuntimeError("Expected-QP mode could not produce any full-file encodes.")

    _print_expected_qp_ladder(rows)
    safe_qp, balanced_qp = _suggest_expected_qps(
        rows=rows,
        min_gain_pct=args.expected_min_gain,
        knee_ratio=args.expected_knee_ratio,
    )
    src_codec = probe_video_codec(args.input)
    src_size = _safe_file_size(args.input) or 0
    selected_qp = _choose_expected_qp(
        args=args,
        rows=rows,
        safe_qp=safe_qp,
        balanced_qp=balanced_qp,
        source_codec=src_codec,
        source_size_bytes=src_size,
    )
    print(
        f"Expected-QP mode suggestions: safe={safe_qp}, balanced={balanced_qp}. "
        f"Selected QP={selected_qp}."
    )

    selected_row = next((row for row in rows if row["qp"] == selected_qp), rows[0])
    selected_path = selected_row["path"]
    selected_ssim = selected_row["ssim"]
    for row in rows:
        path = row["path"]
        if path != selected_path:
            try:
                os.remove(path)
            except OSError:
                pass
    return selected_path, selected_qp, selected_ssim


def _prepare_baseline_and_source(
    args: argparse.Namespace,
    use_size_target: bool,
    crop_filter: str | None,
    pix_fmt: str,
    baseline_tmp: str,
    video_codec: str,
) -> tuple[str, str]:
    if use_size_target:
        if crop_filter:
            baseline_file = _determine_baseline_file(
                input_path=args.input,
                skip_baseline=args.skip_baseline,
                pix_fmt=pix_fmt,
                baseline_tmp=baseline_tmp,
                baseline_qp=args.baseline_qp,
                extra_vf=crop_filter,
                video_codec=video_codec,
            )
            source_ref_path = _resolve_source_ref_path(args, baseline_file)
            return baseline_file, source_ref_path
        baseline_file = args.input
        source_ref_path = args.input
        if not args.skip_baseline:
            print("Skipping baseline generation for size-targeted path.")
        return baseline_file, source_ref_path

    baseline_file = _determine_baseline_file(
        input_path=args.input,
        skip_baseline=args.skip_baseline,
        pix_fmt=pix_fmt,
        baseline_tmp=baseline_tmp,
        baseline_qp=args.baseline_qp,
        extra_vf=crop_filter,
        video_codec=video_codec,
    )
    source_ref_path = _resolve_source_ref_path(args, baseline_file)
    return baseline_file, source_ref_path


def _select_best_qp_for_size_target(
    args: argparse.Namespace,
    baseline_file: str,
    audio_opts: List[str],
    raw_fr: float,
    gop: int,
    scratch_root: str,
    target_bytes: int,
    video_codec: str,
) -> tuple[int, str | None, list[str]]:
    best_qp, samples_tmpdir, samples = _select_best_qp_size(
        args=args,
        baseline_file=baseline_file,
        audio_opts=audio_opts,
        raw_fr=raw_fr,
        gop=gop,
        scratch_root=scratch_root,
        target_bytes=target_bytes,
        tolerance=args.size_tolerance,
        video_codec=video_codec,
    )
    if samples_tmpdir:
        shutil.rmtree(samples_tmpdir, ignore_errors=True)
        samples_tmpdir = None
    return best_qp, samples_tmpdir, samples


def _select_best_qp_for_ssim(
    args: argparse.Namespace,
    baseline_file: str,
    source_ref_path: str,
    audio_opts: List[str],
    raw_fr: float,
    gop: int,
    scratch_root: str,
) -> tuple[int, str | None, list[str]]:
    return _select_best_qp(
        args=args,
        baseline_file=baseline_file,
        source_ref_path=source_ref_path,
        audio_opts=audio_opts,
        raw_fr=raw_fr,
        gop=gop,
        scratch_root=scratch_root,
    )


def _run_size_target_flow(
    args: argparse.Namespace,
    baseline_file: str,
    source_ref_path: str,
    best_qp: int,
    audio_opts: List[str],
    raw_fr: float,
    gop: int,
    tmpdir: str,
    target_bytes: int,
    baseline_tmp: str,
    scratch_root: str,
    video_codec: str,
) -> tuple[str, int, float | None, str | None]:
    final_file, final_qp, final_full_ssim = _run_final_encode_size(
        args=args,
        baseline_file=baseline_file,
        source_ref_path=source_ref_path,
        best_qp=best_qp,
        audio_opts=audio_opts,
        raw_fr=raw_fr,
        gop=gop,
        tmpdir=tmpdir,
        target_bytes=target_bytes,
        tolerance=args.size_tolerance,
        video_codec=video_codec,
    )
    if not has_filter('ssim'):
        print("WARNING: ssim filter not available; skipping SSIM refinement.")
        return final_file, final_qp, final_full_ssim, None

    print("\nRefining with SSIM using size-target output as the source...")
    ssim_input = final_file
    _, _, ssim_pix_fmt = _probe_video_basics(ssim_input)
    ssim_baseline = _determine_baseline_file(
        input_path=ssim_input,
        skip_baseline=True,
        pix_fmt=ssim_pix_fmt,
        baseline_tmp=baseline_tmp,
        baseline_qp=args.baseline_qp,
        video_codec=video_codec,
    )
    original_ssim = args.ssim
    args.ssim = args.h264_compat_ssim
    best_qp, samples_tmpdir, _samples = _select_best_qp(
        args=args,
        baseline_file=ssim_baseline,
        source_ref_path=ssim_input,
        audio_opts=audio_opts,
        raw_fr=raw_fr,
        gop=gop,
        scratch_root=scratch_root,
    )
    if samples_tmpdir:
        shutil.rmtree(samples_tmpdir, ignore_errors=True)
    refine_tmp = tempfile.mkdtemp(prefix="ssim_refine_", dir=scratch_root)
    final_file, final_qp, final_full_ssim = _run_final_encode(
        args=args,
        baseline_file=ssim_baseline,
        source_ref_path=args.input,
        best_qp=best_qp,
        audio_opts=audio_opts,
        raw_fr=raw_fr,
        gop=gop,
        tmpdir=refine_tmp,
        video_codec=video_codec,
    )
    args.ssim = original_ssim
    return final_file, final_qp, final_full_ssim, refine_tmp


def _maybe_rerun_full_ssim(
    args: argparse.Namespace,
    use_size_target: bool,
    final_full_ssim: float | None,
    samples: list[str],
    final_qp: int,
    baseline_file: str,
    source_ref_path: str,
    audio_opts: List[str],
    raw_fr: float,
    gop: int,
    tmpdir: str,
    video_codec: str,
) -> tuple[str | None, int | None, float | None]:
    if use_size_target or args.skip_full_ssim or final_full_ssim is None or not samples:
        return None, None, None
    sample_metric = measure_ssim(
        qp=final_qp,
        samples=samples,
        raw_fr=raw_fr,
        gop=gop,
        audio_opts=audio_opts,
        metric=args.metric,
        video_codec=video_codec,
    )
    discrepancy_threshold = 0.02
    rerun_qp = _prompt_initial_qp_rerun(
        sample_metric=sample_metric,
        full_metric=final_full_ssim,
        final_qp=final_qp,
        min_qp=args.min_qp,
        max_qp=args.max_qp,
        threshold=discrepancy_threshold,
    )
    if rerun_qp is None:
        return None, None, None
    return _encode_with_full_ssim(
        baseline_file=baseline_file,
        best_qp=rerun_qp,
        min_qp=args.min_qp,
        target_ssim=args.ssim,
        audio_opts=audio_opts,
        raw_fr=raw_fr,
        gop=gop,
        video_codec=video_codec,
        ssim_chunk_seconds=args.full_ssim_chunk_seconds,
        tmpdir=tmpdir,
        source_base=os.path.splitext(os.path.basename(source_ref_path))[0],
        source_ext=os.path.splitext(os.path.basename(source_ref_path))[1],
    )


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    args.video_codec = _normalize_video_codec(args.video_codec)

    # ------------------------------------------------------------
    # Logging setup
    # ------------------------------------------------------------
    setup_logging(args.verbose, args.log_file)

    if args.use_baseline_as_source and args.skip_baseline:
        print("WARNING: --use-baseline-as-source requires a baseline; ignoring --skip-baseline.")
        args.skip_baseline = False
    if args.source_quality and args.expected_qp_alias is not None:
        print("WARNING: both --source-quality and --expected-qp provided; using --source-quality.")

    if not os.path.isfile(args.input):
        logging.error("Input file not found: %s", args.input)
        return

    # ------------------------------------------------------------
    # Audio-only path: copy video, process audio, skip SSIM pipeline
    # ------------------------------------------------------------
    if _maybe_run_audio_only_path(args):
        return

    # ------------------------------------------------------------
    # Probe basic video info (for impossible-encode checks)
    # ------------------------------------------------------------
    width, height, pix_fmt = _probe_video_basics(args.input)

    if args.video_codec == "h264":
        if not _confirm_h264_compat(width, height, pix_fmt, decision=args.h264_compat):
            return

    crop_filter = _resolve_crop_filter(args, width, height)

    # ------------------------------------------------------------
    # Decide quality metric path (SSIM vs size-target for modern codecs)
    # ------------------------------------------------------------
    expected_mode = _expected_mode_requested(args)
    quality_mode = _resolve_quality_mode(args, args.input, force_ssim=expected_mode)
    if quality_mode is None:
        return
    codec, use_size_target = quality_mode

    # ------------------------------------------------------------
    # Probe audio/video from original file
    # ------------------------------------------------------------
    audio_opts, raw_fr, args.audio_normalize = _prepare_audio_and_framerate(
        args.input, args.audio_normalize, args.add_stereo_downmix
    )
    _maybe_print_expected_mode_advice(
        args=args,
        input_path=args.input,
        width=width,
        height=height,
        raw_fr=raw_fr,
    )

    # ------------------------------------------------------------
    # Prepare temp directories
    # ------------------------------------------------------------
    scratch_root = _resolve_scratch_root(args.scratch_dir)
    baseline_tmp, tmpdir, samples_tmpdir = _prepare_work_dirs(
        scratch_root, args.input, args.skip_baseline
    )
    refine_tmp: str | None = None

    try:
        # ------------------------------------------------------------
        # STEP 1 — BASELINE (QP=baseline) ENCODE with PROGRESS (+ optional HDR->SDR)
        # ------------------------------------------------------------
        baseline_file, source_ref_path = _prepare_baseline_and_source(
            args=args,
            use_size_target=use_size_target,
            crop_filter=crop_filter,
            pix_fmt=pix_fmt,
            baseline_tmp=baseline_tmp,
            video_codec=args.video_codec,
        )

        target_bytes = _resolve_target_bytes(args, args.input, codec) if use_size_target else 0

        # ------------------------------------------------------------
        # GOP ~ half framerate
        gop = _calculate_gop(raw_fr)

        samples: list[str] = []
        best_qp = args.initial_qp if args.initial_qp is not None else args.min_qp

        # ------------------------------------------------------------
        # STEP 2 — SAMPLE CLIP EXTRACTION (optional)
        # ------------------------------------------------------------
        if not expected_mode:
            if use_size_target:
                best_qp, samples_tmpdir, samples = _select_best_qp_for_size_target(
                    args=args,
                    baseline_file=baseline_file,
                    audio_opts=audio_opts,
                    raw_fr=raw_fr,
                gop=gop,
                scratch_root=scratch_root,
                target_bytes=target_bytes,
                video_codec=args.video_codec,
            )
            else:
                best_qp, samples_tmpdir, samples = _select_best_qp_for_ssim(
                    args=args,
                    baseline_file=baseline_file,
                    source_ref_path=source_ref_path,
                    audio_opts=audio_opts,
                    raw_fr=raw_fr,
                    gop=gop,
                    scratch_root=scratch_root,
                )

        # ------------------------------------------------------------
        # STEP 4 — FULL-FILE FINAL ENCODE DESCENT (CLEANER OUTPUT)
        # ------------------------------------------------------------
        if expected_mode:
            final_file, final_qp, final_full_ssim = _run_expected_qp_mode(
                args=args,
                baseline_file=baseline_file,
                source_ref_path=source_ref_path,
                audio_opts=audio_opts,
                raw_fr=raw_fr,
                gop=gop,
                tmpdir=tmpdir,
                video_codec=args.video_codec,
            )
        elif use_size_target:
            final_file, final_qp, final_full_ssim, refine_tmp = _run_size_target_flow(
                args=args,
                baseline_file=baseline_file,
                source_ref_path=source_ref_path,
                best_qp=best_qp,
                audio_opts=audio_opts,
                raw_fr=raw_fr,
                gop=gop,
                tmpdir=tmpdir,
                target_bytes=target_bytes,
                baseline_tmp=baseline_tmp,
                scratch_root=scratch_root,
                video_codec=args.video_codec,
            )
        else:
            final_file, final_qp, final_full_ssim = _run_final_encode(
                args=args,
                baseline_file=baseline_file,
                source_ref_path=source_ref_path,
                best_qp=best_qp,
                audio_opts=audio_opts,
                raw_fr=raw_fr,
                gop=gop,
                tmpdir=tmpdir,
                video_codec=args.video_codec,
            )

        rerun_file, rerun_qp, rerun_ssim = _maybe_rerun_full_ssim(
            args=args,
            use_size_target=(use_size_target or expected_mode),
            final_full_ssim=final_full_ssim,
            samples=samples,
            final_qp=final_qp,
            baseline_file=baseline_file,
            source_ref_path=source_ref_path,
            audio_opts=audio_opts,
            raw_fr=raw_fr,
            gop=gop,
            tmpdir=tmpdir,
            video_codec=args.video_codec,
        )
        if rerun_file is not None and rerun_qp is not None:
            final_file = rerun_file
            final_qp = rerun_qp
            final_full_ssim = rerun_ssim

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
        if refine_tmp:
            shutil.rmtree(refine_tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
