# config.py
from __future__ import annotations
import argparse
import dataclasses
from typing import Any


@dataclasses.dataclass
class EncodeConfig:
    # --- Positional ---
    input: str

    # --- Core encoding ---
    video_codec: str = "h265"
    ssim: float = 0.986
    min_qp: int = 6
    max_qp: int = 40

    # --- Sampling ---
    sample_percent: Any = "auto"   # str "auto" resolved to float early in pipeline
    sample_count: Any = "auto"     # str "auto" resolved to int early in pipeline
    sample_qp: int = 6
    sampling_mode: str = "motion"
    metric: str = "avg"

    # --- Quality selection ---
    initial_qp: int | None = None
    source_quality: str | None = None
    expected_choice: str = "prompt"
    expected_min_gain: float = 1.0
    expected_max_steps: int = 8
    expected_knee_ratio: float = 1.5

    # --- Batch mode ---
    batch_auto: bool = False
    batch_dry_run: bool = False
    batch_size_guard: bool = False
    batch_bppf_tolerance: float = 0.15
    batch_bitrate_tolerance: float = 0.20
    re_encode_same_codec_video: bool = False

    # --- Output ---
    output_dir: str | None = None
    no_suffix: bool = False

    # --- Baseline ---
    baseline_qp: int = 6
    skip_baseline: bool = False
    use_baseline_as_source: bool = False
    h264_compat: str = "abort"

    # --- Video filters ---
    resize_resolution: str | None = None
    display_ar: str = "auto"
    auto_crop: str = "off"
    target_fps: float | None = None

    # --- Audio ---
    audio_normalize: bool = True
    add_stereo_downmix: bool = False
    add_stereo_downmix_copy_video: bool = False

    # --- Misc ---
    full_ssim_chunk_seconds: float | None = None
    scratch_dir: str | None = None
    log_file: str | None = None
    verbose: bool = False

    # --- Pipeline-internal (set during execution, not from argparse) ---
    skip_baseline_requested: bool = False

    # --- Batch-internal (set per-cluster or per-member) ---
    batch_input_dir: str | None = None           # was _batch_input_dir
    batch_precomputed_baseline_file: str | None = None
    batch_precomputed_segments: list[str] | None = None
    batch_precomputed_segments_tmpdir: str | None = None
    batch_precomputed_rows: list[dict] | None = None
    batch_heuristic_size_limit: int | None = None
    batch_force_fixed_qp_mode: bool = False


def config_from_args(args: argparse.Namespace) -> EncodeConfig:
    """Convert a parsed argparse.Namespace to a typed EncodeConfig."""
    return EncodeConfig(
        input=args.input,
        video_codec=args.video_codec,
        ssim=args.ssim,
        min_qp=args.min_qp,
        max_qp=args.max_qp,
        sample_percent=args.sample_percent,
        sample_count=args.sample_count,
        sample_qp=args.sample_qp,
        sampling_mode=args.sampling_mode,
        metric=args.metric,
        initial_qp=args.initial_qp,
        source_quality=args.source_quality,
        expected_choice=args.expected_choice,
        expected_min_gain=args.expected_min_gain,
        expected_max_steps=args.expected_max_steps,
        expected_knee_ratio=args.expected_knee_ratio,
        batch_auto=args.batch_auto,
        batch_dry_run=args.batch_dry_run,
        batch_size_guard=args.batch_size_guard,
        batch_bppf_tolerance=args.batch_bppf_tolerance,
        batch_bitrate_tolerance=args.batch_bitrate_tolerance,
        re_encode_same_codec_video=args.re_encode_same_codec_video,
        output_dir=args.output_dir,
        no_suffix=args.no_suffix,
        baseline_qp=args.baseline_qp,
        skip_baseline=args.skip_baseline,
        use_baseline_as_source=args.use_baseline_as_source,
        h264_compat=args.h264_compat,
        resize_resolution=getattr(args, 'resize_resolution', None),
        display_ar=args.display_ar,
        auto_crop=args.auto_crop,
        target_fps=args.target_fps,
        audio_normalize=args.audio_normalize,
        add_stereo_downmix=args.add_stereo_downmix,
        add_stereo_downmix_copy_video=args.add_stereo_downmix_copy_video,
        full_ssim_chunk_seconds=args.full_ssim_chunk_seconds,
        scratch_dir=args.scratch_dir,
        log_file=args.log_file,
        verbose=args.verbose,
    )
