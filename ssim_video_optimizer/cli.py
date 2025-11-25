# cli.py
"""
Command-line interface for ssim_video_optimizer.

Coordinates:
  - baseline generation (QP 0, optional HDR->SDR)
  - sampling (uniform / scene / motion)
  - SSIM-based QP search (VMAF stubbed for future)
  - final encoding + full-file SSIM verification
"""

import argparse
import logging
import os
import tempfile
import shutil

from tqdm import tqdm

from .probes import (
    probe_audio_streams,
    probe_video_framerate,
    probe_video_stream_info,
)
from .utils import (
    build_audio_options,
    setup_logging,
    has_filter,
)
from .sampling import extract_samples
from .ssim_search import find_best_qp
from .encoder import encode_final, encode_baseline


def main():
    parser = argparse.ArgumentParser(
        description='Optimize video quality via perceptual metric and QP binary search.'
    )

    parser.add_argument('input', help='Source video file')

    # Quality targets
    parser.add_argument('--ssim', type=float, default=0.99,
                        help='Target SSIM threshold for acceptance (used when metric is SSIM).')
    parser.add_argument('--target-vmaf', type=float, default=95.0,
                        help='(Reserved) Target VMAF score; VMAF not implemented yet in this build.')

    # Metric selection (staged: only SSIM is actually used for now)
    parser.add_argument(
        '--quality-metric',
        choices=['auto', 'ssim', 'vmaf'],
        default='auto',
        help='Preferred metric: auto = VMAF if available, otherwise SSIM.'
    )

    # QP search bounds
    parser.add_argument('--min-qp', type=int, default=16,
                        help='Minimum QP allowed during search.')
    parser.add_argument('--max-qp', type=int, default=34,
                        help='Maximum QP allowed during search.')

    # Sampling configuration
    parser.add_argument('--sample-percent', type=float, default=15,
                        help='Percentage of video duration used for sampling.')
    parser.add_argument('--sample-count', type=int, default=3,
                        help='How many sample clips to extract.')
    parser.add_argument('--sample-qp', type=int, default=15,
                        help='QP used to encode sample clips.')

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
        default='min',
        help='Which SSIM aggregation metric to use across samples.'
    )

    # Logging
    parser.add_argument('--log-file', help='Optional log file path.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose logging.')

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
    vinfo = probe_video_stream_info(args.input)
    width = vinfo['width']
    height = vinfo['height']
    pix_fmt = vinfo['pix_fmt']

    problems = []

    # Conservative H.264/NVENC safe bounds
    if width > 4096 or height > 4096:
        problems.append(f"- Resolution {width}x{height} exceeds 4096 in at least one dimension.")

    # crude but effective: any 10-bit pix_fmt contains "10"
    if '10' in pix_fmt:
        problems.append(f"- Pixel format '{pix_fmt}' is 10-bit; H.264/NVENC 8-bit pipeline may be invalid.")

    if problems:
        print("The source video may not be safely encodable with H.264/NVENC:")
        for p in problems:
            print("  ", p)
        print()
        print("Choose an action:")
        print("  [1] Abort (recommended)")
        print("  [2] Continue anyway with H.264 (may fail or produce non-standard output)")
        while True:
            choice = input("Enter 1 or 2: ").strip()
            if choice in ('1', '2'):
                break

        if choice == '1':
            print("Aborting due to incompatible source for H.264/NVENC.")
            return
        else:
            print("Continuing with H.264/NVENC despite potential incompatibilities...")

    # ------------------------------------------------------------
    # Choose quality metric (staged: everything still uses SSIM)
    # ------------------------------------------------------------
    # For now, we only IMPLEMENT SSIM. VMAF is reserved for future.
    metric_effective = 'ssim'

    if args.quality_metric == 'auto':
        if has_filter('libvmaf'):
            # In future: metric_effective = 'vmaf'
            print("NOTE: libvmaf detected, but VMAF is not implemented yet in this build; using SSIM instead.")
            metric_effective = 'ssim'
        elif has_filter('ssim'):
            metric_effective = 'ssim'
        else:
            print("ERROR: Neither libvmaf nor ssim filter are available in ffmpeg. Cannot measure quality.")
            return
    elif args.quality_metric == 'vmaf':
        if has_filter('libvmaf'):
            print("NOTE: VMAF requested but VMAF mode is not implemented yet; using SSIM instead.")
            metric_effective = 'ssim'
        elif has_filter('ssim'):
            print("WARNING: libvmaf not available; falling back to SSIM.")
            metric_effective = 'ssim'
        else:
            print("ERROR: libvmaf not available and ssim filter missing; cannot measure quality.")
            return
    elif args.quality_metric == 'ssim':
        if has_filter('ssim'):
            metric_effective = 'ssim'
        else:
            if has_filter('libvmaf'):
                print("ERROR: ssim filter missing and VMAF mode is not implemented yet in this build.")
            else:
                print("ERROR: ssim filter missing and libvmaf not available; cannot measure quality.")
            return

    print(f"Quality metric in use: {metric_effective.upper()} (requested={args.quality_metric})")

    # ------------------------------------------------------------
    # Probe audio/video from original file
    # ------------------------------------------------------------
    streams = probe_audio_streams(args.input)
    audio_opts = build_audio_options(streams)
    raw_fr = probe_video_framerate(args.input)

    # ------------------------------------------------------------
    # Prepare temp directories
    # ------------------------------------------------------------
    baseline_tmp = tempfile.mkdtemp(prefix="ssim_baseline_")
    tmpdir = tempfile.mkdtemp(prefix="ssim_final_")

    try:
        # ------------------------------------------------------------
        # STEP 1 — BASELINE (QP=0) ENCODE with PROGRESS (+ optional HDR->SDR)
        # ------------------------------------------------------------
        baseline_file = encode_baseline(args.input, output_dir=baseline_tmp)
        print(f"Baseline file created: {baseline_file}")

        # ------------------------------------------------------------
        # STEP 2 — SAMPLE CLIP EXTRACTION
        # ------------------------------------------------------------
        samples = extract_samples(
            baseline_file,
            percent=args.sample_percent,
            count=args.sample_count,
            sample_qp=args.sample_qp,
            audio_opts=audio_opts,
            raw_fr=raw_fr,
            sampling_mode=args.sampling_mode,
        )

        # GOP ~ half framerate
        gop = max(1, int(round(raw_fr / 2)))

        # ------------------------------------------------------------
        # STEP 3 — SAMPLE-BASED QP SEARCH (currently SSIM-only)
        # ------------------------------------------------------------
        # NOTE: metric_effective is always 'ssim' for now.
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

        # ------------------------------------------------------------
        # STEP 4 — FULL-FILE FINAL ENCODE DESCENT (CLEANER OUTPUT)
        # ------------------------------------------------------------

        prev_file = None
        final_file = None
        final_qp = best_qp

        for qp in range(best_qp, args.min_qp - 1, -1):
            final_qp = qp
            print(f"\nChecking full-file quality at QP={qp}...")

            # Clean up older intermediate file
            if prev_file and os.path.exists(prev_file):
                os.remove(prev_file)

            # Encode full file (this has its own real FFmpeg progress bar)
            output_path, ssim_val = encode_final(
                input_file=baseline_file,
                qp=qp,
                audio_opts=audio_opts,
                raw_fr=raw_fr,
                gop=gop,
                return_ssim=True,
                output_dir=tmpdir
            )

            print(f"  → SSIM={ssim_val:.4f}")

            if ssim_val >= args.ssim:
                print(f"  ✓ Meets target SSIM ≥ {args.ssim}; accepting QP={qp}")
                final_file = output_path
                break

            print("  ✗ Below target; trying lower QP...")
            prev_file = output_path

        else:
            # If loop never broke (none passed threshold)
            logging.warning(
                "Could not meet SSIM target; using sample-based QP=%d",
                best_qp
            )
            final_file = prev_file or encode_final(
                input_file=baseline_file,
                qp=best_qp,
                audio_opts=audio_opts,
                raw_fr=raw_fr,
                gop=gop,
                return_ssim=False,
                output_dir=tmpdir
            )
            final_qp = best_qp

        # ------------------------------------------------------------
        # STEP 5 — MOVE FINAL RESULT TO SOURCE DIRECTORY
        # ------------------------------------------------------------
        dest_name = os.path.basename(final_file)
        dest = os.path.join(os.path.dirname(args.input), dest_name)

        shutil.move(final_file, dest)
        print(f"Optimized file: {dest} (QP={final_qp})")

    finally:
        # Cleanup temporary dirs
        shutil.rmtree(tmpdir, ignore_errors=True)
        shutil.rmtree(baseline_tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
