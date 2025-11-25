# cli.py
import argparse
import logging
import os
import tempfile
import shutil

from tqdm import tqdm

from .probes import probe_audio_streams, probe_video_framerate
from .utils import build_audio_options, setup_logging
from .sampling import extract_samples
from .ssim_search import find_best_qp
from .encoder import encode_final, encode_baseline


def main():
    parser = argparse.ArgumentParser(
        description='Optimize video quality via SSIM and QP binary search.'
    )
    parser.add_argument('input', help='Source video file')
    parser.add_argument('--ssim', type=float, default=0.99)
    parser.add_argument('--min-qp', type=int, default=16)
    parser.add_argument('--max-qp', type=int, default=34)
    parser.add_argument('--sample-percent', type=float, default=15)
    parser.add_argument('--sample-count', type=int, default=3)
    parser.add_argument('--sample-qp', type=int, default=15)
    parser.add_argument(
        '--sampling-mode',
        choices=['uniform', 'scene', 'motion'],
        default='motion'
    )
    parser.add_argument(
        '--metric',
        choices=['avg', 'min', 'max'],
        default='min'
    )
    parser.add_argument('--log-file')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    setup_logging(args.verbose, args.log_file)

    if not os.path.isfile(args.input):
        logging.error("Input file not found: %s", args.input)
        return

    streams = probe_audio_streams(args.input)
    audio_opts = build_audio_options(streams)
    raw_fr = probe_video_framerate(args.input)

    baseline_tmp = tempfile.mkdtemp(prefix="ssim_baseline_")
    tmpdir = tempfile.mkdtemp(prefix="ssim_final_")

    try:
        # Baseline creation (with progress)
        baseline_file = encode_baseline(args.input, output_dir=baseline_tmp)
        print(f"Baseline file created: {baseline_file}")

        # Extract samples (still uses run_cmd, no progress needed)
        samples = extract_samples(
            baseline_file,
            percent=args.sample_percent,
            count=args.sample_count,
            sample_qp=args.sample_qp,
            audio_opts=audio_opts,
            raw_fr=raw_fr,
            sampling_mode=args.sampling_mode
        )

        gop = max(1, int(round(raw_fr / 2)))

        best_qp = find_best_qp(
            samples,
            args.min_qp,
            args.max_qp,
            args.ssim,
            args.metric,
            audio_opts,
            raw_fr,
            gop
        )

        # Final pass descent
        prev_file = None
        final_file = None
        final_qp = best_qp

        for qp in tqdm(range(best_qp, args.min_qp - 1, -1), desc="Full-file QP Scan", unit="qp"):
            final_qp = qp

            if prev_file and os.path.exists(prev_file):
                os.remove(prev_file)

            result = encode_final(
                input_file=baseline_file,
                qp=final_qp,
                audio_opts=audio_opts,
                raw_fr=raw_fr,
                gop=gop,
                return_ssim=True,
                output_dir=tmpdir
            )
            final_temp, ssim_val = result

            if ssim_val >= args.ssim:
                print(f"Final full-file SSIM {ssim_val:.4f} meets target; using QP={final_qp}")
                final_file = final_temp
                break

            prev_file = final_temp

        else:
            logging.warning("Could not meet SSIM target; using sample-based QP=%d", best_qp)
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

        base, ext = os.path.splitext(os.path.basename(args.input))
        dest = os.path.join(os.path.dirname(args.input), f"{base} [h264_nvenc qp {final_qp}]{ext}")
        shutil.move(final_file, dest)
        print(f"Optimized file: {dest} (QP={final_qp})")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        shutil.rmtree(baseline_tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
