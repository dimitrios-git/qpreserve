# encoder.py

import logging
import os
import subprocess

from .utils import run_cmd, run_ffmpeg_progress
from .probes import (
    probe_video_duration,
    detect_hdr,
    probe_video_stream_info,
)


def measure_full_ssim(input_file: str, encoded_file: str) -> float:
    """
    Measure full-file SSIM between input_file (reference) and encoded_file (distorted).
    """
    res = run_cmd([
        'ffmpeg', '-i', input_file, '-i', encoded_file,
        '-filter_complex', 'ssim', '-f', 'null', '-'
    ], capture_output=True)

    for line in res.stderr.splitlines():
        if 'All:' in line:
            return float(line.split('All:')[1].split()[0])
    return 0.0


def build_hdr_sdr_filter(hdr: dict) -> str | None:
    """
    Builds HDR->SDR tonemapping filter chain for ffmpeg, if needed.

    Handles:
      - PQ (HDR10, transfer=smpte2084)
      - HLG (transfer=arib-std-b67)
      - Fallback for other bt2020-tagged HDR-like streams
    """

    prim = hdr["primaries"]
    tr = hdr["transfer"]
    mat = hdr["matrix"]

    # PQ (HDR10)
    if tr == "smpte2084":
        return (
            "zscale=primaries=bt2020:transfer=smpte2084:matrix=bt2020nc,"
            "zscale=t=linear:npl=100,"
            "tonemap=hable,"
            "zscale=primaries=bt709:transfer=bt709:matrix=bt709"
        )

    # HLG
    if tr == "arib-std-b67":
        return (
            "zscale=primaries=bt2020:transfer=arib-std-b67:matrix=bt2020nc,"
            "zscale=t=linear:npl=100,"
            "tonemap=hable,"
            "zscale=primaries=bt709:transfer=bt709:matrix=bt709"
        )

    # Fallback for weird HDR tagging but bt2020-like space
    if hdr["is_hdr"]:
        return (
            "zscale=primaries=bt2020:transfer=smpte2084:matrix=bt2020nc,"
            "zscale=t=linear:npl=100,"
            "tonemap=hable,"
            "zscale=primaries=bt709:transfer=bt709:matrix=bt709"
        )

    return None


def _is_low_res(path: str) -> bool:
    """
    Decide if a video should be treated as 'low-res' for encoder safety.
    Here: max(width, height) <= 720.
    """
    info = probe_video_stream_info(path)
    w = info["width"]
    h = info["height"]
    max_dim = max(w, h)
    return max_dim <= 720


def encode_baseline(input_file: str, output_dir: str | None = None) -> str:
    """
    Step 0 + 1: Create the "baseline" file from the ORIGINAL source.

    - Detect HDR and apply HDR->SDR tonemapping if needed.
    - Normalize to SDR BT.709, yuv420p, SAR=1.
    - For low-res sources (<=720p):
        * Use libx264 lossless-ish baseline (QP=0, preset=veryslow).
        * Avoid all NVENC bugs / crashes on SD.
    - For higher resolutions:
        * Use H.264 NVENC in constqp mode:
            - QP=0
            - preset=p7
            - bf=2
    """

    base, ext = os.path.splitext(os.path.basename(input_file))
    filename = f"{base} [baseline qp 0]{ext}"
    output = os.path.join(output_dir, filename) if output_dir else filename

    total_duration = probe_video_duration(input_file)
    low_res = _is_low_res(input_file)

    if low_res:
        logging.info("Low-resolution source detected → baseline via libx264 (lossless-ish).")
    else:
        logging.info("High-resolution source → baseline via h264_nvenc (QP=0).")

    # HDR detection + tonemap
    hdr_info = detect_hdr(input_file)
    filter_chain = build_hdr_sdr_filter(hdr_info)

    # Base command: map only v/a/s, ignore data/tmcd/etc.
    base_cmd = [
        'ffmpeg', '-y',
        '-i', input_file,
        '-map', '0:v',
        '-map', '0:a?',
        '-map', '0:s?',
        '-map_metadata', '0',
    ]

    if filter_chain:
        print("HDR detected → applying HDR→SDR tone mapping")
        base_cmd += ['-vf', filter_chain]

    base_cmd += [
        '-pix_fmt', 'yuv420p',
        '-color_primaries', 'bt709',
        '-color_trc', 'bt709',
        '-colorspace', 'bt709',
    ]

    nvenc_opts = [
        '-c:v', 'h264_nvenc',
        '-preset', 'p7',
        '-rc', 'constqp',
        '-qp', '0',
        '-bf', '2',
    ]

    x264_opts = [
        '-c:v', 'libx264',
        '-preset', 'veryslow',
        '-qp', '0',
    ]

    if low_res:
        # Completely avoid NVENC for SD content
        cmd = base_cmd + x264_opts
        cmd += [
            '-c:a', 'copy',
            '-c:s', 'copy',
            output
        ]
        run_ffmpeg_progress(cmd, total_duration, desc="Baseline Encode")
        return output
    else:
        # Original fast GPU baseline for HD/4K
        cmd = base_cmd + nvenc_opts
        cmd += [
            '-c:a', 'copy',
            '-c:s', 'copy',
            output
        ]

        try:
            run_ffmpeg_progress(cmd, total_duration, desc="Baseline Encode")
        except subprocess.CalledProcessError as e:
            log_path = getattr(e, "ffmpeg_log", None)
            logging.warning(
                "Baseline NVENC encode failed; retrying with libx264 (QP=0). "
                "See FFmpeg log above for the failure reason."
            )
            if log_path:
                print(f"NVENC baseline failed; FFmpeg log: {log_path}. Falling back to libx264.")
            fallback_cmd = base_cmd + x264_opts
            fallback_cmd += [
                '-c:a', 'copy',
                '-c:s', 'copy',
                output
            ]
            run_ffmpeg_progress(fallback_cmd, total_duration, desc="Baseline Encode (libx264 fallback)")
    return output


def encode_final(
    input_file: str,
    qp: int,
    audio_opts: list,
    raw_fr: float,
    gop: int,
    return_ssim: bool = False,
    output_dir: str | None = None
):
    """
    Final encode (from the baseline file), with FFmpeg progress and optional SSIM.

    - input_file here is the BASELINE file (already normalized to SDR, bt709).
    - For low-res baselines, use libx264 -qp {qp}.
    - For higher-res baselines, use h264_nvenc -qp {qp}.
    """

    base, ext = os.path.splitext(os.path.basename(input_file))
    total_duration = probe_video_duration(input_file)
    low_res = _is_low_res(input_file)

    encoder_tag = "libx264" if low_res else "h264_nvenc"
    filename = f"{base} [{encoder_tag} qp {qp}]{ext}"
    output = os.path.join(output_dir, filename) if output_dir else filename

    common_opts = [
        '-map', '0:v',
        '-map', '0:a?',
        '-map', '0:s?',
        '-map_metadata', '0',
        '-r', str(raw_fr),
        '-g', str(gop),
        '-bf', '2',
        '-pix_fmt', 'yuv420p',
    ]

    if low_res:
        logging.info("Low-resolution baseline → final encode via libx264 (QP=%d).", qp)
        cmd = [
            'ffmpeg', '-y',
            '-i', input_file,
        ] + common_opts + [
            '-c:v', 'libx264',
            '-preset', 'slow',
            '-qp', str(qp),
        ] + audio_opts + ['-c:s', 'copy', output]
        run_ffmpeg_progress(cmd, total_duration, desc=f"Final Encode (QP={qp})")
    else:
        cmd = [
            'ffmpeg', '-y',
            '-hwaccel', 'cuda',
            '-i', input_file,
        ] + common_opts + [
            '-c:v', 'h264_nvenc',
            '-preset', 'p7',
            '-rc', 'constqp',
            '-qp', str(qp),
        ] + audio_opts + ['-c:s', 'copy', output]
        run_ffmpeg_progress(cmd, total_duration, desc=f"Final Encode (QP={qp})")

    if return_ssim:
        ssim_val = measure_full_ssim(input_file, output)
        print(f"Full-file SSIM at QP {qp}: {ssim_val:.4f}")
        return output, ssim_val

    return output
