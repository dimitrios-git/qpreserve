# encoder.py

import logging
import os
import subprocess
import tempfile
from typing import Dict, List, Sequence, Any, Tuple

from .utils import run_ffmpeg_progress
from .probes import (
    probe_video_duration,
    detect_hdr,
    probe_video_stream_info,
)


def _build_ssim_filter_chain(raw_fr: float | None) -> str:
    # Normalize timestamps (and optionally FPS) to avoid large frame sync buffers
    # that can blow up RAM on long or VFR sources.
    if raw_fr and raw_fr > 0:
        fr_str = f"{raw_fr:.6f}".rstrip("0").rstrip(".")
        return (
            f"[0:v]fps={fr_str},setpts=PTS-STARTPTS[ref];"
            f"[1:v]fps={fr_str},setpts=PTS-STARTPTS[dist];"
            "[ref][dist]ssim=stats_file=-"
        )
    return (
        "[0:v]setpts=PTS-STARTPTS[ref];"
        "[1:v]setpts=PTS-STARTPTS[dist];"
        "[ref][dist]ssim=stats_file=-"
    )


def _run_ssim_ffmpeg(
    input_file: str,
    encoded_file: str,
    filter_chain: str,
    extra: Sequence[str] | None = None,
    start: float | None = None,
    duration: float | None = None,
) -> tuple[str, str, subprocess.CompletedProcess[str]]:
    """
    Run ffmpeg, streaming output to a temp file (not memory) to avoid huge RAM use
    on very long videos. Returns captured log text (from temp file) and the process.
    """
    # Skip audio/sub decoding to reduce RAM/CPU. Let ffmpeg decide thread usage.
    cmd: List[str] = [
        'ffmpeg', '-hide_banner', '-nostats', '-loglevel', 'error',
        '-nostdin',
        '-an', '-sn'
    ]
    if extra:
        cmd += extra
    if start is not None:
        cmd += ['-ss', f"{start:.3f}"]
    if duration is not None:
        cmd += ['-t', f"{duration:.3f}"]
    cmd += ['-i', input_file]
    if start is not None:
        cmd += ['-ss', f"{start:.3f}"]
    if duration is not None:
        cmd += ['-t', f"{duration:.3f}"]
    cmd += [
        '-i', encoded_file,
        '-filter_complex', filter_chain,
        '-f', 'null', '-'
    ]

    res_local = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    return res_local.stdout or "", res_local.stderr or "", res_local


def _parse_ssim(text: str) -> float | None:
    for line in text.splitlines():
        if 'All:' in line:
            try:
                return float(line.split('All:')[1].split()[0])
            except (ValueError, IndexError):
                return None
    return None


def _measure_ssim_once(
    input_file: str,
    encoded_file: str,
    filter_chain: str,
    extra: Sequence[str] | None = None,
    start: float | None = None,
    duration: float | None = None,
) -> Tuple[float | None, subprocess.CompletedProcess[str], str, str]:
    stdout_text, stderr_text, res_local = _run_ssim_ffmpeg(
        input_file=input_file,
        encoded_file=encoded_file,
        filter_chain=filter_chain,
        extra=extra,
        start=start,
        duration=duration,
    )
    val_local = _parse_ssim(stdout_text)
    return val_local, res_local, stdout_text, stderr_text


def _persist_full_ssim_logs(
    encoded_file: str,
    start: float | None,
    duration: float | None,
    log_primary: str,
    log_primary_err: str,
    log_fb: str,
    log_fb_err: str,
) -> str:
    log_file = tempfile.NamedTemporaryFile(
        prefix="full_ssim_",
        suffix=".log",
        delete=False,
        mode="w",
        encoding="utf-8"
    )
    if start is not None or duration is not None:
        log_file.write(f"WINDOW: start={start} duration={duration}\n")
    if log_primary:
        log_file.write("PRIMARY LOG:\n")
        log_file.write(log_primary)
        log_file.write("\n")
    if log_primary_err:
        log_file.write("PRIMARY STDERR:\n")
        log_file.write(log_primary_err)
        log_file.write("\n")
    if log_fb:
        log_file.write("FALLBACK LOG:\n")
        log_file.write(log_fb)
        log_file.write("\n")
    if log_fb_err:
        log_file.write("FALLBACK STDERR:\n")
        log_file.write(log_fb_err)
        log_file.write("\n")
    log_path = log_file.name
    log_file.close()

    logging.warning(
        "Could not parse full-file SSIM for %s; treating as unavailable. See log: %s",
        encoded_file, log_path
    )
    return log_path


def _measure_ssim_window(
    input_file: str,
    encoded_file: str,
    filter_chain: str,
    start: float | None = None,
    duration: float | None = None,
) -> float | None:
    val = None
    log_primary = ""
    log_primary_err = ""
    log_fb = ""
    log_fb_err = ""

    try:
        val, _, log_primary, log_primary_err = _measure_ssim_once(
            input_file=input_file,
            encoded_file=encoded_file,
            filter_chain=filter_chain,
            start=start,
            duration=duration,
        )
    except Exception as e:
        logging.warning(
            "Full-file SSIM command failed for %s (%s); retrying verbose.",
            encoded_file, e
        )

    if val is None or val <= 0:
        try:
            val_fb, _, log_fb, log_fb_err = _measure_ssim_once(
                input_file=input_file,
                encoded_file=encoded_file,
                filter_chain=filter_chain,
                extra=['-v', 'info'],
                start=start,
                duration=duration,
            )
            if val_fb is not None and val_fb > 0:
                return val_fb
        except Exception as e:
            logging.warning(
                "Fallback full-file SSIM failed for %s (%s).",
                encoded_file, e
            )

    if val is not None and val > 0:
        return val

    _persist_full_ssim_logs(
        encoded_file=encoded_file,
        start=start,
        duration=duration,
        log_primary=log_primary,
        log_primary_err=log_primary_err,
        log_fb=log_fb,
        log_fb_err=log_fb_err,
    )
    return None


def _measure_ssim_in_chunks(
    input_file: str,
    encoded_file: str,
    filter_chain: str,
    chunk_seconds: float,
) -> float | None:
    duration_total = probe_video_duration(input_file)
    if duration_total <= chunk_seconds:
        return None

    acc = 0.0
    weight = 0.0
    start = 0.0
    while start < duration_total:
        window = min(chunk_seconds, duration_total - start)
        val_window = _measure_ssim_window(
            input_file=input_file,
            encoded_file=encoded_file,
            filter_chain=filter_chain,
            start=start,
            duration=window,
        )
        if val_window is None:
            return None
        acc += val_window * window
        weight += window
        start += window

    if weight > 0:
        return acc / weight
    return None


def measure_full_ssim(
    input_file: str,
    encoded_file: str,
    raw_fr: float | None = None,
    chunk_seconds: float | None = None,
) -> float | None:
    """
    Measure full-file SSIM between input_file (reference) and encoded_file (distorted).
    Returns None if parsing fails or SSIM is invalid.
    """
    print("Measuring full-file SSIM...")
    filter_chain = _build_ssim_filter_chain(raw_fr)

    if chunk_seconds and chunk_seconds > 0:
        chunked = _measure_ssim_in_chunks(
            input_file=input_file,
            encoded_file=encoded_file,
            filter_chain=filter_chain,
            chunk_seconds=chunk_seconds,
        )
        if chunked is not None:
            return chunked

    return _measure_ssim_window(
        input_file=input_file,
        encoded_file=encoded_file,
        filter_chain=filter_chain,
    )


def build_hdr_sdr_filter(hdr: Dict[str, Any]) -> str | None:
    """
    Builds HDR->SDR tonemapping filter chain for ffmpeg, if needed.

    Handles:
      - PQ (HDR10, transfer=smpte2084)
      - HLG (transfer=arib-std-b67)
      - Fallback for other bt2020-tagged HDR-like streams
    """

    tr = hdr["transfer"]

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


def _get_resolution_label(path: str) -> str:
    """
    Return a standardized resolution label based on the video's actual dimensions.
    Maps to common standards: 480p, 720p, 1080p, 1440p, 2160p (4K), etc.
    
    Uses the larger dimension (usually height for landscape, width for portrait)
    to determine the closest standard resolution.
    """
    info = probe_video_stream_info(path)
    w = info["width"]
    h = info["height"]
    
    # Use the smaller dimension for classification (usually height in landscape)
    # This handles both landscape and portrait correctly
    min_dim = min(w, h)
    
    # Map to standard resolutions with some tolerance
    if min_dim <= 360:
        return "360p"
    elif min_dim <= 480:
        return "480p"
    elif min_dim <= 576:
        return "576p"  # PAL SD
    elif min_dim <= 720:
        return "720p"
    elif min_dim <= 1080:
        return "1080p"
    elif min_dim <= 1440:
        return "1440p"  # 2K
    elif min_dim <= 2160:
        return "2160p"  # 4K UHD
    elif min_dim <= 4320:
        return "4320p"  # 8K
    else:
        return f"{min_dim}p"  # Fallback for unusual resolutions


def encode_baseline(input_file: str, output_dir: str | None = None) -> str:
    """
    Step 0 + 1: Create the "baseline" file from the ORIGINAL source.

    - Detect HDR and apply HDR->SDR tonemapping if needed.
    - Normalize to SDR BT.709, yuv420p, SAR=1.
    - Attempt H.264 NVENC in constqp mode:
        - QP=0
        - preset=p7
        - bf=2
    - If NVENC fails, fall back to libx264 lossless-ish baseline:
        - QP=0
        - preset=veryslow
    """

    base, ext = os.path.splitext(os.path.basename(input_file))
    # Normalize container: keep mp4, otherwise use mkv to safely hold codecs.
    out_ext = ".mp4" if ext.lower() == ".mp4" else ".mkv"
    filename = f"{base} [baseline qp 0]{out_ext}"
    output = os.path.join(output_dir, filename) if output_dir else filename

    total_duration = probe_video_duration(input_file)
    low_res = _is_low_res(input_file)

    if low_res:
        logging.info("Low-resolution source detected → baseline via h264_nvenc (QP=0).")
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

    # GPU baseline (force NVENC even for low-res per user request)
    cmd = base_cmd + nvenc_opts
    cmd += [
        '-c:a', 'copy',
        '-c:s', 'copy',
        output
    ]

    try:
        run_ffmpeg_progress(cmd, total_duration, desc="Baseline Encode")
    except subprocess.CalledProcessError as err:
        logging.warning("NVENC baseline failed (%s). Falling back to libx264 QP=0.", err)
        cpu_cmd = base_cmd + [
            '-c:v', 'libx264',
            '-preset', 'veryslow',
            '-qp', '0',
            '-c:a', 'copy',
            '-c:s', 'copy',
            output
        ]
        run_ffmpeg_progress(cpu_cmd, total_duration, desc="Baseline Encode (CPU)")
    return output


def encode_final(
    input_file: str,
    qp: int,
    audio_opts: list[str],
    raw_fr: float,
    gop: int,
    return_ssim: bool = False,
    ssim_chunk_seconds: float | None = None,
    output_dir: str | None = None,
    output_base: str | None = None,
    output_ext: str | None = None,
) -> str | tuple[str, float | None]:
    """
    Final encode (from the baseline file), with FFmpeg progress and optional SSIM.

    - input_file here is the BASELINE file (already normalized to SDR, bt709).
    - Uses h264_nvenc -qp {qp} for all baselines (low-res included).
    """

    base_in, _ = os.path.splitext(os.path.basename(input_file))
    base = output_base or base_in
    # Default to MKV for final optimized output; allow override when explicitly provided.
    if output_ext:
        ext = output_ext if output_ext.startswith(".") else f".{output_ext}"
    else:
        ext = ".mkv"
    total_duration = probe_video_duration(input_file)
    low_res = _is_low_res(input_file)

    # Get resolution label and format framerate
    resolution = _get_resolution_label(input_file)
    fps_int = int(round(raw_fr))
    
    encoder_tag = "h264_nvenc"
    # New format: [codec resolution qp XX].source.ext
    filename = f"{base} [{encoder_tag} {resolution}{fps_int} qp {qp}].source{ext}"
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
        logging.info("Low-resolution baseline → final encode via h264_nvenc (QP=%d).", qp)
    else:
        logging.info("High-resolution baseline → final encode via h264_nvenc (QP=%d).", qp)

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
    run_ffmpeg_progress(cmd, total_duration, desc=f"Final File Encode (QP={qp})")

    if return_ssim:
        ssim_val = measure_full_ssim(
            input_file,
            output,
            raw_fr=raw_fr,
            chunk_seconds=ssim_chunk_seconds,
        )
        if ssim_val is not None:
            print(f"Full-file SSIM at QP {qp}: {ssim_val:.4f}")
        else:
            print(f"Full-file SSIM at QP {qp}: unavailable")
        return output, ssim_val

    return output
